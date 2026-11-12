"""Microbenchmarks of HNet's op patterns on the DirectML EP (pipelined throughput).

    python tools/hnet/bench_ops.py [--field 320x320]

Each case is a tiny torch function exported to ONNX at the level-0 shapes of the field.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Numerics, Ops, partition_windows, reverse_windows  # noqa: E402

ort.set_default_logger_severity(3)
OPS = Ops(Numerics.deploy("veltkamp", "pwl"))
OPS_EXP = Ops(Numerics.deploy("none", "exp"))


class Fn(nn.Module):
    def __init__(self, fn, *consts):
        super().__init__()
        self.fn = fn
        for i, c in enumerate(consts):
            self.register_buffer(f"c{i}", c)
        self.n = len(consts)

    def forward(self, x):
        return self.fn(x, *[getattr(self, f"c{i}") for i in range(self.n)])


def run(path, shape, runs=20):
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(str(path), so, providers=["DmlExecutionProvider"])
    out = sess.get_outputs()[0]
    x = ort.OrtValue.ortvalue_from_numpy(np.random.default_rng(0).normal(0, 0.5, shape).astype(np.float16), "dml", 0)
    binding = sess.io_binding()
    binding.bind_ortvalue_input("x", x)
    binding.bind_output(out.name, "dml")
    for _ in range(3):
        sess.run_with_iobinding(binding)
    binding.synchronize_outputs()
    started = time.perf_counter()
    for _ in range(runs):
        sess.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return (time.perf_counter() - started) * 1000 / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", default="320x320")
    args = parser.parse_args()
    w, h = (int(v) for v in args.field.split("x"))
    windows = (h // 8) * (w // 8)
    rng = np.random.default_rng(1)
    wq = torch.from_numpy(rng.normal(0, 0.2, (32, 96)).astype(np.float16))
    w1 = torch.from_numpy(rng.normal(0, 0.2, (32, 128)).astype(np.float16))
    bias = torch.from_numpy(rng.normal(0, 0.2, (1, 1, 64, 64)).astype(np.float16))
    cases = [
        ("copy (x*1)", (1, h, w, 32), lambda x: x * 1.0),
        ("partition+reverse windows", (1, h, w, 32),
         lambda x: reverse_windows(partition_windows(x) * 1.0, 1, h, w)),
        ("veltkamp e4m3", (1, h, w, 32), lambda x: OPS.e4m3(x)),
        ("gate activation", (1, h, w, 128), lambda x: OPS.gate_activation(x)),
        ("matmul tokens 32->128", (1, h, w, 32), lambda x, m: x @ m, w1),
        ("matmul qkv 32->96 windows", (windows, 64, 32), lambda x, m: x @ m, wq),
        ("scores q@kT (W,1,64,32)", (windows, 1, 64, 32), lambda x: x @ x.transpose(-1, -2)),
        ("softmax pwl (W,1,64,64)", (windows, 1, 64, 64), lambda x: OPS.window_softmax(x)),
        ("softmax exp (W,1,64,64)", (windows, 1, 64, 64), lambda x: OPS_EXP.window_softmax(x)),
        ("scores+bias", (windows, 1, 64, 64), lambda x, b: x + b, bias),
        ("cosine normalize (W,1,64,32)", (windows, 1, 64, 32), lambda x: OPS.cosine_normalize(x)),
        ("pad 4 (shifted window)", (1, h, w, 32),
         lambda x: torch.nn.functional.pad(x, (0, 0, 4, 4, 4, 4))[:, 4:4 + h, 4:4 + w]),
        ("avg pool 2", (1, h, w, 32),
         lambda x: (x[:, 0::2, 0::2] + x[:, 1::2, 0::2] + x[:, 0::2, 1::2] + x[:, 1::2, 1::2]) * 0.25),
    ]
    print(f"field {w}x{h}: {windows} windows at level 0")
    with tempfile.TemporaryDirectory() as tmp:
        for label, shape, fn, *consts in cases:
            module = Fn(fn, *consts).eval()
            path = Path(tmp) / "op.onnx"
            with torch.no_grad():
                torch.onnx.export(module, (torch.zeros(shape, dtype=torch.float16),), dynamo=True,
                                  input_names=["x"], opset_version=20, optimize=True, verbose=False).save(str(path))
            ms = run(path, shape)
            mb = math.prod(shape) * 2 / 2**20
            print(f"  {label:<32} {str(shape):<22} {ms:8.3f} ms   ({mb:6.1f} MiB in, {mb / 1024 / (ms / 1000):6.1f} GiB/s)",
                  flush=True)


if __name__ == "__main__":
    main()
