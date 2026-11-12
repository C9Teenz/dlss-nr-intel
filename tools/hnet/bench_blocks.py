"""Time single HNet blocks on the DirectML EP, to see where a field's time goes.

    python tools/hnet/bench_blocks.py --field 320x320 [--softmax pwl|exp] [--e4m3 veltkamp|none]

Each case is one block (synthetic weights, deploy numerics, fp16) exported at the shape it
meets in the given field, run with IOBinding; the table multiplies by the number of such
blocks in the graph to estimate the share of the whole.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from model import HNet, Numerics  # noqa: E402

ort.set_default_logger_severity(3)


class Block(nn.Module):
    def __init__(self, net, kind, index, heads):
        super().__init__()
        self.net, self.kind, self.index, self.heads = net, kind, index, heads

    def forward(self, x):
        if self.kind == "window":
            return self.net.window_block(x, self.index, heads=self.heads)
        if self.kind == "split":
            return self.net.split_window_block(x, self.index)
        return self.net.global_block(x, self.index)


def time_onnx(path, shape, runs):
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(str(path), so, providers=["DmlExecutionProvider"])
    out_shape = shape
    x = ort.OrtValue.ortvalue_from_numpy(np.random.default_rng(0).normal(0, 0.5, shape).astype(np.float16), "dml", 0)
    y = ort.OrtValue.ortvalue_from_shape_and_type(out_shape, np.float16, "dml", 0)
    binding = sess.io_binding()
    binding.bind_ortvalue_input(sess.get_inputs()[0].name, x)
    binding.bind_ortvalue_output(sess.get_outputs()[0].name, y)
    for _ in range(3):
        sess.run_with_iobinding(binding)
    binding.synchronize_outputs()
    # back to back, as in the whole graph: a sync after every run measures the GPU's ramp
    times = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(runs):
            sess.run_with_iobinding(binding)
        binding.synchronize_outputs()
        times.append((time.perf_counter() - started) * 1000 / runs)
    return float(np.median(times)), float(np.min(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", default="320x320")
    parser.add_argument("--softmax", default="pwl")
    parser.add_argument("--e4m3", default="veltkamp")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    fw, fh = (int(v) for v in args.field.split("x"))
    net = HNet(common.synthetic_weights(), Numerics.deploy(args.e4m3, args.softmax)).half().eval()

    def level(div):
        return -(-fh // div), -(-fw // div)

    # (label, kind, block index, heads, (H, W, C), how many such blocks in the graph)
    cases = [
        ("L0 window 32ch", "window", 0, 1, (*level(1), 32), 2),
        ("L1 window 32ch", "window", 1, 1, (*level(2), 32), 8),
        ("L2 branched 64ch", "window", 5, 2, (*level(4), 64), 8),
        ("L3 branched 128ch", "window", 9, 4, (*level(8), 128), 12),
        ("L4 branched 256ch", "window", 15, 8, (*level(16), 256), 16),
        ("L5 split 512ch", "split", 23, 16, (*level(32), 512), 16),
        ("ViT 1024ch", "global", 31, 32, (-(-level(32)[0] // 16) * 8 // 8, -(-level(32)[1] // 16) * 8 // 8, 1024), 8),
    ]
    # the ViT runs on block 30's output padded to 8 and pooled
    h5, w5 = level(32)
    cases[-1] = ("ViT 1024ch", "global", 31, 32, ((-(-h5 // 8) * 8) // 2, (-(-w5 // 8) * 8) // 2, 1024), 8)

    total = 0.0
    print(f"field {fw}x{fh}, e4m3={args.e4m3}, softmax={args.softmax}")
    with tempfile.TemporaryDirectory() as tmp:
        for label, kind, index, heads, (h, w, c), count in cases:
            module = Block(net, kind, index, heads).eval()
            example = torch.zeros(1, h, w, c, dtype=torch.float16)
            path = Path(tmp) / f"{label.replace(' ', '_')}.onnx"
            with torch.no_grad():
                torch.onnx.export(module, (example,), dynamo=True, input_names=["x"], output_names=["y"],
                                  opset_version=20, optimize=True, verbose=False).save(str(path))
            median, best = time_onnx(path, (1, h, w, c), args.runs)
            total += median * count
            print(f"  {label:<20} {h:4d}x{w:<4d} x{count:<3d} median {median:7.2f} ms (min {best:6.2f})  "
                  f"-> {median * count:8.2f} ms", flush=True)
    print(f"estimated sum over the graph: {total:.1f} ms")


if __name__ == "__main__":
    main()
