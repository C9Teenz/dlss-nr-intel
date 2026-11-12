"""Run an exported HNet ONNX in ONNX Runtime and compare it with the PyTorch port.

    python tools/hnet/check_onnx.py work/onnx/hnet_320x320_vk-pwl_fp32.onnx --provider cpu
    python tools/hnet/check_onnx.py work/onnx/hnet_1920x1152_vk-pwl_fp16.onnx --provider dml --runs 20

The reference is the PyTorch port in the same deploy numerics, run in float32 (and, with
--exact, also the exact numerics — what the vendor computes). The features come from a
synthetic test frame (or --image) through frame.make_features, exactly as the HLSL pre-pass
builds them. With --provider dml it also times the session (`Run` wall time with the
inputs already on the device via IOBinding).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import frame  # noqa: E402
from model import HNet, Numerics  # noqa: E402

NAME = re.compile(r"hnet_(\d+)x(\d+)_(vk|nq)-(pwl|exp)_(fp16|fp32)(_synthetic)?\.onnx$")


def session(path, provider):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "dml":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    started = time.perf_counter()
    sess = ort.InferenceSession(str(path), options, providers=providers)
    print(f"  session ({provider}) created in {time.perf_counter() - started:.1f} s; "
          f"providers {sess.get_providers()}", flush=True)
    return sess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--provider", choices=("cpu", "dml"), default="cpu")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--image", type=Path, help="test frame (default: synthetic)")
    parser.add_argument("--exact", action="store_true", help="also compare with the exact numerics")
    parser.add_argument("--no-reference", action="store_true", help="timing only")
    args = parser.parse_args()

    match = NAME.search(args.onnx.name)
    if not match:
        raise SystemExit("file name must be export_onnx.py's hnet_<W>x<H>_<e4m3>-<softmax>_<dtype>.onnx")
    field_w, field_h = int(match[1]), int(match[2])
    e4m3 = {"vk": "veltkamp", "nq": "none"}[match[3]]
    softmax, dtype, synthetic = match[4], match[5], bool(match[6])

    if args.image:
        color = frame.load_image(args.image)
        geometry = frame.network_geometry(color.shape[1], color.shape[0])
        if (geometry.field_width, geometry.field_height) != (field_w, field_h):
            raise SystemExit(f"image field {geometry.field_width}x{geometry.field_height} != onnx field")
    else:
        color = frame.test_image(field_w, field_h)
        geometry = frame.Geometry(field_w, field_h, field_w, field_h)
    features = frame.make_features(color, geometry)[None]
    np_dtype = np.float16 if dtype == "fp16" else np.float32

    sess = session(args.onnx, args.provider)
    feed = features.astype(np_dtype)
    if args.provider == "dml":
        binding = sess.io_binding()
        device_input = ort.OrtValue.ortvalue_from_numpy(feed, "dml", 0)
        device_output = ort.OrtValue.ortvalue_from_shape_and_type((1, field_h, field_w, 4), np_dtype, "dml", 0)
        binding.bind_ortvalue_input("features", device_input)
        binding.bind_ortvalue_output("head", device_output)
        sess.run_with_iobinding(binding)                   # warm-up (graph build)
        times = []
        for _ in range(args.runs):
            started = time.perf_counter()
            sess.run_with_iobinding(binding)
            binding.synchronize_outputs()
            times.append((time.perf_counter() - started) * 1000.0)
        head = device_output.numpy().astype(np.float32)[0]
    else:
        times = []
        for _ in range(max(1, args.runs)):
            started = time.perf_counter()
            head = sess.run(["head"], {"features": feed})[0].astype(np.float32)[0]
            times.append((time.perf_counter() - started) * 1000.0)
    times = np.array(times)
    print(f"  {args.provider} run: median {np.median(times):.2f} ms, min {times.min():.2f} ms "
          f"over {len(times)} runs at field {field_w}x{field_h} ({dtype})", flush=True)
    if not np.isfinite(head).all():
        print("  FAIL: non-finite values in the head")
        return 1
    if args.no_reference:
        return 0

    weights = common.get_weights("synthetic" if synthetic else "real")
    references = [("deploy", Numerics.deploy(e4m3=e4m3, softmax=softmax))]
    if args.exact:
        references.append(("exact", Numerics.exact()))
    ok = True
    for label, numerics in references:
        with torch.no_grad():
            want = HNet(weights, numerics)(torch.from_numpy(features))[0].numpy()
        diff = np.abs(want - head)
        corr = float(np.corrcoef(want.ravel(), head.ravel())[0, 1])
        out_want, _ = frame.compose(want, color)
        out_got, _ = frame.compose(head, color)
        p = frame.psnr(out_want, out_got)
        print(f"  vs torch {label:<6}: head max |d| {diff.max():.4f}, mean {diff.mean():.5f}, "
              f"corr {corr:.5f}; composed PSNR {p:.2f} dB")
        if label == "deploy":
            limit = 45.0 if dtype == "fp16" else 70.0
            ok = ok and p >= limit
            print(f"  [{'PASS' if p >= limit else 'FAIL'}] composed PSNR >= {limit} dB against the port")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
