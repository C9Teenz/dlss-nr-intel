"""Per-node profile of an HNet ONNX on the DirectML EP (ORT's own profiler).

    python tools/hnet/profile_onnx.py work/onnx/hnet_320x320_vk-pwl_fp16_synthetic.onnx [--top 30]

Prints the time per op type and the slowest nodes, and which nodes ran on the CPU EP
(each of those is a device round trip).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    w, h = (int(v) for v in re.search(r"hnet_(\d+)x(\d+)", args.onnx.name).groups())
    dtype = np.float16 if "fp16" in args.onnx.name else np.float32

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.profile_file_prefix = str(Path("work") / "ort_profile")
    sess = ort.InferenceSession(str(args.onnx), options,
                                providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    feed = {"features": np.zeros((1, h, w, 16), dtype)}
    for _ in range(args.runs):
        sess.run(["head"], feed)
    profile = sess.end_profiling()
    events = json.loads(Path(profile).read_text())
    nodes = [e for e in events if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]
    runs = [e for e in events if e.get("name") == "model_run"]
    last_run = runs[-1]
    start, end = last_run["ts"], last_run["ts"] + last_run["dur"]
    nodes = [e for e in nodes if start <= e["ts"] <= end]
    total = sum(e["dur"] for e in nodes)
    print(f"model_run {last_run['dur'] / 1000:.2f} ms, node kernel time {total / 1000:.2f} ms, {len(nodes)} nodes")
    by_type = collections.Counter()
    by_provider = collections.Counter()
    for e in nodes:
        by_type[e["args"].get("op_name", "?")] += e["dur"]
        by_provider[e["args"].get("provider", "?")] += e["dur"]
    print("by provider:", {k: f"{v / 1000:.2f} ms" for k, v in by_provider.items()})
    print("by op type:")
    for op, dur in by_type.most_common(15):
        print(f"  {op:<22} {dur / 1000:9.2f} ms")
    print("slowest nodes:")
    for e in sorted(nodes, key=lambda e: -e["dur"])[: args.top]:
        a = e["args"]
        print(f"  {e['dur'] / 1000:8.2f} ms  {a.get('op_name')}  {a.get('provider')}  {e['name'][:60]}  "
              f"in {a.get('input_type_shape')}")
    cpu = [e for e in nodes if e["args"].get("provider") == "CPUExecutionProvider"]
    print(f"{len(cpu)} nodes on the CPU EP")


if __name__ == "__main__":
    main()
