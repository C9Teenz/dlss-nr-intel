"""Does the DML EP's fused com.microsoft.MultiHeadAttention beat the decomposed window
attention? Times both on level-0 window shapes for a field.

    python tools/hnet/bench_mha.py --field 320x320 [--heads 1]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ort.set_default_logger_severity(3)


def mha_model(windows, heads, dim, with_bias):
    c = heads * dim
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT16, [windows, 64, c]) for n in ("q", "k", "v")]
    node_inputs = ["q", "k", "v", "", ""]
    if with_bias:
        inputs.append(helper.make_tensor_value_info("bias", TensorProto.FLOAT16, [1, heads, 64, 64]))
        node_inputs.append("bias")
    node = helper.make_node("MultiHeadAttention", node_inputs, ["y"], domain="com.microsoft",
                            num_heads=heads, scale=1.0)
    graph = helper.make_graph([node], "mha", inputs,
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [windows, 64, c])])
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)])


def decomposed_model(windows, heads, dim):
    c = heads * dim
    nodes, inits = [], []
    shape = helper.make_tensor("shape", TensorProto.INT64, [4], [windows, 64, heads, dim])
    back = helper.make_tensor("back", TensorProto.INT64, [3], [windows, 64, c])
    inits += [shape, back]
    for n in ("q", "k", "v"):
        nodes.append(helper.make_node("Reshape", [n, "shape"], [n + "r"]))
        nodes.append(helper.make_node("Transpose", [n + "r"], [n + "t"], perm=[0, 2, 1, 3] if n != "k" else [0, 2, 3, 1]))
    nodes += [helper.make_node("MatMul", ["qt", "kt"], ["s"]),
              helper.make_node("Add", ["s", "bias"], ["sb"]),
              helper.make_node("Softmax", ["sb"], ["p"], axis=-1),
              helper.make_node("MatMul", ["p", "vt"], ["o"]),
              helper.make_node("Transpose", ["o"], ["ot"], perm=[0, 2, 1, 3]),
              helper.make_node("Reshape", ["ot", "back"], ["y"])]
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT16, [windows, 64, c]) for n in ("q", "k", "v")]
    inputs.append(helper.make_tensor_value_info("bias", TensorProto.FLOAT16, [1, heads, 64, 64]))
    graph = helper.make_graph(nodes, "dec", inputs,
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [windows, 64, c])], inits)
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 20)])


def bench(model, feeds, runs=20):
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["DmlExecutionProvider"])
    binding = sess.io_binding()
    for name, value in feeds.items():
        if name in [i.name for i in sess.get_inputs()]:
            binding.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(value, "dml", 0))
    binding.bind_output("y", "dml")
    for _ in range(3):
        sess.run_with_iobinding(binding)
    binding.synchronize_outputs()
    started = time.perf_counter()
    for _ in range(runs):
        sess.run_with_iobinding(binding)
    binding.synchronize_outputs()
    out = binding.get_outputs()[0].numpy()
    return (time.perf_counter() - started) * 1000 / runs, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", default="320x320")
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--div", type=int, default=1, help="level divisor (1 = level 0)")
    args = parser.parse_args()
    w, h = (int(v) for v in args.field.split("x"))
    windows = (h // args.div // 8) * (w // args.div // 8)
    rng = np.random.default_rng(0)
    c = args.heads * 32
    feeds = {n: rng.normal(0, 0.3, (windows, 64, c)).astype(np.float16) for n in ("q", "k", "v")}
    feeds["bias"] = rng.normal(0, 0.3, (1, args.heads, 64, 64)).astype(np.float16)
    t_dec, y_dec = bench(decomposed_model(windows, args.heads, 32), feeds)
    print(f"{windows} windows x {args.heads} heads: decomposed {t_dec:.3f} ms")
    for with_bias in (False, True):
        try:
            t, y = bench(mha_model(windows, args.heads, 32, with_bias), feeds)
            agree = f", max |d| vs decomposed {np.abs(y.astype(np.float32) - y_dec.astype(np.float32)).max():.4f}" if with_bias else ""
            print(f"  MultiHeadAttention{' + bias' if with_bias else ''}: {t:.3f} ms{agree}")
        except Exception as exc:  # noqa: BLE001
            print(f"  MultiHeadAttention{' + bias' if with_bias else ''}: FAILED {str(exc).splitlines()[0][:160]}")


if __name__ == "__main__":
    main()
