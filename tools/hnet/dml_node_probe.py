"""Find the nodes of an ONNX graph the DirectML EP refuses: each distinct node signature
(op, attributes, input shapes/types) is cut out into a one-node model and loaded alone.

    python tools/hnet/dml_node_probe.py work/onnx/hnet_192x128_vk-pwl_fp16_synthetic.onnx
"""
from __future__ import annotations

import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

ort.set_default_logger_severity(4)


def main(path):
    model = onnx.load(path)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    info = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
    inits = {i.name: i for i in graph.initializer}
    consts = {}
    for node in graph.node:
        if node.op_type == "Constant":
            consts[node.output[0]] = node
    seen = {}
    for node in graph.node:
        if node.op_type == "Constant":
            continue
        sig_inputs = []
        for name in node.input:
            if name in inits:
                t = inits[name]
                small = int(np.prod(t.dims)) <= 16
                sig_inputs.append(("init", tuple(t.dims), t.data_type,
                                   numpy_helper.to_array(t).tobytes() if small else None))
            elif name in info:
                tt = info[name].type.tensor_type
                sig_inputs.append(("value", tuple(d.dim_value for d in tt.shape.dim), tt.elem_type))
            else:
                sig_inputs.append(("?", name))
        signature = (node.op_type, str(node.attribute), tuple(sig_inputs))
        if signature in seen:
            seen[signature][1] += 1
            continue
        seen[signature] = [node, 1]

    bad = 0
    for signature, (node, count) in seen.items():
        inputs, initializers, nodes = [], [], []
        for name in node.input:
            if not name:
                continue
            if name in inits:
                initializers.append(inits[name])
            elif name in consts:
                nodes.append(consts[name])
            elif name in info and not any(i.name == name for i in inputs):
                inputs.append(info[name])
        outputs = [info[o] if o in info else helper.make_tensor_value_info(o, onnx.TensorProto.UNDEFINED, None)
                   for o in node.output]
        g = helper.make_graph(nodes + [node], "probe", inputs, outputs, initializers)
        m = helper.make_model(g, opset_imports=model.opset_import)
        m.ir_version = model.ir_version
        try:
            ort.InferenceSession(m.SerializeToString(), providers=["DmlExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            bad += 1
            shapes = [s[1] if len(s) > 1 else s for s in signature[2]]
            print(f"REFUSED x{count}: {node.op_type} {node.name} inputs {shapes} "
                  f"attrs {[(a.name, helper.get_attribute_value(a)) for a in node.attribute]}\n"
                  f"   {str(exc).splitlines()[0][:200]}")
    print(f"{len(seen)} distinct node signatures, {bad} refused")


if __name__ == "__main__":
    main(sys.argv[1])
