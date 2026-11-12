"""Export HNet (deploy numerics) to ONNX for one network field.

    python tools/hnet/export_onnx.py --frame 1920x1080                # field from the vendor rule
    python tools/hnet/export_onnx.py --field 320x320 --dtype fp32     # for the ORT-CPU check
    python tools/hnet/export_onnx.py --frame 960x540 --weights synthetic --softmax exp

The graph is exported with a static field: DirectML plans and fuses best on fixed shapes,
and the field set a game meets is small (one per swapchain size and render scale). The
file name carries the field, numerics and dtype, e.g. work/onnx/hnet_1920x1152_vk-pwl_fp16.onnx;
the C++ runner (src/neural) looks it up by that name.

Input  `features` (1, H, W, 16)  — see tools/hnet/frame.py make_features / neural_pre.hlsl
Output `head`     (1, H, W, 4)   — residual RGB and the temporal gate logit

ONNX files made from real weights embed them: they live in work/ and are never committed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import frame  # noqa: E402
from model import HNet, HNetMiddle, Numerics  # noqa: E402

E4M3_TAGS = {"veltkamp": "vk", "none": "nq", "exact": "ex"}


def parse_extent(text):
    width, height = (int(v) for v in text.lower().split("x"))
    return width, height


def onnx_name(field_w, field_h, e4m3, softmax, dtype, weights="real", part="full"):
    tag = f"{E4M3_TAGS[e4m3]}-{softmax}"
    synthetic = "_synthetic" if weights == "synthetic" else ""
    stem = {"full": "hnet", "middle": "hnet-mid", "middle-deep": "hnet-mid23"}[part]
    return f"{stem}_{field_w}x{field_h}_{tag}_{dtype}{synthetic}.onnx"


def export(weights, field_w, field_h, numerics, dtype, out_path: Path, part="full"):
    """`full`: features (1, H, W, 16) -> head (1, H, W, 4). `middle`: blocks 5-65 only,
    value (1, H/4, W/4, 64) -> value_out (same shape), for the hybrid runner whose HLSL
    kernels run blocks 0-4 and 66-70 (src/compute/neural_blocks.hlsl)."""
    net = HNet(weights, numerics).eval()
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    net = net.to(torch_dtype)
    if part == "middle":
        model = HNetMiddle(net, "shallow").eval()
        example = torch.zeros(1, field_h // 4, field_w // 4, 64, dtype=torch_dtype)
        names = (["value"], ["value_out"])
    elif part == "middle-deep":
        # blocks 23-47 at 1/32: block 22 pads its 1/16 output to a multiple of 8 before pooling
        model = HNetMiddle(net, "deep").eval()
        h5, w5 = -(-(field_h // 16) // 8) * 4, -(-(field_w // 16) // 8) * 4
        example = torch.zeros(1, h5, w5, 512, dtype=torch_dtype)
        names = (["value"], ["value_out"])
    else:
        model = net
        example = torch.zeros(1, field_h, field_w, 16, dtype=torch_dtype)
        names = (["features"], ["head"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.no_grad():
        program = torch.onnx.export(
            model, (example,), dynamo=True, input_names=names[0], output_names=names[1],
            opset_version=20, optimize=True, external_data=False)
    program.save(str(out_path))
    print(f"exported {out_path} ({out_path.stat().st_size / 2**20:.1f} MiB) "
          f"in {time.perf_counter() - started:.1f} s", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame", help="frame WxH handed to the network; the field follows the vendor rule")
    group.add_argument("--field", help="network field WxH directly (multiples of 64)")
    parser.add_argument("--weights", choices=("real", "synthetic"), default="real")
    parser.add_argument("--e4m3", choices=("veltkamp", "none"), default="veltkamp")
    parser.add_argument("--softmax", choices=("pwl", "exp"), default="pwl")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--part", choices=("full", "middle", "middle-deep"), default="full",
                        help="middle: blocks 5-65, middle-deep: blocks 23-47, for the hybrid runner "
                             "(needs --e4m3 none --softmax exp)")
    parser.add_argument("--out", type=Path, help="output path (default work/onnx/<name>)")
    args = parser.parse_args()

    if args.frame:
        geometry = frame.network_geometry(*parse_extent(args.frame))
        field_w, field_h = geometry.field_width, geometry.field_height
        print(f"frame {args.frame} -> field {field_w}x{field_h}")
    else:
        field_w, field_h = parse_extent(args.field)
        if field_w % 64 or field_h % 64:
            raise SystemExit("field sides must be multiples of 64")
    if args.part != "full" and (args.e4m3, args.softmax) != ("none", "exp"):
        raise SystemExit("the hybrid runner's HLSL ends implement nq-exp numerics: use --e4m3 none --softmax exp")
    numerics = Numerics.deploy(e4m3=args.e4m3, softmax=args.softmax)
    out = args.out or common.ONNX_DIR / onnx_name(field_w, field_h, args.e4m3, args.softmax,
                                                  args.dtype, args.weights, args.part)
    export(common.get_weights(args.weights), field_w, field_h, numerics, args.dtype, out, args.part)


if __name__ == "__main__":
    main()
