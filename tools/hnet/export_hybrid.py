"""Weights and reference data for the hybrid runner's HLSL ends (blocks 0-4 and 66-70).

    python tools/hnet/export_hybrid.py --weights synthetic                 # weights blob only
    python tools/hnet/export_hybrid.py --weights synthetic --dump 320x320  # + reference tensors

Weights: work/onnx/hnet-ends_nq-exp[_synthetic].bin — the ten 32-channel window blocks and the
glue around them (and, since the deep split and the native runner, every other block), in the deploy layout the kernels read (attention bias in logical
order, softmax base folded into scale and bias). Every matrix the kernels contract row by row
is stored transposed, (out, in), so an output is one contiguous dot product; `weight2` stays
(128, 32) because the FFN accumulates it one hidden unit (row) at a time. QKV is (96, 32),
rows [q 0-31 | k 32-63 | v 64-95]:

    "HNWB" u32 version=2 u32 count, then count x {char name[64]; u32 dtype (0 float32, 1 float16);
    u32 elements; u64 offset}, then the data (offsets in bytes from its start, 16-byte aligned).

The matrices only the DirectML GEMMs read (blocks 5-65, `gemm_only`) are float16, the precision
the GEMMs read them in anyway: the file is half the size. Everything else stays float32. The
runner still loads version 1 (all float32; entries {name; u32 offset in floats; u32 floats}).

Reference (--dump FIELD [--split deep]): work/hybrid/<W>x<H>_<split>[_synthetic]/ with raw tensors
(NHWC, batch dropped) computed by HNet in the native runner's numerics — exact E4M3 publishes,
the exp softmax, plain cosine norm (Numerics.deploy("exact", "exp")) — float32:
    features.f16              (H, W, 16) half — the kernels' input
    front_value.f32           (H/4, W/4, 64)   after block 4's downsample
    front_skip<i>.f32         encoder level i's skip (0: block 4 at 1/2 ... 3: block 22 at 1/16)
    front_full_skip.f32       (H, W, 32)       block 0's output
    block<N>.f32              every end block's output, for debugging
    middle_value.f32          (H/4, W/4, 64)   after block 65 (what ORT hands back)
    head.f32                  (H, W, 4)
Files made from real weights hold them: they stay in work/.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import frame  # noqa: E402
from model import HNet, Numerics  # noqa: E402

END_BLOCKS = (0, 1, 2, 3, 4, 66, 67, 68, 69, 70)
PER_BLOCK = ("weight1", "weight2", "ffn_cos_skip", "qkv_heads", "attn_scale", "attn_bias",
             "projection_weight", "attn_cos_skip")
EXTRA = ("block0.layer0.input_adapter_weight", "block4.layer0.weight0", "block66.layer0.weight0",
         "block66.layer0.sin", "block70.layer0.inp_merge_sin", "block70.layer0.inp_merge_cos",
         "block70.layer0.out_gain", "block70.layer0.out_conv_weight")
# the deep split's HLSL also runs the branched blocks (C = 64, 128, 256) and their transitions
WIDE_BLOCKS = tuple(range(5, 23)) + tuple(range(48, 66))
WIDE_PER_BLOCK = ("fused_expand", "fused_project", "ffn_output_projection_weight", "ffn_cos_skip",
                  "qkv_heads", "attn_scale", "attn_bias", "projection_weight", "attn_cos_skip")
WIDE_EXTRA = tuple(f"block{b}.layer0.weight0" for b in (8, 14, 22, 48, 56, 62)) + tuple(
    f"block{b}.layer0.sin" for b in (48, 56, 62))
# the native runner also runs the bottleneck (blocks 23-47) itself: the 512-channel split blocks,
# block 30's transition, the ViT and block 39 (GEMMs in DirectML, the rest in HLSL)
SPLIT_BLOCKS = tuple(range(23, 31)) + tuple(range(40, 48))
SPLIT_ROLES = ("layer0.first_projection_weight", "layer0.group_expand_weight", "layer0.group_project_weight",
               "layer1.weight3", "layer1.ffn_cos_skip", "layer2.qkv_heads", "layer2.attn_scale",
               "layer2.attn_bias", "layer3.projection_weight", "layer3.attn_cos_skip")
VIT_ROLES = ("layer0.weight", "layer1.weight", "layer1.ffn_cos_skip", "layer2.qkv_weight", "layer2.attn_scale",
             "layer4.projection_weight", "layer4.attn_cos_skip")
CORE_EXTRA = ("block30.layer4.weight", "block39.layer0.conv_weight", "block39.layer0.inp_upsample_sin")
DUMP_BLOCKS = END_BLOCKS + WIDE_BLOCKS + SPLIT_BLOCKS + tuple(range(31, 40))


GEMM_MATRICES = ("fused_expand", "fused_project", "projection_weight", "weight1", "weight2", "qkv_heads",
                 "qkv_weight", "weight3", "layer0.weight", "layer1.weight", "group_expand_weight",
                 "group_project_weight")


def gemm_only(name):
    """A matrix only a DirectML GEMM reads (HybridEnds::LoadWeights' rule): the 32-channel end
    blocks' kernels read theirs in float."""
    block = int(name[5:name.index(".")])
    return 5 <= block <= 65 and name.endswith(GEMM_MATRICES)


def blob_name(synthetic):
    return f"hnet-ends_nq-exp{'_synthetic' if synthetic else ''}.bin"


def write_blob(net: HNet, path: Path):
    names = [f"block{b}.layer0.{role}" for b in END_BLOCKS for role in PER_BLOCK] + list(EXTRA)
    names += [f"block{b}.layer0.{role}" for b in WIDE_BLOCKS for role in WIDE_PER_BLOCK] + list(WIDE_EXTRA)
    names += [f"block{b}.{role}" for b in SPLIT_BLOCKS for role in SPLIT_ROLES]
    names += [f"block{b}.{role}" for b in range(31, 39) for role in VIT_ROLES] + list(CORE_EXTRA)
    arrays = []
    for name in names:
        value = net.w(name).detach().float().numpy()
        if name.endswith("qkv_heads"):
            value = value.transpose(0, 2, 1)                                   # (H, C, 96) -> (H, 96, C)
            if value.shape[0] == 1:
                value = value[0]
        elif value.ndim == 2 and not name.endswith(("weight2", "attn_bias")):
            value = value.T                                                    # (in, out) -> (out, in)
        arrays.append((name, np.ascontiguousarray(value, dtype="<f2" if gemm_only(name) else "<f4")))
    header = bytearray(b"HNWB" + struct.pack("<II", 2, len(arrays)))
    offset = 0
    for name, value in arrays:
        encoded = name.encode("ascii")
        if len(encoded) >= 64:
            raise SystemExit(f"name too long: {name}")
        header += encoded.ljust(64, b"\0") + struct.pack("<IIQ", int(value.dtype == np.float16), value.size, offset)
        offset += (value.nbytes + 15) // 16 * 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(header)
        for _, value in arrays:
            out.write(value.tobytes().ljust((value.nbytes + 15) // 16 * 16, b"\0"))
    halves = sum(value.size for _, value in arrays if value.dtype == np.float16)
    total = sum(value.size for _, value in arrays)
    print(f"wrote {path} ({len(arrays)} tensors, {total} values, {halves} of them float16, {offset} bytes)")


def dump(net: HNet, field_w, field_h, out_dir: Path, split="shallow"):
    """Reference tensors of one split: front_value / middle_value are the middle's input and
    output, front_skip<i> the encoder skips the front hands the back."""
    out_dir.mkdir(parents=True, exist_ok=True)
    color = frame.test_image(field_w, field_h)
    geometry = frame.Geometry(field_w, field_h, field_w, field_h)
    features = frame.make_features(color, geometry).astype(np.float16)
    capture = {}
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32))[None]
        value, skips, full_skip = net.forward_front(x, capture, split=split)
        middle = net.forward_middle(value, split=split)
        head = net.forward_back(middle, skips, full_skip, capture, split=split)
    features.tofile(out_dir / "features.f16")
    tensors = {"front_value": value, "front_full_skip": full_skip, "middle_value": middle, "head": head}
    tensors.update({f"front_skip{i}": skip for i, skip in enumerate(skips)})
    tensors.update({name: capture[name] for name in (f"block{b}" for b in DUMP_BLOCKS) if name in capture})
    for name, tensor in tensors.items():
        array = tensor[0].numpy().astype("<f4")
        array.tofile(out_dir / f"{name}.f32")
        print(f"  {name:<16} {tuple(array.shape)}")
    color.astype("<f4").tofile(out_dir / "color.f32")
    print(f"dumped reference tensors to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", choices=("real", "synthetic"), default="real")
    parser.add_argument("--dump", help="also dump reference tensors for a field WxH")
    parser.add_argument("--split", choices=("shallow", "deep"), default="shallow")
    args = parser.parse_args()
    synthetic = args.weights == "synthetic"
    net = HNet(common.get_weights(args.weights), Numerics.deploy(e4m3="exact", softmax="exp")).eval()
    write_blob(net, common.ONNX_DIR / blob_name(synthetic))
    if args.dump:
        w, h = (int(v) for v in args.dump.split("x"))
        suffix = "_synthetic" if synthetic else ""
        dump(net, w, h, common.WORK / "hybrid" / f"{w}x{h}_{args.split}{suffix}", args.split)


if __name__ == "__main__":
    main()
