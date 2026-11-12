"""Paths, the logical weight spec and weight loading shared by the HNet tools.

Nothing here contains NVIDIA data. The real weights are extracted locally by
`tools/get_weights.ps1` into `work/mlxw/dlssnr-logical.safetensors` and never leave
`work/`. `synthetic_weights()` builds stand-ins with the same 649 names and shapes so
the port can be checked before (or without) the real ones.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
MLX_DLSS = WORK / "mlx-dlss"
MLX_DLSS_PIN = "06a3e11a8b68817127406ace5c764463543f699b"
WEIGHT_SPEC = MLX_DLSS / "python" / "mlxdlss" / "weight_spec.json"
LOGICAL_WEIGHTS = WORK / "mlxw" / "dlssnr-logical.safetensors"
GOLDEN_DIR = WORK / "golden"
ONNX_DIR = WORK / "onnx"

# The numpy reference (Apache-2.0, uzbekunknown/dlss-nr-on-intel): the vendor graph,
# every rounding point kept. This is the oracle the PyTorch port is checked against.
REFERENCE_SRC = ROOT / "third_party" / "refs" / "dlss-nr-on-intel" / "src" / "ref"

EXPECTED_TENSORS = 649
EXPECTED_PARAMETERS = 145_755_123


def import_reference_model(chunk_tokens: int = 0):
    """Import the reference `nr_model` with chunking set before its module constants bind."""
    os.environ["NR_CHUNK_TOKENS"] = str(chunk_tokens)
    os.environ.setdefault("NR_MLX_DLSS_GRAPH", "0")
    if not (REFERENCE_SRC / "nr_model.py").is_file():
        raise SystemExit(f"missing reference model at {REFERENCE_SRC / 'nr_model.py'}")
    if str(REFERENCE_SRC) not in sys.path:
        sys.path.insert(0, str(REFERENCE_SRC))
    import nr_model  # noqa: E402
    return nr_model


def load_spec() -> dict:
    if not WEIGHT_SPEC.is_file():
        raise SystemExit(
            f"missing {WEIGHT_SPEC}\n  git clone https://github.com/iamwavecut/MLX-DLSS {MLX_DLSS}"
            f" && git -C {MLX_DLSS} checkout {MLX_DLSS_PIN}")
    spec = json.loads(WEIGHT_SPEC.read_text())
    tensors = spec["tensors"]
    if len(tensors) != EXPECTED_TENSORS:
        raise SystemExit(f"weight spec lists {len(tensors)} tensors, expected {EXPECTED_TENSORS}")
    return tensors


def parameter_count(tensors: dict) -> int:
    return sum(int(np.prod(entry["shape"])) for entry in tensors.values())


def _role(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def synthetic_weights(seed: int = 20261023) -> dict[str, np.ndarray]:
    """Stand-in weights with the logical names and shapes and plausible magnitudes.

    Matrices are scaled 1/sqrt(fan_in) and rounded to E4M3 (the real ones are decoded
    FP8); gates and biases are FP16-rounded. The values mean nothing — they only make a
    forward pass finite so two implementations can be compared on it.
    """
    rng = np.random.default_rng(seed)
    tensors = load_spec()
    weights = {}
    for name in sorted(tensors):
        shape = tuple(tensors[name]["shape"])
        role = _role(name)
        if role == "attn_scale":
            value = rng.uniform(4.0, 9.0, shape)
        elif role == "attn_bias":
            value = rng.normal(0.0, 0.5, shape)
        elif role in ("attn_cos_skip", "ffn_cos_skip", "inp_merge_cos"):
            # near 1: the 71 residual blocks keep their activations in a healthy half range
            value = rng.uniform(0.9, 1.0, shape)
        elif role in ("sin", "inp_upsample_sin", "inp_merge_sin"):
            value = rng.uniform(0.2, 0.6, shape)
        elif role == "blend_scale":
            value = np.full(shape, 0.73974609375)
        else:
            fan_in = shape[-2] if len(shape) >= 2 else shape[0]
            if role == "ffn_expand_weight":          # (G, 4, G, 32, 32): sums G inputs
                fan_in = shape[2] * shape[3]
            value = rng.normal(0.0, 0.6 / np.sqrt(fan_in), shape)
        value = value.astype(np.float32)
        if tensors[name]["dtype"] == "F16":
            value = value.astype(np.float16).astype(np.float32)
        weights[name] = value
    return weights


_ST_DTYPES = {"F16": np.float16, "F32": np.float32}


def load_logical(path: Path = LOGICAL_WEIGHTS) -> dict[str, np.ndarray]:
    """Read the logical safetensors into float32 arrays (same contract as nr_model)."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"missing weights {path}\n  run tools/get_weights.ps1 -Dll <your nvngx_dlssnr.dll>")
    with path.open("rb") as source:
        header_length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_length))
    base = 8 + header_length
    metadata = header.pop("__metadata__", {})
    if metadata.get("fully_logical") != "true":
        raise SystemExit("weights must declare fully_logical=true (use the *logical* safetensors)")
    blob = np.memmap(path, dtype=np.uint8, mode="r")
    weights = {}
    for name, entry in header.items():
        dtype = _ST_DTYPES.get(entry["dtype"])
        if dtype is None:
            raise SystemExit(f"{name}: unsupported dtype {entry['dtype']}")
        start, stop = entry["data_offsets"]
        raw = blob[base + start:base + stop].view(dtype)
        weights[name] = np.ascontiguousarray(raw.reshape(entry["shape"]), dtype=np.float32)
    if len(weights) != EXPECTED_TENSORS:
        raise SystemExit(f"{path} has {len(weights)} tensors, expected {EXPECTED_TENSORS}")
    return weights


def get_weights(kind: str) -> dict[str, np.ndarray]:
    """`real` (work/mlxw logical safetensors) or `synthetic`."""
    if kind == "real":
        return load_logical()
    if kind == "synthetic":
        return synthetic_weights()
    raise SystemExit(f"unknown weights kind {kind!r}")


def check_weights(weights: dict[str, np.ndarray]) -> list[str]:
    """Problems with a logical weight set against the spec (empty when it is sound)."""
    tensors = load_spec()
    problems = [f"missing {name}" for name in tensors if name not in weights]
    problems += [f"extra {name}" for name in weights if name not in tensors]
    for name, entry in tensors.items():
        if name in weights and tuple(weights[name].shape) != tuple(entry["shape"]):
            problems.append(f"{name}: shape {weights[name].shape} != {tuple(entry['shape'])}")
        elif name in weights and not np.isfinite(weights[name]).all():
            problems.append(f"{name}: non-finite values")
    return problems


if __name__ == "__main__":
    tensors = load_spec()
    count = parameter_count(tensors)
    print(f"spec: {len(tensors)} tensors, {count:,} parameters "
          f"({'OK' if count == EXPECTED_PARAMETERS else 'MISMATCH, expected %d' % EXPECTED_PARAMETERS})")
    if "--check-weights" in sys.argv:
        problems = check_weights(load_logical())
        for problem in problems[:20]:
            print("  " + problem)
        print(f"{LOGICAL_WEIGHTS}: {'OK' if not problems else '%d problems' % len(problems)}")
        raise SystemExit(1 if problems else 0)
