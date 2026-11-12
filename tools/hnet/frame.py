"""Everything around the network for one frame: field geometry, the 16-channel features,
and the composition of the head — the CPU reference the HLSL pre/post passes must match.

Sources: MLX-DLSS `features.py` (noise, scaled colour, channel layout; loaded from the
pinned clone in `work/mlx-dlss`) and the vendor geometry/composition as written down in
`third_party/refs/dlss-nr-on-intel/src/ref/nr_frame.py` and docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass

import numpy as np

from common import MLX_DLSS

VENDOR_MINIMUM_EXTENT = 320
GRAPH_MINIMUM_EXTENT = 128
BLEND_SCALE = 0.73974609375


def _load_mlx_features():
    path = MLX_DLSS / "python" / "mlxdlss" / "features.py"
    if not path.is_file():
        raise SystemExit(f"missing {path} (clone MLX-DLSS into work/mlx-dlss)")
    spec = importlib.util.spec_from_file_location("mlxdlss_features", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mlxdlss_features"] = module
    spec.loader.exec_module(module)
    return module


features_mod = _load_mlx_features()
deterministic_noise = features_mod.deterministic_noise
scaled_color = features_mod.scaled_color
half = features_mod.half


def _field_alignment(extent: int) -> int:
    reductions, size = 0, extent
    for level in range(6):
        half_size = -(-((size + 1) // 2) // 4) * 4
        reductions += half_size < size
        reductions += level == 0 and half_size % 8 != 0
        size = half_size
    return 1 << reductions


@dataclass(frozen=True)
class Geometry:
    width: int            # the frame handed to the network
    height: int
    field_width: int      # the padded network field
    field_height: int

    def source_rows(self):
        return features_mod.NetworkGeometry.extended_indices(self.field_height, self.height)

    def source_columns(self):
        return features_mod.NetworkGeometry.extended_indices(self.field_width, self.width)


def network_geometry(width: int, height: int, minimum: int = VENDOR_MINIMUM_EXTENT) -> Geometry:
    """The vendor's field for a frame (nr_frame.network_geometry): each side aligned to the
    graph's reductions and at least `minimum`, one alignment wider when both sides are four
    alignments, then rounded up to 64."""
    floor = max(GRAPH_MINIMUM_EXTENT, int(minimum))
    aw, ah = _field_alignment(width), _field_alignment(height)
    fw = max(floor, -(-width // aw) * aw)
    fh = max(floor, -(-height // ah) * ah)
    if fw % (4 * aw) == 0 and fh % (4 * ah) == 0:
        fw += aw
    return Geometry(width, height, -(-fw // 64) * 64, -(-fh // 64) * 64)


PROFILES = {
    "standard": (0.0, 1.0, 1.0),
    "natural": (1.0 / 128, 1.0, 1.0),
    "cinematic": (2.0 / 128, 1.0, 1.0),
    "neutral": (0.0, 0.0, 0.0),
}


def make_features(color, geometry: Geometry, *, history=None, frame_index=0, profile="standard"):
    """(H, W, 3) colour in [0, 1] -> (field_h, field_w, 16) float32 features. `history`
    (H, W, 3), the previous composed prediction, replaces channels 7-9."""
    style, tone, structure = PROFILES[profile]
    ng = features_mod.NetworkGeometry(geometry.width, geometry.height,
                                      geometry.field_width, geometry.field_height)
    features = features_mod.make_features(
        np.asarray(color, np.float32), frame_index=frame_index, geometry=ng,
        normalized_style=style, local_tone_strength=tone, local_structure_strength=structure)
    if history is not None:
        rows, columns = geometry.source_rows(), geometry.source_columns()
        features[..., 7:10] = scaled_color(np.asarray(history, np.float32)[rows[:, None], columns[None, :], :])
    return features


def compose(head, color, history=None, *, intensity=1.0):
    """The head over the frame (vendor composition): returns (output, next_history).

    predicted = clamp(colour + half(head.rgb) * 0.25, 0, 1)
    alpha     = clamp(sigmoid(half(head.a)) * half(blend_scale), 0, 1)     (with history)
    out       = predicted + alpha * (history - predicted)
    result    = clamp(colour + intensity * (out - colour), 0, 1)
    """
    head = np.asarray(head, np.float32)[: color.shape[0], : color.shape[1]]
    color = np.asarray(color, np.float32)
    predicted = np.clip(color + half(head[..., :3]) * np.float32(0.25), 0, 1)
    if history is not None:
        logit = half(head[..., 3:4])
        alpha = np.clip(1.0 / (1.0 + np.exp(-logit)) * half(BLEND_SCALE), 0, 1).astype(np.float32)
        predicted = predicted + alpha * (np.asarray(history, np.float32) - predicted)
    out = np.clip(color + np.float32(intensity) * (predicted - color), 0, 1).astype(np.float32)
    return out, predicted.astype(np.float16).astype(np.float32)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak * peak / mse)


def test_image(width, height, seed=7):
    """A deterministic synthetic test frame: gradients, edges, a disc and texture noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = xx / max(1, width - 1), yy / max(1, height - 1)
    img = np.stack([0.15 + 0.7 * u, 0.2 + 0.6 * v, 0.5 + 0.4 * np.sin(6.28 * (u + v))], -1)
    disc = ((xx - width * 0.6) ** 2 + (yy - height * 0.45) ** 2) < (0.2 * min(width, height)) ** 2
    img[disc] = [0.85, 0.65, 0.5]
    img[(xx.astype(int) // 16 + yy.astype(int) // 16) % 2 == 0] *= 0.9
    img += rng.normal(0, 0.02, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def load_image(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_image(path, rgb):
    from PIL import Image
    Image.fromarray((np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_pfm(path, rgb):
    """Little-endian RGB PFM (bottom-up rows), what the C++ --neural --golden mode reads."""
    rgb = np.ascontiguousarray(np.asarray(rgb, np.float32)[::-1])
    with open(path, "wb") as out:
        out.write(f"PF\n{rgb.shape[1]} {rgb.shape[0]}\n-1.0\n".encode("ascii"))
        out.write(rgb.astype("<f4").tobytes())
