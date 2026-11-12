"""HNet — the recovered DLSS-NR 71-block graph, in PyTorch.

A port of the numpy reference `third_party/refs/dlss-nr-on-intel/src/ref/nr_model.py`
(Apache-2.0; itself a port of iamwavecut/MLX-DLSS `model.py`), following the *vendor*
graph (`NR_MLX_DLSS_GRAPH=0`). One graph, two numeric personalities chosen by `Numerics`:

* `Numerics.exact()` keeps every rounding point of the reference — half roundings, the
  E4M3 publishes, the fragment-tree cosine normalise and the bit-affine softmax — so the
  port can be checked against the numpy oracle value for value (`test_port.py`).
* `Numerics.deploy(...)` replaces the bit tricks with ops ONNX Runtime's DirectML EP can
  run: the softmax as `exp` or as a float emulation of the vendor's piecewise-linear
  exponential (`pwl`), the cosine normalise as a plain L2 norm, and the E4M3 publish as a
  Veltkamp split onto 4 significant bits or left out. These are what `export_onnx.py`
  exports, in FP16.

Weights are the logical tensors (649, `work/mlxw/dlssnr-logical.safetensors`) as float32
numpy arrays; `common.synthetic_weights()` gives stand-ins with the same shapes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

COSINE_NORM_FLOOR = 0.00006198883056640625
# The bases of the vendor's two exponentials, as natural-log multipliers: the window blocks'
# weight is 2^(1.4375 s + c), the ViT's 2^(1.4326 s + c). Deploy `exp` folds them into the
# attention scales and biases so the softmax input needs no multiply of its own.
WINDOW_SOFTMAX_BASE = 1.4375 * math.log(2.0)
VIT_SOFTMAX_BASE = 1.4326171875 * math.log(2.0)


def _sum_squares(value):
    """Sum of squares over the last axis as a GEMM against ones: a half ReduceSum exports
    wrapped in two Casts, and the DML EP runs the MatMul at least as fast."""
    ones = torch.ones(value.shape[-1], 1, dtype=value.dtype)
    return torch.matmul(value * value, ones)


# --------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Numerics:
    e4m3: str = "exact"        # exact | veltkamp | none
    softmax: str = "exact"     # exact | pwl | exp
    norm: str = "exact"        # exact | plain
    round_half: bool = True    # emulate the half roundings (float32 carrier)

    @classmethod
    def exact(cls):
        return cls()

    @classmethod
    def deploy(cls, e4m3="veltkamp", softmax="pwl"):
        return cls(e4m3=e4m3, softmax=softmax, norm="plain", round_half=False)

    @property
    def is_exact(self):
        return self == Numerics.exact()


class Ops:
    """Every numeric primitive of the graph, one implementation per `Numerics` choice."""

    def __init__(self, numerics: Numerics):
        self.n = numerics

    # -- rounding ---------------------------------------------------------

    def h(self, x):
        """Round to half (exact mode); a no-op where the carrier is already the precision."""
        return x.half().float() if self.n.round_half else x

    def e4m3(self, x):
        mode = self.n.e4m3
        if mode == "none":
            return x
        clamped = torch.clamp(x, -448.0, 448.0)
        if mode == "exact":
            # round-half-even onto E4M3, saturating at 448: the reference's `_e4m3_chunk`
            return clamped.to(torch.float8_e4m3fn).to(x.dtype)
        # Veltkamp split: the high part keeps 4 significant bits (E4M3's 1 + 3) in the
        # carrier's own rounding. 2^7 + 1 for half, 2^20 + 1 for float32.
        factor = 129.0 if x.dtype == torch.float16 else 1048577.0
        scaled = clamped * factor
        return scaled - (scaled - clamped)

    # -- matmul -----------------------------------------------------------

    def mm(self, a, b):
        if self.n.is_exact:
            # contiguous operands so the oracle, redirected to torch.matmul on the same
            # contiguous arrays, hits the very same GEMM (test_port.py)
            return torch.matmul(a.contiguous(), b.contiguous())
        return torch.matmul(a, b)

    def mm_nt(self, a, b):
        return self.mm(a, b.transpose(-1, -2))

    # -- activations ------------------------------------------------------

    def gate_activation(self, x):
        """The vendor activation x * gate(x), gate a clamped quadratic."""
        wide = self.h(x)
        clamped = torch.clamp(wide, -4.0, 4.0)
        # max(c, -c) rather than abs(): the exporter wraps a half Abs in two Casts
        magnitude = torch.maximum(clamped, -clamped)
        linear = self.h(magnitude * -0.055908203125 + 0.447265625)
        gate = self.h(linear * clamped + 0.89453125)
        return self.h(gate * wide)

    # -- softmax ----------------------------------------------------------

    def window_softmax(self, scores):
        """Softmax over the last axis of the window attention scores, then published."""
        mode = self.n.softmax
        if mode == "exact":
            return self._vendor_softmax_exact(scores)
        if mode == "exp":
            # The vendor's weight is 2^(1.4375 s + c) over s in [-6, 6]: base e^0.99640.
            # HNet folds that base into the attention scale and bias (`_fold_softmax_base`),
            # so the scores arrive already multiplied and only the clamp is left.
            k = WINDOW_SOFTMAX_BASE
            probs = torch.softmax(torch.clamp(scores, -6.0 * k, 5.98 * k), dim=-1)
        else:
            weights = self._pwl_weight(scores, 1.4375, 9.625, 1.0, 18.21875, 32.0)
            probs = weights / weights.sum(dim=-1, keepdim=True)
        return self.e4m3(probs)

    def _pwl_weight(self, scores, slope, intercept, low, high, steps):
        """The vendor's exponential, emulated in float: t = slope*s + intercept on the half
        grid of its affine (1/steps), then 2^floor(t) * (1 + frac(t)), scaled by 2^-15."""
        t = torch.clamp(scores * slope + intercept, low, high)
        t = torch.round(t * steps) / steps
        whole = torch.floor(t)
        return (1.0 + (t - whole)) * torch.pow(2.0, whole - 15.0)

    def _vendor_softmax_exact(self, value):
        """The fused kernels' half bit-affine softmax approximation (bit-exact port)."""
        affine = self.h(value) * 0.044921875 + 1.30078125
        affine = torch.clamp(affine, 1.03125, 1.5693359375).half()
        bits = affine.view(torch.int16).to(torch.int64) & 0xFFFF
        pairs = bits.reshape(*bits.shape[:-1], bits.shape[-1] // 2, 2)
        packed = pairs[..., 0] | (pairs[..., 1] << 16)
        transformed = ((packed << 5) + 0x7FF88000) & 0xFFFFFFFF
        weight_bits = torch.stack((transformed & 0xFFFF, (transformed >> 16) & 0xFFFF),
                                  dim=-1).reshape(bits.shape)
        weights = weight_bits.to(torch.int32).to(torch.int16).view(torch.float16).float()
        totals = weights.sum(dim=-1, keepdim=True).half().float()
        reciprocal = (1.0 / totals).half().float()
        return self.e4m3(self.h(weights * reciprocal))

    # -- cosine normalise ---------------------------------------------------

    def cosine_normalize(self, value):
        if self.n.norm == "plain":
            squares = torch.clamp(_sum_squares(value), min=COSINE_NORM_FLOOR)
            return value * torch.pow(squares, -0.5)
        return self._cosine_normalize_exact(value)

    def _cosine_normalize_exact(self, value):
        """The fused kernels' half fragment-tree cosine normalisation (head_dim 32)."""
        h = self.h
        half = h(value)
        partial = []
        for lane in range(4):
            lane_partial = []
            for parity in range(2):
                c = lane * 2 + parity
                first = h(half[..., c + 8] * half[..., c + 8] + h(half[..., c] * half[..., c]))
                second = h(half[..., c + 24] * half[..., c + 24]
                           + h(half[..., c + 16] * half[..., c + 16]))
                lane_partial.append(h(first + second))
            partial.append(torch.stack(lane_partial, dim=-1))
        partial = torch.stack(partial, dim=-2)                         # (..., 4, 2)
        xor_two = torch.stack([h(partial[..., lane, :] + partial[..., lane ^ 2, :])
                               for lane in range(4)], dim=-2)
        xor_one = torch.stack([h(xor_two[..., lane, :] + xor_two[..., lane ^ 1, :])
                               for lane in range(4)], dim=-2)
        norm = h(xor_one[..., 0, 0] + xor_one[..., 0, 1])
        norm = torch.clamp(norm, min=float(np.float16(COSINE_NORM_FLOOR)))
        reciprocal = h(1.0 / torch.sqrt(norm))[..., None]
        return h(half * reciprocal)

    def cosine_publish(self, value, scale=None):
        normalized = self.cosine_normalize(value)
        if scale is not None:
            normalized = self.h(normalized * self.h(scale).reshape(1, -1, 1, 1))
        return self.e4m3(normalized)

    # -- the ViT's own attention numerics -------------------------------------

    def vit_norm(self, value):
        """Reciprocal norm of each 32-channel head, (..., 32) -> (...)."""
        h = self.h
        if self.n.norm == "plain":
            squares = torch.clamp(_sum_squares(value)[..., 0], min=COSINE_NORM_FLOOR)
            return torch.pow(squares, -0.5)
        low, high = value[..., :16], value[..., 16:]
        squares = h(low * low + h(high * high))
        eight = h(squares[..., :8] + squares[..., 8:])
        four = h(eight[..., :4] + eight[..., 4:])
        total = h(h(four[..., 0] + four[..., 2]) + h(four[..., 1] + four[..., 3]))
        total = torch.clamp(total, min=float(np.float16(COSINE_NORM_FLOOR)))
        return h(1.0 / torch.sqrt(total))

    def vit_attend(self, query, key, value):
        """Attention of published Q, K, V (..., tokens, 32) -> attended values, published."""
        mode = self.n.softmax
        if mode == "exact":
            return self._vit_attend_exact(query, key, value)
        scores = self.mm_nt(query, key)
        if mode == "exp":
            k = VIT_SOFTMAX_BASE                  # folded into the learned scale by HNet
            probs = torch.softmax(torch.clamp(scores, -3.01 * k, 3.0 * k), dim=-1)
        else:
            weights = self._pwl_weight(scores, 1.4326171875, 11.34375, 7.03125, 15.640625, 64.0)
            probs = weights / weights.sum(dim=-1, keepdim=True)
        return self.e4m3(self.mm(probs, value))

    def _vit_exp_weight(self, score):
        affine = self.h(score * 0.08953857421875 + 1.708984375)
        affine = torch.clamp(affine, 1.439453125, 1.9775390625).half()
        bits = affine.view(torch.int16).to(torch.int32) & 0xFFFF
        out = ((bits << 4) + 0x4000) & 0xFFFF
        return out.to(torch.int16).view(torch.float16).float()

    def _vit_denominator(self, weights, tokens):
        h = self.h
        padded = weights.shape[-1]
        total = torch.zeros(weights.shape[:-1], dtype=torch.float32)
        for base in range(0, padded, 64):
            block = weights[..., base:base + 64]

            def column(k):
                a = h(block[..., k] + block[..., k + 8])
                b = h(block[..., k + 16] + block[..., k + 24])
                c = h(block[..., k + 32] + block[..., k + 40])
                d = h(block[..., k + 48] + block[..., k + 56])
                return h(h(h(a + b) + c) + d)

            sums = []
            for parity in (0, 1):
                running = h(column(parity) + column(2 + parity))
                running = h(running + column(4 + parity))
                sums.append(h(running + column(6 + parity)))
            total = h(total + h(sums[0] + sums[1]))
        if padded > tokens:
            zero_weight = self._vit_exp_weight(torch.zeros(()))
            total = h(total - h(zero_weight * float(padded - tokens)))
        return total

    def _vit_attend_exact(self, query, key, value):
        tokens = key.shape[-2]
        padded = -(-tokens // 64) * 64
        key = F.pad(key, (0, 0, 0, padded - tokens))
        value = F.pad(value, (0, 0, 0, padded - tokens))
        scores = self.h(self.mm_nt(query, key))
        weights = self._vit_exp_weight(scores)
        reciprocal = self.h(1.0 / self._vit_denominator(weights, tokens))
        summed = self.h(self.mm(self.e4m3(weights), value))
        return self.e4m3(self.h(summed * reciprocal[..., None]))


# --------------------------------------------------------------------------
# graph constants (identical to the reference)
# --------------------------------------------------------------------------


def _fragment_swizzle_indices():
    """Stored offset of every logical (query, key) window-bias entry."""
    indices = []
    for entry in range(64 * 64):
        query, key = divmod(entry, 64)
        qy, qx = divmod(query, 8)
        ky, kx = divmod(key, 8)

        def bit(v, p):
            return (v >> p) & 1

        indices.append((bit(qy, 2) << 11) | (bit(qx, 2) << 10) | (bit(ky, 2) << 9)
                       | (bit(kx, 2) << 8) | (bit(qy, 0) << 7) | (bit(qx, 1) << 6)
                       | (bit(qx, 0) << 5) | (bit(ky, 0) << 4) | (bit(kx, 1) << 3)
                       | (bit(ky, 1) << 2) | (bit(qy, 1) << 1) | bit(kx, 0))
    return np.array(indices, dtype=np.intp)


FRAGMENT_SWIZZLE_INDICES = _fragment_swizzle_indices()


def recover_attention_bias_layout(bias: np.ndarray) -> np.ndarray:
    heads = bias.shape[0]
    return bias.reshape(heads, 64 * 64)[:, FRAGMENT_SWIZZLE_INDICES].reshape(heads, 64, 64)


def window_origin(block: int):
    """The vendor window origin (y, x) for one block: shifted windows, phase per stage."""
    if block == 0:
        phase = 0
    elif 1 <= block <= 4:
        phase = block - 1
    elif 5 <= block <= 8:
        phase = block - 5
    elif 9 <= block <= 14:
        phase = block - 9
    elif 15 <= block <= 22:
        phase = block - 15
    elif 23 <= block <= 30:
        phase = block - 23
    elif 40 <= block <= 55:
        phase = block - (40 if block < 48 else 48)
    elif 56 <= block <= 61:
        phase = block - 54
    elif 62 <= block <= 69:
        phase = block - (62 if block < 66 else 66)
    elif block == 70:
        phase = 1
    else:
        return (0, 0)
    return ((0, -4, 0, -4)[phase % 4], (0, -4, -4, 0)[phase % 4])


VIT_BLOCKS = tuple(f"block{i}." for i in range(31, 39))

# encoder levels: (regular blocks, transition block, heads)
ENCODER_LEVELS = ((range(1, 4), 4, 1), (range(5, 8), 8, 2), (range(9, 14), 14, 4),
                  (range(15, 22), 22, 8))
# decoder levels after block 55: (transition, regular blocks, skip index, heads)
DECODER_LEVELS = ((56, range(57, 62), 2, 4), (62, range(63, 66), 1, 2), (66, range(67, 70), 0, 1))


# --------------------------------------------------------------------------
# spatial helpers (NHWC)
#
# Every reshape spells its dimensions out: the exporter writes a -1 with allowzero=1,
# which the DirectML EP refuses ("The parameter is incorrect") — dml_node_probe.py.
# --------------------------------------------------------------------------


def average_pool2(x):
    return (x[:, 0::2, 0::2, :] + x[:, 1::2, 0::2, :]
            + x[:, 0::2, 1::2, :] + x[:, 1::2, 1::2, :]) * 0.25


def pad_spatial_end(x, multiple):
    bottom = (-x.shape[1]) % multiple
    right = (-x.shape[2]) % multiple
    if not (bottom or right):
        return x
    return F.pad(x, (0, 0, 0, right, 0, bottom))


def nearest_upsample2_crop(x, height, width):
    b, h, w, c = x.shape
    up = x[:, :, None, :, None, :].expand(b, h, 2, w, 2, c).reshape(b, 2 * h, 2 * w, c)
    return up[:, :height, :width, :]


def partition_windows(x, size=8):
    b, h, w, c = x.shape
    return (x.reshape(b, h // size, size, w // size, size, c)
            .permute(0, 1, 3, 2, 4, 5).reshape(b * (h // size) * (w // size), size * size, c))


def reverse_windows(windows, b, h, w, size=8):
    c = windows.shape[-1]
    return (windows.reshape(b, h // size, w // size, size, size, c)
            .permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c))


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class HNet(nn.Module):
    """The fixed 71-block graph over external logical weights. Input (1, H, W, 16) features
    on the padded network field (H, W multiples of 64), output the (1, H, W, 4) head."""

    def __init__(self, weights: dict[str, np.ndarray], numerics: Numerics = Numerics.exact()):
        super().__init__()
        self.numerics = numerics
        self.ops = Ops(numerics)
        self._names = {}
        for name, value in weights.items():
            value = np.asarray(value, dtype=np.float32)
            # the single-head window blocks' and the 16-head split blocks' biases are
            # stored in mma fragment order
            if name.endswith("attn_bias") and value.shape[0] in (1, 16):
                value = recover_attention_bias_layout(value)
            self._register(name, value)
        if not numerics.is_exact:
            self._fuse_branched()
            self._split_heads()
            if numerics.softmax == "exp":
                self._fold_softmax_base()

    def _split_heads(self):
        """Each window attention's QKV weight (C, 3C) as (heads, C, 96), [q|k|v] per head: one
        broadcast GEMM then lands Q, K and V already head-major, with no transposes."""
        for name in [n for n in self._names if n.endswith("qkv_weight") and not n.startswith(VIT_BLOCKS)]:
            qkv = self.w(name).numpy()
            c = qkv.shape[0]
            heads = c // 32
            per_head = qkv.reshape(c, 3, heads, 32).transpose(2, 0, 1, 3).reshape(heads, c, 96)
            self._register(name[: -len("qkv_weight")] + "qkv_heads", np.ascontiguousarray(per_head))
            delattr(self, self._names.pop(name))

    def _fold_softmax_base(self):
        """Multiply the softmax's exponential base into what makes the logits: the window
        blocks' attention scale (on the normalised query) and bias, and the ViT's learned scale."""
        for name in [n for n in self._names if n.endswith("attn_scale")]:
            base = VIT_SOFTMAX_BASE if name.startswith(VIT_BLOCKS) else WINDOW_SOFTMAX_BASE
            self._register(name, self.w(name).numpy() * np.float32(base))
            bias = name[: -len("attn_scale")] + "attn_bias"
            if self.has(bias):
                self._register(bias, self.w(bias).numpy() * np.float32(base))

    def _register(self, name, value):
        key = name.replace(".", "__")
        self.register_buffer(key, torch.from_numpy(np.ascontiguousarray(value)), persistent=True)
        self._names[name] = key

    def w(self, name):
        return getattr(self, self._names[name])

    def has(self, name):
        return name in self._names

    def _fuse_branched(self):
        """The branched feed-forward's 4*G^2 small GEMMs as one expansion and G batched
        projections (the same arithmetic, reassociated): see nr_model._fused_branched_weights."""
        for name in [n for n in self._names if n.endswith("ffn_expand_weight")]:
            prefix = name[: -len("ffn_expand_weight")]
            expansion = self.w(name).numpy()                              # (G, 4, G, 32, 32)
            groups = expansion.shape[0]
            fused = expansion.transpose(0, 2, 3, 1, 4).reshape(groups, groups * 32, 128)
            whole = np.ascontiguousarray(fused.transpose(1, 0, 2).reshape(groups * 32, groups * 128))
            projection = self.w(prefix + "ffn_branch_projection_weight").numpy().reshape(groups, 128, 32)
            self._register(prefix + "fused_expand", whole)
            self._register(prefix + "fused_project", np.ascontiguousarray(projection))
            for original in (name, prefix + "ffn_branch_projection_weight"):
                delattr(self, self._names.pop(original))

    def is_branched(self, prefix):
        return self.has(prefix + ".ffn_expand_weight") or self.has(prefix + ".fused_expand")

    # -- attention ----------------------------------------------------------

    def cosine_attention(self, value, *, qkv, scale, bias, projection, heads):
        o = self.ops
        b, t, c = value.shape
        if qkv.dim() == 3:                       # deploy: (heads, C, 96), see _split_heads
            projected = o.mm(value.unsqueeze(1), qkv)                       # (b, heads, t, 96)
            query, key, val = torch.split(projected, 32, dim=-1)
            query = o.cosine_publish(query, scale)
            key = o.cosine_publish(key)
            val = o.e4m3(val)
            scores = o.mm_nt(query, key) + bias.reshape(1, heads, t, t)
            probs = o.window_softmax(scores)
            attended = o.mm(probs, val)                                     # (b, heads, t, 32)
            if heads > 1:
                attended = attended.permute(0, 2, 1, 3)
            return o.mm(o.e4m3(attended.reshape(b, t, c)), projection)
        projected = o.mm(value, qkv)
        query, key, val = torch.split(projected, c, dim=-1)
        shape = (b, t, heads, c // heads)
        query = query.reshape(shape).permute(0, 2, 1, 3)
        key = key.reshape(shape).permute(0, 2, 1, 3)
        val = val.reshape(shape).permute(0, 2, 1, 3)
        query = o.cosine_publish(query.contiguous(), scale)
        key = o.cosine_publish(key.contiguous())
        val = o.e4m3(val.contiguous())
        scores = o.mm_nt(query, key) + bias.reshape(1, heads, t, t)
        probs = o.window_softmax(scores)
        attended = o.mm(probs, val).permute(0, 2, 1, 3).reshape(b, t, c)
        return o.mm(o.e4m3(attended.contiguous()), projection)

    def window_attention(self, x, block, prefix, attn_prefix, *, heads, publish_input):
        o = self.ops
        oy, ox = window_origin(block)
        top, left = -oy, -ox
        b, height, width, c = x.shape
        bottom = (-(height + top)) % 8
        right = (-(width + left)) % 8
        padded = F.pad(x, (0, 0, left, right, top, bottom)) if (top or left or bottom or right) else x
        ph, pw = padded.shape[1], padded.shape[2]
        windows = partition_windows(padded)
        if publish_input:
            windows = o.e4m3(windows)
        attended = self.cosine_attention(
            windows, qkv=self.w(f"{attn_prefix}.qkv_heads" if self.has(f"{attn_prefix}.qkv_heads")
                                 else f"{attn_prefix}.qkv_weight"),
            scale=self.w(f"{attn_prefix}.attn_scale"), bias=self.w(f"{attn_prefix}.attn_bias"),
            projection=self.w(f"{prefix}.projection_weight"), heads=heads)
        out = reverse_windows(attended, b, ph, pw)
        return out[:, top:top + height, left:left + width, :]

    # -- feed-forwards --------------------------------------------------------

    def branched_ffn(self, x, prefix):
        o = self.ops
        if not self.numerics.is_exact:
            groups = x.shape[-1] // 32
            lead = x.shape[:-1]
            hidden = o.mm(x, self.w(prefix + ".fused_expand"))           # (..., G*128)
            hidden = o.e4m3(o.gate_activation(hidden)).reshape(math.prod(lead), groups, 128).transpose(0, 1)
            heads = o.e4m3(o.mm(hidden, self.w(prefix + ".fused_project")))   # (G, N, 32)
            heads = heads.transpose(0, 1).reshape(*lead, groups * 32)
            return o.mm(heads, self.w(prefix + ".ffn_output_projection_weight"))
        expansion = self.w(prefix + ".ffn_expand_weight")
        projection = self.w(prefix + ".ffn_branch_projection_weight")
        groups = expansion.shape[0]
        inputs = torch.split(x, 32, dim=-1)
        outputs = []
        for oh in range(groups):
            branches = []
            for br in range(4):
                expanded = o.mm(inputs[0], expansion[oh, br, 0])
                for ih in range(1, groups):
                    expanded = expanded + o.mm(inputs[ih], expansion[oh, br, ih])
                branches.append(o.mm(o.e4m3(o.gate_activation(expanded)), projection[oh, br]))
            total = branches[0]
            for branch in branches[1:]:
                total = total + branch
            outputs.append(o.e4m3(total))
        return o.mm(torch.cat(outputs, dim=-1), self.w(prefix + ".ffn_output_projection_weight"))

    def split_group_ffn(self, x, prefix):
        o = self.ops
        hidden = o.e4m3(o.mm(x, self.w(prefix + ".first_projection_weight")))
        expand = self.w(prefix + ".group_expand_weight")                    # (8, 64, 256)
        project = self.w(prefix + ".group_project_weight")                  # (8, 256, 64)
        groups = expand.shape[0]
        if not self.numerics.is_exact:
            lead = hidden.shape[:-1]
            grouped = hidden.reshape(math.prod(lead), groups, 64).transpose(0, 1)         # (G, N, 64)
            out = o.mm(o.gate_activation(o.mm(grouped, expand)), project)   # (G, N, 64)
            return o.e4m3(out.transpose(0, 1).reshape(*lead, groups * 64))
        parts = torch.split(hidden, 64, dim=-1)
        outputs = [o.mm(o.gate_activation(o.mm(part, expand[g])), project[g])
                   for g, part in enumerate(parts)]
        return o.e4m3(torch.cat(outputs, dim=-1))

    # -- blocks ---------------------------------------------------------------

    def window_block(self, x, block, *, heads, publish=True):
        o = self.ops
        p = f"block{block}.layer0"
        if self.is_branched(p):
            branch = self.branched_ffn(o.e4m3(x), p)
            ff = o.e4m3(branch + x * self.w(f"{p}.ffn_cos_skip"))
            attention = self.window_attention(ff, block, p, p, heads=heads, publish_input=False)
        else:
            branch = o.mm(o.e4m3(o.gate_activation(o.mm(o.e4m3(x), self.w(f"{p}.weight1")))),
                          self.w(f"{p}.weight2"))
            ff = branch + x * self.w(f"{p}.ffn_cos_skip")
            attention = self.window_attention(ff, block, p, p, heads=heads, publish_input=True)
        out = attention + ff * self.w(f"{p}.attn_cos_skip")
        # block 0 is pooled before its publication and block 70 feeds the head
        return out if block in (0, 70) or not publish else o.e4m3(out)

    def split_window_block(self, x, block, publish=True):
        o = self.ops
        p = f"block{block}"
        branch = o.mm(self.split_group_ffn(o.e4m3(x), f"{p}.layer0"), self.w(f"{p}.layer1.weight3"))
        ff = o.e4m3(branch + x * self.w(f"{p}.layer1.ffn_cos_skip"))
        attention = self.window_attention(ff, block, f"{p}.layer3", f"{p}.layer2",
                                          heads=16, publish_input=False)
        out = attention + ff * self.w(f"{p}.layer3.attn_cos_skip")
        return o.e4m3(out) if publish else out

    def global_block(self, x, block):
        o = self.ops
        p = f"block{block}"
        b, hh, ww, c = x.shape
        tokens = x.reshape(b, hh * ww, c)
        branch = o.mm(o.e4m3(o.gate_activation(o.mm(tokens, self.w(f"{p}.layer0.weight")))),
                      self.w(f"{p}.layer1.weight"))
        ff = o.e4m3(branch + tokens * self.w(f"{p}.layer1.ffn_cos_skip"))
        attention = self.vit_attention(ff, p, heads=32)
        out = attention + ff * self.w(f"{p}.layer4.attn_cos_skip")
        return o.e4m3(out.reshape(b, hh, ww, c))

    def vit_attention(self, x, p, *, heads):
        o = self.ops
        b, t, c = x.shape
        projected = o.h(o.mm(x, self.w(f"{p}.layer2.qkv_weight")))
        query, key, val = torch.split(projected, c, dim=-1)
        shape = (b, t, heads, 32)
        query, key, val = query.reshape(shape), key.reshape(shape), val.reshape(shape)
        learned = o.h(self.w(f"{p}.layer2.attn_scale")).reshape(1, 1, heads, 1)
        root = float(np.float16(math.sqrt(32)))
        query = o.e4m3(o.h(o.h(o.h(query * o.vit_norm(query)[..., None]) * root) * learned))
        key = o.e4m3(o.h(key * o.vit_norm(key)[..., None]))
        val = o.e4m3(val)
        query, key, val = (t_.permute(0, 2, 1, 3).contiguous() for t_ in (query, key, val))
        attended = o.vit_attend(query, key, val)
        attended = attended.permute(0, 2, 1, 3).reshape(b, t, c)
        return o.mm(attended.contiguous(), self.w(f"{p}.layer4.projection_weight"))

    def downsample_window(self, x, block, *, heads):
        o = self.ops
        transformed = self.window_block(x, block, heads=heads, publish=False)
        skip = o.e4m3(transformed)
        if block == 22:
            transformed = pad_spatial_end(transformed, 8)
        pooled = o.e4m3(average_pool2(transformed))
        return skip, o.e4m3(o.mm(pooled, self.w(f"block{block}.layer0.weight0")))

    def upsample_window(self, x, skip, block, *, heads):
        o = self.ops
        p = f"block{block}.layer0"
        projected = o.mm(x, self.w(f"{p}.weight0"))
        merged = nearest_upsample2_crop(projected, skip.shape[1], skip.shape[2]) + skip * self.w(f"{p}.sin")
        if self.is_branched(p):
            merged = o.e4m3(merged)
        return self.window_block(merged, block, heads=heads)

    # -- forward ----------------------------------------------------------------

    # The graph in three parts, the split the hybrid runner uses (src/neural): the window blocks
    # at both ends run as fused HLSL kernels, the middle as ONNX. Two splits:
    #   shallow  front blocks 0-4,  middle blocks 5-65 (64ch at 1/4 in and out),  back 66-70
    #   deep     front blocks 0-22, middle blocks 23-47 (512ch at 1/32),          back 48-70
    # The front returns (value, skips, full_skip): skips[i] is encoder level i's transition
    # output (level 0 = block 4 at 1/2 ... level 3 = block 22 at 1/16).
    SPLIT_LEVELS = {"shallow": 1, "deep": 4}

    def _encoder_level(self, value, level, keep):
        regular, transition, heads = ENCODER_LEVELS[level]
        for index in regular:
            value = keep(f"block{index}", self.window_block(value, index, heads=heads))
        skip, value = self.downsample_window(value, transition, heads=heads)
        keep(f"block{transition}", value)
        return skip, value

    def _bottleneck(self, value, keep):
        """Blocks 23-47: the 512-channel split blocks, the ViT and the decoder's way back up."""
        o = self.ops
        for index in range(23, 30):
            value = keep(f"block{index}", self.split_window_block(value, index))
        raw = keep("block30", self.split_window_block(value, 30, publish=False))
        split_skip = o.e4m3(raw)
        value = o.mm(o.e4m3(average_pool2(pad_spatial_end(raw, 8))), self.w("block30.layer4.weight"))
        value = o.e4m3(value)
        for index in range(31, 39):
            value = keep(f"block{index}", self.global_block(value, index))
        up = o.mm(value, self.w("block39.layer0.conv_weight"))
        value = (nearest_upsample2_crop(up, split_skip.shape[1], split_skip.shape[2])
                 + split_skip * self.w("block39.layer0.inp_upsample_sin"))
        value = keep("block39", o.e4m3(value))
        for index in range(40, 48):
            value = keep(f"block{index}", self.split_window_block(value, index))
        return value

    def _decoder_level(self, value, skip, transition, regular, heads, keep):
        value = keep(f"block{transition}", self.upsample_window(value, skip, transition, heads=heads))
        for index in regular:
            value = keep(f"block{index}", self.window_block(value, index, heads=heads))
        return value

    # the decoder levels after the bottleneck, finest last: (transition, regular, skip index, heads)
    DECODER = ((48, range(49, 56), 3, 8),) + DECODER_LEVELS

    def forward_front(self, features, capture=None, split="shallow"):
        o = self.ops
        keep = self._keeper(capture)
        value = o.mm(features, self.w("block0.layer0.input_adapter_weight"))
        block0 = keep("block0", self.window_block(value, 0, heads=1))
        full_skip = o.e4m3(block0)
        value = o.e4m3(average_pool2(block0))
        skips = []
        for level in range(self.SPLIT_LEVELS[split]):
            skip, value = self._encoder_level(value, level, keep)
            skips.append(skip)
        return value, skips, full_skip

    def forward_middle(self, value, capture=None, split="shallow"):
        keep = self._keeper(capture)
        levels = self.SPLIT_LEVELS[split]
        skips = [None] * levels
        for level in range(levels, len(ENCODER_LEVELS)):
            skip, value = self._encoder_level(value, level, keep)
            skips.append(skip)
        value = self._bottleneck(value, keep)
        for transition, regular, skip_index, heads in self.DECODER[: len(self.DECODER) - levels]:
            value = self._decoder_level(value, skips[skip_index], transition, regular, heads, keep)
        return value

    def forward_back(self, value, skips, full_skip, capture=None, split="shallow"):
        o = self.ops
        keep = self._keeper(capture)
        levels = self.SPLIT_LEVELS[split]
        for transition, regular, skip_index, heads in self.DECODER[len(self.DECODER) - levels:]:
            value = self._decoder_level(value, skips[skip_index], transition, regular, heads, keep)
        value = nearest_upsample2_crop(value, full_skip.shape[1], full_skip.shape[2])
        value = value * self.w("block70.layer0.inp_merge_sin") + full_skip * self.w("block70.layer0.inp_merge_cos")
        value = keep("block70", self.window_block(value, 70, heads=1))
        return (o.mm(value[..., :16], self.w("block70.layer0.out_gain"))
                + o.mm(value[..., 16:], self.w("block70.layer0.out_conv_weight")))

    @staticmethod
    def _keeper(capture):
        def keep(name, value):
            if capture is not None:
                capture[name] = value
            return value
        return keep

    def forward(self, features, capture=None):
        """features (1, H, W, 16) -> head (1, H, W, 4). `capture`, a dict, receives every
        block's output by name (for the per-block comparison with the oracle)."""
        value, skips, full_skip = self.forward_front(features, capture)
        value = self.forward_middle(value, capture)
        return self.forward_back(value, skips, full_skip, capture)


class HNetMiddle(nn.Module):
    """The middle of a split alone: what the hybrid runner hands to ONNX Runtime."""

    def __init__(self, net: HNet, split="shallow"):
        super().__init__()
        self.net = net
        self.split = split

    def forward(self, value):
        return self.net.forward_middle(value, split=self.split)
