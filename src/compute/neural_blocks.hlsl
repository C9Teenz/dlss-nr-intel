// DLSS-NR hybrid runner: the 32-channel high-resolution ends of HNet as fused kernels.
//
// Blocks 0-4 (front) and 66-70 (back) are single-head shifted-window blocks at C = 32, the
// bulk of the field's pixels and, run op by op in ONNX/DirectML, 60 % of the frame while
// doing little arithmetic. Here one thread group is one 8x8 window and one thread one token:
// the token's feed-forward, its Q/K/V, the window's 64x64 attention and the residuals all
// stay in registers and groupshared memory, and a block reads and writes its tensor once.
//
// Reference: tools/hnet/model.py HNet.window_block in deploy numerics `nq-exp` (no E4M3
// publishes, softmax with the vendor's base folded into scale and bias, plain cosine norm),
// computed here in float32 and stored as half between blocks. Weights: the blob from
// tools/hnet/export_hybrid.py, matrices as (out, in) except weight2 (128, 32).
//
//   CSWindowBlock32  one window block; input and output modes below
//   CSPool2          2x2 average pool, 32 channels (after block 0)
//   CSPoolProject    2x2 average pool then 32 -> 64 (block 4's transition into the middle)
//
// Tensors are NHWC half, packed two per uint (RWByteAddressBuffer): 32 channels = 64 bytes.

cbuffer BlockConstants : register(b0) {
    uint out_w;           // extent of this dispatch's tokens / outputs
    uint out_h;
    uint in_w;            // extent of input A where it differs (coarse input, pre-pool input)
    uint in_h;
    uint pad_top;         // the window origin: windows start at (-pad_top, -pad_left)
    uint pad_left;
    uint input_mode;      // 0 tensor, 1 features+adapter, 2 block 66 merge, 3 block 70 merge
    uint output_mode;     // 0 tensor, 1 head (block 70)
    uint o_w1;            // weight offsets in floats
    uint o_w2;
    uint o_ffn_cos;
    uint o_qkv;
    uint o_scale;
    uint o_bias;
    uint o_proj;
    uint o_attn_cos;
    uint o_extra0;        // mode 1 adapter (32,16) | mode 2 weight0 (32,64) | mode 3 merge sin | pool-project weight0 (64,32)
    uint o_extra1;        // mode 2 sin | mode 3 merge cos
    uint o_extra2;        // head: out_gain (4,16)
    uint o_extra3;        // head: out_conv (4,16)
    uint o_w1_16;         // byte offsets of weight1 / weight2 in the half weights (t1)
    uint o_w2_16;
    uint half_ffn;        // 1: the FFN in half — dot2add for x @ w1, half2 FMAs for the w2 sum
    uint publish;         // 1: the output is published (E4M3); 0 for blocks 0, 4 and 70, whose raw
                          // output feeds a pool and a skip that round it themselves
};

static const float kNormFloor = 0.00006198883056640625;
static const float kSoftmaxBase = 0.99639907;       // 1.4375 ln 2, folded into scale and bias
static const float kLogitLow = -6.0 * kSoftmaxBase;
static const float kLogitHigh = 5.98 * kSoftmaxBase;

ByteAddressBuffer g_w : register(t0);
ByteAddressBuffer g_w16 : register(t1);
RWByteAddressBuffer g_a : register(u0);     // input A
RWByteAddressBuffer g_b : register(u1);     // input B (the skip of a merge)
RWByteAddressBuffer g_out : register(u2);

// K and V of the window, 4 channels per element, stored in half. Groupshared memory bounds how
// many windows a core holds: 16 KB in float ran 11.45 ms for the ten blocks at 960x576, these
// 8 KB 9.2 ms with float math (4 KB gains nothing more; the rest is the half -> float converts).
groupshared half4 gs_k[64][8];
groupshared half4 gs_v[64][8];

// ATTN_HALF 1: the attention's arithmetic in half where it reads K and V — each 4-channel slice
// of q.k a half dot (the logit sums the slices in float), and p * V accumulated in half over 4
// keys before it is added to the float sum (p shifted to <= 1 so the half sums stay small).
// 9.2 -> 8.7 ms; the synthetic checks moved by < 4 %. 0 restores float math (re-check with the
// real weights: a larger attn_scale means larger logits and so larger half rounding).
#ifndef ATTN_HALF
#define ATTN_HALF 1
#endif
#if ATTN_HALF
typedef half4 attn4;
typedef half attn1;
#else
typedef float4 attn4;
typedef float attn1;
#endif

// q . k for the key in slot `key` (the bias and the clamp are the caller's)
float logit(attn4 q[8], uint key) {
    float s = 0;
    [unroll] for (uint c = 0; c < 8; ++c) s += (float)dot(q[c], (attn4)gs_k[key][c]);
    return s;
}

#include "neural_e4m3.hlsli"

float W(uint i) { return asfloat(g_w.Load(i * 4)); }
float4 W4(uint i) { return asfloat(g_w.Load4(i * 4)); }

float2 unpack2(uint v) { return float2(f16tof32(v & 0xFFFFu), f16tof32(v >> 16)); }
uint pack2(float a, float b) { return f32tof16(a) | (f32tof16(b) << 16); }

void load32(RWByteAddressBuffer buf, uint pixel, out float v[32]) {
    [unroll] for (uint q = 0; q < 4; ++q) {
        uint4 u = buf.Load4(pixel * 64 + q * 16);
        float2 a = unpack2(u.x), b = unpack2(u.y), c = unpack2(u.z), d = unpack2(u.w);
        v[q * 8 + 0] = a.x; v[q * 8 + 1] = a.y; v[q * 8 + 2] = b.x; v[q * 8 + 3] = b.y;
        v[q * 8 + 4] = c.x; v[q * 8 + 5] = c.y; v[q * 8 + 6] = d.x; v[q * 8 + 7] = d.y;
    }
}

void store32(RWByteAddressBuffer buf, uint pixel, float v[32]) {
    [unroll] for (uint q = 0; q < 4; ++q) {
        uint4 u;
        u.x = pack2(v[q * 8 + 0], v[q * 8 + 1]);
        u.y = pack2(v[q * 8 + 2], v[q * 8 + 3]);
        u.z = pack2(v[q * 8 + 4], v[q * 8 + 5]);
        u.w = pack2(v[q * 8 + 6], v[q * 8 + 7]);
        buf.Store4(pixel * 64 + q * 16, u);
    }
}

float dot32(float v[32], uint row) {
    float s = 0;
    [unroll] for (uint q = 0; q < 8; ++q) {
        float4 w = W4(row + q * 4);
        s += v[q * 4] * w.x + v[q * 4 + 1] * w.y + v[q * 4 + 2] * w.z + v[q * 4 + 3] * w.w;
    }
    return s;
}

// x * gate(x), the gate a clamped quadratic (Ops.gate_activation)
float gate_activation(float x) {
    float c = clamp(x, -4.0, 4.0);
    float slope = abs(c) * -0.055908203125 + 0.447265625;
    return (slope * c + 0.89453125) * x;
}

// Weights by wave broadcast. Every lane of a wave needs the same weight row, so each of the
// 16 lanes fetches a different element (one coalesced load per 16 floats) and the row is read
// back with WaveReadLaneAt at a constant lane, a register broadcast the compiler folds into
// the FMA. Every lane must be active wherever these are used: no divergent control flow —
// padding tokens run the same code on a zero input and only skip the final store.

// WAVE is the wave size the kernels are compiled for (16 by default; -D WAVE=32 for the
// 32-lane build): a 32-float row is two loads per lane at 16, one at 32.
#ifndef WAVE
#define WAVE 16
#endif

float bcast(float2 pair, uint c) {          // element c of a 32-float row fetched by row32
#if WAVE == 32
    return WaveReadLaneAt(pair.x, c);
#else
    return c < 16 ? WaveReadLaneAt(pair.x, c) : WaveReadLaneAt(pair.y, c - 16);
#endif
}

float2 row32(uint row, uint lane) {
#if WAVE == 32
    return float2(W(row + lane), 0);
#else
    return float2(W(row + lane), W(row + 16 + lane));
#endif
}

float wdot16(float v[16], uint row, uint lane) {
    const float a = W(row + (lane & 15));
    float s = 0;
    [unroll] for (uint k = 0; k < 16; ++k) s += v[k] * WaveReadLaneAt(a, k);
    return s;
}

// Four independent partial sums: one 32-long chain of dependent FMAs leaves the kernel waiting
// on FMA latency (measured: one chain 22.1 ms, two 20.1 ms at 960x576 for the whole graph).
float wdot32(float v[32], uint row, uint lane) {
    const float2 r = row32(row, lane);
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    [unroll] for (uint k = 0; k < 32; k += 4) {
        s0 += v[k] * bcast(r, k);
        s1 += v[k + 1] * bcast(r, k + 1);
        s2 += v[k + 2] * bcast(r, k + 2);
        s3 += v[k + 3] * bcast(r, k + 3);
    }
    return (s0 + s1) + (s2 + s3);
}

float wdot64(float v[64], uint row, uint lane) {
    float s = 0;
    [unroll] for (uint q = 0; q < 2; ++q) {
        const float2 r = row32(row + q * 32, lane);
        [unroll] for (uint k = 0; k < 32; ++k) s += v[q * 32 + k] * bcast(r, k);
    }
    return s;
}

// acc += g * row (32 floats)
void waxpy32(inout float acc[32], float g, uint row, uint lane) {
    const float2 r = row32(row, lane);
    [unroll] for (uint k = 0; k < 32; ++k) acc[k] += g * bcast(r, k);
}

// The block's input token at (y, x) of the out extent (coordinates already clamped).
void load_input(uint y, uint x, uint lane, out float v[32]) {
    const uint pixel = y * out_w + x;
    if (input_mode == 0) {
        load32(g_a, pixel, v);
    } else if (input_mode == 1) {
        // the 16 feature channels through the input adapter (32, 16)
        float f[16];
        [unroll] for (uint q = 0; q < 2; ++q) {
            uint4 u = g_a.Load4(pixel * 32 + q * 16);
            float2 a = unpack2(u.x), b = unpack2(u.y), c = unpack2(u.z), d = unpack2(u.w);
            f[q * 8 + 0] = a.x; f[q * 8 + 1] = a.y; f[q * 8 + 2] = b.x; f[q * 8 + 3] = b.y;
            f[q * 8 + 4] = c.x; f[q * 8 + 5] = c.y; f[q * 8 + 6] = d.x; f[q * 8 + 7] = d.y;
        }
        [unroll] for (uint c = 0; c < 32; ++c) v[c] = wdot16(f, o_extra0 + c * 16, lane);
    } else if (input_mode == 2) {
        // block 66: nearest-upsampled (coarse 64ch @ weight0 (32, 64)) + skip * sin
        const uint coarse = (y / 2) * in_w + (x / 2);
        float c64[64];
        [unroll] for (uint q = 0; q < 8; ++q) {
            uint4 u = g_a.Load4(coarse * 128 + q * 16);
            float2 a = unpack2(u.x), b = unpack2(u.y), c = unpack2(u.z), d = unpack2(u.w);
            c64[q * 8 + 0] = a.x; c64[q * 8 + 1] = a.y; c64[q * 8 + 2] = b.x; c64[q * 8 + 3] = b.y;
            c64[q * 8 + 4] = c.x; c64[q * 8 + 5] = c.y; c64[q * 8 + 6] = d.x; c64[q * 8 + 7] = d.y;
        }
        float skip[32];
        load32(g_b, pixel, skip);   // block 4's raw output: the skip is its publication
        const float2 sine = row32(o_extra1, lane);
        [unroll] for (uint c = 0; c < 32; ++c) v[c] = wdot64(c64, o_extra0 + c * 64, lane) + e4m3(skip[c]) * bcast(sine, c);
    } else {
        // block 70: nearest-upsampled coarse * merge_sin + full-resolution skip * merge_cos
        float coarse[32], skip[32];
        load32(g_a, (y / 2) * in_w + (x / 2), coarse);
        load32(g_b, pixel, skip);   // block 0's raw output: full_skip is its publication
        const float2 sine = row32(o_extra0, lane), cosine = row32(o_extra1, lane);
        [unroll] for (uint c = 0; c < 32; ++c) v[c] = coarse[c] * bcast(sine, c) + e4m3(skip[c]) * bcast(cosine, c);
    }
}

[WaveSize(WAVE)]
[numthreads(64, 1, 1)]
void CSWindowBlock32(uint3 group : SV_GroupID, uint token : SV_GroupIndex) {
    const uint lane = WaveGetLaneIndex();
    const int y = int(group.y * 8 + token / 8) - int(pad_top);
    const int x = int(group.x * 8 + token % 8) - int(pad_left);
    const bool inside = y >= 0 && x >= 0 && y < int(out_h) && x < int(out_w);
    const uint cy = uint(clamp(y, 0, int(out_h) - 1)), cx = uint(clamp(x, 0, int(out_w) - 1));

    // feed-forward, token-local: ff = (gate(x @ w1) @ w2) + x * ffn_cos. A padding token runs
    // on a zero input, which the gate keeps at zero all the way through.
    float xin[32];
    load_input(cy, cx, lane, xin);
    const float keep = inside ? 1.0 : 0.0;
    const float2 ffn_cos = row32(o_ffn_cos, lane);
    float ff[32];
    [unroll] for (uint c = 0; c < 32; ++c) {
        xin[c] *= keep;
        ff[c] = xin[c] * bcast(ffn_cos, c);   // the residual takes x as it is
        xin[c] = e4m3(xin[c]);                // the FFN its publication
    }
#if WAVE == 16
    if (half_ffn != 0) {
        // Half operands, as the vendor's tensor cores take them: each lane fetches one half2 of a
        // 32-wide row, dot2add widens the products into a float sum, and the hidden layer's
        // contributions accumulate in half2 (the vendor accumulates this GEMM in half too).
        half2 xh[16];
        [unroll] for (uint k = 0; k < 16; ++k) xh[k] = half2(xin[2 * k], xin[2 * k + 1]);
        half2 acc[16];
        [unroll] for (uint k2 = 0; k2 < 16; ++k2) acc[k2] = half2(0, 0);
        [loop] for (uint j = 0; j < 128; ++j) {
            const half2 a = g_w16.Load<half2>(o_w1_16 + (j * 32 + 2 * lane) * 2);
            float s = 0;
            [unroll] for (uint k3 = 0; k3 < 16; ++k3) s = dot2add(xh[k3], WaveReadLaneAt(a, k3), s);
            const half g = (half)e4m3(gate_activation(s));
            const half2 b = g_w16.Load<half2>(o_w2_16 + (j * 32 + 2 * lane) * 2);
            [unroll] for (uint k4 = 0; k4 < 16; ++k4) acc[k4] = mad(half2(g, g), WaveReadLaneAt(b, k4), acc[k4]);
        }
        [unroll] for (uint k5 = 0; k5 < 16; ++k5) {
            ff[2 * k5] += (float)acc[k5].x;
            ff[2 * k5 + 1] += (float)acc[k5].y;
        }
    } else
#endif
    {
        [loop] for (uint j = 0; j < 128; ++j) {
            const float g = e4m3(gate_activation(wdot32(xin, o_w1 + j * 32, lane)));
            waxpy32(ff, g, o_w2 + j * 32, lane);
        }
    }

    // The attention's residual takes ff as it is, Q/K/V its publication: keep the residual term
    // and round ff in place (no second 32-float array alive through the attention).
    const float2 attn_cos = row32(o_attn_cos, lane);
    float residual[32];
    [unroll] for (uint c0 = 0; c0 < 32; ++c0) {
        residual[c0] = ff[c0] * bcast(attn_cos, c0);
        ff[c0] = e4m3(ff[c0]);
    }

    // K (normalised in float before it is rounded to half), then Q (cosine-normalised, times the
    // folded scale) and V
    {
        float4 kv[8];
        float kk = 0;
        [unroll] for (uint c3 = 0; c3 < 8; ++c3) {
            [unroll] for (uint e = 0; e < 4; ++e) kv[c3][e] = wdot32(ff, o_qkv + (32 + c3 * 4 + e) * 32, lane);
            kk += dot(kv[c3], kv[c3]);
        }
        const float rk = rsqrt(max(kk, kNormFloor));
        [unroll] for (uint c4 = 0; c4 < 8; ++c4) gs_k[token][c4] = (half4)e4m3(kv[c4] * rk);
    }
    float4 qv[8];
    float qq = 0;
    [unroll] for (uint c3 = 0; c3 < 8; ++c3) {
        float4 v4;
        [unroll] for (uint e = 0; e < 4; ++e) {
            qv[c3][e] = wdot32(ff, o_qkv + (c3 * 4 + e) * 32, lane);
            v4[e] = wdot32(ff, o_qkv + (64 + c3 * 4 + e) * 32, lane);
        }
        qq += dot(qv[c3], qv[c3]);
        gs_v[token][c3] = (half4)e4m3(v4);
    }
    const float rq = rsqrt(max(qq, kNormFloor)) * W(o_scale);
    [unroll] for (uint c4 = 0; c4 < 8; ++c4) qv[c4] = e4m3(qv[c4] * rq);
    attn4 qa[8];
    [unroll] for (uint c9 = 0; c9 < 8; ++c9) qa[c9] = (attn4)qv[c9];
    GroupMemoryBarrierWithGroupSync();

    // The window's attention for this query: clamped logits, softmax, values. The vendor publishes
    // the probabilities normalised, so the total comes first — one pass over the keys for it, a
    // second for the values with the logits recomputed (64 kept per thread would cost occupancy).
    // The exponent is shifted by the clamp's top, which cancels in 1 / total and keeps p <= 1.
    // Without E4M3 (the fast variant) one pass, normalised at the end.
    float total = 0;
    float4 o4[8];
    [unroll] for (uint c5 = 0; c5 < 8; ++c5) o4[c5] = 0;
#if E4M3
    [loop] for (uint j = 0; j < 64; j += 4) {
        const float4 bias = W4(o_bias + token * 64 + j);
        [unroll] for (uint jj = 0; jj < 4; ++jj) {
            total += exp(clamp(bias[jj] + logit(qa, j + jj), kLogitLow, kLogitHigh) - kLogitHigh);
        }
    }
    const float inv = 1.0 / total;
#endif
    [loop] for (uint j2 = 0; j2 < 64; j2 += 4) {
        const float4 bias = W4(o_bias + token * 64 + j2);
        attn4 acc[8];   // 4 keys' p * V (half with ATTN_HALF), then into the float sum
        [unroll] for (uint ca = 0; ca < 8; ++ca) acc[ca] = 0;
        [unroll] for (uint jj = 0; jj < 4; ++jj) {
            float p = exp(clamp(bias[jj] + logit(qa, j2 + jj), kLogitLow, kLogitHigh) - kLogitHigh);
#if E4M3
            p = e4m3(p * inv);
#else
            total += p;
#endif
            [unroll] for (uint c6 = 0; c6 < 8; ++c6) acc[c6] = mad((attn1)p, (attn4)gs_v[j2 + jj][c6], acc[c6]);
        }
        [unroll] for (uint cb = 0; cb < 8; ++cb) o4[cb] += (float4)acc[cb];
    }
#if !E4M3
    const float inv = 1.0 / total;
    [unroll] for (uint cn = 0; cn < 8; ++cn) o4[cn] *= inv;
#endif
    float o[32];
    [unroll] for (uint c7 = 0; c7 < 32; ++c7) o[c7] = e4m3(o4[c7 / 4][c7 % 4]);

    float result[32];
    [unroll] for (uint c8 = 0; c8 < 32; ++c8) {
        result[c8] = wdot32(o, o_proj + c8 * 32, lane) + residual[c8];
        if (publish != 0) result[c8] = e4m3(result[c8]);
    }

    const uint pixel = cy * out_w + cx;
    if (output_mode == 0) {
        if (inside) store32(g_out, pixel, result);
    } else {
        // block 70 feeds the head: result[:16] @ out_gain + result[16:] @ out_conv, both (4, 16)
        float low[16], high[16];
        [unroll] for (uint i = 0; i < 16; ++i) {
            low[i] = result[i];
            high[i] = result[16 + i];
        }
        float head[4];
        [unroll] for (uint k2 = 0; k2 < 4; ++k2) {
            head[k2] = wdot16(low, o_extra2 + k2 * 16, lane) + wdot16(high, o_extra3 + k2 * 16, lane);
        }
        if (inside) g_out.Store2(pixel * 8, uint2(pack2(head[0], head[1]), pack2(head[2], head[3])));
    }
}

void pooled32(uint x, uint y, out float v[32]) {
    float a[32], b[32], c[32], d[32];
    load32(g_a, (2 * y) * in_w + 2 * x, a);
    load32(g_a, (2 * y + 1) * in_w + 2 * x, b);
    load32(g_a, (2 * y) * in_w + 2 * x + 1, c);
    load32(g_a, (2 * y + 1) * in_w + 2 * x + 1, d);
    // the pool of a transition's raw output, published (average_pool2 then e4m3)
    [unroll] for (uint i = 0; i < 32; ++i) v[i] = e4m3((a[i] + b[i] + c[i] + d[i]) * 0.25);
}

[numthreads(8, 8, 1)]
void CSPool2(uint3 id : SV_DispatchThreadID) {
    if (id.x >= out_w || id.y >= out_h) return;
    float v[32];
    pooled32(id.x, id.y, v);
    store32(g_out, id.y * out_w + id.x, v);
}

[numthreads(8, 8, 1)]
void CSPoolProject(uint3 id : SV_DispatchThreadID) {
    if (id.x >= out_w || id.y >= out_h) return;
    float v[32];
    pooled32(id.x, id.y, v);
    const uint pixel = id.y * out_w + id.x;
    [unroll] for (uint q = 0; q < 16; ++q) {
        float a = dot32(v, o_extra0 + (q * 4) * 32), b = dot32(v, o_extra0 + (q * 4 + 1) * 32);
        float c = dot32(v, o_extra0 + (q * 4 + 2) * 32), d = dot32(v, o_extra0 + (q * 4 + 3) * 32);
        g_out.Store2(pixel * 128 + q * 8, uint2(pack2(e4m3(a), e4m3(b)), pack2(e4m3(c), e4m3(d))));
    }
}
