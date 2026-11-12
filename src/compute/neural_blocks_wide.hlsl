// DLSS-NR hybrid runner, "deep" split: HNet's branched window blocks at C = 64, 128, 256
// (blocks 5-22 and 48-65) and the transitions between their levels, as HLSL.
//
// A wide block does not fit one thread's registers the way the 32-channel blocks do
// (neural_blocks.hlsl), so it runs as four kernels, each reading and writing its tensors once:
//
//   CSBranchedFFN      hs = branched_ffn(x)                       per (token, output head)
//   CSProjectResidual  out = in @ W + res * cos                   per (token, 32-channel group)
//                        (the FFN's output projection, then the attention's projection)
//   CSWindowAttention  o = window attention of ff, one head        per (window, head)
//   CSProjectResidual  (again)
//
// and the transitions:
//   CSPoolProjectN     2x2 average pool (zero beyond the input, block 22's pad) then C -> C'
//   CSUpMerge          nearest-upsampled (coarse @ W0) + skip * sin   (decoder blocks 48, 56, 62)
//
// Reference: tools/hnet/model.py in deploy numerics nq-exp (no E4M3 publishes), float32 math,
// half storage. Weights from tools/hnet/export_hybrid.py: every matrix as (out, in).
// Tensors are NHWC half packed two per uint; C channels = C * 2 bytes per pixel; a 32-channel
// group g of pixel p sits at p * C * 2 + g * 64.

cbuffer WideConstants : register(b0) {
    uint out_w;        // extent of the tokens / outputs
    uint out_h;
    uint in_w;         // extent of input A where it differs (pool: finer; up-merge: coarser)
    uint in_h;
    uint pad_top;      // window origin of the attention
    uint pad_left;
    uint channels;     // C of the block (pool/up-merge: input channels)
    uint out_channels; // pool/up-merge: output channels
    uint o_a;          // primary matrix (out, in)
    uint o_b;          // secondary: FFN branch projection (G, 128, 32)
    uint o_cos;        // residual gate (C), or up-merge sin
    uint o_scale;      // attention: per-head scale (H)
    uint o_bias;       // attention: bias (H, 64, 64)
    uint publish;      // gate / residual / project-residual: 1 publishes (E4M3) the output
    uint pad1;
    uint pad2;
    uint pad3;
    uint pad4;
    uint pad5;
    uint pad6;
};

static const float kNormFloor = 0.00006198883056640625;
static const float kSoftmaxBase = 0.99639907;   // 1.4375 ln 2, folded into scale and bias
static const float kLogitLow = -6.0 * kSoftmaxBase;
static const float kLogitHigh = 5.98 * kSoftmaxBase;

ByteAddressBuffer g_w : register(t0);
RWByteAddressBuffer g_a : register(u0);
RWByteAddressBuffer g_b : register(u1);
RWByteAddressBuffer g_out : register(u2);

// K and V of 64 keys, 4 channels per element, in half (8 KB). Unlike the 32-channel blocks these
// kernels were not bound by groupshared (float, 16 KB, ran the same): the half copy is there for
// the half math below (ATTN_HALF).
groupshared half4 gs_k[64][8];
groupshared half4 gs_v[64][8];

#include "neural_e4m3.hlsli"

float W(uint i) { return asfloat(g_w.Load(i * 4)); }
float4 W4(uint i) { return asfloat(g_w.Load4(i * 4)); }
float2 unpack2(uint v) { return float2(f16tof32(v & 0xFFFFu), f16tof32(v >> 16)); }
uint pack2(float a, float b) { return f32tof16(a) | (f32tof16(b) << 16); }

// 32 channels of group `g` of pixel `p` in a C-channel tensor
void load_group(RWByteAddressBuffer buf, uint p, uint c, uint g, out float v[32]) {
    const uint base = p * c * 2 + g * 64;
    [unroll] for (uint q = 0; q < 4; ++q) {
        uint4 u = buf.Load4(base + q * 16);
        float2 a = unpack2(u.x), b = unpack2(u.y), cc = unpack2(u.z), d = unpack2(u.w);
        v[q * 8 + 0] = a.x; v[q * 8 + 1] = a.y; v[q * 8 + 2] = b.x; v[q * 8 + 3] = b.y;
        v[q * 8 + 4] = cc.x; v[q * 8 + 5] = cc.y; v[q * 8 + 6] = d.x; v[q * 8 + 7] = d.y;
    }
}

void store_group(RWByteAddressBuffer buf, uint p, uint c, uint g, float v[32]) {
    const uint base = p * c * 2 + g * 64;
    [unroll] for (uint q = 0; q < 4; ++q) {
        buf.Store4(base + q * 16, uint4(pack2(v[q * 8 + 0], v[q * 8 + 1]), pack2(v[q * 8 + 2], v[q * 8 + 3]),
                                        pack2(v[q * 8 + 4], v[q * 8 + 5]), pack2(v[q * 8 + 6], v[q * 8 + 7])));
    }
}

// Weights by wave broadcast (see neural_blocks.hlsl): all lanes active wherever these run.
float wdot32(float v[32], uint row, uint lane) {
    const float a = W(row + lane), b = W(row + 16 + lane);
    float s = 0;
    [unroll] for (uint k = 0; k < 16; ++k) s += v[k] * WaveReadLaneAt(a, k) + v[16 + k] * WaveReadLaneAt(b, k);
    return s;
}

void waxpy32(inout float acc[32], float g, uint row, uint lane) {
    const float a = W(row + lane), b = W(row + 16 + lane);
    [unroll] for (uint k = 0; k < 16; ++k) {
        acc[k] += g * WaveReadLaneAt(a, k);
        acc[16 + k] += g * WaveReadLaneAt(b, k);
    }
}

float bcast(float2 pair, uint c) { return c < 16 ? WaveReadLaneAt(pair.x, c) : WaveReadLaneAt(pair.y, c - 16); }

void store_kv(uint slot, float k[32], float rk, float v[32]) {
    [unroll] for (uint c = 0; c < 8; ++c) {
        gs_k[slot][c] = (half4)e4m3(float4(k[4 * c], k[4 * c + 1], k[4 * c + 2], k[4 * c + 3]) * rk);
        gs_v[slot][c] = (half4)e4m3(float4(v[4 * c], v[4 * c + 1], v[4 * c + 2], v[4 * c + 3]));
    }
}

// ATTN_HALF 1 (as in neural_blocks.hlsl): half math where K and V are read — q.k per 4-channel
// slice as a half dot, p * V accumulated in half over 4 keys (p shifted to <= 1). Window
// attentions 2.58 -> 2.18 ms at 960x576. 0: the same loops in float.
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

float dot_key(attn4 q[8], uint slot) {
    float s = 0;
    [unroll] for (uint c = 0; c < 8; ++c) s += (float)dot(q[c], (attn4)gs_k[slot][c]);
    return s;
}

void add_value(inout attn4 acc[8], attn1 e, uint slot) {
    [unroll] for (uint c = 0; c < 8; ++c) acc[c] = mad(e, (attn4)gs_v[slot][c], acc[c]);
}

void unpack_o(float4 o4[8], float inv, out float o[32]) {
    [unroll] for (uint c = 0; c < 32; ++c) o[c] = e4m3(o4[c / 4][c % 4] * inv);
}

float gate_activation(float x) {
    float c = clamp(x, -4.0, 4.0);
    float slope = abs(c) * -0.055908203125 + 0.447265625;
    return (slope * c + 0.89453125) * x;
}

// out[32] = in (C channels of pixel p) @ rows [row0, row0 + 32) of the (out, C) matrix at `mat`
void project_group(RWByteAddressBuffer buf, uint p, uint c, uint mat, uint row0, uint lane, out float acc[32]) {
    [unroll] for (uint k = 0; k < 32; ++k) acc[k] = 0;
    [loop] for (uint i = 0; i < c / 32; ++i) {
        float chunk[32];
        load_group(buf, p, c, i, chunk);
        [unroll] for (uint k2 = 0; k2 < 32; ++k2) acc[k2] += wdot32(chunk, mat + (row0 + k2) * c + i * 32, lane);
    }
}

// hs[t][head oh] = sum over 128 hidden units u of gate(x . E[oh*128 + u]) * P[oh][u]
// E is the fused expansion (G*128, C), P the fused branch projection (G, 128, 32).
[WaveSize(16)]
[numthreads(64, 1, 1)]
void CSBranchedFFN(uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    const uint lane = WaveGetLaneIndex();
    const uint tokens = out_w * out_h;
    const uint t = min(group.x * 64 + index, tokens - 1);
    const uint oh = group.y;
    const uint c = channels;
    float hb[32];
    [unroll] for (uint k = 0; k < 32; ++k) hb[k] = 0;
    [loop] for (uint ub = 0; ub < 4; ++ub) {
        float acc[32];
        [unroll] for (uint k2 = 0; k2 < 32; ++k2) acc[k2] = 0;
        [loop] for (uint i = 0; i < c / 32; ++i) {
            float chunk[32];
            load_group(g_a, t, c, i, chunk);
            [unroll] for (uint u = 0; u < 32; ++u) acc[u] += wdot32(chunk, o_a + (oh * 128 + ub * 32 + u) * c + i * 32, lane);
        }
        [unroll] for (uint u2 = 0; u2 < 32; ++u2) {
            waxpy32(hb, e4m3(gate_activation(acc[u2])), o_b + (oh * 128 + ub * 32 + u2) * 32, lane);
        }
    }
    [unroll] for (uint k3 = 0; k3 < 32; ++k3) hb[k3] = e4m3(hb[k3]);
    if (group.x * 64 + index < tokens) store_group(g_out, t, c, oh, hb);
}

// out[t][g] = in[t] @ W[g rows] + res[t][g] * cos[g]
[WaveSize(16)]
[numthreads(64, 1, 1)]
void CSProjectResidual(uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    const uint lane = WaveGetLaneIndex();
    const uint tokens = out_w * out_h;
    const uint t = min(group.x * 64 + index, tokens - 1);
    const uint g = group.y;
    float acc[32];
    project_group(g_a, t, channels, o_a, g * 32, lane, acc);
    float res[32];
    load_group(g_b, t, channels, g, res);
    const float2 gate = float2(W(o_cos + g * 32 + lane), W(o_cos + g * 32 + 16 + lane));
    [unroll] for (uint k = 0; k < 32; ++k) {
        acc[k] += res[k] * bcast(gate, k);
        if (publish != 0) acc[k] = e4m3(acc[k]);
    }
    if (group.x * 64 + index < tokens) store_group(g_out, t, channels, g, acc);
}

// One head of one shifted window: q, k, v from ff (C channels), cosine-normalised, the 64x64
// logits plus bias, clamped, softmax, values. qkv rows: (H, 96, C), [q|k|v] per head.
[WaveSize(16)]
[numthreads(64, 1, 1)]
void CSWindowAttention(uint3 group : SV_GroupID, uint token : SV_GroupIndex) {
    const uint lane = WaveGetLaneIndex();
    const uint h = group.z;
    const uint c = channels;
    const int y = int(group.y * 8 + token / 8) - int(pad_top);
    const int x = int(group.x * 8 + token % 8) - int(pad_left);
    const bool inside = y >= 0 && x >= 0 && y < int(out_h) && x < int(out_w);
    const uint p = uint(clamp(y, 0, int(out_h) - 1)) * out_w + uint(clamp(x, 0, int(out_w) - 1));
    const float keep = inside ? 1.0 : 0.0;   // padding tokens: ff = 0, so q = k = v = 0

    float q[32], k[32], v[32];
    [unroll] for (uint z = 0; z < 32; ++z) { q[z] = 0; k[z] = 0; v[z] = 0; }
    const uint rows = o_a + h * 96 * c;
    [loop] for (uint i = 0; i < c / 32; ++i) {
        float chunk[32];
        load_group(g_a, p, c, i, chunk);
        [unroll] for (uint m = 0; m < 32; ++m) chunk[m] *= keep;
        [unroll] for (uint r = 0; r < 32; ++r) {
            q[r] += wdot32(chunk, rows + r * c + i * 32, lane);
            k[r] += wdot32(chunk, rows + (32 + r) * c + i * 32, lane);
            v[r] += wdot32(chunk, rows + (64 + r) * c + i * 32, lane);
        }
    }
    float qq = 0, kk = 0;
    [unroll] for (uint r2 = 0; r2 < 32; ++r2) { qq += q[r2] * q[r2]; kk += k[r2] * k[r2]; }
    const float rq = rsqrt(max(qq, kNormFloor)) * W(o_scale + h);
    const float rk = rsqrt(max(kk, kNormFloor));
    float4 q4[8];
    [unroll] for (uint r3 = 0; r3 < 8; ++r3) q4[r3] = e4m3(float4(q[4 * r3], q[4 * r3 + 1], q[4 * r3 + 2], q[4 * r3 + 3]) * rq);
    store_kv(token, k, rk, v);
    GroupMemoryBarrierWithGroupSync();

    float4 o4[8];
    [unroll] for (uint r4 = 0; r4 < 8; ++r4) o4[r4] = 0;
    float total = 0;
    const uint bias = o_bias + (h * 64 + token) * 64;
    attn4 qh[8];
    [unroll] for (uint r9 = 0; r9 < 8; ++r9) qh[r9] = (attn4)q4[r9];
    // the probabilities are published normalised (as in neural_blocks.hlsl): the total first,
    // then the values with the logits recomputed; without E4M3 one pass, normalised at the end
#if E4M3
    [loop] for (uint j = 0; j < 64; j += 4) {
        const float4 b4 = W4(bias + j);
        [unroll] for (uint jj = 0; jj < 4; ++jj) {
            total += exp(clamp(b4[jj] + dot_key(qh, j + jj), kLogitLow, kLogitHigh) - kLogitHigh);
        }
    }
    const float inv = 1.0 / total;
#endif
    [loop] for (uint j2 = 0; j2 < 64; j2 += 4) {
        const float4 b4 = W4(bias + j2);
        attn4 acc[8];
        [unroll] for (uint ca = 0; ca < 8; ++ca) acc[ca] = 0;
        [unroll] for (uint jj = 0; jj < 4; ++jj) {
            float e = exp(clamp(b4[jj] + dot_key(qh, j2 + jj), kLogitLow, kLogitHigh) - kLogitHigh);
#if E4M3
            e = e4m3(e * inv);
#else
            total += e;
#endif
            add_value(acc, (attn1)e, j2 + jj);
        }
        [unroll] for (uint cb = 0; cb < 8; ++cb) o4[cb] += (float4)acc[cb];
    }
    float o[32];
#if E4M3
    unpack_o(o4, 1.0, o);
#else
    unpack_o(o4, 1.0 / total, o);
#endif
    if (inside) store_group(g_out, p, c, h, o);
}

// out[p][g'] = pool2(in)[p] @ W0[g' rows] — C_in = channels, C_out = out_channels; input pixels
// beyond (in_w, in_h) read as zero (block 22 pads to a multiple of 8 before it pools)
[WaveSize(16)]
[numthreads(64, 1, 1)]
void CSPoolProjectN(uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    const uint lane = WaveGetLaneIndex();
    const uint tokens = out_w * out_h;
    const uint t = min(group.x * 64 + index, tokens - 1);
    const uint ox = t % out_w, oy = t / out_w;
    const uint g = group.y;
    const uint c = channels;
    float acc[32];
    [unroll] for (uint k = 0; k < 32; ++k) acc[k] = 0;
    [loop] for (uint i = 0; i < c / 32; ++i) {
        float pooled[32];
        [unroll] for (uint m = 0; m < 32; ++m) pooled[m] = 0;
        // (0,0) + (1,0) + (0,1) + (1,1), as average_pool2 adds them
        [unroll] for (uint tap = 0; tap < 4; ++tap) {
            const uint ix = 2 * ox + tap / 2, iy = 2 * oy + tap % 2;
            float part[32];
            load_group(g_a, min(iy, in_h - 1) * in_w + min(ix, in_w - 1), c, i, part);
            const float valid = (ix < in_w && iy < in_h) ? 1.0 : 0.0;
            [unroll] for (uint m2 = 0; m2 < 32; ++m2) pooled[m2] += part[m2] * valid;
        }
        [unroll] for (uint m3 = 0; m3 < 32; ++m3) pooled[m3] = e4m3(pooled[m3] * 0.25);
        [unroll] for (uint r = 0; r < 32; ++r) acc[r] += wdot32(pooled, o_a + (g * 32 + r) * c + i * 32, lane);
    }
    [unroll] for (uint k = 0; k < 32; ++k) acc[k] = e4m3(acc[k]);
    if (group.x * 64 + index < tokens) store_group(g_out, t, out_channels, g, acc);
}

// merged[p][g'] = (coarse[p / 2] @ W0[g' rows]) + skip[p][g'] * sin[g'] — C_in = channels
// (coarse), C_out = out_channels (fine, the skip's)
[WaveSize(16)]
[numthreads(64, 1, 1)]
void CSUpMerge(uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    const uint lane = WaveGetLaneIndex();
    const uint tokens = out_w * out_h;
    const uint t = min(group.x * 64 + index, tokens - 1);
    const uint x = t % out_w, y = t / out_w;
    const uint g = group.y;
    float acc[32];
    project_group(g_a, (y / 2) * in_w + (x / 2), channels, o_a, g * 32, lane, acc);
    float skip[32];
    load_group(g_b, t, out_channels, g, skip);
    const float2 sine = float2(W(o_cos + g * 32 + lane), W(o_cos + g * 32 + 16 + lane));
    [unroll] for (uint k = 0; k < 32; ++k) acc[k] = e4m3(acc[k] + e4m3(skip[k]) * bcast(sine, k));
    if (group.x * 64 + index < tokens) store_group(g_out, t, out_channels, g, acc);
}

// ---------------------------------------------------------------------------------------
// With the projections on DirectML GEMMs (XMX, src/neural/dml_gemm.cpp) the wide block is:
//   GEMM x @ expand -> h; CSGate(h); batched GEMM h @ branch -> hs; GEMM hs @ out_proj -> t;
//   CSResidual ff = t + x * ffn_cos; GEMM ff @ qkv -> qkv; CSWindowAttentionQKV -> o;
//   GEMM o @ proj -> t; CSResidual out = t + ff * attn_cos.
// Elementwise kernels take 8 halves (one uint4) per thread.

[numthreads(256, 1, 1)]
void CSGate(uint3 id : SV_DispatchThreadID) {
    const uint i = id.y * 65535 * 256 + id.x;     // out_w = count of uint4s
    if (i >= out_w) return;
    uint4 u = g_out.Load4(i * 16);
    uint words[4] = {u.x, u.y, u.z, u.w};
    [unroll] for (uint k = 0; k < 4; ++k) {
        float2 v = unpack2(words[k]);
        float2 g = float2(gate_activation(v.x), gate_activation(v.y));
        if (publish != 0) g = float2(e4m3(g.x), e4m3(g.y));
        words[k] = pack2(g.x, g.y);
    }
    g_out.Store4(i * 16, uint4(words[0], words[1], words[2], words[3]));
}

// in place: out = e4m3(out); out_w = count of uint4s
[numthreads(256, 1, 1)]
void CSQuantize(uint3 id : SV_DispatchThreadID) {
    const uint i = id.y * 65535 * 256 + id.x;
    if (i >= out_w) return;
    uint4 u = g_out.Load4(i * 16);
    uint words[4] = {u.x, u.y, u.z, u.w};
    [unroll] for (uint k = 0; k < 4; ++k) {
        const float2 v = unpack2(words[k]);
        words[k] = pack2(e4m3(v.x), e4m3(v.y));
    }
    g_out.Store4(i * 16, uint4(words[0], words[1], words[2], words[3]));
}

// out = a + b * cos[channel]; out_w = count of uint4s, channels = C
[numthreads(256, 1, 1)]
void CSResidual(uint3 id : SV_DispatchThreadID) {
    const uint i = id.y * 65535 * 256 + id.x;
    if (i >= out_w) return;
    const uint channel = (i * 8) % channels;
    const uint4 a = g_a.Load4(i * 16), b = g_b.Load4(i * 16);
    const uint aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
    uint words[4];
    [unroll] for (uint k = 0; k < 4; ++k) {
        const float2 av = unpack2(aw[k]), bv = unpack2(bw[k]);
        float2 r = float2(av.x + bv.x * W(o_cos + channel + 2 * k), av.y + bv.y * W(o_cos + channel + 2 * k + 1));
        if (publish != 0) r = float2(e4m3(r.x), e4m3(r.y));
        words[k] = pack2(r.x, r.y);
    }
    g_out.Store4(i * 16, uint4(words[0], words[1], words[2], words[3]));
}

// One head of one shifted window from the GEMM's q|k|v rows (T, 3C; head h at h * 96):
// cosine norm (q times the folded scale), logits + bias, clamp, softmax, values -> o (T, C).
[numthreads(64, 1, 1)]
void CSWindowAttentionQKV(uint3 group : SV_GroupID, uint token : SV_GroupIndex) {
    const uint h = group.z;
    const uint c = channels;
    const int y = int(group.y * 8 + token / 8) - int(pad_top);
    const int x = int(group.x * 8 + token % 8) - int(pad_left);
    const bool inside = y >= 0 && x >= 0 && y < int(out_h) && x < int(out_w);
    const uint p = uint(clamp(y, 0, int(out_h) - 1)) * out_w + uint(clamp(x, 0, int(out_w) - 1));
    const float keep = inside ? 1.0 : 0.0;   // padding tokens: ff = 0, so q = k = v = 0

    float q[32], k[32], v[32];
    load_group(g_a, p * 3, c, h * 3, q);        // row p of a 3C-wide tensor = "pixel" 3p of a C-wide one
    load_group(g_a, p * 3, c, h * 3 + 1, k);
    load_group(g_a, p * 3, c, h * 3 + 2, v);
    float qq = 0, kk = 0;
    [unroll] for (uint r = 0; r < 32; ++r) {
        q[r] *= keep; k[r] *= keep; v[r] *= keep;
        qq += q[r] * q[r];
        kk += k[r] * k[r];
    }
    const float rq = rsqrt(max(qq, kNormFloor)) * W(o_scale + h);
    const float rk = rsqrt(max(kk, kNormFloor));
    float4 q4[8];
    [unroll] for (uint r3 = 0; r3 < 8; ++r3) q4[r3] = e4m3(float4(q[4 * r3], q[4 * r3 + 1], q[4 * r3 + 2], q[4 * r3 + 3]) * rq);
    store_kv(token, k, rk, v);
    GroupMemoryBarrierWithGroupSync();

    float4 o4[8];
    [unroll] for (uint r4 = 0; r4 < 8; ++r4) o4[r4] = 0;
    float total = 0;
    const uint bias = o_bias + (h * 64 + token) * 64;
    attn4 qh[8];
    [unroll] for (uint r9 = 0; r9 < 8; ++r9) qh[r9] = (attn4)q4[r9];
    // the probabilities are published normalised (as in neural_blocks.hlsl): the total first,
    // then the values with the logits recomputed; without E4M3 one pass, normalised at the end
#if E4M3
    [loop] for (uint j = 0; j < 64; j += 4) {
        const float4 b4 = W4(bias + j);
        [unroll] for (uint jj = 0; jj < 4; ++jj) {
            total += exp(clamp(b4[jj] + dot_key(qh, j + jj), kLogitLow, kLogitHigh) - kLogitHigh);
        }
    }
    const float inv = 1.0 / total;
#endif
    [loop] for (uint j2 = 0; j2 < 64; j2 += 4) {
        const float4 b4 = W4(bias + j2);
        attn4 acc[8];
        [unroll] for (uint ca = 0; ca < 8; ++ca) acc[ca] = 0;
        [unroll] for (uint jj = 0; jj < 4; ++jj) {
            float e = exp(clamp(b4[jj] + dot_key(qh, j2 + jj), kLogitLow, kLogitHigh) - kLogitHigh);
#if E4M3
            e = e4m3(e * inv);
#else
            total += e;
#endif
            add_value(acc, (attn1)e, j2 + jj);
        }
        [unroll] for (uint cb = 0; cb < 8; ++cb) o4[cb] += (float4)acc[cb];
    }
    float o[32];
#if E4M3
    unpack_o(o4, 1.0, o);
#else
    unpack_o(o4, 1.0 / total, o);
#endif
    if (inside) store_group(g_out, p, c, h, o);
}

// ---------------------------------------------------------------------------------------
// The ViT bottleneck (blocks 31-38): attention over every token of the 1/64 field, 32 heads.
// qkv rows (T, 3072) = [q 1024 | k 1024 | v 1024], head h at h * 32 inside each third.
// q is normalised and scaled by half(sqrt(32)) and the learned scale (the softmax base
// folded in); the logits are clamped to the ViT's range; padding keys do not exist here.
// One thread per query token, one group z per head; keys stream through groupshared in
// tiles of 64. out_w = tokens, o_scale = the 32 learned scales.

static const float kVitBase = 0.99301456;          // 1.4326171875 ln 2
static const float kVitRoot = 5.65625;             // half(sqrt(32))

groupshared float gs_valid[64];

[numthreads(64, 1, 1)]
void CSGlobalAttention(uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    const uint tokens = out_w;
    const uint h = group.z;
    const uint t = min(group.x * 64 + index, tokens - 1);
    float q[32];
    load_group(g_a, t * 3, 1024, h, q);           // row t of a 3072-wide tensor, group h of its first third
    float qq = 0;
    [unroll] for (uint r = 0; r < 32; ++r) qq += q[r] * q[r];
    const float rq = rsqrt(max(qq, kNormFloor)) * kVitRoot * W(o_scale + h);
    float4 q4[8];
    [unroll] for (uint r2 = 0; r2 < 8; ++r2) q4[r2] = e4m3(float4(q[4 * r2], q[4 * r2 + 1], q[4 * r2 + 2], q[4 * r2 + 3]) * rq);

    attn4 qh[8];
    [unroll] for (uint r9 = 0; r9 < 8; ++r9) qh[r9] = (attn4)q4[r9];
    float4 o4[8];
    [unroll] for (uint r3 = 0; r3 < 8; ++r3) o4[r3] = 0;
    float total = 0;
    for (uint base = 0; base < tokens; base += 64) {
        const uint key = base + index;
        const uint kt = min(key, tokens - 1);
        float k[32], v[32];
        load_group(g_a, kt * 3, 1024, 32 + h, k);   // the second third: 32 groups of 32 further on
        load_group(g_a, kt * 3, 1024, 64 + h, v);
        float kk = 0;
        [unroll] for (uint r4 = 0; r4 < 32; ++r4) kk += k[r4] * k[r4];
        const float rk = rsqrt(max(kk, kNormFloor));
        GroupMemoryBarrierWithGroupSync();
        store_kv(index, k, rk, v);
        gs_valid[index] = key < tokens ? 1.0 : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [loop] for (uint j = 0; j < 64; j += 4) {
            attn4 acc[8];
            [unroll] for (uint ca = 0; ca < 8; ++ca) acc[ca] = 0;
            [unroll] for (uint jj = 0; jj < 4; ++jj) {
                const float e = exp(clamp(dot_key(qh, j + jj), -3.01 * kVitBase, 3.0 * kVitBase) - 3.0 * kVitBase) *
                                gs_valid[j + jj];
                total += e;
                add_value(acc, (attn1)e, j + jj);
            }
            [unroll] for (uint cb = 0; cb < 8; ++cb) o4[cb] += (float4)acc[cb];
        }
    }
    float o[32];
    unpack_o(o4, 1.0 / total, o);
    if (group.x * 64 + index < tokens) store_group(g_out, t, 1024, h, o);
}
