// E4M3 ("publish"), shared by neural_blocks.hlsl and neural_blocks_wide.hlsl.
//
// The vendor rounds activations to FP8 E4M3 — every GEMM's input, Q/K/V, the attention output,
// the block outputs (tools/hnet/model.py, Ops.e4m3 and its callers). Skipping it cost the most
// fidelity of all the deploy shortcuts with the real weights (the effect against the oracle:
// corr 0.79 without, ~0.96 with; PROGRESS-neural.md, 2026-11-04). Round half to even,
// saturating at 448 like the reference's `_e4m3_chunk` (the same algorithm checked bit for bit
// against torch.float8_e4m3fn over every half value and 5 M random floats). Every E4M3 value is
// exact in half, so published values survive the half storage between kernels.
// E4M3 0 compiles it out (the old `nq` numerics).

#ifndef E4M3
#define E4M3 1
#endif

float e4m3(float x) {
#if E4M3
    // on the signed value: the mantissa rounding leaves the sign bit alone and round() is symmetric
    const float c = clamp(x, -448.0, 448.0);
    const uint u = asuint(c);
    const float normal = asfloat((u + 0x7FFFFu + ((u >> 20) & 1u)) & 0xFFF00000u);   // 3 mantissa bits
    const float subnormal = round(c * 512.0) * (1.0 / 512.0);                         // multiples of 2^-9
    return abs(c) >= 0.015625 ? normal : subnormal;
#else
    return x;
#endif
}

float4 e4m3(float4 v) { return float4(e4m3(v.x), e4m3(v.y), e4m3(v.z), e4m3(v.w)); }
