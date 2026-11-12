cbuffer ImageParams : register(b0) {
    uint g_width;
    uint g_height;
    float g_sharpness;
    float g_blend_factor;
};

Texture2D<float4> InputTex0 : register(t0);
Texture2D<float4> InputTex1 : register(t1);
RWTexture2D<float4> OutputTex : register(u0);

[numthreads(16, 16, 1)]
void CSImageSpatialFilter(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= g_width || dtid.y >= g_height) {
        return;
    }

    int2 pos = int2(dtid.xy);
    int2 max_pos = int2(int(g_width) - 1, int(g_height) - 1);

    // 3x3 Gaussian-like spatial kernel
    static const float weights[3][3] = {
        {0.0625f, 0.1250f, 0.0625f},
        {0.1250f, 0.2500f, 0.1250f},
        {0.0625f, 0.1250f, 0.0625f}
    };

    float4 accum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            int2 coord = clamp(pos + int2(dx, dy), int2(0, 0), max_pos);
            float4 sample_val = InputTex0.Load(int3(coord, 0));
            accum += sample_val * weights[dy + 1][dx + 1];
        }
    }

    OutputTex[pos] = float4(accum.rgb, 1.0f);
}

[numthreads(16, 16, 1)]
void CSImageResidualReconstruct(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= g_width || dtid.y >= g_height) {
        return;
    }

    int2 pos = int2(dtid.xy);
    float4 filtered = InputTex0.Load(int3(pos, 0));
    float4 original = InputTex1.Load(int3(pos, 0));

    // Unsharp-mask + residual blend representative of neural detail reconstruction post-pass
    float3 detail = original.rgb - filtered.rgb;
    float3 reconstructed = saturate(lerp(original.rgb, filtered.rgb, g_blend_factor) + detail * g_sharpness);

    OutputTex[pos] = float4(reconstructed, 1.0f);
}
