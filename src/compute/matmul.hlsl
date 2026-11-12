cbuffer MatMulParams : register(b0) {
    uint g_M;
    uint g_N;
    uint g_K;
    uint g_flags;
};

StructuredBuffer<float> MatrixA : register(t0);
StructuredBuffer<float> MatrixB : register(t1);
RWStructuredBuffer<float> MatrixC : register(u0);

// 1. Naive Global-Memory FP32 Matrix Multiplication
[numthreads(16, 16, 1)]
void CSMatMulNaive(uint3 dtid : SV_DispatchThreadID) {
    uint row = dtid.y;
    uint col = dtid.x;
    if (row >= g_M || col >= g_N) {
        return;
    }

    float sum = 0.0f;
    uint row_offset = row * g_K;
    for (uint k = 0; k < g_K; ++k) {
        sum = mad(MatrixA[row_offset + k], MatrixB[k * g_N + col], sum);
    }
    MatrixC[row * g_N + col] = sum;
}

// 2. Shared-Memory Tiled 16x16 FP32 Matrix Multiplication
#define TILE_DIM 16
groupshared float gs_A16[TILE_DIM][TILE_DIM];
groupshared float gs_B16[TILE_DIM][TILE_DIM];

[numthreads(TILE_DIM, TILE_DIM, 1)]
void CSMatMulTiled16(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    uint row = dtid.y;
    uint col = dtid.x;
    uint lx = gtid.x;
    uint ly = gtid.y;

    float sum = 0.0f;
    uint num_tiles = (g_K + TILE_DIM - 1) / TILE_DIM;

    for (uint t = 0; t < num_tiles; ++t) {
        uint a_col = t * TILE_DIM + lx;
        uint b_row = t * TILE_DIM + ly;

        gs_A16[ly][lx] = (row < g_M && a_col < g_K) ? MatrixA[row * g_K + a_col] : 0.0f;
        gs_B16[ly][lx] = (b_row < g_K && col < g_N) ? MatrixB[b_row * g_N + col] : 0.0f;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE_DIM; ++k) {
            sum = mad(gs_A16[ly][k], gs_B16[k][lx], sum);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < g_M && col < g_N) {
        MatrixC[row * g_N + col] = sum;
    }
}

// 3. Register-Blocked 32x32 Tile (2x2 outputs per thread) Shared-Memory FP32 Kernel
#define BLOCK_TILE 32
#define BLOCK_K 16
groupshared float gs_A32[BLOCK_TILE][BLOCK_K];
groupshared float gs_B32[BLOCK_K][BLOCK_TILE];

[numthreads(16, 16, 1)]
void CSMatMulRegBlocked32(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint lx = gtid.x;
    uint ly = gtid.y;

    uint base_row = gid.y * BLOCK_TILE + ly * 2;
    uint base_col = gid.x * BLOCK_TILE + lx * 2;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    uint num_tiles = (g_K + BLOCK_K - 1) / BLOCK_K;
    for (uint t = 0; t < num_tiles; ++t) {
        uint k_col = t * BLOCK_K + lx;
        uint r0 = base_row;
        uint r1 = base_row + 1;

        gs_A32[ly * 2 + 0][lx] = (r0 < g_M && k_col < g_K) ? MatrixA[r0 * g_K + k_col] : 0.0f;
        gs_A32[ly * 2 + 1][lx] = (r1 < g_M && k_col < g_K) ? MatrixA[r1 * g_K + k_col] : 0.0f;

        uint k_row = t * BLOCK_K + ly;
        uint c0 = base_col;
        uint c1 = base_col + 1;

        gs_B32[ly][lx * 2 + 0] = (k_row < g_K && c0 < g_N) ? MatrixB[k_row * g_N + c0] : 0.0f;
        gs_B32[ly][lx * 2 + 1] = (k_row < g_K && c1 < g_N) ? MatrixB[k_row * g_N + c1] : 0.0f;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < BLOCK_K; ++k) {
            float a0 = gs_A32[ly * 2 + 0][k];
            float a1 = gs_A32[ly * 2 + 1][k];
            float b0 = gs_B32[k][lx * 2 + 0];
            float b1 = gs_B32[k][lx * 2 + 1];

            acc00 = mad(a0, b0, acc00);
            acc01 = mad(a0, b1, acc01);
            acc10 = mad(a1, b0, acc10);
            acc11 = mad(a1, b1, acc11);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (base_row + 0 < g_M && base_col + 0 < g_N) MatrixC[(base_row + 0) * g_N + (base_col + 0)] = acc00;
    if (base_row + 0 < g_M && base_col + 1 < g_N) MatrixC[(base_row + 0) * g_N + (base_col + 1)] = acc01;
    if (base_row + 1 < g_M && base_col + 0 < g_N) MatrixC[(base_row + 1) * g_N + (base_col + 0)] = acc10;
    if (base_row + 1 < g_M && base_col + 1 < g_N) MatrixC[(base_row + 1) * g_N + (base_col + 1)] = acc11;
}

// 4. FP16-Quantized Shared-Memory Tiled Kernel with FP32 Accumulation
groupshared uint gs_A16_packed[TILE_DIM][TILE_DIM];
groupshared uint gs_B16_packed[TILE_DIM][TILE_DIM];

[numthreads(TILE_DIM, TILE_DIM, 1)]
void CSMatMulFP16AccumFP32(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    uint row = dtid.y;
    uint col = dtid.x;
    uint lx = gtid.x;
    uint ly = gtid.y;

    float sum = 0.0f;
    uint num_tiles = (g_K + TILE_DIM - 1) / TILE_DIM;

    for (uint t = 0; t < num_tiles; ++t) {
        uint a_col = t * TILE_DIM + lx;
        uint b_row = t * TILE_DIM + ly;

        float a_val = (row < g_M && a_col < g_K) ? MatrixA[row * g_K + a_col] : 0.0f;
        float b_val = (b_row < g_K && col < g_N) ? MatrixB[b_row * g_N + col] : 0.0f;

        gs_A16_packed[ly][lx] = f32tof16(a_val);
        gs_B16_packed[ly][lx] = f32tof16(b_val);

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE_DIM; ++k) {
            float a_fp16 = f16tof32(gs_A16_packed[ly][k]);
            float b_fp16 = f16tof32(gs_B16_packed[k][lx]);
            sum = mad(a_fp16, b_fp16, sum);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < g_M && col < g_N) {
        MatrixC[row * g_N + col] = sum;
    }
}

// 5. Representative Neural-Network Layer: Conv2D 3x3 + Bias + LeakyReLU
// g_M = H * W (packed as (H << 16) | W), g_N = C_out, g_K = C_in
[numthreads(16, 16, 1)]
void CSConv2D3x3LeakyReLU(uint3 dtid : SV_DispatchThreadID) {
    uint H = (g_M >> 16) & 0xFFFF;
    uint W = g_M & 0xFFFF;
    uint spatial_idx = dtid.x;
    uint oc = dtid.y;

    if (spatial_idx >= H * W || oc >= g_N) {
        return;
    }

    int y = int(spatial_idx / W);
    int x = int(spatial_idx % W);
    uint C_in = g_K;

    // Bias is stored after the 3x3 weights in MatrixB: offset = g_N * C_in * 9 + oc
    float sum = MatrixB[g_N * C_in * 9 + oc];

    for (uint ic = 0; ic < C_in; ++ic) {
        uint weight_base = (oc * C_in + ic) * 9;
        uint in_channel_base = ic * H * W;

        [unroll]
        for (int ky = -1; ky <= 1; ++ky) {
            int sy = y + ky;
            if (sy >= 0 && sy < int(H)) {
                [unroll]
                for (int kx = -1; kx <= 1; ++kx) {
                    int sx = x + kx;
                    if (sx >= 0 && sx < int(W)) {
                        float val = MatrixA[in_channel_base + uint(sy) * W + uint(sx)];
                        float w = MatrixB[weight_base + uint(ky + 1) * 3 + uint(kx + 1)];
                        sum = mad(val, w, sum);
                    }
                }
            }
        }
    }

    // LeakyReLU(alpha = 0.1)
    float activated = (sum >= 0.0f) ? sum : (sum * 0.1f);
    MatrixC[oc * H * W + spatial_idx] = activated;
}

