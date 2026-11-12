# Backend Decisions & Benchmark Measurements (M1 - M4/M5)

> **2026-10-23:** this document describes the M0-M5 proof of concept. Its M5 "inference graph" was a
> placeholder (random weights, not DLSS-NR) and is retired to `archive/placeholder-graph/`. The real
> network runs in `src/neural/`; see `README.md` and `PROGRESS-neural.md`.


## 1. Hardware & Runtime Summary

- **Target GPU:** Intel(R) Arc(TM) B580 Graphics (`0x8086:0xE20B`)
- **Driver Version:** `32.0.101.9150`
- **D3D12 Feature Level / Shader Model:** `FL 12_2` / `SM 6.8`
- **Shader Compiler:** `DXC 1.8 (SM 6.6 DXIL -O3)`
- **WaveMMATier (`D3D12_OPTIONS9`):** `0`
- **Driver-Exposed D3D12 MetaCommands:** `14`
- **Peak PoC Buffer/Texture Allocation:** `384.70 MB`

## 2. Phase 2 — Basic Compute Kernels (M1)

| Kernel | Elements | Status | Max Abs Error | Warm-up GPU (ms) | Steady GPU (ms) | CPU Submit (ms) | Upload (ms) | Readback (ms) |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CSVectorAdd` | 1048576 | PASS | 0.00E+00 | 0.0071 | 0.0068 | 0.0106 | 0.7621 | 0.3414 |
| `CSVectorMul` | 1048576 | PASS | 0.00E+00 | 0.0076 | 0.0068 | 0.0073 | 0.7621 | 0.3406 |
| `CSVectorChainedAddMul_GPU_Resident` | 1048576 | PASS | 0.00E+00 | 0.0156 | 0.0156 | 0.0069 | 0.7621 | 0.3419 |

## 3. Phase 3 — Matrix Multiplication Benchmark (M2)

| Kernel / Backend | Shape (`M x N x K`) | Status | Max Abs Error | Warm-up GPU (ms) | Steady GPU (ms) | Throughput (GFLOPS) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: |
| `CSMatMulNaive` | `256x256x256` | PASS | 0.00E+00 | 0.025 | 0.023 | **1436.0** |
| `CSMatMulTiled16` | `256x256x256` | PASS | 0.00E+00 | 0.027 | 0.022 | **1528.5** |
| `CSMatMulRegBlocked32` | `256x256x256` | PASS | 0.00E+00 | 0.015 | 0.013 | **2489.8** |
| `CSMatMulFP16AccumFP32` | `256x256x256` | PASS | 4.21E-03 | 0.020 | 0.018 | **1858.0** |
| `DirectML_GEMM_FP32` | `256x256x256` | PASS | 3.34E-06 | 0.007 | 0.005 | **6927.4** |
| `DirectML_GEMM_FP16` | `256x256x256` | PASS | 5.23E-03 | 0.004 | 0.002 | **19522.6** |
| `CSMatMulNaive` | `512x512x512` | PASS | 0.00E+00 | 0.156 | 0.155 | **1729.9** |
| `CSMatMulTiled16` | `512x512x512` | PASS | 0.00E+00 | 0.135 | 0.143 | **1883.1** |
| `CSMatMulRegBlocked32` | `512x512x512` | PASS | 0.00E+00 | 0.097 | 0.099 | **2725.2** |
| `CSMatMulFP16AccumFP32` | `512x512x512` | PASS | 6.09E-03 | 0.125 | 0.124 | **2163.8** |
| `DirectML_GEMM_FP32` | `512x512x512` | PASS | 0.00E+00 | 0.028 | 0.025 | **10919.4** |
| `DirectML_GEMM_FP16` | `512x512x512` | PASS | 6.86E-03 | 0.008 | 0.004 | **62757.5** |
| `CSMatMulNaive` | `1024x1024x1024` | PASS | 0.00E+00 | 1.136 | 1.132 | **1896.8** |
| `CSMatMulTiled16` | `1024x1024x1024` | PASS | 0.00E+00 | 0.997 | 1.052 | **2042.2** |
| `CSMatMulRegBlocked32` | `1024x1024x1024` | PASS | 0.00E+00 | 0.487 | 0.482 | **4454.6** |
| `CSMatMulFP16AccumFP32` | `1024x1024x1024` | PASS | 8.59E-03 | 0.949 | 1.013 | **2119.3** |
| `DirectML_GEMM_FP32` | `1024x1024x1024` | PASS | 1.72E-05 | 0.178 | 0.174 | **12336.0** |
| `DirectML_GEMM_FP16` | `1024x1024x1024` | PASS | 1.09E-02 | 0.035 | 0.032 | **66262.3** |
| `CSMatMulNaive` | `2048x256x256` | PASS | 0.00E+00 | 0.140 | 0.138 | **1943.7** |
| `CSMatMulTiled16` | `2048x256x256` | PASS | 0.00E+00 | 0.124 | 0.123 | **2187.8** |
| `CSMatMulRegBlocked32` | `2048x256x256` | PASS | 0.00E+00 | 0.068 | 0.066 | **4094.5** |
| `CSMatMulFP16AccumFP32` | `2048x256x256` | PASS | 4.28E-03 | 0.118 | 0.117 | **2303.3** |
| `DirectML_GEMM_FP32` | `2048x256x256` | PASS | 0.00E+00 | 0.025 | 0.023 | **11673.8** |
| `DirectML_GEMM_FP16` | `2048x256x256` | PASS | 5.18E-03 | 0.009 | 0.007 | **39156.4** |

## 4. Phase 4 — End-to-End GPU-Resident Texture Pipeline (M3)

- **Resolution:** `1920x1080` (`DXGI_FORMAT_R32G32B32A32_FLOAT`)
- **Correctness vs. CPU Reference:** `PASS` (Max Abs Error: `1.19E-07`, PSNR: `154.99 dB`)
- **Upload GPU Time:** `2.864 ms`
- **Pass 1 (`CSImageSpatialFilter`) GPU Time:** `0.124 ms`
- **Pass 2 (`CSImageResidualReconstruct`) GPU Time:** `0.194 ms`
- **Chained End-to-End GPU Time (Zero CPU Round-Trip):** `0.317 ms`
- **Readback GPU Time:** `2.911 ms`

## 5. Phase 5 — Representative Neural-Network Layers (M4)

| Layer | Shape (`H x W, C_in -> C_out`) | Status | Max Abs Error | Warm-up (ms) | Steady (ms) | Throughput (GFLOPS) | Temp Buffer (KB) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `Conv2D_3x3_LeakyReLU_128x128_C16to16` | `128x128, 16->16` | PASS | 0.00E+00 | 0.090 | 0.089 | **846.3** | 2057.1 |
| `Conv2D_3x3_LeakyReLU_128x128_C32to32` | `128x128, 32->32` | PASS | 0.00E+00 | 0.316 | 0.318 | **948.5** | 4132.1 |

## 6. Go / No-Go Gate Decision

- **Verdict:** **GO (Hybrid DirectML MetaCommand + HLSL Compute Backend)**
- **Rationale:** All numerical accuracy checks passed. Register-blocked HLSL compute reached 4454.6 GFLOPS, while DirectML FP16 GEMM (utilizing driver-exposed D3D12 MetaCommands on Intel Xe2) reached 66262.3 GFLOPS (FP32: 12336.0 GFLOPS). Recommended architecture for Phase 6 full graph: use DirectML/MetaCommands for heavy GEMM/Conv2D layers and native DXIL compute shaders for fused pre/post-processing and element-wise ops.
