# DLSS-NR Native DirectX 12 PoC Architecture

> **2026-10-23:** this document describes the M0-M5 proof of concept. Its M5 "inference graph" was a
> placeholder (random weights, not DLSS-NR) and is retired to `archive/placeholder-graph/`. The real
> network runs in `src/neural/`; see `README.md` and `PROGRESS-neural.md`.


## 1. Overview

This repository implements the standalone native DirectX 12 compute proof of concept (Milestones **M0** through **M4/M5**) targeting Intel Xe2 GPUs (**Intel Arc B580** `PCI\VEN_8086&DEV_E20B` and **Intel Arc 140V**).

Unlike game-interception layers that hook `IDXGISwapChain::Present` in DirectX 12 while delegating neural-network execution to Vulkan (`VK_KHR_cooperative_matrix`), this PoC executes all compute dispatches, tensor buffers, descriptor heaps, resource state barriers, and optional DirectML operator graphs natively on a single `ID3D12Device` and `ID3D12CommandQueue`.

```text
Standalone DX12 PoC (dlss_nr_dx12_poc)
    |
    +-- dx12::Device (src/dx12/device.h, src/dx12/device.cpp)
    |     +-- DXGI adapter enumeration & explicit Intel Xe2 selection
    |     +-- D3D12 feature, Shader Model, WaveOps, WaveMMA, FP16 & Format probing
    |     +-- ID3D12Device5::EnumerateMetaCommands inspection
    |     +-- DirectML.dll runtime availability check
    |
    +-- dx12::CommandQueue (src/dx12/command_queue.h, src/dx12/command_queue.cpp)
    |     +-- ID3D12CommandQueue, ID3D12CommandAllocator, ID3D12GraphicsCommandList
    |     +-- Explicit fence synchronization (ID3D12Fence + Win32 Event)
    |     +-- Hardware GPU timestamp queries (ID3D12QueryHeap + readback buffer)
    |
    +-- dx12::ResourceManager (src/dx12/resources.h, src/dx12/resources.cpp)
    |     +-- Default-heap UAV buffers & Texture2D resources
    |     +-- Upload & Readback staging heaps (256-byte pitch aligned for textures)
    |     +-- Shader-visible CBV_SRV_UAV descriptor heap management
    |     +-- Explicit state transitions & UAV barriers (no per-op CPU round-trip)
    |
    +-- Compute Kernels (src/compute/*.hlsl) & DirectML Baseline
    |     +-- vector_add.hlsl    : CSVectorAdd, CSVectorMul, CSVectorChainedAddMul
    |     +-- matmul.hlsl        : CSMatMulNaive, CSMatMulTiled16,
    |     |                        CSMatMulRegBlocked32, CSMatMulFP16AccumFP32,
    |     |                        CSConv2D3x3LeakyReLU (representative neural layer)
    |     +-- image_process.hlsl : CSImageSpatialFilter, CSImageResidualReconstruct
    |     +-- DirectML GEMM      : DML_GEMM_OPERATOR_DESC (FP32 and FP16) on D3D12
    |
    +-- Validation & Profiling (src/benchmark/benchmark.h, src/benchmark/benchmark.cpp)
          +-- CPU reference oracle (FP64/FP32 exact reference + Python NumPy oracle script)
          +-- Separate warm-up vs steady-state GPU timestamp measurements
          +-- Automated Go/No-Go gate evaluation
```

## 2. Resource & Synchronization Model

1. **Zero Per-Operation CPU Readback in the Primary Path:**
   - Multi-pass operations (such as chained vector math `CSVectorAdd -> CSVectorChainedAddMul` and multi-pass image processing `CSImageSpatialFilter -> CSImageResidualReconstruct`) remain resident on the GPU in `D3D12_HEAP_TYPE_DEFAULT` memory.
   - Intermediate synchronization uses `D3D12_RESOURCE_BARRIER_TYPE_UAV` and `D3D12_RESOURCE_BARRIER_TYPE_TRANSITION` (`UNORDERED_ACCESS -> NON_PIXEL_SHADER_RESOURCE`) within a single command list submission.
2. **Root Signature Layout:**
   - Compute pipelines use a compact root signature:
     - Parameter 0: `32BitConstants` (`register(b0)`, 4 DWORDs) for dimensions and scalar parameters.
     - Parameter 1: Descriptor table of `SRV`s (`register(t0..t1)`) pointing into the shader-visible `CBV_SRV_UAV` heap.
     - Parameter 2: Descriptor table of `UAV`s (`register(u0)`) pointing into the shader-visible `CBV_SRV_UAV` heap.
3. **DirectML Interop on Shared D3D12 Resources:**
   - When DirectML is benchmarked (`RunDirectMLGemm`), `IDMLDevice` is created directly on the PoC's `ID3D12Device`, and `IDMLCommandRecorder::RecordDispatch` records directly into the PoC's `ID3D12GraphicsCommandList` binding the same `D3D12_HEAP_TYPE_DEFAULT` buffers.
