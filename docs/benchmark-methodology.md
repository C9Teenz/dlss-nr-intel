# Benchmark Methodology

## 1. Timing Separation & Instrumentation

To avoid conflating CPU driver submission overhead, PCIe staging transfers, one-time shader/operator initialization, and steady-state GPU kernel execution, this PoC measures each component independently:

1. **GPU Kernel Execution Time (`ID3D12QueryHeap` / `D3D12_QUERY_TYPE_TIMESTAMP`):**
   - Before and after each `Dispatch` (or `IDMLCommandRecorder::RecordDispatch`), `EndQuery` writes hardware timestamps into a `D3D12_QUERY_HEAP_TYPE_TIMESTAMP` heap.
   - `ResolveQueryData` copies the 64-bit tick counters into a `D3D12_HEAP_TYPE_READBACK` buffer after the timed passes.
   - Elapsed GPU time in milliseconds is computed as:
     $$\Delta t_{\text{GPU (ms)}} = \frac{(T_{\text{end}} - T_{\text{start}}) \times 1000.0}{f_{\text{timestamp}}}$$
     where $f_{\text{timestamp}}$ is queried via `ID3D12CommandQueue::GetTimestampFrequency`.
2. **Warm-up vs. Steady-State Separation:**
   - Every benchmark executes a dedicated set of warm-up iterations (`warmup_iters`) first and records the initial warm-up GPU execution time separately.
   - Steady-state performance (`steady_gpu_ms`) is averaged across `steady_iters` subsequent dispatches.
3. **CPU Submission Overhead vs. Fence Wait:**
   - `CommandQueue::ExecuteAndWait()` records high-resolution CPU timestamps around `Close() + ExecuteCommandLists()` (`cpu_submit_ms`) separately from `SetEventOnCompletion() + WaitForSingleObject()` (`cpu_wait_ms`).
4. **Transfer & Readback Cost Isolation:**
   - Upload (`D3D12_HEAP_TYPE_UPLOAD` $\to$ `D3D12_HEAP_TYPE_DEFAULT`) and readback (`D3D12_HEAP_TYPE_DEFAULT` $\to$ `D3D12_HEAP_TYPE_READBACK`) are timed in dedicated command lists using GPU timestamp pairs, strictly outside the compute dispatch loop.

## 2. Numerical Validation & Tolerances

Every GPU kernel output is validated against a deterministic CPU reference implementation (and cross-checked via `tests/verify_cpu_oracle.py` using NumPy):

| Operation | Data Type | Error Metric | Acceptance Tolerance |
| :--- | :--- | :--- | :--- |
| `CSVectorAdd` / `CSVectorMul` | FP32 | $\max |y_{\text{GPU}} - y_{\text{CPU}}|$ | $\le 1.0 \times 10^{-5}$ |
| `CSVectorChainedAddMul` (GPU-resident) | FP32 | $\max |y_{\text{GPU}} - y_{\text{CPU}}|$ | $\le 1.0 \times 10^{-5}$ |
| `CSMatMulNaive` / `CSMatMulTiled16` / `CSMatMulRegBlocked32` | FP32 | $\max |C_{\text{GPU}} - C_{\text{CPU}}|$ | $\le 1.0 \times 10^{-3}$ |
| `CSMatMulFP16AccumFP32` | FP16 $\to$ FP32 | $\max |C_{\text{GPU}} - C_{\text{CPU}}|$ | $\le 5.0 \times 10^{-2}$ |
| `DirectML GEMM` (FP32) | FP32 | $\max |C_{\text{GPU}} - C_{\text{CPU}}|$ | $\le 1.0 \times 10^{-3}$ |
| `DirectML GEMM` (FP16) | FP16 | $\max |C_{\text{GPU}} - C_{\text{CPU}}|$ | $\le 5.0 \times 10^{-2}$ |
| `CSImageSpatialFilter` + `CSImageResidualReconstruct` | RGBA32F | $\max |\text{px}_{\text{GPU}} - \text{px}_{\text{CPU}}|$ | $\le 1.0 \times 10^{-4}$ ($\text{PSNR} \ge 70\text{ dB}$) |
| `CSConv2D3x3LeakyReLU` (Representative Layer) | FP32 | $\max |Y_{\text{GPU}} - Y_{\text{CPU}}|$ | $\le 1.0 \times 10^{-3}$ |

## 3. Throughput Calculation

- **Matrix Multiplication (`M x N x K`):**
  $$\text{GFLOPS} = \frac{2 \times M \times N \times K}{\Delta t_{\text{steady\_gpu (ms)}} \times 10^6}$$
- **2D Convolution (`H x W`, `C_in -> C_out`, $3 \times 3$ kernel):**
  $$\text{GFLOPS} = \frac{2 \times H \times W \times C_{\text{in}} \times C_{\text{out}} \times 9}{\Delta t_{\text{steady\_gpu (ms)}} \times 10^6}$$
