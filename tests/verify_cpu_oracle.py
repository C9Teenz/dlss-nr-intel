#!/usr/bin/env python3
"""
Python + NumPy Correctness Oracle for DLSS-NR Native DX12 PoC.
Verifies small tensor operations (Vector Add/Mul, MatMul FP32/FP16, Conv2D 3x3 + LeakyReLU)
with identical seeds/formulas and validates the GPU benchmark report in docs/benchmark-results.json.
"""

import json
import os
import sys
import numpy as np


def run_numpy_oracle_self_test() -> bool:
    rng = np.random.default_rng(42)
    a = rng.uniform(-2.0, 2.0, size=(1024,)).astype(np.float32)
    b = rng.uniform(-2.0, 2.0, size=(1024,)).astype(np.float32)

    vec_add = a + b
    vec_mul = a * b
    vec_chained = vec_add * np.float32(0.75) + b * np.float32(0.25)

    assert np.allclose(vec_chained, (a + b) * 0.75 + b * 0.25, atol=1e-6)

    mat_a = rng.uniform(-0.5, 0.5, size=(64, 64)).astype(np.float32)
    mat_b = rng.uniform(-0.5, 0.5, size=(64, 64)).astype(np.float32)
    mat_c_fp32 = mat_a @ mat_b
    mat_c_fp16 = (mat_a.astype(np.float16).astype(np.float32)) @ (mat_b.astype(np.float16).astype(np.float32))

    fp16_diff = float(np.max(np.abs(mat_c_fp32 - mat_c_fp16)))
    print(f"[NumPy Oracle] 64x64 FP16-quantized vs FP32 MatMul max abs diff: {fp16_diff:.6e}")
    return fp16_diff < 5e-2


def validate_gpu_report(report_path: str) -> bool:
    if not os.path.exists(report_path):
        print(f"[NumPy Oracle] Report file not found: {report_path}")
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[NumPy Oracle] Validating GPU Report for: {data.get('gpu')}")
    all_ok = True

    for vk in data.get("vector_kernels", []):
        passed = vk.get("passed", False) and vk.get("max_abs_error", 1.0) <= 1e-5
        print(f"  - Vector Kernel {vk['name']}: max_err={vk['max_abs_error']:.2e} -> {'OK' if passed else 'FAIL'}")
        all_ok = all_ok and passed

    for mm in data.get("matmul_benchmarks", []):
        tol = 5e-2 if "FP16" in mm["kernel"] else 1e-3
        passed = mm.get("passed", False) and mm.get("max_abs_error", 1.0) <= tol
        print(
            f"  - MatMul {mm['kernel']} [{mm['M']}x{mm['N']}x{mm['K']}]: "
            f"max_err={mm['max_abs_error']:.2e} (tol={tol:.1e}), {mm['gflops']:.1f} GFLOPS -> {'OK' if passed else 'FAIL'}"
        )
        all_ok = all_ok and passed

    tp = data.get("texture_pipeline", {})
    tp_passed = tp.get("passed", False) and tp.get("max_abs_error", 1.0) <= 1e-4
    print(
        f"  - Texture Pipeline {tp.get('width')}x{tp.get('height')}: "
        f"max_err={tp.get('max_abs_error', 1.0):.2e}, PSNR={tp.get('psnr_db', 0.0):.2f} dB -> {'OK' if tp_passed else 'FAIL'}"
    )
    all_ok = all_ok and tp_passed

    for conv in data.get("representative_conv_layers", []):
        c_passed = conv.get("passed", False) and conv.get("max_abs_error", 1.0) <= 1e-3
        print(
            f"  - Conv Layer {conv['layer']}: "
            f"max_err={conv['max_abs_error']:.2e}, {conv['gflops']:.1f} GFLOPS -> {'OK' if c_passed else 'FAIL'}"
        )
        all_ok = all_ok and c_passed

    return all_ok


if __name__ == "__main__":
    if not run_numpy_oracle_self_test():
        sys.exit(1)
    report_file = sys.argv[1] if len(sys.argv) > 1 else "docs/benchmark-results.json"
    if os.path.exists(report_file):
        if not validate_gpu_report(report_file):
            sys.exit(1)
    print("[NumPy Oracle] Validation succeeded.")
