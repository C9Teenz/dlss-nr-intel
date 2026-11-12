#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "dx12/command_queue.h"
#include "dx12/device.h"
#include "dx12/resources.h"

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    std::string shader_dir = "src/compute";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--shader-dir" && i + 1 < argc) {
            shader_dir = argv[++i];
        }
    }

    std::cout << "=== [Test] DLSS-NR Native DX12 Vector & Matrix Compute Correctness ===\n";

    dlss_nr::dx12::Device device;
    if (!device.Initialize(false, true)) {
        std::cerr << "[FAIL] Failed to initialize D3D12 device.\n";
        return 1;
    }

    const auto& cap = device.GetCapabilityReport();
    std::cout << "[INFO] Selected GPU: " << cap.selected_adapter.description
              << " (Vendor=0x" << std::hex << cap.selected_adapter.vendor_id
              << ", Device=0x" << cap.selected_adapter.device_id << std::dec << ")\n";

    dlss_nr::dx12::CommandQueue queue;
    if (!queue.Initialize(device.GetD3D12Device(), D3D12_COMMAND_LIST_TYPE_DIRECT, 128)) {
        std::cerr << "[FAIL] Failed to initialize CommandQueue.\n";
        return 1;
    }

    dlss_nr::dx12::ResourceManager resources;
    if (!resources.Initialize(device.GetD3D12Device(), 512)) {
        std::cerr << "[FAIL] Failed to initialize ResourceManager.\n";
        return 1;
    }

    dlss_nr::benchmark::BenchmarkRunner runner(device, queue, resources, shader_dir);

    std::vector<dlss_nr::benchmark::VectorTestResult> vec_results;
    if (!runner.RunVectorKernels(65536, vec_results)) {
        std::cerr << "[FAIL] RunVectorKernels failed.\n";
        return 1;
    }

    for (const auto& vr : vec_results) {
        std::cout << "[VECTOR] " << vr.name << " (N=" << vr.num_elements << "): "
                  << (vr.passed ? "PASS" : "FAIL")
                  << " | max_abs_err=" << vr.max_abs_error
                  << " | steady_gpu_ms=" << vr.steady_gpu_ms << " ms\n";
        if (!vr.passed) {
            return 1;
        }
    }

    const std::vector<std::string> variants = {
        "CSMatMulNaive",
        "CSMatMulTiled16",
        "CSMatMulRegBlocked32",
        "CSMatMulFP16AccumFP32"
    };

    for (const auto& variant : variants) {
        dlss_nr::benchmark::MatMulBenchmarkResult res{};
        if (!runner.RunMatMulSingle(variant, 256, 256, 256, 2, 5, res)) {
            std::cerr << "[FAIL] RunMatMulSingle failed for " << variant << "\n";
            return 1;
        }
        std::cout << "[MATMUL] " << res.kernel_name << " (256x256x256): "
                  << (res.passed ? "PASS" : "FAIL")
                  << " | max_abs_err=" << res.max_abs_error
                  << " (tol=" << res.tolerance << ")"
                  << " | steady_gpu_ms=" << res.steady_gpu_ms << " ms"
                  << " | " << res.gflops << " GFLOPS\n";
        if (!res.passed) {
            return 1;
        }
    }

    if (cap.directml_available) {
        for (bool fp16 : {false, true}) {
            dlss_nr::benchmark::MatMulBenchmarkResult dml_res{};
            if (runner.RunDirectMLGemm(256, 256, 256, fp16, 2, 5, dml_res)) {
                std::cout << "[DIRECTML] " << dml_res.kernel_name << " (256x256x256): "
                          << (dml_res.passed ? "PASS" : "FAIL")
                          << " | max_abs_err=" << dml_res.max_abs_error
                          << " | steady_gpu_ms=" << dml_res.steady_gpu_ms << " ms"
                          << " | " << dml_res.gflops << " GFLOPS\n";
                if (!dml_res.passed) {
                    return 1;
                }
            }
        }
    }

    dlss_nr::benchmark::ConvLayerBenchmarkResult conv_res{};
    if (!runner.RunRepresentativeConvLayer(64, 64, 16, 16, 2, 5, conv_res) || !conv_res.passed) {
        std::cerr << "[FAIL] Representative Conv2D 3x3 layer test failed.\n";
        return 1;
    }
    std::cout << "[CONV2D] " << conv_res.layer_name << ": PASS"
              << " | max_abs_err=" << conv_res.max_abs_error
              << " | steady_gpu_ms=" << conv_res.steady_gpu_ms << " ms\n";

    std::cout << "=== All Vector, MatMul, and Conv2D tests PASSED ===\n";
    return 0;
}
