#include <iostream>
#include <string>

#include "benchmark/benchmark.h"
#include "dx12/command_queue.h"
#include "dx12/device.h"
#include "dx12/resources.h"

int main(int argc, char** argv) {
    std::string shader_dir = "src/compute";
    std::string output_dir = "docs";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--shader-dir" && i + 1 < argc) {
            shader_dir = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        }
    }

    std::cout << "=== [Test] DLSS-NR Native DX12 End-to-End Texture Pipeline ===\n";

    dlss_nr::dx12::Device device;
    if (!device.Initialize(false, true)) {
        std::cerr << "[FAIL] Failed to initialize D3D12 device.\n";
        return 1;
    }

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
    dlss_nr::benchmark::TexturePipelineResult tex_res{};
    if (!runner.RunTexturePipeline(1920, 1080, output_dir, tex_res)) {
        std::cerr << "[FAIL] RunTexturePipeline failed.\n";
        return 1;
    }

    std::cout << "[TEXTURE] Resolution: " << tex_res.width << "x" << tex_res.height << "\n"
              << "  Status: " << (tex_res.passed ? "PASS" : "FAIL") << "\n"
              << "  Max Abs Error vs CPU Ref: " << tex_res.max_abs_error << "\n"
              << "  Mean Abs Error vs CPU Ref: " << tex_res.mean_abs_error << "\n"
              << "  PSNR vs CPU Ref: " << tex_res.psnr_db << " dB\n"
              << "  Upload GPU Time: " << tex_res.upload_gpu_ms << " ms\n"
              << "  Pass 1 (Spatial Filter) GPU Time: " << tex_res.pass1_spatial_filter_gpu_ms << " ms\n"
              << "  Pass 2 (Residual Reconstruct) GPU Time: " << tex_res.pass2_reconstruct_gpu_ms << " ms\n"
              << "  Chained End-to-End GPU Time (Zero CPU Round-Trip): " << tex_res.chained_end_to_end_gpu_ms << " ms\n"
              << "  Readback GPU Time: " << tex_res.readback_gpu_ms << " ms\n"
              << "  Saved Output BMP: " << tex_res.output_bmp_path << "\n";

    if (!tex_res.passed) {
        return 1;
    }

    std::cout << "=== End-to-End Texture Pipeline Test PASSED ===\n";
    return 0;
}
