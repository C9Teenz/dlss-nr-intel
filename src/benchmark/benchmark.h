#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dx12/command_queue.h"
#include "dx12/device.h"
#include "dx12/resources.h"

namespace dlss_nr::benchmark {

struct VectorTestResult {
    std::string name;
    uint32_t num_elements = 0;
    bool passed = false;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double warmup_gpu_ms = 0.0;
    double steady_gpu_ms = 0.0;
    double cpu_submit_ms = 0.0;
    double upload_gpu_ms = 0.0;
    double readback_gpu_ms = 0.0;
};

struct MatMulBenchmarkResult {
    std::string kernel_name;
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    bool passed = false;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double tolerance = 0.0;
    double warmup_gpu_ms = 0.0;
    double steady_gpu_ms = 0.0;
    double cpu_submit_ms = 0.0;
    double gflops = 0.0;
};

struct TexturePipelineResult {
    uint32_t width = 0;
    uint32_t height = 0;
    bool passed = false;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double psnr_db = 0.0;
    double upload_gpu_ms = 0.0;
    double pass1_spatial_filter_gpu_ms = 0.0;
    double pass2_reconstruct_gpu_ms = 0.0;
    double chained_end_to_end_gpu_ms = 0.0;
    double readback_gpu_ms = 0.0;
    double cpu_submit_ms = 0.0;
    std::string output_bmp_path;
    std::string output_ppm_path;
};

struct ConvLayerBenchmarkResult {
    std::string layer_name;
    uint32_t H = 0;
    uint32_t W = 0;
    uint32_t C_in = 0;
    uint32_t C_out = 0;
    bool passed = false;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double warmup_gpu_ms = 0.0;
    double steady_gpu_ms = 0.0;
    double cpu_submit_ms = 0.0;
    double gflops = 0.0;
    uint64_t temp_memory_bytes = 0;
};

struct BenchmarkSuiteReport {
    std::vector<VectorTestResult> vector_results;
    std::vector<MatMulBenchmarkResult> matmul_results;
    TexturePipelineResult texture_result;
    std::vector<ConvLayerBenchmarkResult> conv_results;

    uint64_t peak_allocated_bytes = 0;
    uint64_t vram_current_usage_bytes = 0;
    uint64_t vram_budget_bytes = 0;

    std::string shader_compiler_used;
    std::string go_no_go_verdict;
    std::string go_no_go_rationale;
};

class BenchmarkRunner {
public:
    BenchmarkRunner(dx12::Device& device,
                    dx12::CommandQueue& queue,
                    dx12::ResourceManager& resources,
                    const std::string& shader_dir);

    bool RunVectorKernels(uint32_t num_elements, std::vector<VectorTestResult>& out_results);

    bool RunMatMulSingle(const std::string& kernel_variant,
                         uint32_t M,
                         uint32_t N,
                         uint32_t K,
                         uint32_t warmup_iters,
                         uint32_t steady_iters,
                         MatMulBenchmarkResult& out_result);

    bool RunDirectMLGemm(uint32_t M,
                         uint32_t N,
                         uint32_t K,
                         bool use_fp16,
                         uint32_t warmup_iters,
                         uint32_t steady_iters,
                         MatMulBenchmarkResult& out_result);

    bool RunTexturePipeline(uint32_t width,
                            uint32_t height,
                            const std::string& output_dir,
                            TexturePipelineResult& out_result);

    bool RunRepresentativeConvLayer(uint32_t H,
                                    uint32_t W,
                                    uint32_t C_in,
                                    uint32_t C_out,
                                    uint32_t warmup_iters,
                                    uint32_t steady_iters,
                                    ConvLayerBenchmarkResult& out_result);

    bool RunFullSuite(const std::string& output_dir, BenchmarkSuiteReport& out_report);

    static bool SaveSuiteReportJson(const dx12::CapabilityReport& cap,
                                    const BenchmarkSuiteReport& report,
                                    const std::string& path);

    static bool SaveDecisionsMarkdown(const dx12::CapabilityReport& cap,
                                      const BenchmarkSuiteReport& report,
                                      const std::string& path);

    const std::string& GetShaderCompilerUsed() const { return shader_compiler_used_; }

private:
    dx12::Device& device_;
    dx12::CommandQueue& queue_;
    dx12::ResourceManager& resources_;
    std::string shader_dir_;
    std::string shader_compiler_used_ = "D3DCompile (D3DCompiler_47.dll)";
};

} // namespace dlss_nr::benchmark
