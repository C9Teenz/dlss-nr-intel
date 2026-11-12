#include <iomanip>
#include <iostream>
#include <string>

#include "benchmark/benchmark.h"
#include "dx12/command_queue.h"
#include "dx12/device.h"
#include "dx12/resources.h"
#include "neural/neural_app.h"

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--neural") {
            return dlss_nr::neural::RunNeuralMode(argc, argv);
        }
    }
    std::string shader_dir = "src/compute";
    std::string output_dir = "docs";
    bool enable_debug = false;
    bool probe_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--shader-dir" && i + 1 < argc) {
            shader_dir = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--debug-layer") {
            enable_debug = true;
        } else if (arg == "--probe-only") {
            probe_only = true;
        }
    }

    std::cout << "====================================================================\n";
    std::cout << "  DLSS-NR Native DirectX 12 PoC for Intel Xe2 (Arc B580 / Arc 140V)\n";
    std::cout << "====================================================================\n\n";

    // Phase 0 & Phase 1: Device initialization & Capability Probe
    dlss_nr::dx12::Device device;
    if (!device.Initialize(enable_debug, false)) {
        std::cerr << "[WARN] Strict Intel GPU selection failed; retrying with fallback enabled...\n";
        if (!device.Initialize(enable_debug, true)) {
            std::cerr << "[FATAL] Failed to initialize any D3D12 hardware adapter.\n";
            return 1;
        }
    }

    const auto& cap = device.GetCapabilityReport();
    std::cout << "[Phase 0/1] Adapter & Capability Probe:\n";
    std::cout << "  OS Version           : " << cap.os_version << "\n";
    std::cout << "  CPU                  : " << cap.cpu_brand << "\n";
    std::cout << "  Selected GPU         : " << cap.selected_adapter.description << "\n";
    std::cout << "  PCI Vendor/Device ID : 0x" << std::hex << std::uppercase << std::setw(4) << std::setfill('0')
              << cap.selected_adapter.vendor_id << " / 0x" << std::setw(4) << cap.selected_adapter.device_id
              << std::dec << std::setfill(' ') << "\n";
    std::cout << "  Driver Version       : " << cap.driver_version << "\n";
    std::cout << "  Dedicated VRAM       : " << (cap.selected_adapter.dedicated_video_memory_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  Max Feature Level    : " << cap.max_feature_level << "\n";
    std::cout << "  Highest Shader Model : " << cap.highest_shader_model << "\n";
    std::cout << "  Native 16-bit Ops    : " << (cap.native_16bit_shader_ops ? "Yes" : "No") << "\n";
    std::cout << "  Wave Ops (Lanes)     : " << (cap.wave_ops_supported ? "Yes" : "No")
              << " (" << cap.wave_lane_count_min << ".." << cap.wave_lane_count_max << ")\n";
    std::cout << "  WaveMMA Tier         : " << cap.wave_mma_tier << "\n";
    std::cout << "  D3D12 MetaCommands   : " << cap.metacommand_count << " exposed by driver\n";
    std::cout << "  DirectML Runtime     : " << cap.directml_status << "\n\n";

    std::string cap_json_path = output_dir + "/capability-report.json";
    std::string cap_md_path = output_dir + "/capability-report.md";
    device.SaveCapabilityReportJson(cap_json_path);
    device.SaveCapabilityReportMarkdown(cap_md_path);
    std::cout << "  Saved capability reports:\n"
              << "    - " << cap_json_path << "\n"
              << "    - " << cap_md_path << "\n\n";

    if (probe_only) {
        return 0;
    }

    dlss_nr::dx12::CommandQueue queue;
    if (!queue.Initialize(device.GetD3D12Device(), D3D12_COMMAND_LIST_TYPE_DIRECT, 256)) {
        std::cerr << "[FATAL] Failed to initialize D3D12 CommandQueue.\n";
        return 1;
    }

    dlss_nr::dx12::ResourceManager resources;
    if (!resources.Initialize(device.GetD3D12Device(), 1024)) {
        std::cerr << "[FATAL] Failed to initialize D3D12 ResourceManager.\n";
        return 1;
    }

    dlss_nr::benchmark::BenchmarkRunner runner(device, queue, resources, shader_dir);
    dlss_nr::benchmark::BenchmarkSuiteReport suite_report{};
    if (!runner.RunFullSuite(output_dir, suite_report)) {
        std::cerr << "[FATAL] Benchmark suite failed.\n";
        return 1;
    }

    std::string bench_json_path = output_dir + "/benchmark-results.json";
    std::string decisions_md_path = output_dir + "/decisions.md";
    dlss_nr::benchmark::BenchmarkRunner::SaveSuiteReportJson(cap, suite_report, bench_json_path);
    dlss_nr::benchmark::BenchmarkRunner::SaveDecisionsMarkdown(cap, suite_report, decisions_md_path);

    std::cout << "\n====================================================================\n";
    std::cout << "  Go/No-Go Verdict: " << suite_report.go_no_go_verdict << "\n";
    std::cout << "  Rationale       : " << suite_report.go_no_go_rationale << "\n";
    std::cout << "  Reports saved to: " << bench_json_path << " and " << decisions_md_path << "\n";
    std::cout << "====================================================================\n";

    return 0;
}
