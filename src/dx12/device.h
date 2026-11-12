#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dlss_nr::dx12 {

using Microsoft::WRL::ComPtr;

struct AdapterSummary {
    uint32_t index = 0;
    std::string description;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t subsys_id = 0;
    uint32_t revision = 0;
    uint64_t dedicated_video_memory_bytes = 0;
    uint64_t dedicated_system_memory_bytes = 0;
    uint64_t shared_system_memory_bytes = 0;
    bool is_software = false;
    bool is_intel = false;
    bool is_xe2_candidate = false;
};

struct FormatCapability {
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::string name;
    bool buffer_support = false;
    bool typed_uav_load = false;
    bool typed_uav_store = false;
    bool texture2d_support = false;
    bool render_target_support = false;
    uint32_t support1_raw = 0;
    uint32_t support2_raw = 0;
};

struct CapabilityReport {
    std::string os_version;
    std::string cpu_brand;
    std::string driver_version;
    AdapterSummary selected_adapter;
    std::vector<AdapterSummary> all_adapters;

    std::string max_feature_level;
    std::string highest_shader_model;

    uint32_t resource_binding_tier = 0;
    uint32_t resource_heap_tier = 0;
    bool double_precision_float_shader_ops = false;
    bool native_16bit_shader_ops = false;
    bool min_precision_fp16_support = false;

    bool wave_ops_supported = false;
    uint32_t wave_lane_count_min = 0;
    uint32_t wave_lane_count_max = 0;
    uint32_t total_lane_count = 0;
    bool int64_shader_ops = false;

    uint32_t wave_mma_tier = 0;
    uint32_t raytracing_tier = 0;
    uint32_t mesh_shader_tier = 0;
    uint32_t variable_shading_rate_tier = 0;
    bool enhanced_barriers_supported = false;
    bool gpu_upload_heap_supported = false;

    bool directml_available = false;
    std::string directml_status;
    uint32_t metacommand_count = 0;
    std::vector<std::string> metacommand_names;

    std::vector<FormatCapability> format_capabilities;

    uint64_t timestamp_frequency_hz = 0;
};

class Device {
public:
    Device() = default;
    ~Device() = default;

    bool Initialize(bool enable_debug_layer = false, bool allow_fallback_adapter = false);

    ID3D12Device* GetD3D12Device() const { return device_.Get(); }
    IDXGIFactory6* GetDXGIFactory() const { return factory_.Get(); }
    IDXGIAdapter1* GetDXGIAdapter() const { return adapter_.Get(); }

    const CapabilityReport& GetCapabilityReport() const { return report_; }

    bool QueryVideoMemory(uint64_t& out_current_usage_bytes, uint64_t& out_budget_bytes) const;

    bool SaveCapabilityReportJson(const std::string& path) const;
    bool SaveCapabilityReportMarkdown(const std::string& path) const;

private:
    void ProbeCapabilities(ID3D12CommandQueue* temp_queue);
    void ProbeFormats();
    void ProbeMetaCommands();
    void ProbeDirectML();

    ComPtr<IDXGIFactory6> factory_;
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<ID3D12Device> device_;
    CapabilityReport report_;
};

} // namespace dlss_nr::dx12
