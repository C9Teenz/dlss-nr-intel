#include "dx12/device.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace dlss_nr::dx12 {

namespace {

std::string WideToUtf8(const wchar_t* wstr) {
    if (!wstr || !wstr[0]) return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1) return "";
    std::string out(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, out.data(), len, nullptr, nullptr);
    return out;
}

std::string QueryRegistryString(HKEY root, const char* subkey, const char* value_name) {
    HKEY hkey = nullptr;
    if (RegOpenKeyExA(root, subkey, 0, KEY_READ, &hkey) != ERROR_SUCCESS) {
        return "";
    }
    char buf[512] = {};
    DWORD size = sizeof(buf) - 1;
    DWORD type = 0;
    std::string result;
    if (RegQueryValueExA(hkey, value_name, nullptr, &type, reinterpret_cast<LPBYTE>(buf), &size) == ERROR_SUCCESS) {
        if (type == REG_SZ || type == REG_EXPAND_SZ) {
            result = buf;
        }
    }
    RegCloseKey(hkey);
    return result;
}

std::string QueryOSVersionString() {
    std::string product = QueryRegistryString(
        HKEY_LOCAL_MACHINE,
        "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
        "ProductName");
    std::string build = QueryRegistryString(
        HKEY_LOCAL_MACHINE,
        "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
        "CurrentBuildNumber");
    std::string display_ver = QueryRegistryString(
        HKEY_LOCAL_MACHINE,
        "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
        "DisplayVersion");
    std::ostringstream oss;
    oss << (product.empty() ? "Windows" : product);
    if (!display_ver.empty()) oss << " (" << display_ver << ")";
    if (!build.empty()) oss << " Build " << build;
    return oss.str();
}

std::string QueryCPUBrandString() {
    std::string cpu = QueryRegistryString(
        HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        "ProcessorNameString");
    return cpu.empty() ? "Unknown CPU" : cpu;
}

std::string QueryIntelDriverVersion() {
    // Check Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000..0008 for Vendor 8086
    for (int i = 0; i < 8; ++i) {
        char subkey[256];
        std::snprintf(subkey, sizeof(subkey),
                      "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\%04d",
                      i);
        std::string matching_id = QueryRegistryString(HKEY_LOCAL_MACHINE, subkey, "MatchingDeviceId");
        std::string provider = QueryRegistryString(HKEY_LOCAL_MACHINE, subkey, "ProviderName");
        if (matching_id.find("8086") != std::string::npos ||
            matching_id.find("VEN_8086") != std::string::npos ||
            provider.find("Intel") != std::string::npos) {
            std::string ver = QueryRegistryString(HKEY_LOCAL_MACHINE, subkey, "DriverVersion");
            if (!ver.empty()) return ver;
        }
    }
    return "Unknown";
}

bool IsXe2DeviceId(uint32_t vendor_id, uint32_t device_id) {
    if (vendor_id != 0x8086) return false;
    // Battlemage (BMG G21: 0xE202..0xE21F) and Lunar Lake (LNL: 0x6420..0x64AF)
    if ((device_id >= 0xE200 && device_id <= 0xE2FF) ||
        (device_id >= 0x6400 && device_id <= 0x64FF)) {
        return true;
    }
    return false;
}

std::string GuidToString(const GUID& guid) {
    char buf[64];
    std::snprintf(buf, sizeof(buf),
                  "{%08lX-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
                  guid.Data1, guid.Data2, guid.Data3,
                  guid.Data4[0], guid.Data4[1], guid.Data4[2], guid.Data4[3],
                  guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]);
    return buf;
}

std::string EscapeJson(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << c; break;
        }
    }
    return oss.str();
}

// Raw struct for D3D12_FEATURE_D3D12_OPTIONS9 (feature ID 27) in case SDK headers are older
struct RawD3D12Options9 {
    BOOL MeshShaderPipelineStatsSupported;
    BOOL MeshShaderSupportsFullRangeRenderTargetArrayIndex;
    BOOL AtomicInt64OnTypedResourceSupported;
    BOOL AtomicInt64OnGroupSharedSupported;
    BOOL DerivativesInMeshAndAmplificationShadersSupported;
    uint32_t WaveMMATier;
};

// Raw struct for D3D12_FEATURE_D3D12_OPTIONS12 (feature ID 38)
struct RawD3D12Options12 {
    uint32_t MSPrimitivesPipelineStatisticIncludesCulledPrimitives;
    BOOL EnhancedBarriersSupported;
    BOOL RelaxedFormatCastingSupported;
};

// Raw struct for D3D12_FEATURE_D3D12_OPTIONS16 (feature ID 43)
struct RawD3D12Options16 {
    BOOL DynamicDepthBiasSupported;
    BOOL GPUUploadHeapSupported;
};

} // namespace

bool Device::Initialize(bool enable_debug_layer, bool allow_fallback_adapter) {
    UINT dxgi_flags = 0;

    if (enable_debug_layer) {
        ComPtr<ID3D12Debug> debug_controller;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller)))) {
            debug_controller->EnableDebugLayer();
            dxgi_flags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }

    ComPtr<IDXGIFactory4> factory4;
    if (FAILED(CreateDXGIFactory2(dxgi_flags, IID_PPV_ARGS(&factory4)))) {
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory4)))) {
            std::cerr << "[DX12] Failed to create DXGIFactory2\n";
            return false;
        }
    }

    if (FAILED(factory4.As(&factory_))) {
        std::cerr << "[DX12] Failed to query IDXGIFactory6\n";
        return false;
    }

    report_.os_version = QueryOSVersionString();
    report_.cpu_brand = QueryCPUBrandString();
    report_.driver_version = QueryIntelDriverVersion();
    report_.all_adapters.clear();

    ComPtr<IDXGIAdapter1> best_intel_xe2;
    AdapterSummary best_intel_xe2_summary{};
    ComPtr<IDXGIAdapter1> best_intel_any;
    AdapterSummary best_intel_any_summary{};
    ComPtr<IDXGIAdapter1> best_fallback;
    AdapterSummary best_fallback_summary{};

    for (UINT idx = 0;; ++idx) {
        ComPtr<IDXGIAdapter1> candidate;
        if (factory_->EnumAdapters1(idx, &candidate) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(candidate->GetDesc1(&desc))) {
            continue;
        }

        AdapterSummary summary{};
        summary.index = idx;
        summary.description = WideToUtf8(desc.Description);
        summary.vendor_id = desc.VendorId;
        summary.device_id = desc.DeviceId;
        summary.subsys_id = desc.SubSysId;
        summary.revision = desc.Revision;
        summary.dedicated_video_memory_bytes = desc.DedicatedVideoMemory;
        summary.dedicated_system_memory_bytes = desc.DedicatedSystemMemory;
        summary.shared_system_memory_bytes = desc.SharedSystemMemory;
        summary.is_software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        summary.is_intel = (desc.VendorId == 0x8086);
        summary.is_xe2_candidate = IsXe2DeviceId(desc.VendorId, desc.DeviceId);

        report_.all_adapters.push_back(summary);

        if (summary.is_software) continue;

        // Verify D3D12 device creation at FL 11_0 minimum
        if (FAILED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_11_0,
                                     __uuidof(ID3D12Device), nullptr))) {
            continue;
        }

        if (summary.is_xe2_candidate && !best_intel_xe2) {
            best_intel_xe2 = candidate;
            best_intel_xe2_summary = summary;
        }
        if (summary.is_intel && !best_intel_any) {
            best_intel_any = candidate;
            best_intel_any_summary = summary;
        }
        if (!best_fallback) {
            best_fallback = candidate;
            best_fallback_summary = summary;
        }
    }

    if (best_intel_xe2) {
        adapter_ = best_intel_xe2;
        report_.selected_adapter = best_intel_xe2_summary;
    } else if (best_intel_any) {
        adapter_ = best_intel_any;
        report_.selected_adapter = best_intel_any_summary;
    } else if (allow_fallback_adapter && best_fallback) {
        adapter_ = best_fallback;
        report_.selected_adapter = best_fallback_summary;
    } else {
        std::cerr << "[DX12] No suitable Intel D3D12 hardware adapter found.\n";
        return false;
    }

    if (FAILED(D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_12_0,
                                 IID_PPV_ARGS(&device_)))) {
        if (FAILED(D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&device_)))) {
            std::cerr << "[DX12] D3D12CreateDevice failed on selected adapter.\n";
            return false;
        }
    }

    // Create a temporary direct command queue to query GPU timestamp frequency
    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qdesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ComPtr<ID3D12CommandQueue> temp_queue;
    device_->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&temp_queue));

    ProbeCapabilities(temp_queue.Get());
    ProbeFormats();
    ProbeMetaCommands();
    ProbeDirectML();

    return true;
}

void Device::ProbeCapabilities(ID3D12CommandQueue* temp_queue) {
    if (temp_queue) {
        UINT64 freq = 0;
        if (SUCCEEDED(temp_queue->GetTimestampFrequency(&freq))) {
            report_.timestamp_frequency_hz = freq;
        }
    }

    // 1. Feature Levels
    const D3D_FEATURE_LEVEL levels[] = {
        static_cast<D3D_FEATURE_LEVEL>(0xc200), // D3D_FEATURE_LEVEL_12_2
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    D3D12_FEATURE_DATA_FEATURE_LEVELS fl_data{};
    fl_data.NumFeatureLevels = static_cast<UINT>( sizeof(levels) / sizeof(levels[0]) );
    fl_data.pFeatureLevelsRequested = levels;
    if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &fl_data, sizeof(fl_data)))) {
        switch (static_cast<uint32_t>(fl_data.MaxSupportedFeatureLevel)) {
            case 0xc200: report_.max_feature_level = "12_2"; break;
            case D3D_FEATURE_LEVEL_12_1: report_.max_feature_level = "12_1"; break;
            case D3D_FEATURE_LEVEL_12_0: report_.max_feature_level = "12_0"; break;
            case D3D_FEATURE_LEVEL_11_1: report_.max_feature_level = "11_1"; break;
            case D3D_FEATURE_LEVEL_11_0: report_.max_feature_level = "11_0"; break;
            default: report_.max_feature_level = "Unknown"; break;
        }
    }

    // 2. Highest Shader Model (probe from 6.8 down to 5.1)
    struct SMProbe {
        uint32_t code;
        const char* name;
    };
    const SMProbe sm_probes[] = {
        {0x68, "6.8"},
        {0x67, "6.7"},
        {0x66, "6.6"},
        {0x65, "6.5"},
        {0x64, "6.4"},
        {0x63, "6.3"},
        {0x62, "6.2"},
        {0x61, "6.1"},
        {0x60, "6.0"},
        {0x51, "5.1"},
    };
    report_.highest_shader_model = "5.1";
    for (const auto& sm : sm_probes) {
        D3D12_FEATURE_DATA_SHADER_MODEL sm_data{};
        sm_data.HighestShaderModel = static_cast<D3D_SHADER_MODEL>(sm.code);
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm_data, sizeof(sm_data)))) {
            for (const auto& match : sm_probes) {
                if (static_cast<uint32_t>(sm_data.HighestShaderModel) == match.code) {
                    report_.highest_shader_model = match.name;
                    break;
                }
            }
            break;
        }
    }

    // 3. D3D12_OPTIONS
    D3D12_FEATURE_DATA_D3D12_OPTIONS opts0{};
    if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts0, sizeof(opts0)))) {
        report_.double_precision_float_shader_ops = opts0.DoublePrecisionFloatShaderOps != FALSE;
        report_.min_precision_fp16_support = (opts0.MinPrecisionSupport & D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT) != 0;
        report_.resource_binding_tier = static_cast<uint32_t>(opts0.ResourceBindingTier);
        report_.resource_heap_tier = static_cast<uint32_t>(opts0.ResourceHeapTier);
    }

    // 4. D3D12_OPTIONS1 (WaveOps)
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts1{};
    if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts1, sizeof(opts1)))) {
        report_.wave_ops_supported = opts1.WaveOps != FALSE;
        report_.wave_lane_count_min = opts1.WaveLaneCountMin;
        report_.wave_lane_count_max = opts1.WaveLaneCountMax;
        report_.total_lane_count = opts1.TotalLaneCount;
        report_.int64_shader_ops = opts1.Int64ShaderOps != FALSE;
    }

    // 5. D3D12_OPTIONS4 (Native 16-bit shader ops)
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 opts4{};
    if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &opts4, sizeof(opts4)))) {
        report_.native_16bit_shader_ops = opts4.Native16BitShaderOpsSupported != FALSE;
    }

    // 6. D3D12_OPTIONS5 (RaytracingTier)
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5{};
    if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5)))) {
        report_.raytracing_tier = static_cast<uint32_t>(opts5.RaytracingTier);
    }

    // 7. D3D12_OPTIONS6 (VariableShadingRateTier)
    D3D12_FEATURE_DATA_D3D12_OPTIONS6 opts6{};
    if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &opts6, sizeof(opts6)))) {
        report_.variable_shading_rate_tier = static_cast<uint32_t>(opts6.VariableShadingRateTier);
    }

    // 8. D3D12_OPTIONS7 (MeshShaderTier)
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 opts7{};
    if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &opts7, sizeof(opts7)))) {
        report_.mesh_shader_tier = static_cast<uint32_t>(opts7.MeshShaderTier);
    }

    // 9. D3D12_OPTIONS9 (WaveMMATier, feature ID 27)
    RawD3D12Options9 opts9{};
    if (SUCCEEDED(device_->CheckFeatureSupport(static_cast<D3D12_FEATURE>(27), &opts9, sizeof(opts9)))) {
        report_.wave_mma_tier = opts9.WaveMMATier;
    }

    // 10. D3D12_OPTIONS12 (EnhancedBarriers, feature ID 38)
    RawD3D12Options12 opts12{};
    if (SUCCEEDED(device_->CheckFeatureSupport(static_cast<D3D12_FEATURE>(38), &opts12, sizeof(opts12)))) {
        report_.enhanced_barriers_supported = opts12.EnhancedBarriersSupported != FALSE;
    }

    // 11. D3D12_OPTIONS16 (GPUUploadHeapSupported, feature ID 43)
    RawD3D12Options16 opts16{};
    if (SUCCEEDED(device_->CheckFeatureSupport(static_cast<D3D12_FEATURE>(43), &opts16, sizeof(opts16)))) {
        report_.gpu_upload_heap_supported = opts16.GPUUploadHeapSupported != FALSE;
    }
}

void Device::ProbeFormats() {
    struct FormatEntry {
        DXGI_FORMAT fmt;
        const char* name;
    };
    const FormatEntry formats[] = {
        {DXGI_FORMAT_R16_FLOAT, "R16_FLOAT"},
        {DXGI_FORMAT_R16G16B16A16_FLOAT, "R16G16B16A16_FLOAT"},
        {DXGI_FORMAT_R32_FLOAT, "R32_FLOAT"},
        {DXGI_FORMAT_R32G32B32A32_FLOAT, "R32G32B32A32_FLOAT"},
        {DXGI_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM"},
        {DXGI_FORMAT_B8G8R8A8_UNORM, "B8G8R8A8_UNORM"},
        {DXGI_FORMAT_R10G10B10A2_UNORM, "R10G10B10A2_UNORM"},
        {DXGI_FORMAT_R16_UINT, "R16_UINT"},
        {DXGI_FORMAT_R32_UINT, "R32_UINT"},
    };

    report_.format_capabilities.clear();
    for (const auto& entry : formats) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{};
        fs.Format = entry.fmt;
        FormatCapability cap{};
        cap.format = entry.fmt;
        cap.name = entry.name;
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs)))) {
            cap.support1_raw = static_cast<uint32_t>(fs.Support1);
            cap.support2_raw = static_cast<uint32_t>(fs.Support2);
            cap.buffer_support = (fs.Support1 & D3D12_FORMAT_SUPPORT1_BUFFER) != 0;
            cap.texture2d_support = (fs.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) != 0;
            cap.render_target_support = (fs.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET) != 0;
            cap.typed_uav_load = (fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD) != 0;
            cap.typed_uav_store = (fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0;
        }
        report_.format_capabilities.push_back(cap);
    }
}

void Device::ProbeMetaCommands() {
    report_.metacommand_count = 0;
    report_.metacommand_names.clear();

    ComPtr<ID3D12Device5> dev5;
    if (FAILED(device_.As(&dev5)) || !dev5) {
        return;
    }

    UINT num_commands = 0;
    if (FAILED(dev5->EnumerateMetaCommands(&num_commands, nullptr)) || num_commands == 0) {
        return;
    }

    std::vector<D3D12_META_COMMAND_DESC> descs(num_commands);
    if (SUCCEEDED(dev5->EnumerateMetaCommands(&num_commands, descs.data()))) {
        report_.metacommand_count = num_commands;
        for (UINT i = 0; i < num_commands; ++i) {
            std::string name = WideToUtf8(descs[i].Name);
            std::string guid_str = GuidToString(descs[i].Id);
            report_.metacommand_names.push_back(name.empty() ? guid_str : (name + " " + guid_str));
        }
    }
}

void Device::ProbeDirectML() {
    HMODULE dml_mod = LoadLibraryA("DirectML.dll");
    if (!dml_mod) {
        report_.directml_available = false;
        report_.directml_status = "DirectML.dll not found";
        return;
    }

    using PFN_DMLCreateDevice = HRESULT(WINAPI*)(ID3D12Device*, int, REFIID, void**);
    auto create_fn = reinterpret_cast<PFN_DMLCreateDevice>(GetProcAddress(dml_mod, "DMLCreateDevice"));
    if (!create_fn) {
        report_.directml_available = false;
        report_.directml_status = "DMLCreateDevice export missing";
        FreeLibrary(dml_mod);
        return;
    }

    // IID_IDMLDevice: 6dbd6437-96fd-423f-a98c-ae5e7c2a573f
    static const GUID IID_IDMLDevice_Local = {
        0x6dbd6437, 0x96fd, 0x423f, {0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f}
    };
    ComPtr<IUnknown> dml_dev;
    HRESULT hr = create_fn(device_.Get(), 0 /* DML_CREATE_DEVICE_FLAG_NONE */,
                           IID_IDMLDevice_Local, reinterpret_cast<void**>(dml_dev.GetAddressOf()));
    if (SUCCEEDED(hr) && dml_dev) {
        report_.directml_available = true;
        report_.directml_status = "DirectML device initialized on selected Intel D3D12 adapter";
    } else {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "DMLCreateDevice failed (HRESULT=0x%08lX)", static_cast<unsigned long>(hr));
        report_.directml_available = false;
        report_.directml_status = buf;
    }

    dml_dev.Reset();
    FreeLibrary(dml_mod);
}

bool Device::QueryVideoMemory(uint64_t& out_current_usage_bytes, uint64_t& out_budget_bytes) const {
    out_current_usage_bytes = 0;
    out_budget_bytes = 0;
    if (!adapter_) return false;
    ComPtr<IDXGIAdapter3> adapter3;
    if (FAILED(adapter_.As(&adapter3))) return false;
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        return false;
    }
    out_current_usage_bytes = info.CurrentUsage;
    out_budget_bytes = info.Budget;
    return true;
}

bool Device::SaveCapabilityReportJson(const std::string& path) const {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    const auto& r = report_;
    out << "{\n";
    out << "  \"os_version\": \"" << EscapeJson(r.os_version) << "\",\n";
    out << "  \"cpu_brand\": \"" << EscapeJson(r.cpu_brand) << "\",\n";
    out << "  \"driver_version\": \"" << EscapeJson(r.driver_version) << "\",\n";
    out << "  \"selected_adapter\": {\n";
    out << "    \"index\": " << r.selected_adapter.index << ",\n";
    out << "    \"description\": \"" << EscapeJson(r.selected_adapter.description) << "\",\n";
    out << "    \"vendor_id\": \"0x" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << r.selected_adapter.vendor_id << std::dec << "\",\n";
    out << "    \"device_id\": \"0x" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << r.selected_adapter.device_id << std::dec << "\",\n";
    out << "    \"subsys_id\": \"0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << r.selected_adapter.subsys_id << std::dec << "\",\n";
    out << "    \"revision\": " << r.selected_adapter.revision << ",\n";
    out << "    \"dedicated_video_memory_mb\": " << (r.selected_adapter.dedicated_video_memory_bytes / (1024 * 1024)) << ",\n";
    out << "    \"shared_system_memory_mb\": " << (r.selected_adapter.shared_system_memory_bytes / (1024 * 1024)) << ",\n";
    out << "    \"is_intel\": " << (r.selected_adapter.is_intel ? "true" : "false") << ",\n";
    out << "    \"is_xe2_candidate\": " << (r.selected_adapter.is_xe2_candidate ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"d3d12_features\": {\n";
    out << "    \"max_feature_level\": \"" << r.max_feature_level << "\",\n";
    out << "    \"highest_shader_model\": \"" << r.highest_shader_model << "\",\n";
    out << "    \"resource_binding_tier\": " << r.resource_binding_tier << ",\n";
    out << "    \"resource_heap_tier\": " << r.resource_heap_tier << ",\n";
    out << "    \"double_precision_float_shader_ops\": " << (r.double_precision_float_shader_ops ? "true" : "false") << ",\n";
    out << "    \"native_16bit_shader_ops\": " << (r.native_16bit_shader_ops ? "true" : "false") << ",\n";
    out << "    \"min_precision_fp16_support\": " << (r.min_precision_fp16_support ? "true" : "false") << ",\n";
    out << "    \"wave_ops_supported\": " << (r.wave_ops_supported ? "true" : "false") << ",\n";
    out << "    \"wave_lane_count_min\": " << r.wave_lane_count_min << ",\n";
    out << "    \"wave_lane_count_max\": " << r.wave_lane_count_max << ",\n";
    out << "    \"total_lane_count\": " << r.total_lane_count << ",\n";
    out << "    \"int64_shader_ops\": " << (r.int64_shader_ops ? "true" : "false") << ",\n";
    out << "    \"wave_mma_tier\": " << r.wave_mma_tier << ",\n";
    out << "    \"raytracing_tier\": " << r.raytracing_tier << ",\n";
    out << "    \"mesh_shader_tier\": " << r.mesh_shader_tier << ",\n";
    out << "    \"variable_shading_rate_tier\": " << r.variable_shading_rate_tier << ",\n";
    out << "    \"enhanced_barriers_supported\": " << (r.enhanced_barriers_supported ? "true" : "false") << ",\n";
    out << "    \"gpu_upload_heap_supported\": " << (r.gpu_upload_heap_supported ? "true" : "false") << ",\n";
    out << "    \"timestamp_frequency_hz\": " << r.timestamp_frequency_hz << "\n";
    out << "  },\n";
    out << "  \"matrix_acceleration_paths\": {\n";
    out << "    \"directml_available\": " << (r.directml_available ? "true" : "false") << ",\n";
    out << "    \"directml_status\": \"" << EscapeJson(r.directml_status) << "\",\n";
    out << "    \"metacommand_count\": " << r.metacommand_count << ",\n";
    out << "    \"metacommands\": [\n";
    for (size_t i = 0; i < r.metacommand_names.size(); ++i) {
        out << "      \"" << EscapeJson(r.metacommand_names[i]) << "\""
            << (i + 1 < r.metacommand_names.size() ? "," : "") << "\n";
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"formats\": [\n";
    for (size_t i = 0; i < r.format_capabilities.size(); ++i) {
        const auto& f = r.format_capabilities[i];
        out << "    {\n";
        out << "      \"name\": \"" << f.name << "\",\n";
        out << "      \"buffer_support\": " << (f.buffer_support ? "true" : "false") << ",\n";
        out << "      \"texture2d_support\": " << (f.texture2d_support ? "true" : "false") << ",\n";
        out << "      \"render_target_support\": " << (f.render_target_support ? "true" : "false") << ",\n";
        out << "      \"typed_uav_load\": " << (f.typed_uav_load ? "true" : "false") << ",\n";
        out << "      \"typed_uav_store\": " << (f.typed_uav_store ? "true" : "false") << "\n";
        out << "    }" << (i + 1 < r.format_capabilities.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return true;
}

bool Device::SaveCapabilityReportMarkdown(const std::string& path) const {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    const auto& r = report_;
    out << "# DX12 Adapter & Capability Report (M0 / M1)\n\n";
    out << "## 1. System & Selected Adapter\n\n";
    out << "| Property | Value |\n";
    out << "| :--- | :--- |\n";
    out << "| **OS Version** | " << r.os_version << " |\n";
    out << "| **CPU** | " << r.cpu_brand << " |\n";
    out << "| **Selected GPU** | " << r.selected_adapter.description << " |\n";
    out << "| **PCI Vendor / Device ID** | `0x" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << r.selected_adapter.vendor_id
        << "` / `0x" << std::setw(4) << r.selected_adapter.device_id << std::dec << "` |\n";
    out << "| **Driver Version** | `" << r.driver_version << "` |\n";
    out << "| **Dedicated VRAM** | " << (r.selected_adapter.dedicated_video_memory_bytes / (1024 * 1024)) << " MB |\n";
    out << "| **Shared System Memory** | " << (r.selected_adapter.shared_system_memory_bytes / (1024 * 1024)) << " MB |\n";
    out << "| **Xe2 Architecture Candidate** | " << (r.selected_adapter.is_xe2_candidate ? "Yes (Battlemage / Lunar Lake)" : "No") << " |\n\n";

    out << "## 2. Direct3D 12 Feature Support\n\n";
    out << "| Feature | Status / Value |\n";
    out << "| :--- | :--- |\n";
    out << "| **Max Feature Level** | `D3D_FEATURE_LEVEL_" << r.max_feature_level << "` |\n";
    out << "| **Highest Shader Model** | `SM " << r.highest_shader_model << "` |\n";
    out << "| **Resource Binding Tier** | Tier " << r.resource_binding_tier << " |\n";
    out << "| **Resource Heap Tier** | Tier " << r.resource_heap_tier << " |\n";
    out << "| **Native 16-bit Shader Ops (FP16)** | " << (r.native_16bit_shader_ops ? "Supported" : "Not Supported") << " |\n";
    out << "| **Min-Precision 16-bit Support** | " << (r.min_precision_fp16_support ? "Supported" : "Not Supported") << " |\n";
    out << "| **Wave Intrinsics (`WaveOps`)** | " << (r.wave_ops_supported ? "Supported" : "Not Supported") << " |\n";
    out << "| **Wave Lane Count (Min .. Max)** | " << r.wave_lane_count_min << " .. " << r.wave_lane_count_max << " |\n";
    out << "| **Total Lane Count** | " << r.total_lane_count << " |\n";
    out << "| **Int64 Shader Ops** | " << (r.int64_shader_ops ? "Supported" : "Not Supported") << " |\n";
    out << "| **WaveMMA Tier (`D3D12_OPTIONS9`)** | " << r.wave_mma_tier << " |\n";
    out << "| **Enhanced Barriers (`D3D12_OPTIONS12`)** | " << (r.enhanced_barriers_supported ? "Supported" : "Not Supported") << " |\n";
    out << "| **GPU Upload Heap (`D3D12_OPTIONS16`)** | " << (r.gpu_upload_heap_supported ? "Supported" : "Not Supported") << " |\n";
    out << "| **GPU Timestamp Frequency** | " << r.timestamp_frequency_hz << " Hz |\n\n";

    out << "## 3. Candidate Matrix Acceleration Paths\n\n";
    out << "- **HLSL Compute Baseline (FP32 / Packed FP16 / Shared-Memory Tiled / Vec4 Register-Blocked):** Available (`SM " << r.highest_shader_model << "`)\n";
    out << "- **DirectML Runtime (`DirectML.dll`):** " << (r.directml_available ? "Available" : "Unavailable") << " (" << r.directml_status << ")\n";
    out << "- **Driver-Exposed D3D12 MetaCommands (`ID3D12Device5::EnumerateMetaCommands`):** " << r.metacommand_count << " found\n\n";

    if (!r.metacommand_names.empty()) {
        out << "### Enumerated D3D12 MetaCommands\n\n";
        for (const auto& mc : r.metacommand_names) {
            out << "- `" << mc << "`\n";
        }
        out << "\n";
    }

    out << "## 4. Format Support Matrix\n\n";
    out << "| DXGI Format | Buffer | Texture2D | RenderTarget | Typed UAV Load | Typed UAV Store |\n";
    out << "| :--- | :---: | :---: | :---: | :---: | :---: |\n";
    for (const auto& f : r.format_capabilities) {
        out << "| `" << f.name << "` | "
            << (f.buffer_support ? "Yes" : "No") << " | "
            << (f.texture2d_support ? "Yes" : "No") << " | "
            << (f.render_target_support ? "Yes" : "No") << " | "
            << (f.typed_uav_load ? "Yes" : "No") << " | "
            << (f.typed_uav_store ? "Yes" : "No") << " |\n";
    }
    out << "\n";
    return true;
}

} // namespace dlss_nr::dx12
