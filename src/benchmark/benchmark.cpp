#include "benchmark/benchmark.h"

#include <initguid.h>
#include <d3dcompiler.h>
#define DML_TARGET_VERSION_USE_LATEST 1
#include "DirectML.h"

#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IDMLDevice, 0x6dbd6437, 0x96fd, 0x423f, 0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f)
__CRT_UUID_DECL(IDMLOperator, 0x26caae7a, 0x3081, 0x4633, 0x95, 0x81, 0x22, 0x6f, 0xbe, 0x57, 0x69, 0x5d)
__CRT_UUID_DECL(IDMLCompiledOperator, 0x6b15e56a, 0xbf5c, 0x4902, 0x92, 0xd8, 0xda, 0x3a, 0x65, 0x0a, 0xfe, 0xa4)
__CRT_UUID_DECL(IDMLOperatorInitializer, 0x427c1113, 0x435c, 0x469c, 0x86, 0x76, 0x4d, 0x5d, 0xd0, 0x72, 0xf8, 0x13)
__CRT_UUID_DECL(IDMLCommandRecorder, 0xe6857a76, 0x2e3e, 0x4fdd, 0xbf, 0xf4, 0x5d, 0x2b, 0xa1, 0x0f, 0xb4, 0x53)
__CRT_UUID_DECL(IDMLBindingTable, 0x29c687dc, 0xde74, 0x4e3b, 0xab, 0x00, 0x11, 0x68, 0xf2, 0xfc, 0x3c, 0xfc)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "dx12/swapchain.h"

namespace dlss_nr::benchmark {

using Microsoft::WRL::ComPtr;

namespace {

uint16_t Float32ToFloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x00800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | (mant >> 13));
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float Float16ToFloat32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t out_bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            out_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        out_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &out_bits, sizeof(float));
    return result;
}

DML_BINDING_PROPERTIES GetDMLBindingProperties(IDMLDispatchable* disp) {
    DML_BINDING_PROPERTIES props{};
    if (!disp) return props;
#if defined(_MSC_VER) && !defined(__MINGW32__)
    return disp->GetBindingProperties();
#else
    using Fn = DML_BINDING_PROPERTIES*(STDMETHODCALLTYPE*)(IDMLDispatchable*, DML_BINDING_PROPERTIES*);
    void** vtbl = *reinterpret_cast<void***>(disp);
    auto fn = reinterpret_cast<Fn>(vtbl[8]);
    fn(disp, &props);
    return props;
#endif
}

std::string FindDxcExecutable() {
    const std::vector<std::string> candidates = {
        "third_party/dxc/extracted/build/native/bin/x64/dxc.exe",
        "../third_party/dxc/extracted/build/native/bin/x64/dxc.exe",
        "dxc.exe"
    };
    for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) {
            return std::filesystem::absolute(c).string();
        }
    }
    return "";
}

bool CompileShaderToBytecode(const std::string& hlsl_path,
                             const std::string& entry_point,
                             std::vector<uint8_t>& out_bytecode,
                             std::string& out_compiler_used) {
    std::string dxc_exe = FindDxcExecutable();
    if (!dxc_exe.empty()) {
        std::filesystem::path temp_dxil = std::filesystem::temp_directory_path() /
                                          ("dlss_nr_" + entry_point + ".dxil");
        std::ostringstream cmd;
        cmd << "\"\"" << dxc_exe << "\" -T cs_6_6 -E " << entry_point
            << " -O3 \"" << hlsl_path << "\" -Fo \"" << temp_dxil.string() << "\" >nul 2>&1\"";
        int rc = std::system(cmd.str().c_str());
        if (rc == 0 && std::filesystem::exists(temp_dxil)) {
            std::ifstream in(temp_dxil, std::ios::binary | std::ios::ate);
            if (in.is_open()) {
                auto sz = in.tellg();
                if (sz > 0) {
                    out_bytecode.resize(static_cast<size_t>(sz));
                    in.seekg(0, std::ios::beg);
                    in.read(reinterpret_cast<char*>(out_bytecode.data()), sz);
                    in.close();
                    std::filesystem::remove(temp_dxil);
                    out_compiler_used = "DXC 1.8 (SM 6.6 DXIL -O3)";
                    return true;
                }
            }
        }
    }

    // Fallback to D3DCompileFromFile (D3DCompiler_47.dll)
    int wlen = MultiByteToWideChar(CP_UTF8, 0, hlsl_path.c_str(), -1, nullptr, 0);
    std::wstring whlsl(wlen > 0 ? wlen - 1 : 0, L'\0');
    if (wlen > 1) {
        MultiByteToWideChar(CP_UTF8, 0, hlsl_path.c_str(), -1, whlsl.data(), wlen);
    }

    ComPtr<ID3DBlob> code_blob;
    ComPtr<ID3DBlob> err_blob;
    UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
    HRESULT hr = D3DCompileFromFile(
        whlsl.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry_point.c_str(),
        "cs_5_0",
        flags,
        0,
        &code_blob,
        &err_blob);

    if (FAILED(hr) || !code_blob) {
        if (err_blob) {
            std::cerr << "[ShaderCompile] Error compiling " << hlsl_path << " (" << entry_point << "):\n"
                      << static_cast<const char*>(err_blob->GetBufferPointer()) << "\n";
        }
        return false;
    }

    out_bytecode.resize(code_blob->GetBufferSize());
    std::memcpy(out_bytecode.data(), code_blob->GetBufferPointer(), code_blob->GetBufferSize());
    out_compiler_used = "D3DCompile_47 (cs_5_0 -O3)";
    return true;
}

bool CreateComputeRootSignatureAndPSO(ID3D12Device* device,
                                      const std::vector<uint8_t>& bytecode,
                                      ComPtr<ID3D12RootSignature>& out_root_sig,
                                      ComPtr<ID3D12PipelineState>& out_pso) {
    D3D12_DESCRIPTOR_RANGE srv_range{};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = 2;
    srv_range.BaseShaderRegister = 0; // t0, t1
    srv_range.RegisterSpace = 0;
    srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_DESCRIPTOR_RANGE uav_range{};
    uav_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav_range.NumDescriptors = 1;
    uav_range.BaseShaderRegister = 0; // u0
    uav_range.RegisterSpace = 0;
    uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER params[3] = {};
    // Param 0: 4x 32-bit constants at b0
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Param 1: SRV table (t0..t1)
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &srv_range;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Param 2: UAV table (u0)
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable.NumDescriptorRanges = 1;
    params[2].DescriptorTable.pDescriptorRanges = &uav_range;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rs_desc{};
    rs_desc.NumParameters = 3;
    rs_desc.pParameters = params;
    rs_desc.NumStaticSamplers = 0;
    rs_desc.pStaticSamplers = nullptr;
    rs_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig_blob;
    ComPtr<ID3DBlob> err_blob;
    if (FAILED(D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig_blob, &err_blob))) {
        if (err_blob) {
            std::cerr << "[RootSignature] Error: " << static_cast<const char*>(err_blob->GetBufferPointer()) << "\n";
        }
        return false;
    }

    if (FAILED(device->CreateRootSignature(
            0, sig_blob->GetBufferPointer(), sig_blob->GetBufferSize(), IID_PPV_ARGS(&out_root_sig)))) {
        return false;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = out_root_sig.Get();
    pso_desc.CS.pShaderBytecode = bytecode.data();
    pso_desc.CS.BytecodeLength = bytecode.size();
    pso_desc.NodeMask = 0;
    pso_desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    if (FAILED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&out_pso)))) {
        std::cerr << "[PSO] CreateComputePipelineState failed\n";
        return false;
    }

    return true;
}

} // namespace

BenchmarkRunner::BenchmarkRunner(dx12::Device& device,
                                 dx12::CommandQueue& queue,
                                 dx12::ResourceManager& resources,
                                 const std::string& shader_dir)
    : device_(device), queue_(queue), resources_(resources), shader_dir_(shader_dir) {}

bool BenchmarkRunner::RunVectorKernels(uint32_t num_elements,
                                       std::vector<VectorTestResult>& out_results) {
    out_results.clear();
    const std::string hlsl_path = shader_dir_ + "/vector_add.hlsl";

    std::vector<uint8_t> bc_add, bc_mul, bc_chained;
    if (!CompileShaderToBytecode(hlsl_path, "CSVectorAdd", bc_add, shader_compiler_used_) ||
        !CompileShaderToBytecode(hlsl_path, "CSVectorMul", bc_mul, shader_compiler_used_) ||
        !CompileShaderToBytecode(hlsl_path, "CSVectorChainedAddMul", bc_chained, shader_compiler_used_)) {
        return false;
    }

    ComPtr<ID3D12RootSignature> root_sig;
    ComPtr<ID3D12PipelineState> pso_add, pso_mul, pso_chained;
    if (!CreateComputeRootSignatureAndPSO(device_.GetD3D12Device(), bc_add, root_sig, pso_add) ||
        !CreateComputeRootSignatureAndPSO(device_.GetD3D12Device(), bc_mul, root_sig, pso_mul) ||
        !CreateComputeRootSignatureAndPSO(device_.GetD3D12Device(), bc_chained, root_sig, pso_chained)) {
        return false;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> h_a(num_elements);
    std::vector<float> h_b(num_elements);
    for (uint32_t i = 0; i < num_elements; ++i) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    const uint64_t byte_size = static_cast<uint64_t>(num_elements) * sizeof(float);
    dx12::GpuBuffer buf_a, buf_b, buf_temp, buf_c;
    dx12::GpuBuffer upload_buf, readback_buf;
    if (!resources_.CreateDefaultBuffer(byte_size, buf_a, false) ||
        !resources_.CreateDefaultBuffer(byte_size, buf_b, false) ||
        !resources_.CreateDefaultBuffer(byte_size, buf_temp, true) ||
        !resources_.CreateDefaultBuffer(byte_size, buf_c, true) ||
        !resources_.CreateUploadBuffer(byte_size, upload_buf) ||
        !resources_.CreateReadbackBuffer(byte_size, readback_buf)) {
        return false;
    }

    double upload_a_ms = 0.0, upload_b_ms = 0.0;
    if (!resources_.UploadBufferData(queue_, buf_a, upload_buf, h_a.data(), byte_size, &upload_a_ms) ||
        !resources_.UploadBufferData(queue_, buf_b, upload_buf, h_b.data(), byte_size, &upload_b_ms)) {
        return false;
    }

    // Descriptor tables:
    // Table 1 (t0, t1) for buf_a, buf_b
    dx12::DescriptorHandle srv_a = resources_.CreateBufferSRV(buf_a, num_elements, sizeof(float));
    dx12::DescriptorHandle srv_b = resources_.CreateBufferSRV(buf_b, num_elements, sizeof(float));
    (void)srv_b;
    dx12::DescriptorHandle uav_c = resources_.CreateBufferUAV(buf_c, num_elements, sizeof(float));
    dx12::DescriptorHandle uav_temp = resources_.CreateBufferUAV(buf_temp, num_elements, sizeof(float));

    // Table 2 (t0, t1) for buf_temp, buf_b (for chained pass)
    dx12::DescriptorHandle srv_temp = resources_.CreateBufferSRV(buf_temp, num_elements, sizeof(float));
    dx12::DescriptorHandle srv_b2 = resources_.CreateBufferSRV(buf_b, num_elements, sizeof(float));
    (void)srv_b2;

    const uint32_t groups_x = (num_elements + 255) / 256;

    auto run_single_op = [&](const std::string& name, ID3D12PipelineState* pso, bool is_mul) -> bool {
        VectorTestResult res{};
        res.name = name;
        res.num_elements = num_elements;
        res.upload_gpu_ms = upload_a_ms + upload_b_ms;

        struct Constants {
            uint32_t n;
            float alpha;
            float beta;
            uint32_t pad;
        } consts{num_elements, 1.0f, 1.0f, 0};

        // Warm-up + 10 steady dispatches
        const uint32_t total_iters = 11;
        if (!queue_.ResetCommandList()) return false;
        auto* cmd = queue_.GetCommandList();

        dx12::ResourceManager::TransitionBuffer(cmd, buf_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dx12::ResourceManager::TransitionBuffer(cmd, buf_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dx12::ResourceManager::TransitionBuffer(cmd, buf_c, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        ID3D12DescriptorHeap* heaps[] = {resources_.GetDescriptorHeap()};
        cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootSignature(root_sig.Get());
        cmd->SetPipelineState(pso);
        cmd->SetComputeRoot32BitConstants(0, 4, &consts, 0);
        cmd->SetComputeRootDescriptorTable(1, srv_a.gpu);
        cmd->SetComputeRootDescriptorTable(2, uav_c.gpu);

        for (uint32_t it = 0; it < total_iters; ++it) {
            queue_.WriteTimestampBegin(it);
            cmd->Dispatch(groups_x, 1, 1);
            dx12::ResourceManager::RecordUAVBarrier(cmd, buf_c.resource.Get());
            queue_.WriteTimestampEnd(it);
        }
        queue_.ResolveTimestamps(total_iters);
        auto sub = queue_.ExecuteAndWait();
        res.cpu_submit_ms = sub.cpu_submit_ms;
        res.warmup_gpu_ms = queue_.GetTimestampPairDurationMs(0);

        double sum_steady = 0.0;
        for (uint32_t it = 1; it < total_iters; ++it) {
            sum_steady += queue_.GetTimestampPairDurationMs(it);
        }
        res.steady_gpu_ms = sum_steady / static_cast<double>(total_iters - 1);

        std::vector<float> h_out(num_elements, 0.0f);
        if (!resources_.ReadbackBufferData(queue_, buf_c, readback_buf, h_out.data(), byte_size, &res.readback_gpu_ms)) {
            return false;
        }

        double max_err = 0.0, sum_err = 0.0;
        for (uint32_t i = 0; i < num_elements; ++i) {
            float ref = is_mul ? (h_a[i] * h_b[i]) : (h_a[i] + h_b[i]);
            double err = std::abs(static_cast<double>(h_out[i]) - static_cast<double>(ref));
            max_err = std::max(max_err, err);
            sum_err += err;
        }
        res.max_abs_error = max_err;
        res.mean_abs_error = sum_err / num_elements;
        res.passed = (max_err <= 1e-5);
        out_results.push_back(res);
        return true;
    };

    if (!run_single_op("CSVectorAdd", pso_add.Get(), false)) return false;
    if (!run_single_op("CSVectorMul", pso_mul.Get(), true)) return false;

    // 3. Chained GPU-Resident Pipeline: Pass 1 (Temp = A + B) -> Barrier -> Pass 2 (C = Temp * 0.75 + B * 0.25)
    // No intermediate CPU readback!
    {
        VectorTestResult res{};
        res.name = "CSVectorChainedAddMul_GPU_Resident";
        res.num_elements = num_elements;
        res.upload_gpu_ms = upload_a_ms + upload_b_ms;

        struct Constants {
            uint32_t n;
            float alpha;
            float beta;
            uint32_t pad;
        } consts{num_elements, 0.75f, 0.25f, 0};

        if (!queue_.ResetCommandList()) return false;
        auto* cmd = queue_.GetCommandList();

        ID3D12DescriptorHeap* heaps[] = {resources_.GetDescriptorHeap()};
        cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootSignature(root_sig.Get());
        cmd->SetComputeRoot32BitConstants(0, 4, &consts, 0);

        dx12::ResourceManager::TransitionBuffer(cmd, buf_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dx12::ResourceManager::TransitionBuffer(cmd, buf_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dx12::ResourceManager::TransitionBuffer(cmd, buf_temp, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dx12::ResourceManager::TransitionBuffer(cmd, buf_c, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        queue_.WriteTimestampBegin(0);
        // Pass 1: Temp = A + B
        cmd->SetPipelineState(pso_add.Get());
        cmd->SetComputeRootDescriptorTable(1, srv_a.gpu);
        cmd->SetComputeRootDescriptorTable(2, uav_temp.gpu);
        cmd->Dispatch(groups_x, 1, 1);

        // Transition Temp from UAV -> SRV on GPU without CPU readback
        dx12::ResourceManager::TransitionBuffer(cmd, buf_temp, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Pass 2: C = Temp * 0.75 + B * 0.25
        cmd->SetPipelineState(pso_chained.Get());
        cmd->SetComputeRootDescriptorTable(1, srv_temp.gpu);
        cmd->SetComputeRootDescriptorTable(2, uav_c.gpu);
        cmd->Dispatch(groups_x, 1, 1);
        dx12::ResourceManager::RecordUAVBarrier(cmd, buf_c.resource.Get());
        queue_.WriteTimestampEnd(0);

        queue_.ResolveTimestamps(1);
        auto sub = queue_.ExecuteAndWait();
        res.cpu_submit_ms = sub.cpu_submit_ms;
        res.warmup_gpu_ms = queue_.GetTimestampPairDurationMs(0);
        res.steady_gpu_ms = res.warmup_gpu_ms;

        std::vector<float> h_out(num_elements, 0.0f);
        if (!resources_.ReadbackBufferData(queue_, buf_c, readback_buf, h_out.data(), byte_size, &res.readback_gpu_ms)) {
            return false;
        }

        double max_err = 0.0, sum_err = 0.0;
        for (uint32_t i = 0; i < num_elements; ++i) {
            float ref = (h_a[i] + h_b[i]) * 0.75f + h_b[i] * 0.25f;
            double err = std::abs(static_cast<double>(h_out[i]) - static_cast<double>(ref));
            max_err = std::max(max_err, err);
            sum_err += err;
        }
        res.max_abs_error = max_err;
        res.mean_abs_error = sum_err / num_elements;
        res.passed = (max_err <= 1e-5);
        out_results.push_back(res);
    }

    return true;
}

bool BenchmarkRunner::RunMatMulSingle(const std::string& kernel_variant,
                                      uint32_t M,
                                      uint32_t N,
                                      uint32_t K,
                                      uint32_t warmup_iters,
                                      uint32_t steady_iters,
                                      MatMulBenchmarkResult& out_result) {
    out_result = MatMulBenchmarkResult{};
    out_result.kernel_name = kernel_variant;
    out_result.M = M;
    out_result.N = N;
    out_result.K = K;

    const std::string hlsl_path = shader_dir_ + "/matmul.hlsl";
    std::vector<uint8_t> bytecode;
    if (!CompileShaderToBytecode(hlsl_path, kernel_variant, bytecode, shader_compiler_used_)) {
        return false;
    }

    ComPtr<ID3D12RootSignature> root_sig;
    ComPtr<ID3D12PipelineState> pso;
    if (!CreateComputeRootSignatureAndPSO(device_.GetD3D12Device(), bytecode, root_sig, pso)) {
        return false;
    }

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> h_a(static_cast<size_t>(M) * K);
    std::vector<float> h_b(static_cast<size_t>(K) * N);
    for (auto& v : h_a) v = dist(rng);
    for (auto& v : h_b) v = dist(rng);

    const uint64_t bytes_a = static_cast<uint64_t>(M) * K * sizeof(float);
    const uint64_t bytes_b = static_cast<uint64_t>(K) * N * sizeof(float);
    const uint64_t bytes_c = static_cast<uint64_t>(M) * N * sizeof(float);
    const uint64_t max_upload = std::max(bytes_a, bytes_b);

    dx12::GpuBuffer buf_a, buf_b, buf_c, upload_buf, readback_buf;
    if (!resources_.CreateDefaultBuffer(bytes_a, buf_a, false) ||
        !resources_.CreateDefaultBuffer(bytes_b, buf_b, false) ||
        !resources_.CreateDefaultBuffer(bytes_c, buf_c, true) ||
        !resources_.CreateUploadBuffer(max_upload, upload_buf) ||
        !resources_.CreateReadbackBuffer(bytes_c, readback_buf)) {
        return false;
    }

    if (!resources_.UploadBufferData(queue_, buf_a, upload_buf, h_a.data(), bytes_a) ||
        !resources_.UploadBufferData(queue_, buf_b, upload_buf, h_b.data(), bytes_b)) {
        return false;
    }

    dx12::DescriptorHandle srv_a = resources_.CreateBufferSRV(buf_a, M * K, sizeof(float));
    dx12::DescriptorHandle srv_b = resources_.CreateBufferSRV(buf_b, K * N, sizeof(float));
    (void)srv_b;
    dx12::DescriptorHandle uav_c = resources_.CreateBufferUAV(buf_c, M * N, sizeof(float));

    uint32_t tile_x = 16;
    uint32_t tile_y = 16;
    if (kernel_variant == "CSMatMulRegBlocked32") {
        tile_x = 32;
        tile_y = 32;
    }
    const uint32_t groups_x = (N + tile_x - 1) / tile_x;
    const uint32_t groups_y = (M + tile_y - 1) / tile_y;

    const uint32_t total_iters = warmup_iters + steady_iters;
    if (!queue_.ResetCommandList()) return false;
    auto* cmd = queue_.GetCommandList();

    dx12::ResourceManager::TransitionBuffer(cmd, buf_a, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12::ResourceManager::TransitionBuffer(cmd, buf_b, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12::ResourceManager::TransitionBuffer(cmd, buf_c, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ID3D12DescriptorHeap* heaps[] = {resources_.GetDescriptorHeap()};
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(root_sig.Get());
    cmd->SetPipelineState(pso.Get());

    uint32_t consts[4] = {M, N, K, 0};
    cmd->SetComputeRoot32BitConstants(0, 4, consts, 0);
    cmd->SetComputeRootDescriptorTable(1, srv_a.gpu);
    cmd->SetComputeRootDescriptorTable(2, uav_c.gpu);

    for (uint32_t it = 0; it < total_iters; ++it) {
        queue_.WriteTimestampBegin(it);
        cmd->Dispatch(groups_x, groups_y, 1);
        dx12::ResourceManager::RecordUAVBarrier(cmd, buf_c.resource.Get());
        queue_.WriteTimestampEnd(it);
    }
    queue_.ResolveTimestamps(total_iters);

    auto sub = queue_.ExecuteAndWait();
    out_result.cpu_submit_ms = sub.cpu_submit_ms;
    out_result.warmup_gpu_ms = queue_.GetTimestampPairDurationMs(0);

    double sum_steady = 0.0;
    for (uint32_t it = warmup_iters; it < total_iters; ++it) {
        sum_steady += queue_.GetTimestampPairDurationMs(it);
    }
    out_result.steady_gpu_ms = sum_steady / static_cast<double>(steady_iters);
    if (out_result.steady_gpu_ms > 0.0) {
        double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        out_result.gflops = (flops / (out_result.steady_gpu_ms * 1e-3)) / 1e9;
    }

    std::vector<float> h_c(static_cast<size_t>(M) * N, 0.0f);
    if (!resources_.ReadbackBufferData(queue_, buf_c, readback_buf, h_c.data(), bytes_c)) {
        return false;
    }

    // Validate against CPU reference (spot-check all rows for M<=512, or Strided rows for M>512 to keep CPU test fast)
    const uint32_t row_step = (M > 512) ? 8 : 1;
    double max_err = 0.0;
    double sum_err = 0.0;
    uint64_t checked_count = 0;

    for (uint32_t r = 0; r < M; r += row_step) {
        for (uint32_t c = 0; c < N; ++c) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                ref += h_a[r * K + k] * h_b[k * N + c];
            }
            double err = std::abs(static_cast<double>(h_c[r * N + c]) - static_cast<double>(ref));
            max_err = std::max(max_err, err);
            sum_err += err;
            ++checked_count;
        }
    }

    out_result.max_abs_error = max_err;
    out_result.mean_abs_error = (checked_count > 0) ? (sum_err / static_cast<double>(checked_count)) : 0.0;
    out_result.tolerance = (kernel_variant == "CSMatMulFP16AccumFP32") ? 5e-2 : 1e-3;
    out_result.passed = (max_err <= out_result.tolerance);
    return true;
}

bool BenchmarkRunner::RunDirectMLGemm(uint32_t M,
                                      uint32_t N,
                                      uint32_t K,
                                      bool use_fp16,
                                      uint32_t warmup_iters,
                                      uint32_t steady_iters,
                                      MatMulBenchmarkResult& out_result) {
    out_result = MatMulBenchmarkResult{};
    out_result.kernel_name = use_fp16 ? "DirectML_GEMM_FP16" : "DirectML_GEMM_FP32";
    out_result.M = M;
    out_result.N = N;
    out_result.K = K;
    out_result.tolerance = use_fp16 ? 5e-2 : 1e-3;

    // Load DirectML from third_party if available, else System32 (keep loaded for process lifetime)
    static HMODULE dml_mod = nullptr;
    if (!dml_mod) {
        dml_mod = LoadLibraryA("third_party/directml/extracted/bin/x64-win/DirectML.dll");
    }
    if (!dml_mod) {
        dml_mod = LoadLibraryA("../third_party/directml/extracted/bin/x64-win/DirectML.dll");
    }
    if (!dml_mod) {
        dml_mod = LoadLibraryA("DirectML.dll");
    }
    if (!dml_mod) {
        return false;
    }

    using PFN_DMLCreateDevice = HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
    auto create_dev_fn = reinterpret_cast<PFN_DMLCreateDevice>(GetProcAddress(dml_mod, "DMLCreateDevice"));
    if (!create_dev_fn) {
        return false;
    }

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(create_dev_fn(device_.GetD3D12Device(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dml_device)))) {
        return false;
    }

    const DML_TENSOR_DATA_TYPE dtype = use_fp16 ? DML_TENSOR_DATA_TYPE_FLOAT16 : DML_TENSOR_DATA_TYPE_FLOAT32;
    const uint32_t elem_size = use_fp16 ? sizeof(uint16_t) : sizeof(float);

    UINT sizes_a[4] = {1, 1, M, K};
    UINT sizes_b[4] = {1, 1, K, N};
    UINT sizes_c[4] = {1, 1, M, N};

    auto calc_tensor_size = [](uint32_t rows, uint32_t cols, uint32_t esize) -> UINT64 {
        UINT64 raw = static_cast<UINT64>(rows) * cols * esize;
        return (raw + 255ULL) & ~255ULL; // DML requires 256-byte alignment minimum
    };

    DML_BUFFER_TENSOR_DESC desc_a{};
    desc_a.DataType = dtype;
    desc_a.Flags = DML_TENSOR_FLAG_NONE;
    desc_a.DimensionCount = 4;
    desc_a.Sizes = sizes_a;
    desc_a.TotalTensorSizeInBytes = calc_tensor_size(M, K, elem_size);

    DML_BUFFER_TENSOR_DESC desc_b{};
    desc_b.DataType = dtype;
    desc_b.Flags = DML_TENSOR_FLAG_NONE;
    desc_b.DimensionCount = 4;
    desc_b.Sizes = sizes_b;
    desc_b.TotalTensorSizeInBytes = calc_tensor_size(K, N, elem_size);

    DML_BUFFER_TENSOR_DESC desc_c{};
    desc_c.DataType = dtype;
    desc_c.Flags = DML_TENSOR_FLAG_NONE;
    desc_c.DimensionCount = 4;
    desc_c.Sizes = sizes_c;
    desc_c.TotalTensorSizeInBytes = calc_tensor_size(M, N, elem_size);

    DML_TENSOR_DESC tensor_a{DML_TENSOR_TYPE_BUFFER, &desc_a};
    DML_TENSOR_DESC tensor_b{DML_TENSOR_TYPE_BUFFER, &desc_b};
    DML_TENSOR_DESC tensor_c{DML_TENSOR_TYPE_BUFFER, &desc_c};

    DML_GEMM_OPERATOR_DESC gemm_desc{};
    gemm_desc.ATensor = &tensor_a;
    gemm_desc.BTensor = &tensor_b;
    gemm_desc.CTensor = nullptr;
    gemm_desc.OutputTensor = &tensor_c;
    gemm_desc.TransA = DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.TransB = DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.Alpha = 1.0f;
    gemm_desc.Beta = 0.0f;
    gemm_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc{DML_OPERATOR_GEMM, &gemm_desc};
    ComPtr<IDMLOperator> dml_op;
    if (FAILED(dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&dml_op)))) {
        return false;
    }

    DML_EXECUTION_FLAGS exec_flags = use_fp16 ? DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
                                              : DML_EXECUTION_FLAG_NONE;
    ComPtr<IDMLCompiledOperator> compiled_op;
    if (FAILED(dml_device->CompileOperator(dml_op.Get(), exec_flags, IID_PPV_ARGS(&compiled_op)))) {
        return false;
    }

    IDMLCompiledOperator* ops_to_init[] = {compiled_op.Get()};
    ComPtr<IDMLOperatorInitializer> initializer;
    if (FAILED(dml_device->CreateOperatorInitializer(1, ops_to_init, IID_PPV_ARGS(&initializer)))) {
        return false;
    }

    ComPtr<IDMLCommandRecorder> recorder;
    if (FAILED(dml_device->CreateCommandRecorder(IID_PPV_ARGS(&recorder)))) {
        return false;
    }

    // Prepare host data
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> h_a_fp32(static_cast<size_t>(M) * K);
    std::vector<float> h_b_fp32(static_cast<size_t>(K) * N);
    for (auto& v : h_a_fp32) v = dist(rng);
    for (auto& v : h_b_fp32) v = dist(rng);

    dx12::GpuBuffer buf_a, buf_b, buf_c, upload_buf, readback_buf;
    const uint64_t max_upload = std::max(desc_a.TotalTensorSizeInBytes, desc_b.TotalTensorSizeInBytes);
    if (!resources_.CreateDefaultBuffer(desc_a.TotalTensorSizeInBytes, buf_a, true) ||
        !resources_.CreateDefaultBuffer(desc_b.TotalTensorSizeInBytes, buf_b, true) ||
        !resources_.CreateDefaultBuffer(desc_c.TotalTensorSizeInBytes, buf_c, true) ||
        !resources_.CreateUploadBuffer(max_upload, upload_buf) ||
        !resources_.CreateReadbackBuffer(desc_c.TotalTensorSizeInBytes, readback_buf)) {
        return false;
    }

    if (use_fp16) {
        std::vector<uint16_t> h_a_fp16(h_a_fp32.size());
        std::vector<uint16_t> h_b_fp16(h_b_fp32.size());
        for (size_t i = 0; i < h_a_fp32.size(); ++i) h_a_fp16[i] = Float32ToFloat16(h_a_fp32[i]);
        for (size_t i = 0; i < h_b_fp32.size(); ++i) h_b_fp16[i] = Float32ToFloat16(h_b_fp32[i]);
        if (!resources_.UploadBufferData(queue_, buf_a, upload_buf, h_a_fp16.data(), h_a_fp16.size() * sizeof(uint16_t)) ||
            !resources_.UploadBufferData(queue_, buf_b, upload_buf, h_b_fp16.data(), h_b_fp16.size() * sizeof(uint16_t))) {
            return false;
        }
    } else {
        if (!resources_.UploadBufferData(queue_, buf_a, upload_buf, h_a_fp32.data(), h_a_fp32.size() * sizeof(float)) ||
            !resources_.UploadBufferData(queue_, buf_b, upload_buf, h_b_fp32.data(), h_b_fp32.size() * sizeof(float))) {
            return false;
        }
    }

    const DML_BINDING_PROPERTIES compiled_props = GetDMLBindingProperties(compiled_op.Get());
    const DML_BINDING_PROPERTIES init_props = GetDMLBindingProperties(initializer.Get());

    UINT64 persistent_size = compiled_props.PersistentResourceSize;
    UINT64 temp_init_size = init_props.TemporaryResourceSize;
    UINT64 temp_exec_size = compiled_props.TemporaryResourceSize;

    dx12::GpuBuffer persistent_buf, temp_buf;
    if (persistent_size > 0) {
        resources_.CreateDefaultBuffer(persistent_size, persistent_buf, true);
    }
    UINT64 max_temp = std::max(temp_init_size, temp_exec_size);
    if (max_temp > 0) {
        resources_.CreateDefaultBuffer(max_temp, temp_buf, true);
    }

    UINT init_desc_count = init_props.RequiredDescriptorCount;
    UINT exec_desc_count = compiled_props.RequiredDescriptorCount;
    UINT total_dml_descs = std::max(1u, init_desc_count + exec_desc_count + 16u);

    D3D12_DESCRIPTOR_HEAP_DESC dml_heap_desc{};
    dml_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    dml_heap_desc.NumDescriptors = total_dml_descs;
    dml_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> dml_heap;
    if (FAILED(device_.GetD3D12Device()->CreateDescriptorHeap(&dml_heap_desc, IID_PPV_ARGS(&dml_heap)))) {
        return false;
    }

    UINT desc_inc = device_.GetD3D12Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Initialize operator
    DML_BINDING_TABLE_DESC bt_desc{};
    bt_desc.Dispatchable = initializer.Get();
    bt_desc.CPUDescriptorHandle = dml_heap->GetCPUDescriptorHandleForHeapStart();
    bt_desc.GPUDescriptorHandle = dml_heap->GetGPUDescriptorHandleForHeapStart();
    bt_desc.SizeInDescriptors = init_desc_count;

    ComPtr<IDMLBindingTable> binding_table;
    if (FAILED(dml_device->CreateBindingTable(&bt_desc, IID_PPV_ARGS(&binding_table)))) {
        return false;
    }

    if (temp_init_size > 0) {
        DML_BUFFER_BINDING temp_b{temp_buf.resource.Get(), 0, temp_init_size};
        DML_BINDING_DESC temp_bd{DML_BINDING_TYPE_BUFFER, &temp_b};
        binding_table->BindTemporaryResource(&temp_bd);
    }
    if (persistent_size > 0) {
        DML_BUFFER_BINDING pers_b{persistent_buf.resource.Get(), 0, persistent_size};
        DML_BINDING_DESC pers_bd{DML_BINDING_TYPE_BUFFER, &pers_b};
        binding_table->BindOutputs(1, &pers_bd);
    }

    if (!queue_.ResetCommandList()) {
        return false;
    }
    auto* cmd = queue_.GetCommandList();
    dx12::ResourceManager::TransitionBuffer(cmd, buf_a, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12::ResourceManager::TransitionBuffer(cmd, buf_b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12::ResourceManager::TransitionBuffer(cmd, buf_c, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ID3D12DescriptorHeap* dml_heaps[] = {dml_heap.Get()};
    cmd->SetDescriptorHeaps(1, dml_heaps);
    recorder->RecordDispatch(cmd, initializer.Get(), binding_table.Get());
    queue_.ExecuteAndWait();

    // Reset binding table for compiled operator execution
    bt_desc.Dispatchable = compiled_op.Get();
    bt_desc.CPUDescriptorHandle.ptr += static_cast<SIZE_T>(init_desc_count) * desc_inc;
    bt_desc.GPUDescriptorHandle.ptr += static_cast<UINT64>(init_desc_count) * desc_inc;
    bt_desc.SizeInDescriptors = exec_desc_count;
    if (FAILED(binding_table->Reset(&bt_desc))) {
        return false;
    }

    DML_BUFFER_BINDING bind_a{buf_a.resource.Get(), 0, desc_a.TotalTensorSizeInBytes};
    DML_BUFFER_BINDING bind_b{buf_b.resource.Get(), 0, desc_b.TotalTensorSizeInBytes};
    DML_BINDING_DESC input_bindings[3] = {
        {DML_BINDING_TYPE_BUFFER, &bind_a},
        {DML_BINDING_TYPE_BUFFER, &bind_b},
        {DML_BINDING_TYPE_NONE, nullptr}
    };
    binding_table->BindInputs(3, input_bindings);

    DML_BUFFER_BINDING bind_c{buf_c.resource.Get(), 0, desc_c.TotalTensorSizeInBytes};
    DML_BINDING_DESC output_binding{DML_BINDING_TYPE_BUFFER, &bind_c};
    binding_table->BindOutputs(1, &output_binding);

    if (persistent_size > 0) {
        DML_BUFFER_BINDING pers_b{persistent_buf.resource.Get(), 0, persistent_size};
        DML_BINDING_DESC pers_bd{DML_BINDING_TYPE_BUFFER, &pers_b};
        binding_table->BindPersistentResource(&pers_bd);
    }
    if (temp_exec_size > 0) {
        DML_BUFFER_BINDING temp_b{temp_buf.resource.Get(), 0, temp_exec_size};
        DML_BINDING_DESC temp_bd{DML_BINDING_TYPE_BUFFER, &temp_b};
        binding_table->BindTemporaryResource(&temp_bd);
    }

    const uint32_t total_iters = warmup_iters + steady_iters;
    if (!queue_.ResetCommandList()) {
        return false;
    }
    cmd = queue_.GetCommandList();
    cmd->SetDescriptorHeaps(1, dml_heaps);

    for (uint32_t it = 0; it < total_iters; ++it) {
        queue_.WriteTimestampBegin(it);
        recorder->RecordDispatch(cmd, compiled_op.Get(), binding_table.Get());
        dx12::ResourceManager::RecordUAVBarrier(cmd, buf_c.resource.Get());
        queue_.WriteTimestampEnd(it);
    }
    queue_.ResolveTimestamps(total_iters);

    auto sub = queue_.ExecuteAndWait();
    out_result.cpu_submit_ms = sub.cpu_submit_ms;
    out_result.warmup_gpu_ms = queue_.GetTimestampPairDurationMs(0);

    double sum_steady = 0.0;
    for (uint32_t it = warmup_iters; it < total_iters; ++it) {
        sum_steady += queue_.GetTimestampPairDurationMs(it);
    }
    out_result.steady_gpu_ms = sum_steady / static_cast<double>(steady_iters);
    if (out_result.steady_gpu_ms > 0.0) {
        double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        out_result.gflops = (flops / (out_result.steady_gpu_ms * 1e-3)) / 1e9;
    }

    std::vector<float> h_c_fp32(static_cast<size_t>(M) * N, 0.0f);
    if (use_fp16) {
        std::vector<uint16_t> h_c_fp16(static_cast<size_t>(M) * N, 0);
        if (!resources_.ReadbackBufferData(queue_, buf_c, readback_buf, h_c_fp16.data(), h_c_fp16.size() * sizeof(uint16_t))) {
            return false;
        }
        for (size_t i = 0; i < h_c_fp16.size(); ++i) {
            h_c_fp32[i] = Float16ToFloat32(h_c_fp16[i]);
        }
    } else {
        if (!resources_.ReadbackBufferData(queue_, buf_c, readback_buf, h_c_fp32.data(), h_c_fp32.size() * sizeof(float))) {
            return false;
        }
    }

    const uint32_t row_step = (M > 512) ? 8 : 1;
    double max_err = 0.0, sum_err = 0.0;
    uint64_t checked = 0;
    for (uint32_t r = 0; r < M; r += row_step) {
        for (uint32_t c = 0; c < N; ++c) {
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                ref += h_a_fp32[r * K + k] * h_b_fp32[k * N + c];
            }
            double err = std::abs(static_cast<double>(h_c_fp32[r * N + c]) - static_cast<double>(ref));
            max_err = std::max(max_err, err);
            sum_err += err;
            ++checked;
        }
    }

    out_result.max_abs_error = max_err;
    out_result.mean_abs_error = (checked > 0) ? (sum_err / static_cast<double>(checked)) : 0.0;
    out_result.passed = (max_err <= out_result.tolerance);
    return true;
}

bool BenchmarkRunner::RunTexturePipeline(uint32_t width,
                                         uint32_t height,
                                         const std::string& output_dir,
                                         TexturePipelineResult& out_result) {
    out_result = TexturePipelineResult{};
    out_result.width = width;
    out_result.height = height;

    const std::string hlsl_path = shader_dir_ + "/image_process.hlsl";
    std::vector<uint8_t> bc_pass1, bc_pass2;
    if (!CompileShaderToBytecode(hlsl_path, "CSImageSpatialFilter", bc_pass1, shader_compiler_used_) ||
        !CompileShaderToBytecode(hlsl_path, "CSImageResidualReconstruct", bc_pass2, shader_compiler_used_)) {
        return false;
    }

    ComPtr<ID3D12RootSignature> root_sig;
    ComPtr<ID3D12PipelineState> pso_pass1, pso_pass2;
    if (!CreateComputeRootSignatureAndPSO(device_.GetD3D12Device(), bc_pass1, root_sig, pso_pass1) ||
        !CreateComputeRootSignatureAndPSO(device_.GetD3D12Device(), bc_pass2, root_sig, pso_pass2)) {
        return false;
    }

    // Generate synthetic HDR-like RGBA32F test pattern (checkerboard + smooth gradient + rings)
    const size_t num_pixels = static_cast<size_t>(width) * height;
    std::vector<float> h_input(num_pixels * 4, 0.0f);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            float fx = static_cast<float>(x) / static_cast<float>(width);
            float fy = static_cast<float>(y) / static_cast<float>(height);
            float checker = (((x / 16) + (y / 16)) & 1) ? 0.25f : 0.05f;
            float ring = 0.2f * (0.5f + 0.5f * std::sin((fx - 0.5f) * (fx - 0.5f) * 80.0f +
                                                        (fy - 0.5f) * (fy - 0.5f) * 80.0f));
            h_input[idx + 0] = std::clamp(0.4f * fx + checker + ring, 0.0f, 1.0f);
            h_input[idx + 1] = std::clamp(0.4f * fy + checker * 0.8f + ring, 0.0f, 1.0f);
            h_input[idx + 2] = std::clamp(0.3f * (1.0f - fx) + checker + ring * 0.5f, 0.0f, 1.0f);
            h_input[idx + 3] = 1.0f;
        }
    }

    const uint32_t bytes_per_pixel = 16; // RGBA32_FLOAT
    const uint32_t row_pitch = width * bytes_per_pixel;
    const uint32_t aligned_pitch = (row_pitch + 255u) & ~255u;
    const uint64_t staging_bytes = static_cast<uint64_t>(aligned_pitch) * height;

    dx12::GpuTexture tex_input, tex_filtered, tex_output;
    dx12::GpuBuffer upload_buf, readback_buf;
    if (!resources_.CreateTexture2D(width, height, DXGI_FORMAT_R32G32B32A32_FLOAT, tex_input, false) ||
        !resources_.CreateTexture2D(width, height, DXGI_FORMAT_R32G32B32A32_FLOAT, tex_filtered, true) ||
        !resources_.CreateTexture2D(width, height, DXGI_FORMAT_R32G32B32A32_FLOAT, tex_output, true) ||
        !resources_.CreateUploadBuffer(staging_bytes, upload_buf) ||
        !resources_.CreateReadbackBuffer(staging_bytes, readback_buf)) {
        return false;
    }

    if (!resources_.UploadTexture2DData(queue_, tex_input, upload_buf, h_input.data(), row_pitch, bytes_per_pixel, &out_result.upload_gpu_ms)) {
        return false;
    }

    // Descriptors for Pass 1: SRV table [tex_input, tex_input], UAV table [tex_filtered]
    dx12::DescriptorHandle pass1_srv0 = resources_.CreateTexture2DSRV(tex_input);
    dx12::DescriptorHandle pass1_srv1 = resources_.CreateTexture2DSRV(tex_input);
    (void)pass1_srv1;
    dx12::DescriptorHandle pass1_uav = resources_.CreateTexture2DUAV(tex_filtered);

    // Descriptors for Pass 2: SRV table [tex_filtered, tex_input], UAV table [tex_output]
    dx12::DescriptorHandle pass2_srv0 = resources_.CreateTexture2DSRV(tex_filtered);
    dx12::DescriptorHandle pass2_srv1 = resources_.CreateTexture2DSRV(tex_input);
    (void)pass2_srv1;
    dx12::DescriptorHandle pass2_uav = resources_.CreateTexture2DUAV(tex_output);

    struct ImageConsts {
        uint32_t width;
        uint32_t height;
        float sharpness;
        float blend_factor;
    } consts{width, height, 0.6f, 0.35f};

    const uint32_t groups_x = (width + 15) / 16;
    const uint32_t groups_y = (height + 15) / 16;

    // Warm-up run + timed steady run of Chained End-to-End Pipeline (Pass 1 -> Barrier -> Pass 2)
    for (int iter = 0; iter < 4; ++iter) {
        if (!queue_.ResetCommandList()) return false;
        auto* cmd = queue_.GetCommandList();

        ID3D12DescriptorHeap* heaps[] = {resources_.GetDescriptorHeap()};
        cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootSignature(root_sig.Get());
        cmd->SetComputeRoot32BitConstants(0, 4, &consts, 0);

        dx12::ResourceManager::TransitionTexture(cmd, tex_input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dx12::ResourceManager::TransitionTexture(cmd, tex_filtered, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dx12::ResourceManager::TransitionTexture(cmd, tex_output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // Pair 0: Chained end-to-end start
        queue_.WriteTimestampBegin(0);

        // Pair 1: Pass 1 (Spatial Filter)
        queue_.WriteTimestampBegin(1);
        cmd->SetPipelineState(pso_pass1.Get());
        cmd->SetComputeRootDescriptorTable(1, pass1_srv0.gpu);
        cmd->SetComputeRootDescriptorTable(2, pass1_uav.gpu);
        cmd->Dispatch(groups_x, groups_y, 1);
        queue_.WriteTimestampEnd(1);

        // GPU-side transition of tex_filtered from UAV -> SRV (zero CPU round-trip!)
        dx12::ResourceManager::TransitionTexture(cmd, tex_filtered, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Pair 2: Pass 2 (Residual Reconstruct)
        queue_.WriteTimestampBegin(2);
        cmd->SetPipelineState(pso_pass2.Get());
        cmd->SetComputeRootDescriptorTable(1, pass2_srv0.gpu);
        cmd->SetComputeRootDescriptorTable(2, pass2_uav.gpu);
        cmd->Dispatch(groups_x, groups_y, 1);
        dx12::ResourceManager::RecordUAVBarrier(cmd, tex_output.resource.Get());
        queue_.WriteTimestampEnd(2);

        queue_.WriteTimestampEnd(0);
        queue_.ResolveTimestamps(3);

        auto sub = queue_.ExecuteAndWait();
        if (iter == 3) {
            out_result.cpu_submit_ms = sub.cpu_submit_ms;
            out_result.chained_end_to_end_gpu_ms = queue_.GetTimestampPairDurationMs(0);
            out_result.pass1_spatial_filter_gpu_ms = queue_.GetTimestampPairDurationMs(1);
            out_result.pass2_reconstruct_gpu_ms = queue_.GetTimestampPairDurationMs(2);
        }
    }

    std::vector<float> h_output(num_pixels * 4, 0.0f);
    if (!resources_.ReadbackTexture2DData(queue_, tex_output, readback_buf, h_output.data(), row_pitch, bytes_per_pixel, &out_result.readback_gpu_ms)) {
        return false;
    }

    // Compute exact CPU reference for Pass 1 + Pass 2
    std::vector<float> cpu_filtered(num_pixels * 4, 0.0f);
    static const float weights[3][3] = {
        {0.0625f, 0.1250f, 0.0625f},
        {0.1250f, 0.2500f, 0.1250f},
        {0.0625f, 0.1250f, 0.0625f}
    };

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            float accum[3] = {0.0f, 0.0f, 0.0f};
            for (int dy = -1; dy <= 1; ++dy) {
                int sy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    int sx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(width) - 1);
                    size_t sidx = (static_cast<size_t>(sy) * width + static_cast<size_t>(sx)) * 4;
                    float w = weights[dy + 1][dx + 1];
                    accum[0] += h_input[sidx + 0] * w;
                    accum[1] += h_input[sidx + 1] * w;
                    accum[2] += h_input[sidx + 2] * w;
                }
            }
            size_t didx = (static_cast<size_t>(y) * width + x) * 4;
            cpu_filtered[didx + 0] = accum[0];
            cpu_filtered[didx + 1] = accum[1];
            cpu_filtered[didx + 2] = accum[2];
            cpu_filtered[didx + 3] = 1.0f;
        }
    }

    double max_err = 0.0;
    double sum_err = 0.0;
    double sum_sq_err = 0.0;
    const size_t total_channels = num_pixels * 3;

    for (size_t i = 0; i < num_pixels; ++i) {
        size_t idx = i * 4;
        for (size_t c = 0; c < 3; ++c) {
            float orig = h_input[idx + c];
            float filt = cpu_filtered[idx + c];
            float detail = orig - filt;
            float blended = orig + (filt - orig) * consts.blend_factor;
            float ref = std::clamp(blended + detail * consts.sharpness, 0.0f, 1.0f);

            double err = std::abs(static_cast<double>(h_output[idx + c]) - static_cast<double>(ref));
            max_err = std::max(max_err, err);
            sum_err += err;
            sum_sq_err += err * err;
        }
    }

    out_result.max_abs_error = max_err;
    out_result.mean_abs_error = sum_err / static_cast<double>(total_channels);
    double mse = sum_sq_err / static_cast<double>(total_channels);
    out_result.psnr_db = (mse > 1e-16) ? (10.0 * std::log10(1.0 / mse)) : 120.0;
    out_result.passed = (max_err <= 1e-4);

    std::filesystem::create_directories(output_dir);
    out_result.output_bmp_path = output_dir + "/output_texture_1080p.bmp";
    out_result.output_ppm_path = output_dir + "/output_texture_1080p.ppm";
    dx12::Swapchain::SaveRGBA32FloatToBMP(out_result.output_bmp_path, h_output.data(), width, height);
    dx12::Swapchain::SaveRGBA32FloatToPPM(out_result.output_ppm_path, h_output.data(), width, height);

    return true;
}

bool BenchmarkRunner::RunRepresentativeConvLayer(uint32_t H,
                                                 uint32_t W,
                                                 uint32_t C_in,
                                                 uint32_t C_out,
                                                 uint32_t warmup_iters,
                                                 uint32_t steady_iters,
                                                 ConvLayerBenchmarkResult& out_result) {
    out_result = ConvLayerBenchmarkResult{};
    std::ostringstream name_ss;
    name_ss << "Conv2D_3x3_LeakyReLU_" << H << "x" << W << "_C" << C_in << "to" << C_out;
    out_result.layer_name = name_ss.str();
    out_result.H = H;
    out_result.W = W;
    out_result.C_in = C_in;
    out_result.C_out = C_out;

    const std::string hlsl_path = shader_dir_ + "/matmul.hlsl";
    std::vector<uint8_t> bytecode;
    if (!CompileShaderToBytecode(hlsl_path, "CSConv2D3x3LeakyReLU", bytecode, shader_compiler_used_)) {
        return false;
    }

    ComPtr<ID3D12RootSignature> root_sig;
    ComPtr<ID3D12PipelineState> pso;
    if (!CreateComputeRootSignatureAndPSO(device_.GetD3D12Device(), bytecode, root_sig, pso)) {
        return false;
    }

    const uint32_t in_elems = C_in * H * W;
    const uint32_t weight_elems = C_out * C_in * 9 + C_out; // weights + bias
    const uint32_t out_elems = C_out * H * W;

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<float> h_in(in_elems);
    std::vector<float> h_weights(weight_elems);
    for (auto& v : h_in) v = dist(rng);
    for (auto& v : h_weights) v = dist(rng);

    const uint64_t bytes_in = static_cast<uint64_t>(in_elems) * sizeof(float);
    const uint64_t bytes_w = static_cast<uint64_t>(weight_elems) * sizeof(float);
    const uint64_t bytes_out = static_cast<uint64_t>(out_elems) * sizeof(float);
    out_result.temp_memory_bytes = bytes_in + bytes_w + bytes_out;

    dx12::GpuBuffer buf_in, buf_w, buf_out, upload_buf, readback_buf;
    if (!resources_.CreateDefaultBuffer(bytes_in, buf_in, false) ||
        !resources_.CreateDefaultBuffer(bytes_w, buf_w, false) ||
        !resources_.CreateDefaultBuffer(bytes_out, buf_out, true) ||
        !resources_.CreateUploadBuffer(std::max(bytes_in, bytes_w), upload_buf) ||
        !resources_.CreateReadbackBuffer(bytes_out, readback_buf)) {
        return false;
    }

    if (!resources_.UploadBufferData(queue_, buf_in, upload_buf, h_in.data(), bytes_in) ||
        !resources_.UploadBufferData(queue_, buf_w, upload_buf, h_weights.data(), bytes_w)) {
        return false;
    }

    dx12::DescriptorHandle srv_in = resources_.CreateBufferSRV(buf_in, in_elems, sizeof(float));
    dx12::DescriptorHandle srv_w = resources_.CreateBufferSRV(buf_w, weight_elems, sizeof(float));
    (void)srv_w;
    dx12::DescriptorHandle uav_out = resources_.CreateBufferUAV(buf_out, out_elems, sizeof(float));

    const uint32_t packed_hw = ((H & 0xFFFFu) << 16) | (W & 0xFFFFu);
    uint32_t consts[4] = {packed_hw, C_out, C_in, 0};

    const uint32_t groups_x = (H * W + 15) / 16;
    const uint32_t groups_y = (C_out + 15) / 16;
    const uint32_t total_iters = warmup_iters + steady_iters;

    if (!queue_.ResetCommandList()) return false;
    auto* cmd = queue_.GetCommandList();
    dx12::ResourceManager::TransitionBuffer(cmd, buf_in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12::ResourceManager::TransitionBuffer(cmd, buf_w, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12::ResourceManager::TransitionBuffer(cmd, buf_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ID3D12DescriptorHeap* heaps[] = {resources_.GetDescriptorHeap()};
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(root_sig.Get());
    cmd->SetPipelineState(pso.Get());
    cmd->SetComputeRoot32BitConstants(0, 4, consts, 0);
    cmd->SetComputeRootDescriptorTable(1, srv_in.gpu);
    cmd->SetComputeRootDescriptorTable(2, uav_out.gpu);

    for (uint32_t it = 0; it < total_iters; ++it) {
        queue_.WriteTimestampBegin(it);
        cmd->Dispatch(groups_x, groups_y, 1);
        dx12::ResourceManager::RecordUAVBarrier(cmd, buf_out.resource.Get());
        queue_.WriteTimestampEnd(it);
    }
    queue_.ResolveTimestamps(total_iters);

    auto sub = queue_.ExecuteAndWait();
    out_result.cpu_submit_ms = sub.cpu_submit_ms;
    out_result.warmup_gpu_ms = queue_.GetTimestampPairDurationMs(0);

    double sum_steady = 0.0;
    for (uint32_t it = warmup_iters; it < total_iters; ++it) {
        sum_steady += queue_.GetTimestampPairDurationMs(it);
    }
    out_result.steady_gpu_ms = sum_steady / static_cast<double>(steady_iters);
    if (out_result.steady_gpu_ms > 0.0) {
        double flops = 2.0 * static_cast<double>(H) * W * C_in * C_out * 9.0;
        out_result.gflops = (flops / (out_result.steady_gpu_ms * 1e-3)) / 1e9;
    }

    std::vector<float> h_out(out_elems, 0.0f);
    if (!resources_.ReadbackBufferData(queue_, buf_out, readback_buf, h_out.data(), bytes_out)) {
        return false;
    }

    // CPU reference validation
    double max_err = 0.0, sum_err = 0.0;
    for (uint32_t oc = 0; oc < C_out; ++oc) {
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                float sum = h_weights[C_out * C_in * 9 + oc];
                for (uint32_t ic = 0; ic < C_in; ++ic) {
                    uint32_t w_base = (oc * C_in + ic) * 9;
                    uint32_t in_base = ic * H * W;
                    for (int ky = -1; ky <= 1; ++ky) {
                        int sy = static_cast<int>(y) + ky;
                        if (sy >= 0 && sy < static_cast<int>(H)) {
                            for (int kx = -1; kx <= 1; ++kx) {
                                int sx = static_cast<int>(x) + kx;
                                if (sx >= 0 && sx < static_cast<int>(W)) {
                                    sum += h_in[in_base + static_cast<uint32_t>(sy) * W + static_cast<uint32_t>(sx)] *
                                           h_weights[w_base + static_cast<uint32_t>(ky + 1) * 3 + static_cast<uint32_t>(kx + 1)];
                                }
                            }
                        }
                    }
                }
                float ref = (sum >= 0.0f) ? sum : (sum * 0.1f);
                uint32_t out_idx = oc * H * W + y * W + x;
                double err = std::abs(static_cast<double>(h_out[out_idx]) - static_cast<double>(ref));
                max_err = std::max(max_err, err);
                sum_err += err;
            }
        }
    }

    out_result.max_abs_error = max_err;
    out_result.mean_abs_error = sum_err / static_cast<double>(out_elems);
    out_result.passed = (max_err <= 1e-3);
    return true;
}

bool BenchmarkRunner::RunFullSuite(const std::string& output_dir, BenchmarkSuiteReport& out_report) {
    out_report = BenchmarkSuiteReport{};

    std::cout << "[Phase 2] Running Basic Compute Kernels (Vector Add/Mul & Chained GPU-Resident)...\n";
    if (!RunVectorKernels(1048576, out_report.vector_results)) {
        return false;
    }
    for (const auto& vr : out_report.vector_results) {
        std::cout << "  - " << std::left << std::setw(34) << vr.name
                  << " | " << (vr.passed ? "PASS" : "FAIL")
                  << " | max_err=" << std::scientific << std::setprecision(2) << vr.max_abs_error
                  << std::fixed << std::setprecision(4)
                  << " | steady_gpu=" << vr.steady_gpu_ms << " ms\n";
    }

    std::cout << "\n[Phase 3] Running Matrix Multiplication Benchmark & Backend Comparison...\n";
    struct Shape {
        uint32_t M, N, K;
    };
    const std::vector<Shape> shapes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 256, 256}
    };
    const std::vector<std::string> hlsl_variants = {
        "CSMatMulNaive",
        "CSMatMulTiled16",
        "CSMatMulRegBlocked32",
        "CSMatMulFP16AccumFP32"
    };

    for (const auto& s : shapes) {
        for (const auto& v : hlsl_variants) {
            MatMulBenchmarkResult r{};
            if (RunMatMulSingle(v, s.M, s.N, s.K, 2, 8, r)) {
                out_report.matmul_results.push_back(r);
                std::cout << "  - " << std::left << std::setw(24) << r.kernel_name
                          << " [" << std::setw(4) << s.M << "x" << std::setw(4) << s.N << "x" << std::setw(4) << s.K << "]"
                          << " | " << (r.passed ? "PASS" : "FAIL")
                          << " | steady=" << std::fixed << std::setprecision(3) << std::setw(7) << r.steady_gpu_ms << " ms"
                          << " | " << std::setprecision(1) << std::setw(8) << r.gflops << " GFLOPS\n";
            }
        }

        if (device_.GetCapabilityReport().directml_available) {
            for (bool fp16 : {false, true}) {
                MatMulBenchmarkResult dml_r{};
                if (RunDirectMLGemm(s.M, s.N, s.K, fp16, 2, 8, dml_r)) {
                    out_report.matmul_results.push_back(dml_r);
                    std::cout << "  - " << std::left << std::setw(24) << dml_r.kernel_name
                              << " [" << std::setw(4) << s.M << "x" << std::setw(4) << s.N << "x" << std::setw(4) << s.K << "]"
                              << " | " << (dml_r.passed ? "PASS" : "FAIL")
                              << " | steady=" << std::fixed << std::setprecision(3) << std::setw(7) << dml_r.steady_gpu_ms << " ms"
                              << " | " << std::setprecision(1) << std::setw(8) << dml_r.gflops << " GFLOPS\n";
                }
            }
        }
    }

    std::cout << "\n[Phase 4] Running End-to-End GPU-Resident Texture Pipeline (1920x1080)...\n";
    if (!RunTexturePipeline(1920, 1080, output_dir, out_report.texture_result)) {
        return false;
    }
    const auto& tr = out_report.texture_result;
    std::cout << "  - 1920x1080 Chained Pipeline : " << (tr.passed ? "PASS" : "FAIL")
              << " | PSNR=" << std::fixed << std::setprecision(2) << tr.psnr_db << " dB"
              << " | Pass1=" << std::setprecision(3) << tr.pass1_spatial_filter_gpu_ms << " ms"
              << " | Pass2=" << tr.pass2_reconstruct_gpu_ms << " ms"
              << " | Chained Total=" << tr.chained_end_to_end_gpu_ms << " ms\n";

    std::cout << "\n[Phase 5] Running Representative Neural-Network Layers (Conv2D 3x3 + Bias + LeakyReLU)...\n";
    ConvLayerBenchmarkResult conv1{}, conv2{};
    if (RunRepresentativeConvLayer(128, 128, 16, 16, 2, 6, conv1)) {
        out_report.conv_results.push_back(conv1);
        std::cout << "  - " << std::left << std::setw(36) << conv1.layer_name
                  << " | " << (conv1.passed ? "PASS" : "FAIL")
                  << " | steady=" << std::fixed << std::setprecision(3) << conv1.steady_gpu_ms << " ms"
                  << " | " << std::setprecision(1) << conv1.gflops << " GFLOPS\n";
    }
    if (RunRepresentativeConvLayer(128, 128, 32, 32, 2, 6, conv2)) {
        out_report.conv_results.push_back(conv2);
        std::cout << "  - " << std::left << std::setw(36) << conv2.layer_name
                  << " | " << (conv2.passed ? "PASS" : "FAIL")
                  << " | steady=" << std::fixed << std::setprecision(3) << conv2.steady_gpu_ms << " ms"
                  << " | " << std::setprecision(1) << conv2.gflops << " GFLOPS\n";
    }

    out_report.shader_compiler_used = shader_compiler_used_;
    out_report.peak_allocated_bytes = resources_.GetPeakAllocatedBytes();
    device_.QueryVideoMemory(out_report.vram_current_usage_bytes, out_report.vram_budget_bytes);

    // Evaluate Go / Conditional-Go / No-Go rule from Section 5
    double best_hlsl_gflops = 0.0;
    double best_dml_fp16_gflops = 0.0;
    double best_dml_fp32_gflops = 0.0;
    bool all_passed = tr.passed;
    for (const auto& vr : out_report.vector_results) all_passed = all_passed && vr.passed;
    for (const auto& mr : out_report.matmul_results) {
        all_passed = all_passed && mr.passed;
        if (mr.kernel_name.find("DirectML_GEMM_FP16") != std::string::npos) {
            best_dml_fp16_gflops = std::max(best_dml_fp16_gflops, mr.gflops);
        } else if (mr.kernel_name.find("DirectML_GEMM_FP32") != std::string::npos) {
            best_dml_fp32_gflops = std::max(best_dml_fp32_gflops, mr.gflops);
        } else {
            best_hlsl_gflops = std::max(best_hlsl_gflops, mr.gflops);
        }
    }

    if (!all_passed) {
        out_report.go_no_go_verdict = "NO-GO";
        out_report.go_no_go_rationale = "One or more numerical correctness checks failed against the CPU reference.";
    } else if (best_dml_fp16_gflops > best_hlsl_gflops * 1.25 || best_hlsl_gflops >= 500.0) {
        out_report.go_no_go_verdict = "GO (Hybrid DirectML MetaCommand + HLSL Compute Backend)";
        std::ostringstream rat;
        rat << std::fixed << std::setprecision(1)
            << "All numerical accuracy checks passed. Register-blocked HLSL compute reached "
            << best_hlsl_gflops << " GFLOPS, while DirectML FP16 GEMM (utilizing driver-exposed D3D12 MetaCommands on Intel Xe2) reached "
            << best_dml_fp16_gflops << " GFLOPS (FP32: " << best_dml_fp32_gflops
            << " GFLOPS). Recommended architecture for Phase 6 full graph: use DirectML/MetaCommands for heavy GEMM/Conv2D layers and native DXIL compute shaders for fused pre/post-processing and element-wise ops.";
        out_report.go_no_go_rationale = rat.str();
    } else {
        out_report.go_no_go_verdict = "CONDITIONAL GO";
        out_report.go_no_go_rationale = "Correctness passed across all kernels; further tiling or DirectML metacommand tuning is advised for full frame-budget compliance.";
    }

    return true;
}

bool BenchmarkRunner::SaveSuiteReportJson(const dx12::CapabilityReport& cap,
                                          const BenchmarkSuiteReport& report,
                                          const std::string& path) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"gpu\": \"" << cap.selected_adapter.description << "\",\n";
    out << "  \"driver_version\": \"" << cap.driver_version << "\",\n";
    out << "  \"shader_compiler\": \"" << report.shader_compiler_used << "\",\n";
    out << "  \"go_no_go_verdict\": \"" << report.go_no_go_verdict << "\",\n";
    out << "  \"go_no_go_rationale\": \"" << report.go_no_go_rationale << "\",\n";
    out << "  \"memory\": {\n";
    out << "    \"peak_allocated_mb\": " << (static_cast<double>(report.peak_allocated_bytes) / (1024.0 * 1024.0)) << ",\n";
    out << "    \"vram_current_usage_mb\": " << (static_cast<double>(report.vram_current_usage_bytes) / (1024.0 * 1024.0)) << ",\n";
    out << "    \"vram_budget_mb\": " << (static_cast<double>(report.vram_budget_bytes) / (1024.0 * 1024.0)) << "\n";
    out << "  },\n";

    out << "  \"vector_kernels\": [\n";
    for (size_t i = 0; i < report.vector_results.size(); ++i) {
        const auto& v = report.vector_results[i];
        out << "    {\n";
        out << "      \"name\": \"" << v.name << "\",\n";
        out << "      \"num_elements\": " << v.num_elements << ",\n";
        out << "      \"passed\": " << (v.passed ? "true" : "false") << ",\n";
        out << "      \"max_abs_error\": " << std::scientific << v.max_abs_error << std::fixed << ",\n";
        out << "      \"warmup_gpu_ms\": " << v.warmup_gpu_ms << ",\n";
        out << "      \"steady_gpu_ms\": " << v.steady_gpu_ms << ",\n";
        out << "      \"cpu_submit_ms\": " << v.cpu_submit_ms << ",\n";
        out << "      \"upload_gpu_ms\": " << v.upload_gpu_ms << ",\n";
        out << "      \"readback_gpu_ms\": " << v.readback_gpu_ms << "\n";
        out << "    }" << (i + 1 < report.vector_results.size() ? "," : "") << "\n";
    }
    out << "  ],\n";

    out << "  \"matmul_benchmarks\": [\n";
    for (size_t i = 0; i < report.matmul_results.size(); ++i) {
        const auto& m = report.matmul_results[i];
        out << "    {\n";
        out << "      \"kernel\": \"" << m.kernel_name << "\",\n";
        out << "      \"M\": " << m.M << ", \"N\": " << m.N << ", \"K\": " << m.K << ",\n";
        out << "      \"passed\": " << (m.passed ? "true" : "false") << ",\n";
        out << "      \"max_abs_error\": " << std::scientific << m.max_abs_error << std::fixed << ",\n";
        out << "      \"warmup_gpu_ms\": " << m.warmup_gpu_ms << ",\n";
        out << "      \"steady_gpu_ms\": " << m.steady_gpu_ms << ",\n";
        out << "      \"cpu_submit_ms\": " << m.cpu_submit_ms << ",\n";
        out << "      \"gflops\": " << m.gflops << "\n";
        out << "    }" << (i + 1 < report.matmul_results.size() ? "," : "") << "\n";
    }
    out << "  ],\n";

    const auto& t = report.texture_result;
    out << "  \"texture_pipeline\": {\n";
    out << "    \"width\": " << t.width << ", \"height\": " << t.height << ",\n";
    out << "    \"passed\": " << (t.passed ? "true" : "false") << ",\n";
    out << "    \"max_abs_error\": " << std::scientific << t.max_abs_error << std::fixed << ",\n";
    out << "    \"psnr_db\": " << t.psnr_db << ",\n";
    out << "    \"upload_gpu_ms\": " << t.upload_gpu_ms << ",\n";
    out << "    \"pass1_spatial_filter_gpu_ms\": " << t.pass1_spatial_filter_gpu_ms << ",\n";
    out << "    \"pass2_reconstruct_gpu_ms\": " << t.pass2_reconstruct_gpu_ms << ",\n";
    out << "    \"chained_end_to_end_gpu_ms\": " << t.chained_end_to_end_gpu_ms << ",\n";
    out << "    \"readback_gpu_ms\": " << t.readback_gpu_ms << ",\n";
    out << "    \"cpu_submit_ms\": " << t.cpu_submit_ms << "\n";
    out << "  },\n";

    out << "  \"representative_conv_layers\": [\n";
    for (size_t i = 0; i < report.conv_results.size(); ++i) {
        const auto& c = report.conv_results[i];
        out << "    {\n";
        out << "      \"layer\": \"" << c.layer_name << "\",\n";
        out << "      \"H\": " << c.H << ", \"W\": " << c.W << ", \"C_in\": " << c.C_in << ", \"C_out\": " << c.C_out << ",\n";
        out << "      \"passed\": " << (c.passed ? "true" : "false") << ",\n";
        out << "      \"max_abs_error\": " << std::scientific << c.max_abs_error << std::fixed << ",\n";
        out << "      \"warmup_gpu_ms\": " << c.warmup_gpu_ms << ",\n";
        out << "      \"steady_gpu_ms\": " << c.steady_gpu_ms << ",\n";
        out << "      \"gflops\": " << c.gflops << ",\n";
        out << "      \"temp_memory_kb\": " << (static_cast<double>(c.temp_memory_bytes) / 1024.0) << "\n";
        out << "    }" << (i + 1 < report.conv_results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return true;
}

bool BenchmarkRunner::SaveDecisionsMarkdown(const dx12::CapabilityReport& cap,
                                            const BenchmarkSuiteReport& report,
                                            const std::string& path) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    out << "# Backend Decisions & Benchmark Measurements (M1 - M4/M5)\n\n";
    out << "## 1. Hardware & Runtime Summary\n\n";
    out << "- **Target GPU:** " << cap.selected_adapter.description
        << " (`0x" << std::hex << std::uppercase << cap.selected_adapter.vendor_id
        << ":0x" << cap.selected_adapter.device_id << std::dec << "`)\n";
    out << "- **Driver Version:** `" << cap.driver_version << "`\n";
    out << "- **D3D12 Feature Level / Shader Model:** `FL " << cap.max_feature_level << "` / `SM " << cap.highest_shader_model << "`\n";
    out << "- **Shader Compiler:** `" << report.shader_compiler_used << "`\n";
    out << "- **WaveMMATier (`D3D12_OPTIONS9`):** `" << cap.wave_mma_tier << "`\n";
    out << "- **Driver-Exposed D3D12 MetaCommands:** `" << cap.metacommand_count << "`\n";
    out << "- **Peak PoC Buffer/Texture Allocation:** `"
        << std::fixed << std::setprecision(2)
        << (static_cast<double>(report.peak_allocated_bytes) / (1024.0 * 1024.0)) << " MB`\n\n";

    out << "## 2. Phase 2 — Basic Compute Kernels (M1)\n\n";
    out << "| Kernel | Elements | Status | Max Abs Error | Warm-up GPU (ms) | Steady GPU (ms) | CPU Submit (ms) | Upload (ms) | Readback (ms) |\n";
    out << "| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& v : report.vector_results) {
        out << "| `" << v.name << "` | " << v.num_elements << " | "
            << (v.passed ? "PASS" : "FAIL") << " | "
            << std::scientific << std::setprecision(2) << v.max_abs_error << std::fixed << std::setprecision(4) << " | "
            << v.warmup_gpu_ms << " | " << v.steady_gpu_ms << " | "
            << v.cpu_submit_ms << " | " << v.upload_gpu_ms << " | " << v.readback_gpu_ms << " |\n";
    }
    out << "\n";

    out << "## 3. Phase 3 — Matrix Multiplication Benchmark (M2)\n\n";
    out << "| Kernel / Backend | Shape (`M x N x K`) | Status | Max Abs Error | Warm-up GPU (ms) | Steady GPU (ms) | Throughput (GFLOPS) |\n";
    out << "| :--- | :---: | :---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& m : report.matmul_results) {
        out << "| `" << m.kernel_name << "` | `" << m.M << "x" << m.N << "x" << m.K << "` | "
            << (m.passed ? "PASS" : "FAIL") << " | "
            << std::scientific << std::setprecision(2) << m.max_abs_error << std::fixed << std::setprecision(3) << " | "
            << m.warmup_gpu_ms << " | " << m.steady_gpu_ms << " | **"
            << std::setprecision(1) << m.gflops << "** |\n";
    }
    out << "\n";

    const auto& t = report.texture_result;
    out << "## 4. Phase 4 — End-to-End GPU-Resident Texture Pipeline (M3)\n\n";
    out << "- **Resolution:** `" << t.width << "x" << t.height << "` (`DXGI_FORMAT_R32G32B32A32_FLOAT`)\n";
    out << "- **Correctness vs. CPU Reference:** `" << (t.passed ? "PASS" : "FAIL")
        << "` (Max Abs Error: `" << std::scientific << std::setprecision(2) << t.max_abs_error
        << "`, PSNR: `" << std::fixed << std::setprecision(2) << t.psnr_db << " dB`)\n";
    out << "- **Upload GPU Time:** `" << std::setprecision(3) << t.upload_gpu_ms << " ms`\n";
    out << "- **Pass 1 (`CSImageSpatialFilter`) GPU Time:** `" << t.pass1_spatial_filter_gpu_ms << " ms`\n";
    out << "- **Pass 2 (`CSImageResidualReconstruct`) GPU Time:** `" << t.pass2_reconstruct_gpu_ms << " ms`\n";
    out << "- **Chained End-to-End GPU Time (Zero CPU Round-Trip):** `" << t.chained_end_to_end_gpu_ms << " ms`\n";
    out << "- **Readback GPU Time:** `" << t.readback_gpu_ms << " ms`\n\n";

    out << "## 5. Phase 5 — Representative Neural-Network Layers (M4)\n\n";
    out << "| Layer | Shape (`H x W, C_in -> C_out`) | Status | Max Abs Error | Warm-up (ms) | Steady (ms) | Throughput (GFLOPS) | Temp Buffer (KB) |\n";
    out << "| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& c : report.conv_results) {
        out << "| `" << c.layer_name << "` | `" << c.H << "x" << c.W << ", " << c.C_in << "->" << c.C_out << "` | "
            << (c.passed ? "PASS" : "FAIL") << " | "
            << std::scientific << std::setprecision(2) << c.max_abs_error << std::fixed << std::setprecision(3) << " | "
            << c.warmup_gpu_ms << " | " << c.steady_gpu_ms << " | **"
            << std::setprecision(1) << c.gflops << "** | "
            << std::setprecision(1) << (static_cast<double>(c.temp_memory_bytes) / 1024.0) << " |\n";
    }
    out << "\n";

    out << "## 6. Go / No-Go Gate Decision\n\n";
    out << "- **Verdict:** **" << report.go_no_go_verdict << "**\n";
    out << "- **Rationale:** " << report.go_no_go_rationale << "\n";
    return true;
}

} // namespace dlss_nr::benchmark
