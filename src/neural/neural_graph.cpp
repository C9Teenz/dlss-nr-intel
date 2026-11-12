#include "neural/neural_graph.h"
#include "neural/embedded_neural_dxil.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace dlss_nr::neural {

namespace {

namespace fs = std::filesystem;

constexpr float kBlendScale = 0.73974609375f;   // half(blend_scale) of the recovered package
constexpr uint32_t kRootConstants = 19;          // NeuralConstants in neural_passes.hlsl
constexpr uint32_t kTableSize = 6;               // t0 t1 u0 u1 u2 u3
constexpr uint32_t kTimestampsPerSlot = 4;       // pre begin/end, post begin/end

using PFN_D3D12SerializeRootSignature = HRESULT(WINAPI*)(
    const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);

uint32_t FloatBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    return bits;
}

void UavBarrier(ID3D12GraphicsCommandList* cmd) {
    dx12::ResourceManager::RecordUAVBarrier(cmd, nullptr);
}

} // namespace

bool ApplyProfile(const std::string& name, NeuralOptions& options) {
    // style index / 128, local tone, local structure (MLX-DLSS PROFILES)
    if (name == "standard") {
        options.style = 0.0f; options.tone = 1.0f; options.structure = 1.0f;
    } else if (name == "natural") {
        options.style = 1.0f / 128.0f; options.tone = 1.0f; options.structure = 1.0f;
    } else if (name == "cinematic") {
        options.style = 2.0f / 128.0f; options.tone = 1.0f; options.structure = 1.0f;
    } else if (name == "neutral") {
        options.style = 0.0f; options.tone = 0.0f; options.structure = 0.0f;
    } else {
        return false;
    }
    return true;
}

std::string NeuralGraph::ModelFileName(uint32_t field_w, uint32_t field_h, const std::string& tag, bool synthetic) {
    std::ostringstream name;
    name << "hnet_" << field_w << "x" << field_h << "_" << tag << "_fp16" << (synthetic ? "_synthetic" : "") << ".onnx";
    return name.str();
}

NeuralGraph::~NeuralGraph() {
    ReleaseSession();
    if (ort_.IsLoaded() && dml_memory_) ort_.Api()->ReleaseMemoryInfo(dml_memory_);
    dml_memory_ = nullptr;
}

bool NeuralGraph::Fail(const std::string& message) {
    error_ = message;
    return false;
}

void NeuralGraph::SetConditioning(float style, float tone, float structure) {
    options_.style = style;
    options_.tone = tone;
    options_.structure = structure;
}

bool NeuralGraph::Initialize(ID3D12Device* device, ID3D12CommandQueue* queue, const NeuralOptions& options) {
    if (!device || !queue) return Fail("NeuralGraph needs a device and a queue");
    device_ = device;
    queue_ = queue;
    options_ = options;

    if (!ort_.Load(options_.ort_dirs, options_.directml_dirs, &error_, &ort_error_)) return false;
    if (!ort_.CreateDmlDevice(device_, dml_device_, &error_)) return false;
    if (ort_.IsLoaded() &&
        !ort_.Check(ort_.Api()->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &dml_memory_),
                    "CreateMemoryInfo(DML)", &error_)) {
        return false;
    }
    if (!upload_queue_.Initialize(device_, D3D12_COMMAND_LIST_TYPE_DIRECT, 1)) return Fail("upload queue");
    if (!resources_.Initialize(device_, 64)) return Fail("descriptor heap");
    if (!CreatePipelines()) return false;
    std::string hybrid_error;
    hybrid_ready_ = options_.allow_hybrid &&
                    hybrid_.Initialize(device_, options_.dml_gemms ? dml_device_.Get() : nullptr, &hybrid_error);

    D3D12_QUERY_HEAP_DESC qh{};
    qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qh.Count = kTimingSlots * kTimestampsPerSlot;
    if (FAILED(device_->CreateQueryHeap(&qh, IID_PPV_ARGS(&timestamp_heap_)))) return Fail("timestamp heap");
    D3D12_HEAP_PROPERTIES rb{};
    rb.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sizeof(uint64_t) * qh.Count;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device_->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                nullptr, IID_PPV_ARGS(&timestamp_readback_)))) {
        return Fail("timestamp readback");
    }
    if (FAILED(queue_->GetTimestampFrequency(&timestamp_freq_hz_)) || timestamp_freq_hz_ == 0) {
        timestamp_freq_hz_ = 0;
    }
    return true;
}

bool NeuralGraph::CreatePipelines() {
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;                    // t0 input, t1 noise
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 6;   // u0 features, u1 head, u2 history, u3 output, u4 stabiliser, u5 warp
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;

    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = kRootConstants;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 2;
    desc.pParameters = params;

    HMODULE d3d12 = GetModuleHandleA("d3d12.dll");
    if (!d3d12) d3d12 = LoadLibraryA("d3d12.dll");
    auto serialize = d3d12 ? reinterpret_cast<PFN_D3D12SerializeRootSignature>(
                                 GetProcAddress(d3d12, "D3D12SerializeRootSignature"))
                           : nullptr;
    ComPtr<ID3DBlob> blob, err;
    if (!serialize || FAILED(serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
        return Fail("serialize neural root signature");
    }
    if (FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                            IID_PPV_ARGS(&root_sig_)))) {
        return Fail("create neural root signature");
    }

    auto make = [&](const uint8_t* code, size_t size, ComPtr<ID3D12PipelineState>& out) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root_sig_.Get();
        pso.CS.pShaderBytecode = code;
        pso.CS.BytecodeLength = size;
        return SUCCEEDED(device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&out)));
    };
    if (!make(embedded::kNeuralPreDxil, sizeof(embedded::kNeuralPreDxil), pso_pre_) ||
        !make(embedded::kNeuralPostDxil, sizeof(embedded::kNeuralPostDxil), pso_post_) ||
        !make(embedded::kNeuralStabilizeDxil, sizeof(embedded::kNeuralStabilizeDxil), pso_stabilize_) ||
        !make(embedded::kNeuralMotionDxil, sizeof(embedded::kNeuralMotionDxil), pso_motion_) ||
        !make(embedded::kNeuralCommitDxil, sizeof(embedded::kNeuralCommitDxil), pso_commit_) ||
        !make(embedded::kNeuralFieldColourDxil, sizeof(embedded::kNeuralFieldColourDxil), pso_field_colour_) ||
        !make(embedded::kNeuralRefCommitDxil, sizeof(embedded::kNeuralRefCommitDxil), pso_ref_commit_) ||
        !make(embedded::kNeuralWarpMotionDxil, sizeof(embedded::kNeuralWarpMotionDxil), pso_warp_motion_) ||
        !make(embedded::kNeuralHeadWarpDxil, sizeof(embedded::kNeuralHeadWarpDxil), pso_head_warp_) ||
        !make(embedded::kNeuralGuideFitDxil, sizeof(embedded::kNeuralGuideFitDxil), pso_guide_fit_) ||
        !make(embedded::kNeuralGuideSmoothDxil, sizeof(embedded::kNeuralGuideSmoothDxil), pso_guide_smooth_) ||
        !make(embedded::kNeuralCoarseSumsDxil, sizeof(embedded::kNeuralCoarseSumsDxil), pso_coarse_sums_) ||
        !make(embedded::kNeuralCoarseFitDxil, sizeof(embedded::kNeuralCoarseFitDxil), pso_coarse_fit_) ||
        !make(embedded::kNeuralQuarterCurDxil, sizeof(embedded::kNeuralQuarterCurDxil), pso_quarter_cur_) ||
        !make(embedded::kNeuralQuarterRefDxil, sizeof(embedded::kNeuralQuarterRefDxil), pso_quarter_ref_) ||
        !make(embedded::kNeuralWarpPyramidDxil, sizeof(embedded::kNeuralWarpPyramidDxil), pso_warp_pyramid_) ||
        !make(embedded::kNeuralStepMotionDxil, sizeof(embedded::kNeuralStepMotionDxil), pso_step_motion_) ||
        !make(embedded::kNeuralChainDxil, sizeof(embedded::kNeuralChainDxil), pso_chain_) ||
        !make(embedded::kNeuralKeepLumaDxil, sizeof(embedded::kNeuralKeepLumaDxil), pso_keep_luma_)) {
        return Fail("create neural pipelines (DXIL rejected?)");
    }
    return true;
}

void NeuralGraph::ReleaseSession() {
    if (!ort_.IsLoaded()) return;
    const OrtApi* api = ort_.Api();
    if (binding_) api->ReleaseIoBinding(binding_);
    if (features_value_) api->ReleaseValue(features_value_);
    if (head_value_) api->ReleaseValue(head_value_);
    if (features_alloc_) ort_.Dml()->FreeGPUAllocation(features_alloc_);
    if (head_alloc_) ort_.Dml()->FreeGPUAllocation(head_alloc_);
    if (session_) api->ReleaseSession(session_);
    binding_ = nullptr;
    features_value_ = head_value_ = nullptr;
    features_alloc_ = head_alloc_ = nullptr;
    session_ = nullptr;
}

std::string NeuralGraph::FindModel(const std::string& name) const {
    for (const auto& dir : options_.model_dirs) {
        std::error_code ec;
        if (fs::is_regular_file(fs::path(dir) / name, ec)) return (fs::path(dir) / name).string();
    }
    return "";
}

bool NeuralGraph::CreateSession(const std::string& path) {
    if (!ort_.IsLoaded()) return Fail("ONNX model " + path + " needs ONNX Runtime: " + ort_error_);
    const OrtApi* api = ort_.Api();
    auto started = std::chrono::high_resolution_clock::now();

    OrtSessionOptions* so = nullptr;
    if (!ort_.Check(api->CreateSessionOptions(&so), "CreateSessionOptions", &error_)) return false;
    bool ok = ort_.Check(api->DisableMemPattern(so), "DisableMemPattern", &error_) &&
              ort_.Check(api->SetSessionExecutionMode(so, ORT_SEQUENTIAL), "SetSessionExecutionMode", &error_) &&
              ort_.Check(api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL), "SetOptimizationLevel", &error_) &&
              ort_.Check(ort_.Dml()->SessionOptionsAppendExecutionProvider_DML1(so, dml_device_.Get(), queue_.Get()),
                         "AppendExecutionProvider_DML1", &error_);
    if (ok) {
        std::wstring wpath = fs::path(path).wstring();
        ok = ort_.Check(api->CreateSession(ort_.Env(), wpath.c_str(), so, &session_), "CreateSession", &error_);
    }
    api->ReleaseSessionOptions(so);
    if (!ok) return false;

    // the whole graph reads the features and writes the head; the hybrid middle goes from
    // the front kernels' 1/4-extent 64-channel value to the back kernels' input
    const int64_t fh = extent_.field_h, fw = extent_.field_w;
    const int64_t mh = hybrid_.MidHeight(), mw = hybrid_.MidWidth(), mc = hybrid_.MidChannels();
    const int64_t in_shape[4] = {1, hybrid_active_ ? mh : fh, hybrid_active_ ? mw : fw, hybrid_active_ ? mc : 16};
    const int64_t out_shape[4] = {1, in_shape[1], in_shape[2], hybrid_active_ ? mc : 4};
    dx12::GpuBuffer& in_buf = hybrid_active_ ? hybrid_.MidIn() : features_buf_;
    dx12::GpuBuffer& out_buf = hybrid_active_ ? hybrid_.MidOut() : head_buf_;
    const char* in_name = hybrid_active_ ? "value" : "features";
    const char* out_name = hybrid_active_ ? "value_out" : "head";
    const size_t in_bytes = static_cast<size_t>(in_shape[1] * in_shape[2] * in_shape[3] * 2);
    const size_t out_bytes = static_cast<size_t>(out_shape[1] * out_shape[2] * out_shape[3] * 2);
    ok = ort_.Check(ort_.Dml()->CreateGPUAllocationFromD3DResource(in_buf.resource.Get(), &features_alloc_),
                    "CreateGPUAllocation(input)", &error_) &&
         ort_.Check(ort_.Dml()->CreateGPUAllocationFromD3DResource(out_buf.resource.Get(), &head_alloc_),
                    "CreateGPUAllocation(output)", &error_) &&
         ort_.Check(api->CreateTensorWithDataAsOrtValue(dml_memory_, features_alloc_, in_bytes, in_shape, 4,
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &features_value_),
                    "CreateTensor(input)", &error_) &&
         ort_.Check(api->CreateTensorWithDataAsOrtValue(dml_memory_, head_alloc_, out_bytes, out_shape, 4,
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &head_value_),
                    "CreateTensor(output)", &error_) &&
         ort_.Check(api->CreateIoBinding(session_, &binding_), "CreateIoBinding", &error_) &&
         ort_.Check(api->BindInput(binding_, in_name, features_value_), "BindInput", &error_) &&
         ort_.Check(api->BindOutput(binding_, out_name, head_value_), "BindOutput", &error_);
    session_create_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - started).count();
    return ok;
}

bool NeuralGraph::EnsureExtent(uint32_t output_w, uint32_t output_h, DXGI_FORMAT format) {
    NeuralExtent wanted = ComputeExtent(output_w, output_h, options_.render_scale);
    if (IsReady() && wanted == extent_ && format == format_ && extent_scale_ == options_.render_scale) return true;
    const auto setup_started = std::chrono::high_resolution_clock::now();

    // the model for the field must exist before anything is torn down
    // the hybrid graph when its middle and the ends' weights exist, else the whole graph in ONNX
    std::string blob;
    std::string path;
    HybridSplit split = HybridSplit::kDeep;
    if (hybrid_ready_ && options_.model_tag == "nq-exp") {
        blob = FindModel(HybridEnds::BlobFileName(options_.synthetic_weights));
        // native: the whole graph in our kernels and DirectML GEMMs, the weight blob is the model
        if (!blob.empty() && options_.native && hybrid_.UsesDmlGemms()) {
            split = HybridSplit::kNative;
            path = blob;
        }
        // else the deepest split whose middle exists: more of the graph in the fused kernels
        for (HybridSplit s : {HybridSplit::kDeep, HybridSplit::kShallow}) {
            if (!path.empty()) break;
            if (blob.empty() || (s == HybridSplit::kDeep && !options_.deep_hybrid)) continue;
            path = FindModel(HybridEnds::MiddleFileName(s, wanted.field_w, wanted.field_h, options_.synthetic_weights));
            if (!path.empty()) {
                split = s;
                break;
            }
        }
    }
    const bool hybrid = !path.empty();
    const std::string name = ModelFileName(wanted.field_w, wanted.field_h, options_.model_tag, options_.synthetic_weights);
    if (!hybrid) path = FindModel(name);
    if (path.empty()) {
        std::ostringstream msg;
        msg << "no ONNX model " << name << " for output " << output_w << "x" << output_h << " (inner "
            << wanted.inner_w << "x" << wanted.inner_h << "); export it with tools/hnet/export_onnx.py --field "
            << wanted.field_w << "x" << wanted.field_h;
        return Fail(msg.str());
    }

    // Same field and model: the session, the features/head buffers and the noise carry over,
    // and only what lives at the output extent is rebuilt (a resize without a new graph).
    const bool keep_field = IsReady() && path == model_path_ && wanted.field_w == extent_.field_w &&
                            wanted.field_h == extent_.field_h;
    if (!keep_field) {
        ReleaseSession();
        InvalidateNetworkLists();   // they point at the old field's buffers
        native_active_ = false;
        noise_buf_ = {};
        features_buf_ = {};
        head_buf_ = {};
        hybrid_active_ = hybrid;
        hybrid_.SetPrecise(options_.precise);
        if (hybrid && (!hybrid_.LoadWeights(blob, resources_, upload_queue_, &error_) ||
                       !hybrid_.Allocate(split, wanted.field_w, wanted.field_h, features_buf_, head_buf_, resources_,
                                         upload_queue_, &error_))) {
            return false;
        }
    }
    input_tex_ = {};
    output_tex_ = {};
    history_buf_ = {};
    history_valid_ = false;
    stable_valid_ = false;
    head_valid_ = false;   // the first frame runs the whole network
    ref_valid_ = false;
    capture_valid_ = false;
    phase_ = 0;
    extent_ = wanted;
    format_ = format;
    extent_scale_ = options_.render_scale;

    const uint64_t field_pixels = static_cast<uint64_t>(extent_.field_w) * extent_.field_h;
    const uint64_t output_pixels = static_cast<uint64_t>(extent_.output_w) * extent_.output_h;
    const uint64_t blocks = static_cast<uint64_t>((extent_.field_w + 7) / 8) * ((extent_.field_h + 7) / 8);
    // warp_buf_ (neural_passes.hlsl): 32 B per field pixel + a motion per block, then the colour
    // slope (CSGuideFit): two fits of 36 B and a 24 B range per 2x2 cell, the head kept for
    // CSRefCommit's blend (8 B per field pixel) and its count of misses, then the regional model:
    // 112 B of sums and a 72 B fit per 16x16 cell, then the motion pyramid: two quarter-resolution
    // lumas (4 B per 4x4 field pixels) and a vector per 32x32, then the motion chain: last frame's
    // luma (4 B per field pixel), a step and two halves of two sums per block (neural_passes.hlsl)
    const uint64_t cells = static_cast<uint64_t>((extent_.field_w + 1) / 2) * ((extent_.field_h + 1) / 2);
    const uint64_t coarse = static_cast<uint64_t>((extent_.field_w + 15) / 16) * ((extent_.field_h + 15) / 16);
    const uint64_t quarter_w = (extent_.field_w + 3) / 4, quarter_h = (extent_.field_h + 3) / 4;
    const uint64_t pyramid = ((quarter_w + 7) / 8) * ((quarter_h + 7) / 8);
    const uint64_t warp_bytes = ((field_pixels * 32 + blocks * 8 + 15) & ~15ull) + cells * 96 + field_pixels * 8 + 32 +
                                coarse * 184 + quarter_w * quarter_h * 8 + pyramid * 8 + field_pixels * 4 + blocks * 40;
    if (!resources_.CreateTexture2D(output_w, output_h, format, input_tex_, false) ||
        !resources_.CreateTexture2D(output_w, output_h, format, output_tex_, true) ||
        !resources_.CreateDefaultBuffer(output_pixels * 8, history_buf_, true) ||
        !resources_.CreateDefaultBuffer(field_pixels * 32 + blocks * 8, stable_buf_, true) ||
        !resources_.CreateDefaultBuffer(warp_bytes, warp_buf_, true)) {
        return Fail("allocate neural textures");
    }
    if (!keep_field && (!resources_.CreateDefaultBuffer(field_pixels * 8, noise_buf_, false) ||
                        !resources_.CreateDefaultBuffer(field_pixels * 32, features_buf_, true) ||
                        !resources_.CreateDefaultBuffer(field_pixels * 8, head_buf_, true))) {
        return Fail("allocate neural buffers");
    }

    if (!keep_field) {
        // the noise depends on the field and the frame index only; the live index stays 0
        std::vector<uint16_t> noise = DeterministicNoise(extent_.field_w, extent_.field_h, 0);
        dx12::GpuBuffer staging;
        if (!resources_.CreateUploadBuffer(noise_buf_.byte_size, staging) ||
            !resources_.UploadBufferData(upload_queue_, noise_buf_, staging, noise.data(), noise_buf_.byte_size)) {
            return Fail("upload noise table");
        }
    }

    // the buffers the DML EP sees must be in UNORDERED_ACCESS, and stay there
    if (!upload_queue_.ResetCommandList()) return Fail("upload queue reset");
    ID3D12GraphicsCommandList* cmd = upload_queue_.GetCommandList();
    dx12::ResourceManager::TransitionBuffer(cmd, noise_buf_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12::ResourceManager::TransitionBuffer(cmd, features_buf_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12::ResourceManager::TransitionBuffer(cmd, head_buf_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12::ResourceManager::TransitionBuffer(cmd, history_buf_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12::ResourceManager::TransitionBuffer(cmd, stable_buf_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dx12::ResourceManager::TransitionBuffer(cmd, warp_buf_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    upload_queue_.ExecuteAndWait();

    resources_.ResetDescriptors();
    table_ = resources_.CreateTexture2DSRV(input_tex_);
    D3D12_SHADER_RESOURCE_VIEW_DESC raw_srv{};
    raw_srv.Format = DXGI_FORMAT_R32_TYPELESS;
    raw_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    raw_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    raw_srv.Buffer.NumElements = static_cast<UINT>(noise_buf_.byte_size / 4);
    raw_srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    device_->CreateShaderResourceView(noise_buf_.resource.Get(), &raw_srv, resources_.AllocateDescriptor().cpu);
    for (dx12::GpuBuffer* buf : {&features_buf_, &head_buf_, &history_buf_}) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC raw_uav{};
        raw_uav.Format = DXGI_FORMAT_R32_TYPELESS;
        raw_uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        raw_uav.Buffer.NumElements = static_cast<UINT>(buf->byte_size / 4);
        raw_uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_->CreateUnorderedAccessView(buf->resource.Get(), nullptr, &raw_uav, resources_.AllocateDescriptor().cpu);
    }
    resources_.CreateTexture2DUAV(output_tex_);
    for (dx12::GpuBuffer* buf : {&stable_buf_, &warp_buf_}) {   // u4, u5 after the output
        D3D12_UNORDERED_ACCESS_VIEW_DESC raw_uav{};
        raw_uav.Format = DXGI_FORMAT_R32_TYPELESS;
        raw_uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        raw_uav.Buffer.NumElements = static_cast<UINT>(buf->byte_size / 4);
        raw_uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_->CreateUnorderedAccessView(buf->resource.Get(), nullptr, &raw_uav, resources_.AllocateDescriptor().cpu);
    }

    if (keep_field) {
        session_create_ms_ = 0.0;
        if (Warping() && part_begin_.size() != options_.interval + 1 && !BalanceParts()) return false;
        return true;
    }
    model_path_ = path;
    if (hybrid_active_ && hybrid_.Split() == HybridSplit::kNative) {
        // no session: the "session" time is the whole setup (weights on first use, GEMM compile)
        native_active_ = true;
        part_begin_.clear();
        if (Warping() && !BalanceParts()) return false;
        session_create_ms_ = std::chrono::duration<double, std::milli>(
                                 std::chrono::high_resolution_clock::now() - setup_started).count();
        return true;
    }
    return CreateSession(path);
}

void NeuralGraph::SetConstants(ID3D12GraphicsCommandList* cmd) const {
    const uint32_t constants[kRootConstants] = {
        extent_.field_w, extent_.field_h, extent_.inner_w, extent_.inner_h, extent_.output_w, extent_.output_h,
        ((history_valid_ && options_.temporal) ? 1u : 0u) | (stable_valid_ ? 2u : 0u) | (stable_parity_ ? 4u : 0u) |
            (options_.warp_model ? 8u : 0u) | (capture_now_ ? 16u : 0u) | (blend_now_ ? 32u : 0u) |
            (options_.warp_pyramid ? 64u : 0u) | chain_flags_ | (options_.warp_fade_none ? 4096u : 0u),
        FloatBits(options_.style), FloatBits(options_.tone), FloatBits(options_.structure),
        FloatBits(options_.intensity), FloatBits(kBlendScale), FloatBits(options_.stabilize),
        FloatBits(options_.stabilize_tolerance), FloatBits(options_.stabilize_motion), FloatBits(warp_predict_),
        FloatBits(options_.warp_blend), FloatBits(fade_lo_), FloatBits(fade_hi_)};
    cmd->SetComputeRoot32BitConstants(0, kRootConstants, constants, 0);
}

// The op sequence cut into options_.interval parts of about equal GPU time: three profiled runs of
// the whole network on the upload queue, each op's cost the least of them, the boundaries where
// the running total passes k/interval of it. Without timestamps: equal op counts.
// Every part holds at least one op and the last part the last op, which alone writes the head:
// the frames before it show the warped reference in the head buffer, so a head completed by an
// earlier part would be overwritten before it becomes the reference. That happened when the
// profile was taken while another process used the GPU (a game): time slicing booked one op with
// most of the total, the share was reached early, the trailing parts were empty, and the
// reference stopped updating (a stale correction; the interval check then failed at random).
bool NeuralGraph::BalanceParts() {
    const uint32_t parts = std::clamp<uint32_t>(options_.interval, 1u, kMaxInterval);
    const size_t ops = hybrid_.OpCount();
    if (ops < parts) return Fail("the network has fewer ops than parts");
    std::vector<double> cost;
    const bool was_profiling = hybrid_.Profiling();
    if ((was_profiling || hybrid_.EnableProfiling(upload_queue_.GetQueue()))) {
        for (int run = 0; run < 3; ++run) {
            if (!upload_queue_.ResetCommandList()) {
                cost.clear();
                break;
            }
            ID3D12GraphicsCommandList* cmd = upload_queue_.GetCommandList();
            hybrid_.RecordFront(cmd);
            hybrid_.RecordCore(cmd);
            hybrid_.RecordBack(cmd);
            upload_queue_.ExecuteAndWait();
            std::vector<std::pair<std::string, double>> profile;
            if (!hybrid_.ReadProfile(profile) || profile.size() != ops) {
                cost.clear();
                break;
            }
            // the least of the runs: a run that shared the GPU only ever adds time
            if (cost.empty()) cost.assign(ops, 1e30);
            for (size_t i = 0; i < ops; ++i) {
                const double c = profile[i].second;
                cost[i] = std::min(cost[i], c > 0.0 && c < 1e3 ? c : 0.0);
            }
        }
        if (!was_profiling) hybrid_.DisableProfiling();
    }
    double total = 0.0;
    for (double c : cost) total += c;
    if (cost.size() != ops || total <= 0.0) {
        cost.assign(ops, 1.0);
        total = static_cast<double>(ops);
    }
    part_begin_.assign(parts + 1, ops);
    part_begin_[0] = 0;
    double running = 0.0;
    size_t i = 0;
    for (uint32_t k = 1; k < parts; ++k) {
        // an op goes into the earlier part while its middle lies before the share
        while (i < ops && running + 0.5 * cost[i] <= total * k / parts) running += cost[i++];
        part_begin_[k] = std::clamp(i, part_begin_[k - 1] + 1, ops - (parts - k));
        while (i < part_begin_[k]) running += cost[i++];
    }
    std::ostringstream summary;
    summary << std::fixed << std::setprecision(2) << "network parts (ops, profiled ms):";
    for (uint32_t k = 0; k < parts; ++k) {
        double ms = 0.0;
        for (size_t j = part_begin_[k]; j < part_begin_[k + 1]; ++j) ms += cost[j];
        summary << " [" << part_begin_[k] << ", " << part_begin_[k + 1] << ") " << ms;
    }
    part_summary_ = summary.str();
    InvalidateNetworkLists();
    return true;
}

void NeuralGraph::RecordPre(ID3D12GraphicsCommandList* cmd, uint32_t timing_slot) {
    const uint32_t ts = (timing_slot % kTimingSlots) * kTimestampsPerSlot;
    // interval N once a head exists: phase 0 takes this frame's features and runs part 0, phase k
    // part k, phase N-1 completes the head. The parts' outputs wait in their buffers in between.
    const uint32_t parts = part_begin_.empty() ? 0u : static_cast<uint32_t>(part_begin_.size()) - 1;
    const bool split = Warping() && head_valid_ && parts == options_.interval;
    if (!split) phase_ = 0;
    run_front_ = !split || phase_ == 0;
    run_back_ = !split || phase_ + 1 == parts;
    part_ = split ? 1 + phase_ : 0;
    slot_ = timing_slot % kTimingSlots;
    cmd->EndQuery(timestamp_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts + 0);
    if (run_front_) {
        dx12::ResourceManager::TransitionTexture(cmd, input_tex_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ID3D12DescriptorHeap* heaps[] = {resources_.GetDescriptorHeap()};
        cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootSignature(root_sig_.Get());
        SetConstants(cmd);
        cmd->SetComputeRootDescriptorTable(1, table_.gpu);
        cmd->SetPipelineState(pso_pre_.Get());
        cmd->Dispatch((extent_.field_w + 7) / 8, (extent_.field_h + 7) / 8, 1);
        UavBarrier(cmd);
        // the native graph's blocks are RunNetwork's prerecorded lists; the ORT split records its ends here
        if (hybrid_active_ && !native_active_) hybrid_.RecordFront(cmd);
    }
    cmd->EndQuery(timestamp_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts + 1);
}

void NeuralGraph::InvalidateNetworkLists() {
    for (auto& slot : network_lists_) {
        for (auto& l : slot) l.valid = false;   // allocators are reset when re-recorded
    }
}

bool NeuralGraph::RunNetwork() {
    if (native_active_) {
        // Recording the ~570 dispatches took ~4 ms of CPU per frame; they never change within an
        // extent, so each (slot, part) is recorded on first use and resubmitted after that.
        if (hybrid_.Profiling() != lists_profiled_) {
            InvalidateNetworkLists();
            lists_profiled_ = hybrid_.Profiling();
        }
        const uint32_t part = part_;
        NetworkList& n = network_lists_[slot_][part];
        if (!n.valid) {
            if (!n.allocator &&
                (FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&n.allocator))) ||
                 FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, n.allocator.Get(), nullptr,
                                                   IID_PPV_ARGS(&n.list))) ||
                 FAILED(n.list->Close()))) {
                return Fail("network command list");
            }
            // only reached once the slot's previous frame completed (the callers' contract)
            if (FAILED(n.allocator->Reset()) || FAILED(n.list->Reset(n.allocator.Get(), nullptr))) {
                return Fail("reset network command list");
            }
            if (part == 0) {
                hybrid_.RecordFront(n.list.Get());
                hybrid_.RecordCore(n.list.Get());
                hybrid_.RecordBack(n.list.Get());
            } else {
                hybrid_.RecordRange(n.list.Get(), part_begin_[part - 1], part_begin_[part]);
            }
            if (FAILED(n.list->Close())) return Fail("close network command list");
            n.valid = true;
        }
        ID3D12CommandList* lists[] = {n.list.Get()};
        queue_->ExecuteCommandLists(1, lists);
        return true;
    }
    if (!session_ || !binding_) return Fail("no ORT session");
    return ort_.Check(ort_.Api()->RunWithBinding(session_, nullptr, binding_), "RunWithBinding", &error_);
}

void NeuralGraph::RecordPost(ID3D12GraphicsCommandList* cmd, uint32_t timing_slot) {
    const uint32_t ts = (timing_slot % kTimingSlots) * kTimestampsPerSlot;
    // the age (frames) of the picture the shown head was computed from: a head completed now by
    // the last part is interval - 1 frames old, by the whole network 0; then one more each frame
    const uint32_t age = run_back_ ? (part_ != 0 ? options_.interval - 1 : 0) : warp_age_ + 1;
    warp_predict_ = warp_age_valid_ && warp_age_ > 0 ? static_cast<float>(age) / static_cast<float>(warp_age_) : 0.0f;
    // the detail fade by speed: the motion since the reference over its age (CSHeadWarp)
    const bool fading = options_.warp_fade_full > 0.0f;
    fade_hi_ = fading ? options_.warp_fade_full * static_cast<float>(age) : 0.0f;
    fade_lo_ = fading ? std::min(options_.warp_fade_start, options_.warp_fade_full) * static_cast<float>(age) : 0.0f;
    // the blend of a new reference with the last one (CSRefCommit): the frame that takes the next
    // head's features keeps what it shows (the last reference moved onto exactly that picture), the
    // frame that completes that head blends with it
    const bool blending = Warping() && options_.warp && options_.warp_model && options_.warp_blend < 1.0f;
    capture_now_ = blending && part_ == 1 && ref_valid_ && !run_back_;
    blend_now_ = blending && run_back_ && part_ != 0 && capture_valid_;
    cmd->EndQuery(timestamp_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts + 2);
    if (hybrid_active_ && !native_active_) hybrid_.RecordBack(cmd);
    dx12::ResourceManager::TransitionTexture(cmd, input_tex_, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dx12::ResourceManager::TransitionTexture(cmd, output_tex_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[] = {resources_.GetDescriptorHeap()};
    cmd->SetDescriptorHeaps(1, heaps);
    // the motion chain this frame: a step needs last frame's luma; the pending sum restarts on a
    // features frame and is whole while every frame since had a step; it becomes the active one
    // when its head completes
    const bool chaining = Warping() && options_.warp && options_.warp_chain;
    bool pending_after = false, active_after = false;
    chain_flags_ = 0;
    if (chaining) {
        pending_after = run_front_ || (chain_pending_ && chain_prev_);
        active_after = run_back_ ? pending_after : (chain_active_ && chain_prev_);
        chain_flags_ = (chain_prev_ ? 256u : 0u) | (run_front_ ? 512u : 0u) | (run_back_ ? 1024u : 0u) |
                       (chain_parity_ ? 2048u : 0u) | (active_after ? 128u : 0u);
    }
    cmd->SetComputeRootSignature(root_sig_.Get());
    SetConstants(cmd);
    cmd->SetComputeRootDescriptorTable(1, table_.gpu);
    if (options_.stabilize > 0.0f && run_back_) {
        // a new head: the motion since the last one (CSMotion, one group per 8x8 block), then the
        // head averaged with the last ones along it (CSHeadStabilize)
        if (stable_valid_) {
            cmd->SetPipelineState(pso_motion_.Get());
            cmd->Dispatch((extent_.field_w + 7) / 8, (extent_.field_h + 7) / 8, 1);
            UavBarrier(cmd);
        }
        cmd->SetPipelineState(pso_stabilize_.Get());
        cmd->Dispatch((extent_.field_w + 7) / 8, (extent_.field_h + 7) / 8, 1);
        UavBarrier(cmd);
        cmd->SetPipelineState(pso_commit_.Get());
        cmd->Dispatch((extent_.field_w + 7) / 8, (extent_.field_h + 7) / 8, 1);
        UavBarrier(cmd);
        // from the next frame on (this frame's constants are already set): the state just written
        stable_valid_ = true;
        stable_parity_ = !stable_parity_;
    }
    if (Warping() && options_.warp) {
        const UINT gx = (extent_.field_w + 7) / 8, gy = (extent_.field_h + 7) / 8;
        if (run_back_) {   // a new complete head: the reference from now on
            cmd->SetPipelineState(pso_ref_commit_.Get());
            cmd->Dispatch(gx, gy, 1);
            UavBarrier(cmd);
            // ... its quarter-resolution luma for the motion pyramid (always: the pyramid may be
            // switched on by an ini reload while this reference stays)
            cmd->SetPipelineState(pso_quarter_ref_.Get());
            cmd->Dispatch(((extent_.field_w + 3) / 4 + 7) / 8, ((extent_.field_h + 3) / 4 + 7) / 8, 1);
            UavBarrier(cmd);
            if (options_.warp_model) {   // ... and its colour slope (2x2 cells, 8x8 cells a group)
                const UINT cx = ((extent_.field_w + 1) / 2 + 7) / 8, cy = ((extent_.field_h + 1) / 2 + 7) / 8;
                cmd->SetPipelineState(pso_guide_fit_.Get());
                cmd->Dispatch(cx, cy, 1);
                UavBarrier(cmd);
                cmd->SetPipelineState(pso_guide_smooth_.Get());
                cmd->Dispatch(cx, cy, 1);
                UavBarrier(cmd);
                // ... and the regional model (16x16 cells, one group each, then 8x8 cells a group)
                const UINT kx = (extent_.field_w + 15) / 16, ky = (extent_.field_h + 15) / 16;
                cmd->SetPipelineState(pso_coarse_sums_.Get());
                cmd->Dispatch(kx, ky, 1);
                UavBarrier(cmd);
                cmd->SetPipelineState(pso_coarse_fit_.Get());
                cmd->Dispatch((kx + 7) / 8, (ky + 7) / 8, 1);
                UavBarrier(cmd);
            }
            ref_valid_ = true;
            capture_valid_ = false;
        }
        if (capture_now_) capture_valid_ = true;   // written by CSHeadWarp below
        warp_age_ = age;
        warp_age_valid_ = ref_valid_;
        if (ref_valid_ || chaining) {   // this frame's colour and luma on the field
            cmd->SetPipelineState(pso_field_colour_.Get());
            cmd->Dispatch(gx, gy, 1);
            UavBarrier(cmd);
        }
        if (chaining) {   // one frame of motion per block, summed into the displacements (CSChain)
            cmd->SetPipelineState(pso_step_motion_.Get());
            cmd->Dispatch(gx, gy, 1);
            UavBarrier(cmd);
            cmd->SetPipelineState(pso_chain_.Get());
            cmd->Dispatch((gx + 7) / 8, (gy + 7) / 8, 1);
            UavBarrier(cmd);
        }
        if (ref_valid_) {   // the reference moved to this frame, into the head the post reads
            if (options_.warp_pyramid) {   // the coarse search over +-96 field pixels (CSWarpPyramid)
                const UINT qx = (extent_.field_w + 3) / 4, qy = (extent_.field_h + 3) / 4;
                cmd->SetPipelineState(pso_quarter_cur_.Get());
                cmd->Dispatch((qx + 7) / 8, (qy + 7) / 8, 1);
                UavBarrier(cmd);
                cmd->SetPipelineState(pso_warp_pyramid_.Get());
                cmd->Dispatch((qx + 7) / 8, (qy + 7) / 8, 1);
                UavBarrier(cmd);
            }
            cmd->SetPipelineState(pso_warp_motion_.Get());
            cmd->Dispatch(gx, gy, 1);
            UavBarrier(cmd);
            cmd->SetPipelineState(pso_head_warp_.Get());
            cmd->Dispatch(gx, gy, 1);
            UavBarrier(cmd);
        }
        if (chaining) {   // this frame's luma for the next frame's step
            cmd->SetPipelineState(pso_keep_luma_.Get());
            cmd->Dispatch(gx, gy, 1);
            UavBarrier(cmd);
        }
    }
    if (chaining) {
        chain_prev_ = true;
        chain_pending_ = pending_after;
        chain_active_ = active_after;
        chain_parity_ = !chain_parity_;
    } else {
        chain_prev_ = chain_pending_ = chain_active_ = false;
    }
    cmd->SetPipelineState(pso_post_.Get());
    cmd->Dispatch((extent_.output_w + 7) / 8, (extent_.output_h + 7) / 8, 1);
    UavBarrier(cmd);
    cmd->EndQuery(timestamp_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts + 3);
    cmd->ResolveQueryData(timestamp_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ts, kTimestampsPerSlot,
                          timestamp_readback_.Get(), ts * sizeof(uint64_t));
    history_valid_ = options_.temporal;
    if (run_back_) head_valid_ = true;
    if (part_ != 0) phase_ = (phase_ + 1) % options_.interval;
}

bool NeuralGraph::ReadTimings(uint32_t timing_slot, NeuralTimings& out) const {
    if (!timestamp_readback_ || timestamp_freq_hz_ == 0) return false;
    const uint32_t ts = (timing_slot % kTimingSlots) * kTimestampsPerSlot;
    D3D12_RANGE range{ts * sizeof(uint64_t), (ts + kTimestampsPerSlot) * sizeof(uint64_t)};
    void* mapped = nullptr;
    if (FAILED(timestamp_readback_->Map(0, &range, &mapped)) || !mapped) return false;
    const auto* t = static_cast<const uint64_t*>(mapped) + ts;
    const uint64_t v[4] = {t[0], t[1], t[2], t[3]};
    D3D12_RANGE none{0, 0};
    timestamp_readback_->Unmap(0, &none);
    if (!(v[1] >= v[0] && v[2] >= v[1] && v[3] >= v[2])) return false;
    const double scale = 1000.0 / static_cast<double>(timestamp_freq_hz_);
    out.pre_ms = (v[1] - v[0]) * scale;
    out.network_ms = (v[2] - v[1]) * scale;
    out.post_ms = (v[3] - v[2]) * scale;
    out.total_ms = (v[3] - v[0]) * scale;
    return true;
}

} // namespace dlss_nr::neural

namespace dlss_nr::neural {

namespace {

bool ReadRaw(const std::string& path, std::vector<char>& out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    out.resize(static_cast<size_t>(in.tellg()));
    in.seekg(0);
    in.read(out.data(), static_cast<std::streamsize>(out.size()));
    return static_cast<bool>(in);
}

// E4M3 rounding as the kernels do it (src/compute/neural_e4m3.hlsli), for tensors the reference
// holds published while the runner keeps them raw (the transitions' skips)
std::vector<float> Published(std::vector<float> v) {
    for (float& x : v) {
        const float a = std::min(std::abs(x), 448.0f);
        uint32_t u;
        std::memcpy(&u, &a, 4);
        float r;
        if (a >= 0.015625f) {
            const uint32_t bits = (u + 0x7FFFFu + ((u >> 20) & 1u)) & 0xFFF00000u;
            std::memcpy(&r, &bits, 4);
        } else {
            r = std::nearbyint(a * 512.0f) * (1.0f / 512.0f);   // the default rounding mode: half to even
        }
        x = std::signbit(x) ? -r : r;
    }
    return v;
}

// max |a - b| relative to max |b|, and the correlation, over two float arrays
std::string Compare(const char* label, const std::vector<float>& got, const std::vector<float>& want, bool& ok,
                    double tolerance, double min_corr) {
    std::ostringstream out;
    if (got.size() != want.size()) {
        out << "  " << label << ": size " << got.size() << " != " << want.size() << "\n";
        ok = false;
        return out.str();
    }
    double max_diff = 0, max_ref = 0, sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double a = got[i], b = want[i];
        max_diff = std::max(max_diff, std::abs(a - b));
        max_ref = std::max(max_ref, std::abs(b));
        sa += a; sb += b; sab += a * b; saa += a * a; sbb += b * b;
    }
    const double n = static_cast<double>(got.size());
    const double cov = sab / n - (sa / n) * (sb / n);
    const double var = std::sqrt(std::max(0.0, saa / n - (sa / n) * (sa / n)) * std::max(0.0, sbb / n - (sb / n) * (sb / n)));
    const double corr = var > 0 ? cov / var : 1.0;
    const double rel = max_ref > 0 ? max_diff / max_ref : max_diff;
    const bool pass = rel <= tolerance && corr >= min_corr;
    ok = ok && pass;
    out << "  " << (pass ? "[PASS] " : "[FAIL] ") << label << ": max |d| " << max_diff << " (rel " << rel
        << "), corr " << corr << "\n";
    return out.str();
}

} // namespace

bool NeuralGraph::CheckHybridEnds(const std::string& dir, std::string& report) {
    if (!hybrid_active_) return Fail("CheckHybridEnds: the current extent does not run the hybrid graph");
    const uint64_t field = static_cast<uint64_t>(extent_.field_w) * extent_.field_h;
    std::vector<char> features;
    if (!ReadRaw(dir + "/features.f16", features) || features.size() != field * 32) {
        return Fail("CheckHybridEnds: " + dir + "/features.f16 missing or not this field's size");
    }
    auto upload = [&](dx12::GpuBuffer& dst, const void* data, uint64_t bytes) {
        dx12::GpuBuffer staging;
        return resources_.CreateUploadBuffer(bytes, staging) &&
               resources_.UploadBufferData(upload_queue_, dst, staging, data, bytes);
    };
    auto download = [&](dx12::GpuBuffer& src, uint64_t halves) {
        std::vector<uint16_t> raw(halves);
        dx12::GpuBuffer staging;
        std::vector<float> out;
        if (!resources_.CreateReadbackBuffer(halves * 2, staging) ||
            !resources_.ReadbackBufferData(upload_queue_, src, staging, raw.data(), halves * 2)) {
            return out;
        }
        out.reserve(halves);
        for (uint16_t h : raw) out.push_back(HalfToFloat(h));
        return out;
    };
    auto reference = [&](const std::string& name) {
        std::vector<char> raw;
        std::vector<float> out;
        if (ReadRaw(dir + "/" + name, raw)) {
            out.resize(raw.size() / 4);
            std::memcpy(out.data(), raw.data(), out.size() * 4);
        }
        return out;
    };
    auto upload_reference = [&](dx12::GpuBuffer& dst, const std::string& name) {
        std::vector<float> v = reference(name);
        std::vector<uint16_t> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = FloatToHalf(v[i]);
        return !h.empty() && h.size() * 2 <= dst.byte_size && upload(dst, h.data(), h.size() * 2);
    };
    auto run = [&](bool front) {
        if (!upload_queue_.ResetCommandList()) return false;
        ID3D12GraphicsCommandList* cmd = upload_queue_.GetCommandList();
        std::vector<dx12::GpuBuffer*> all = hybrid_.Buffers();
        all.push_back(&features_buf_);
        all.push_back(&head_buf_);
        for (dx12::GpuBuffer* b : all) {
            dx12::ResourceManager::TransitionBuffer(cmd, *b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (front) hybrid_.RecordFront(cmd);
        else hybrid_.RecordBack(cmd);
        upload_queue_.ExecuteAndWait();
        return true;
    };
    // halves in encoder level l's skip: level l + 1 of the field, 32 << l channels (levels 0-1 are 32)
    auto skip_halves = [&](uint32_t l) { return (field >> (2 * (l + 1))) * (l == 0 ? 32u : (32u << l)); };

    bool ok = true;
    std::ostringstream log;
    // With E4M3 publishes the kernels and PyTorch round the same values, but a value a hair from a
    // rounding boundary can land on either side (accumulation order, half math) and moves by an
    // E4M3 step (up to 12.5 %), which later blocks amplify: the network as a whole agrees with the
    // exact oracle as well as PyTorch does (--golden, effect corr), not value for value. So the
    // gate is the correlation; `max |d|` only catches gross errors (a wrong weight, a lost tile).
    const double tol = 0.5, corr_mid = 0.95, corr_head = 0.995;
    const uint64_t mid = static_cast<uint64_t>(hybrid_.MidWidth()) * hybrid_.MidHeight() * hybrid_.MidChannels();
    if (!upload(features_buf_, features.data(), features.size()) || !run(true)) return Fail("front run failed");
    log << Compare("front full_skip (block 0)", Published(download(hybrid_.FullSkip(), field * 32)),
                   reference("front_full_skip.f32"), ok, tol, corr_mid);
    for (uint32_t l = 0; l < hybrid_.SkipCount(); ++l) {
        const std::string label = "front skip" + std::to_string(l);
        log << Compare(label.c_str(), Published(download(*hybrid_.Skip(l), skip_halves(l))),
                       reference("front_skip" + std::to_string(l) + ".f32"), ok, tol, corr_mid);
    }
    log << Compare("front value (into the middle)", download(hybrid_.MidIn(), mid), reference("front_value.f32"), ok,
                   tol, corr_mid);

    if (native_active_) {
        auto run_core = [&]() {
            if (!upload_queue_.ResetCommandList()) return false;
            ID3D12GraphicsCommandList* cmd = upload_queue_.GetCommandList();
            for (dx12::GpuBuffer* b : hybrid_.Buffers()) {
                dx12::ResourceManager::TransitionBuffer(cmd, *b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            hybrid_.RecordCore(cmd);
            upload_queue_.ExecuteAndWait();
            return true;
        };
        if (!upload_reference(hybrid_.MidIn(), "front_value.f32") || !run_core()) return Fail("core run failed");
        log << Compare("core value (blocks 23-47)", download(hybrid_.MidOut(), mid), reference("middle_value.f32"), ok,
                       tol, corr_mid);
    }

    // the back from the reference middle output and the reference skips, so it is judged alone
    bool loaded = upload_reference(hybrid_.MidOut(), "middle_value.f32") &&
                  upload_reference(hybrid_.FullSkip(), "front_full_skip.f32");
    for (uint32_t l = 0; loaded && l < hybrid_.SkipCount(); ++l) {
        loaded = upload_reference(*hybrid_.Skip(l), "front_skip" + std::to_string(l) + ".f32");
    }
    if (!loaded || !run(false)) return Fail("back run failed (reference tensors missing?)");
    log << Compare("back head", download(head_buf_, field * 4), reference("head.f32"), ok, tol, corr_head);
    run(true);   // leaves every buffer in UNORDERED_ACCESS again
    report = log.str();
    return ok;
}

} // namespace dlss_nr::neural
