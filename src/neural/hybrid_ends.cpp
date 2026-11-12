#include "neural/hybrid_ends.h"
#include "neural/embedded_neural_dxil.h"
#include "neural/neural_field.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>

namespace dlss_nr::neural {

namespace {

using PFN_D3D12SerializeRootSignature = HRESULT(WINAPI*)(
    const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);

constexpr uint32_t kConstants = 24;   // BlockConstants / WideConstants (which reads the first 20)

// The vendor window origin (y, x) of a block: shifted windows, the phase per stage
// (tools/hnet/model.py window_origin).
void WindowOrigin(int block, int& oy, int& ox) {
    int phase = 0;
    if (block == 0) phase = 0;
    else if (block <= 4) phase = block - 1;
    else if (block <= 8) phase = block - 5;
    else if (block <= 14) phase = block - 9;
    else if (block <= 22) phase = block - 15;
    else if (block <= 30) phase = block - 23;
    else if (block >= 40 && block <= 55) phase = block - (block < 48 ? 40 : 48);
    else if (block >= 56 && block <= 61) phase = block - 54;
    else if (block >= 62 && block <= 69) phase = block - (block < 66 ? 62 : 66);
    else if (block == 70) phase = 1;
    static const int ys[4] = {0, -4, 0, -4};
    static const int xs[4] = {0, -4, -4, 0};
    oy = ys[phase % 4];
    ox = xs[phase % 4];
}

void Padded(int block, uint32_t w, uint32_t h, uint32_t& top, uint32_t& left, uint32_t& pw, uint32_t& ph) {
    int oy = 0, ox = 0;
    WindowOrigin(block, oy, ox);
    top = static_cast<uint32_t>(-oy);
    left = static_cast<uint32_t>(-ox);
    ph = h + top + ((8 - (h + top) % 8) % 8);
    pw = w + left + ((8 - (w + left) % 8) % 8);
}

uint32_t Groups64(uint64_t tokens) { return static_cast<uint32_t>((tokens + 63) / 64); }

} // namespace

std::string HybridEnds::BlobFileName(bool synthetic) {
    return std::string("hnet-ends_nq-exp") + (synthetic ? "_synthetic" : "") + ".bin";
}

std::string HybridEnds::MiddleFileName(HybridSplit split, uint32_t field_w, uint32_t field_h, bool synthetic) {
    std::ostringstream name;
    name << (split == HybridSplit::kDeep ? "hnet-mid23_" : "hnet-mid_") << field_w << "x" << field_h
         << "_nq-exp_fp16" << (synthetic ? "_synthetic" : "") << ".onnx";
    return name.str();
}

void HybridEnds::DeepExtent(uint32_t field_w, uint32_t field_h, uint32_t& w, uint32_t& h) {
    w = (field_w / 16 + 7) / 8 * 4;
    h = (field_h / 16 + 7) / 8 * 4;
}

bool HybridEnds::Initialize(ID3D12Device* device, IDMLDevice* dml, std::string* error) {
    device_ = device;
    use_dml_ = dml && gemm_.Initialize(device, dml);
    D3D12_ROOT_PARAMETER params[6] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = kConstants;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;      // t0 weights
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (uint32_t i = 0; i < 3; ++i) {                            // u0 a, u1 b, u2 out
        params[2 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[2 + i].Descriptor.ShaderRegister = i;
        params[2 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;      // t1 half weights
    params[5].Descriptor.ShaderRegister = 1;
    params[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 6;
    desc.pParameters = params;

    HMODULE d3d12 = GetModuleHandleA("d3d12.dll");
    if (!d3d12) d3d12 = LoadLibraryA("d3d12.dll");
    auto serialize = d3d12 ? reinterpret_cast<PFN_D3D12SerializeRootSignature>(
                                 GetProcAddress(d3d12, "D3D12SerializeRootSignature"))
                           : nullptr;
    ComPtr<ID3DBlob> blob, err;
    if (!serialize || FAILED(serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
        FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                            IID_PPV_ARGS(&root_sig_)))) {
        if (error) *error = "hybrid root signature";
        return false;
    }
    auto make = [&](const uint8_t* code, size_t size, ComPtr<ID3D12PipelineState>& out) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root_sig_.Get();
        pso.CS.pShaderBytecode = code;
        pso.CS.BytecodeLength = size;
        return SUCCEEDED(device_->CreateComputePipelineState(&pso, IID_PPV_ARGS(&out)));
    };
#define HYBRID_PSO(set, field, name, suffix) \
    make(embedded::k##name##suffix##Dxil, sizeof(embedded::k##name##suffix##Dxil), set.field)
#define HYBRID_SET(set, suffix)                                                                       \
    (HYBRID_PSO(set, block32, WindowBlock32, suffix) && HYBRID_PSO(set, pool32, Pool2, suffix) &&      \
     HYBRID_PSO(set, ffn, BranchedFfn, suffix) && HYBRID_PSO(set, project, ProjectResidual, suffix) && \
     HYBRID_PSO(set, attention, WindowAttention, suffix) &&                                           \
     HYBRID_PSO(set, pool_project, PoolProjectN, suffix) && HYBRID_PSO(set, up_merge, UpMerge, suffix) && \
     HYBRID_PSO(set, gate, Gate, suffix) && HYBRID_PSO(set, residual, Residual, suffix) &&             \
     HYBRID_PSO(set, attention_qkv, WindowAttentionQkv, suffix) &&                                   \
     HYBRID_PSO(set, global_attention, GlobalAttention, suffix) && HYBRID_PSO(set, quantize, Quantize, suffix))
    // [0] the E4M3 publishes (precision accurate), [1] without them (fast): Allocate picks one
    if (!HYBRID_SET(sets_[0], ) || !HYBRID_SET(sets_[1], Fast)) {
        if (error) *error = "hybrid pipelines (WaveSize(16) / SM 6.6 rejected?)";
        return false;
    }
#undef HYBRID_SET
#undef HYBRID_PSO
    return true;
}

namespace {

// A read-only view of a whole file (CreateFileMapping); closed with the object.
struct MappedFile {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    const uint8_t* data = nullptr;
    uint64_t size = 0;

    bool Open(const std::string& path) {
        file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        LARGE_INTEGER length{};
        if (file == INVALID_HANDLE_VALUE || !GetFileSizeEx(file, &length) || length.QuadPart <= 0) return false;
        size = static_cast<uint64_t>(length.QuadPart);
        mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) return false;
        data = static_cast<const uint8_t*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
        return data != nullptr;
    }
    ~MappedFile() {
        if (data) UnmapViewOfFile(data);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }
};

} // namespace

bool HybridEnds::LoadWeights(const std::string& path, dx12::ResourceManager& rm, dx12::CommandQueue& upload,
                             std::string* error) {
    if (weights_.resource && path == weights_path_) return true;
    // The blob is mapped, not read: tensors are copied once, from the file cache straight into
    // the upload heaps (a read into a private buffer first cost ~180 ms for 300 MB).
    MappedFile file;
    if (!file.Open(path)) {
        if (error) *error = "cannot open weight blob: " + path;
        return false;
    }
    uint64_t at = 0;
    auto take = [&](void* dst, uint64_t n) {
        if (at + n > file.size) return false;
        std::memcpy(dst, file.data + at, static_cast<size_t>(n));
        at += n;
        return true;
    };
    char magic[4] = {};
    uint32_t version = 0, count = 0;
    if (!take(magic, 4) || !take(&version, 4) || !take(&count, 4) || std::memcmp(magic, "HNWB", 4) != 0 ||
        (version != 1 && version != 2) || count == 0 || count > 65536) {
        if (error) *error = "not an HNWB v1/v2 weight blob: " + path;
        return false;
    }
    // v1: {name; u32 offset in floats; u32 floats}, all float. v2: {name; u32 dtype (1 = half);
    // u32 elements; u64 offset in bytes}, the GEMM-only matrices in half.
    struct Tensor {
        uint64_t offset = 0;   // bytes into the data
        uint32_t size = 0;     // elements
        bool half = false;
    };
    std::map<std::string, Tensor> tensors;
    uint64_t bytes_in_file = 0;
    for (uint32_t i = 0; i < count; ++i) {
        char name[64] = {};
        Tensor t;
        bool ok = take(name, 64);
        if (version == 1) {
            uint32_t offset = 0;
            ok = ok && take(&offset, 4) && take(&t.size, 4);
            t.offset = static_cast<uint64_t>(offset) * 4;
        } else {
            uint32_t dtype = 0;
            ok = ok && take(&dtype, 4) && take(&t.size, 4) && take(&t.offset, 8);
            t.half = dtype == 1;
        }
        if (!ok) {
            if (error) *error = "truncated weight blob header: " + path;
            return false;
        }
        name[63] = 0;
        tensors[name] = t;
        bytes_in_file = std::max<uint64_t>(bytes_in_file, t.offset + static_cast<uint64_t>(t.size) * (t.half ? 2 : 4));
    }
    const uint8_t* data = file.data + at;
    if (at + bytes_in_file > file.size) {
        if (error) *error = "truncated weight blob: " + path;
        return false;
    }
    auto float_at = [data](const Tensor& t, uint64_t i) {
        if (!t.half) {
            float v;
            std::memcpy(&v, data + t.offset + i * 4, 4);
            return v;
        }
        uint16_t h;
        std::memcpy(&h, data + t.offset + i * 2, 2);
        return HalfToFloat(h);
    };
    auto half_at = [&](const Tensor& t, uint64_t i) {
        if (!t.half) return FloatToHalf(float_at(t, i));
        uint16_t h;
        std::memcpy(&h, data + t.offset + i * 2, 2);
        return h;
    };

    auto ends_with = [](const std::string& s, const char* tail) {
        const size_t n = std::strlen(tail);
        return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
    };
    // the 32-channel blocks' kernels read every one of their weights in float
    auto end_block = [](const std::string& name) {
        const int block = std::atoi(name.c_str() + 5);   // "block<N>."
        return block <= 4 || block >= 66;
    };

    // The DirectML GEMMs read their matrices in half, (N, K) per batch: the blob's (out, in)
    // matrices as they are, the (groups, in, out) ones transposed per group. The layout comes
    // first, then the halves go straight into the mapped upload heap (no ~300 MB staging copy).
    std::map<std::string, bool> gemm_only;
    if (use_dml_) {
        struct Placed {
            const Tensor* t;
            uint64_t at;               // in halves
            uint64_t rows, cols;       // a grouped matrix to transpose per group, else 0
        };
        std::vector<Placed> placed;
        offsets16_.clear();
        uint64_t halves = 0;
        for (const auto& [name, t] : tensors) {
            uint64_t rows = 0, cols = 0;
            if (ends_with(name, "fused_project")) { rows = 128; cols = 32; }
            if (ends_with(name, "group_expand_weight")) { rows = 64; cols = 256; }
            if (ends_with(name, "group_project_weight")) { rows = 256; cols = 64; }
            const bool matrix = ends_with(name, "fused_expand") || ends_with(name, "projection_weight") ||
                                ends_with(name, "weight1") || ends_with(name, "weight2") ||
                                ends_with(name, "qkv_heads") || ends_with(name, "qkv_weight") ||
                                ends_with(name, "weight3") || ends_with(name, "layer0.weight") ||
                                ends_with(name, "layer1.weight");
            if (!(rows || matrix)) continue;
            gemm_only[name] = !end_block(name);
            halves = (halves + 7) / 8 * 8;                 // 16-byte aligned for DirectML
            offsets16_[name] = halves * 2;
            placed.push_back({&t, halves, rows, cols});
            halves += t.size;
        }
        weights16_ = {};
        dx12::GpuBuffer staging16;
        const uint64_t bytes16 = std::max<uint64_t>(16, halves * 2);
        void* mapped = nullptr;
        const D3D12_RANGE nothing_read{0, 0};
        if (!rm.CreateDefaultBuffer(bytes16, weights16_, true) || !rm.CreateUploadBuffer(bytes16, staging16) ||
            FAILED(staging16.resource->Map(0, &nothing_read, &mapped)) || !mapped) {
            if (error) *error = "upload half GEMM weights";
            return false;
        }
        uint16_t* out = static_cast<uint16_t*>(mapped);
        for (const Placed& p : placed) {
            const Tensor& t = *p.t;
            uint16_t* dst = out + p.at;
            if (p.rows) {
                const uint64_t groups = t.size / (p.rows * p.cols);
                for (uint64_t g = 0; g < groups; ++g) {
                    for (uint64_t col = 0; col < p.cols; ++col) {
                        for (uint64_t row = 0; row < p.rows; ++row) {
                            *dst++ = half_at(t, g * p.rows * p.cols + row * p.cols + col);
                        }
                    }
                }
            } else if (t.half) {
                std::memcpy(dst, data + t.offset, static_cast<size_t>(t.size) * 2);
            } else {
                for (uint64_t i = 0; i < t.size; ++i) dst[i] = half_at(t, i);
            }
        }
        staging16.resource->Unmap(0, nullptr);
        if (!upload.ResetCommandList()) return false;
        ID3D12GraphicsCommandList* cmd = upload.GetCommandList();
        dx12::ResourceManager::TransitionBuffer(cmd, weights16_, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyBufferRegion(weights16_.resource.Get(), 0, staging16.resource.Get(), 0, bytes16);
        dx12::ResourceManager::TransitionBuffer(cmd, weights16_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        upload.ExecuteAndWait();
    }

    // The float weights the HLSL kernels read: everything, less the matrices only a GEMM reads
    // (with the GEMMs that is most of the 145.8 M parameters: ~550 MB of VRAM saved).
    offsets_.clear();
    // (Without the GEMMs a v2 blob's half matrices come back as float, rounded: only a fallback.)
    std::vector<float> kept;
    for (const auto& [name, t] : tensors) {
        if (gemm_only.count(name) && gemm_only[name]) continue;
        kept.resize((kept.size() + 3) / 4 * 4);           // 16-byte aligned for Load4
        offsets_[name] = static_cast<uint32_t>(kept.size());
        const size_t start = kept.size();
        kept.resize(start + t.size);
        if (t.half) {
            for (uint32_t i = 0; i < t.size; ++i) kept[start + i] = float_at(t, i);
        } else {
            std::memcpy(kept.data() + start, data + t.offset, static_cast<size_t>(t.size) * 4);
        }
    }
    kept.resize(std::max<size_t>(4, (kept.size() + 3) / 4 * 4));
    const uint64_t bytes = kept.size() * sizeof(float);
    dx12::GpuBuffer staging;
    weights_ = {};
    if (!rm.CreateDefaultBuffer(bytes, weights_, false) || !rm.CreateUploadBuffer(bytes, staging) ||
        !rm.UploadBufferData(upload, weights_, staging, kept.data(), bytes)) {
        if (error) *error = "upload hybrid weights";
        return false;
    }
    if (!upload.ResetCommandList()) return false;
    dx12::ResourceManager::TransitionBuffer(upload.GetCommandList(), weights_,
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    upload.ExecuteAndWait();
    weights_path_ = path;
    return true;
}

uint64_t HybridEnds::Offset16(const std::string& name) const {
    auto it = offsets16_.find(name);
    return it == offsets16_.end() ? 0u : it->second;
}

uint32_t HybridEnds::Offset(const std::string& name) const {
    auto it = offsets_.find(name);
    return it == offsets_.end() ? 0u : it->second;
}

dx12::GpuBuffer* HybridEnds::NewBuffer(uint64_t bytes) {
    buffers_.push_back(std::make_unique<dx12::GpuBuffer>());
    buffers_.back()->byte_size = bytes;   // allocated in Allocate once the list is complete
    return buffers_.back().get();
}

std::vector<dx12::GpuBuffer*> HybridEnds::Buffers() {
    std::vector<dx12::GpuBuffer*> out;
    for (auto& b : buffers_) out.push_back(b.get());
    return out;
}

void HybridEnds::Block32(std::vector<Op>& ops, int block, uint32_t w, uint32_t h, uint32_t in_w, uint32_t in_h,
                         uint32_t input_mode, uint32_t output_mode, dx12::GpuBuffer* a, dx12::GpuBuffer* b,
                         dx12::GpuBuffer* out) {
    uint32_t top, left, pw, ph;
    Padded(block, w, h, top, left, pw, ph);
    const std::string p = "block" + std::to_string(block) + ".layer0.";
    Op op;
    op.pso = pso_block32_.Get();
    const uint32_t c[kConstants] = {
        w, h, in_w, in_h, top, left, input_mode, output_mode,
        Offset(p + "weight1"), Offset(p + "weight2"), Offset(p + "ffn_cos_skip"), Offset(p + "qkv_heads"),
        Offset(p + "attn_scale"), Offset(p + "attn_bias"), Offset(p + "projection_weight"),
        Offset(p + "attn_cos_skip"),
        input_mode == 1 ? Offset("block0.layer0.input_adapter_weight")
                        : input_mode == 2 ? Offset(p + "weight0")
                                          : input_mode == 3 ? Offset("block70.layer0.inp_merge_sin") : 0u,
        input_mode == 2 ? Offset(p + "sin") : input_mode == 3 ? Offset("block70.layer0.inp_merge_cos") : 0u,
        output_mode == 1 ? Offset("block70.layer0.out_gain") : 0u,
        output_mode == 1 ? Offset("block70.layer0.out_conv_weight") : 0u,
        // the FFN in half (dot2add, half2 accumulation) when the half weights exist
        static_cast<uint32_t>(Offset16(p + "weight1")), static_cast<uint32_t>(Offset16(p + "weight2")),
        use_dml_ && half_ffn_ ? 1u : 0u, Publishes(block) ? 1u : 0u};
    std::memcpy(op.constants, c, sizeof(c));
    op.a = a;
    op.b = b;
    op.out = out;
    op.groups[0] = pw / 8;
    op.groups[1] = ph / 8;
    op.label = "b" + std::to_string(block) + " fused32";
    ops.push_back(op);
}

void HybridEnds::Pool32(std::vector<Op>& ops, uint32_t in_w, uint32_t in_h, dx12::GpuBuffer* a,
                        dx12::GpuBuffer* out) {
    Op op;
    op.pso = pso_pool32_.Get();
    op.constants[0] = in_w / 2;
    op.constants[1] = in_h / 2;
    op.constants[2] = in_w;
    op.constants[3] = in_h;
    op.a = a;
    op.out = out;
    op.groups[0] = (in_w / 2 + 7) / 8;
    op.groups[1] = (in_h / 2 + 7) / 8;
    op.label = "pool32";
    ops.push_back(op);
}

HybridEnds::Op HybridEnds::Elementwise(ID3D12PipelineState* pso, uint64_t halves, uint32_t channels, uint32_t o_cos,
                                       dx12::GpuBuffer* a, dx12::GpuBuffer* b, dx12::GpuBuffer* out,
                                       const std::string& label, uint32_t publish) {
    Op op;
    op.pso = pso;
    const uint32_t vectors = static_cast<uint32_t>(halves / 8);
    op.constants[0] = vectors;
    op.constants[6] = channels;
    op.constants[10] = o_cos;
    op.constants[13] = publish;
    op.a = a ? a : out;
    op.b = b;
    op.out = out;
    const uint32_t groups = (vectors + 255) / 256;
    op.groups[0] = std::min<uint32_t>(groups, 65535);
    op.groups[1] = (groups + 65534) / 65535;
    op.label = label;
    return op;
}

HybridEnds::Op HybridEnds::GemmOp(const DmlGemmSet::Gemm& gemm, const std::string& label) {
    Op op;
    op.gemm = gemm_.Add(gemm);
    op.label = label;
    return op;
}

void HybridEnds::WideBlockDml(std::vector<Op>& ops, int block, uint32_t w, uint32_t h, uint32_t channels,
                              const Level& lv, dx12::GpuBuffer* out) {
    const std::string p = "block" + std::to_string(block) + ".layer0.";
    const std::string tag = "b" + std::to_string(block) + " C" + std::to_string(channels) + " ";
    const uint32_t t = w * h, c = channels, g = channels / 32;
    const uint64_t tokens = static_cast<uint64_t>(t);
    DmlGemmSet::Gemm gemm;
    // h1 = x @ expand (G*128 hidden), gate, hs = per-group h1 @ branch (batched over the groups)
    gemm = {};
    gemm.m = t; gemm.k = c; gemm.n = g * 128;
    gemm.a = lv.x; gemm.b = &weights16_; gemm.b_offset = Offset16(p + "fused_expand"); gemm.out = lv.h1;
    ops.push_back(GemmOp(gemm, tag + "gemm expand"));
    ops.push_back(Elementwise(pso_gate_.Get(), tokens * g * 128, 0, 0, nullptr, nullptr, lv.h1, tag + "gate", 1));
    gemm = {};
    gemm.batch = g; gemm.m = t; gemm.k = 128; gemm.n = 32;
    gemm.a_row = g * 128; gemm.a_batch = 128; gemm.out_row = c; gemm.out_batch = 32;
    gemm.a = lv.h1; gemm.b = &weights16_; gemm.b_offset = Offset16(p + "fused_project"); gemm.out = lv.sc;
    ops.push_back(GemmOp(gemm, tag + "gemm branch"));
    if (precise_) ops.push_back(Elementwise(pso_quantize_.Get(), tokens * c, 0, 0, nullptr, nullptr, lv.sc, tag + "publish"));
    // ff = hs @ out_proj + x * ffn_cos
    gemm = {};
    gemm.m = t; gemm.k = c; gemm.n = c;
    gemm.a = lv.sc; gemm.b = &weights16_; gemm.b_offset = Offset16(p + "ffn_output_projection_weight"); gemm.out = lv.t;
    ops.push_back(GemmOp(gemm, tag + "gemm ffn-out"));
    ops.push_back(Elementwise(pso_residual_.Get(), tokens * c, c, Offset(p + "ffn_cos_skip"), lv.t, lv.x, lv.ff,
                              tag + "residual", 1));
    // qkv = ff @ qkv (per head [q|k|v]), the window attention -> o
    gemm = {};
    gemm.m = t; gemm.k = c; gemm.n = 3 * c;
    gemm.a = lv.ff; gemm.b = &weights16_; gemm.b_offset = Offset16(p + "qkv_heads"); gemm.out = lv.qkv;
    ops.push_back(GemmOp(gemm, tag + "gemm qkv"));
    uint32_t top, left, pw, ph;
    Padded(block, w, h, top, left, pw, ph);
    Op attention;
    attention.pso = pso_attention_qkv_.Get();
    const uint32_t ac[kConstants] = {w, h, w, h, top, left, c, c, 0, 0, 0, Offset(p + "attn_scale"),
                                     Offset(p + "attn_bias")};
    std::memcpy(attention.constants, ac, sizeof(ac));
    attention.a = lv.qkv;
    attention.out = lv.sc;
    attention.groups[0] = pw / 8;
    attention.groups[1] = ph / 8;
    attention.groups[2] = g;
    attention.label = tag + "attention";
    ops.push_back(attention);
    // out = o @ proj + ff * attn_cos
    gemm = {};
    gemm.m = t; gemm.k = c; gemm.n = c;
    gemm.a = lv.sc; gemm.b = &weights16_; gemm.b_offset = Offset16(p + "projection_weight"); gemm.out = lv.t;
    ops.push_back(GemmOp(gemm, tag + "gemm proj"));
    ops.push_back(Elementwise(pso_residual_.Get(), tokens * c, c, Offset(p + "attn_cos_skip"), lv.t, lv.ff, out,
                              tag + "residual", Publishes(block) ? 1u : 0u));
}

void HybridEnds::WideBlock(std::vector<Op>& ops, int block, uint32_t w, uint32_t h, uint32_t channels,
                           const Level& lv, dx12::GpuBuffer* out) {
    if (use_dml_) {
        WideBlockDml(ops, block, w, h, channels, lv, out);
        return;
    }
    dx12::GpuBuffer* x = lv.x;
    dx12::GpuBuffer* ff = lv.ff;
    dx12::GpuBuffer* scratch = lv.sc;
    const std::string p = "block" + std::to_string(block) + ".layer0.";
    const uint32_t groups = channels / 32;
    const uint64_t tokens = static_cast<uint64_t>(w) * h;
    auto wide = [&](ID3D12PipelineState* pso, dx12::GpuBuffer* a, dx12::GpuBuffer* b, dx12::GpuBuffer* o,
                    uint32_t oa, uint32_t ob, uint32_t ocos) {
        Op op;
        op.pso = pso;
        const uint32_t c[kConstants] = {w, h, w, h, 0, 0, channels, channels, oa, ob, ocos};
        std::memcpy(op.constants, c, sizeof(c));
        op.a = a;
        op.b = b;
        op.out = o;
        op.groups[0] = Groups64(tokens);
        op.groups[1] = groups;
        op.label = "b" + std::to_string(block) + " C" + std::to_string(channels);
        return op;
    };
    // the feed-forward: hs = branched(x), ff = hs @ out_proj + x * ffn_cos
    ops.push_back(wide(pso_ffn_.Get(), x, nullptr, scratch, Offset(p + "fused_expand"), Offset(p + "fused_project"), 0));
    ops.back().label += " ffn";
    ops.push_back(wide(pso_project_.Get(), scratch, x, ff, Offset(p + "ffn_output_projection_weight"), 0,
                       Offset(p + "ffn_cos_skip")));
    ops.back().constants[13] = 1;   // ff published
    ops.back().label += " ffn-proj";
    // the attention, one head per group z, then out = o @ proj + ff * attn_cos
    uint32_t top, left, pw, ph;
    Padded(block, w, h, top, left, pw, ph);
    Op attention = wide(pso_attention_.Get(), ff, nullptr, scratch, Offset(p + "qkv_heads"), 0, 0);
    attention.constants[4] = top;
    attention.constants[5] = left;
    attention.constants[11] = Offset(p + "attn_scale");
    attention.constants[12] = Offset(p + "attn_bias");
    attention.groups[0] = pw / 8;
    attention.groups[1] = ph / 8;
    attention.groups[2] = groups;   // heads = channels / 32
    attention.label += " attention";
    ops.push_back(attention);
    ops.push_back(wide(pso_project_.Get(), scratch, ff, out, Offset(p + "projection_weight"), 0,
                       Offset(p + "attn_cos_skip")));
    ops.back().constants[13] = Publishes(block) ? 1u : 0u;
    ops.back().label += " attn-proj";
}

void HybridEnds::PoolProject(std::vector<Op>& ops, const std::string& weight, uint32_t in_w, uint32_t in_h,
                             uint32_t out_w, uint32_t out_h, uint32_t channels, uint32_t out_channels,
                             dx12::GpuBuffer* a, dx12::GpuBuffer* out) {
    Op op;
    op.pso = pso_pool_project_.Get();
    const uint32_t c[kConstants] = {out_w, out_h, in_w, in_h, 0, 0, channels, out_channels, Offset(weight)};
    std::memcpy(op.constants, c, sizeof(c));
    op.a = a;
    op.out = out;
    op.groups[0] = Groups64(static_cast<uint64_t>(out_w) * out_h);
    op.groups[1] = out_channels / 32;
    op.label = "pool-project " + std::to_string(channels) + "->" + std::to_string(out_channels);
    ops.push_back(op);
}

void HybridEnds::UpMerge(std::vector<Op>& ops, const std::string& weight, const std::string& sine, uint32_t w,
                         uint32_t h, uint32_t coarse_w, uint32_t coarse_h, uint32_t channels, uint32_t out_channels,
                         dx12::GpuBuffer* coarse, dx12::GpuBuffer* skip, dx12::GpuBuffer* out,
                         const std::string& label) {
    Op op;
    op.pso = pso_up_merge_.Get();
    const uint32_t c[kConstants] = {w, h, coarse_w, coarse_h, 0, 0, channels, out_channels,
                                    Offset(weight), 0, Offset(sine)};
    std::memcpy(op.constants, c, sizeof(c));
    op.a = coarse;
    op.b = skip;
    op.out = out;
    op.groups[0] = Groups64(static_cast<uint64_t>(w) * h);
    op.groups[1] = out_channels / 32;
    op.label = label;
    ops.push_back(op);
}

void HybridEnds::Gemm(std::vector<Op>& ops, const std::string& label, uint32_t m, uint32_t k, uint32_t n,
                      dx12::GpuBuffer* a, const std::string& weight, dx12::GpuBuffer* out, uint32_t batch,
                      uint32_t a_row, uint32_t a_batch, uint32_t out_row, uint32_t out_batch) {
    DmlGemmSet::Gemm g;
    g.batch = batch;
    g.m = m;
    g.k = k;
    g.n = n;
    g.a_row = a_row;
    g.a_batch = a_batch;
    g.out_row = out_row;
    g.out_batch = out_batch;
    g.b_batch = batch > 1 ? n * k : 0;
    g.a = a;
    g.b = &weights16_;
    g.b_offset = Offset16(weight);
    g.out = out;
    ops.push_back(GemmOp(g, label));
}

// A 512-channel split block (23-30, 40-47): x @ first, per-64-group MLP 64 -> 256 -> 64 (gate
// between), @ weight3, + x * ffn_cos; then the 16-head window attention and its residual.
void HybridEnds::SplitBlock(std::vector<Op>& ops, int block, uint32_t w, uint32_t h, const Level& lv,
                            dx12::GpuBuffer* out) {
    const std::string p = "block" + std::to_string(block) + ".";
    const std::string tag = "b" + std::to_string(block) + " C512 ";
    const uint32_t t = w * h;
    const uint64_t tokens = t;
    Gemm(ops, tag + "gemm first", t, 512, 512, lv.x, p + "layer0.first_projection_weight", lv.t);
    if (precise_) ops.push_back(Elementwise(pso_quantize_.Get(), tokens * 512, 0, 0, nullptr, nullptr, lv.t, tag + "publish"));
    Gemm(ops, tag + "gemm group-expand", t, 64, 256, lv.t, p + "layer0.group_expand_weight", lv.h1, 8, 512, 64,
         2048, 256);
    ops.push_back(Elementwise(pso_gate_.Get(), tokens * 2048, 0, 0, nullptr, nullptr, lv.h1, tag + "gate"));
    Gemm(ops, tag + "gemm group-project", t, 256, 64, lv.h1, p + "layer0.group_project_weight", lv.sc, 8, 2048, 256,
         512, 64);
    if (precise_) ops.push_back(Elementwise(pso_quantize_.Get(), tokens * 512, 0, 0, nullptr, nullptr, lv.sc, tag + "publish"));
    Gemm(ops, tag + "gemm weight3", t, 512, 512, lv.sc, p + "layer1.weight3", lv.t);
    ops.push_back(Elementwise(pso_residual_.Get(), tokens * 512, 512, Offset(p + "layer1.ffn_cos_skip"), lv.t, lv.x,
                              lv.ff, tag + "residual", 1));
    Gemm(ops, tag + "gemm qkv", t, 512, 1536, lv.ff, p + "layer2.qkv_heads", lv.qkv);
    uint32_t top, left, pw, ph;
    Padded(block, w, h, top, left, pw, ph);
    Op attention;
    attention.pso = pso_attention_qkv_.Get();
    const uint32_t ac[kConstants] = {w, h, w, h, top, left, 512, 512, 0, 0, 0, Offset(p + "layer2.attn_scale"),
                                     Offset(p + "layer2.attn_bias")};
    std::memcpy(attention.constants, ac, sizeof(ac));
    attention.a = lv.qkv;
    attention.out = lv.sc;
    attention.groups[0] = pw / 8;
    attention.groups[1] = ph / 8;
    attention.groups[2] = 16;
    attention.label = tag + "attention";
    ops.push_back(attention);
    Gemm(ops, tag + "gemm proj", t, 512, 512, lv.sc, p + "layer3.projection_weight", lv.t);
    ops.push_back(Elementwise(pso_residual_.Get(), tokens * 512, 512, Offset(p + "layer3.attn_cos_skip"), lv.t, lv.ff,
                              out, tag + "residual", Publishes(block) ? 1u : 0u));
}

// A ViT block (31-38) on every token of the 1/64 field, in place: FFN 1024 -> 4096 -> 1024 with
// the gate, + x * ffn_cos; then the 32-head global attention and its residual.
void HybridEnds::VitBlock(std::vector<Op>& ops, int block, uint32_t tokens, const Level& lv) {
    const std::string p = "block" + std::to_string(block) + ".";
    const std::string tag = "b" + std::to_string(block) + " ViT ";
    Gemm(ops, tag + "gemm expand", tokens, 1024, 4096, lv.x, p + "layer0.weight", lv.h1);
    ops.push_back(Elementwise(pso_gate_.Get(), static_cast<uint64_t>(tokens) * 4096, 0, 0, nullptr, nullptr, lv.h1,
                              tag + "gate", 1));
    Gemm(ops, tag + "gemm contract", tokens, 4096, 1024, lv.h1, p + "layer1.weight", lv.t);
    ops.push_back(Elementwise(pso_residual_.Get(), static_cast<uint64_t>(tokens) * 1024, 1024,
                              Offset(p + "layer1.ffn_cos_skip"), lv.t, lv.x, lv.ff, tag + "residual", 1));
    Gemm(ops, tag + "gemm qkv", tokens, 1024, 3072, lv.ff, p + "layer2.qkv_weight", lv.qkv);
    Op attention;
    attention.pso = pso_global_attention_.Get();
    attention.constants[0] = tokens;
    attention.constants[11] = Offset(p + "layer2.attn_scale");
    attention.a = lv.qkv;
    attention.out = lv.sc;
    attention.groups[0] = Groups64(tokens);
    attention.groups[2] = 32;
    attention.label = tag + "attention";
    ops.push_back(attention);
    Gemm(ops, tag + "gemm proj", tokens, 1024, 1024, lv.sc, p + "layer4.projection_weight", lv.t);
    ops.push_back(Elementwise(pso_residual_.Get(), static_cast<uint64_t>(tokens) * 1024, 1024,
                              Offset(p + "layer4.attn_cos_skip"), lv.t, lv.ff, lv.x, tag + "residual", 1));
}

bool HybridEnds::Allocate(HybridSplit split, uint32_t field_w, uint32_t field_h, dx12::GpuBuffer& features,
                          dx12::GpuBuffer& head, dx12::ResourceManager& rm, dx12::CommandQueue& upload,
                          std::string* error) {
    if (split == HybridSplit::kNative && !use_dml_) {
        if (error) *error = "the native split needs the DirectML GEMMs";
        return false;
    }
    split_ = split;
    const PipelineSet& set = sets_[precise_ ? 0 : 1];
    pso_block32_ = set.block32;
    pso_pool32_ = set.pool32;
    pso_ffn_ = set.ffn;
    pso_project_ = set.project;
    pso_attention_ = set.attention;
    pso_pool_project_ = set.pool_project;
    pso_up_merge_ = set.up_merge;
    pso_gate_ = set.gate;
    pso_residual_ = set.residual;
    pso_attention_qkv_ = set.attention_qkv;
    pso_global_attention_ = set.global_attention;
    pso_quantize_ = set.quantize;
    buffers_.clear();
    gemm_.Clear();
    core_.clear();
    front_.clear();
    back_.clear();
    std::fill(std::begin(skips_), std::end(skips_), nullptr);

    // level l extent: the field halved l times; level 5 (512ch) is block 22's padded pool
    uint32_t ew[6], eh[6];
    for (int l = 0; l < 5; ++l) {
        ew[l] = field_w >> l;
        eh[l] = field_h >> l;
    }
    DeepExtent(field_w, field_h, ew[5], eh[5]);
    const uint32_t channels[6] = {32, 32, 64, 128, 256, 512};
    auto bytes = [&](int l, uint32_t c) { return static_cast<uint64_t>(ew[l]) * eh[l] * c * 2; };

    full_skip_ = NewBuffer(bytes(0, 32));
    dx12::GpuBuffer* t32a = NewBuffer(bytes(1, 32));
    dx12::GpuBuffer* t32b = NewBuffer(bytes(1, 32));
    skips_[0] = NewBuffer(bytes(1, 32));

    // front, level 0-1: blocks 0-4, all 32 channels
    Block32(front_, 0, ew[0], eh[0], ew[0], eh[0], 1, 0, &features, nullptr, full_skip_);
    Pool32(front_, ew[0], eh[0], full_skip_, t32a);
    Block32(front_, 1, ew[1], eh[1], ew[1], eh[1], 0, 0, t32a, nullptr, t32b);
    Block32(front_, 2, ew[1], eh[1], ew[1], eh[1], 0, 0, t32b, nullptr, t32a);
    Block32(front_, 3, ew[1], eh[1], ew[1], eh[1], 0, 0, t32a, nullptr, t32b);
    Block32(front_, 4, ew[1], eh[1], ew[1], eh[1], 0, 0, t32b, nullptr, skips_[0]);

    dx12::GpuBuffer* coarse = nullptr;   // what the back's first step reads
    uint32_t coarse_level = 2;
    if (split == HybridSplit::kShallow) {
        mid_in_ = NewBuffer(bytes(2, 64));
        mid_out_ = NewBuffer(bytes(2, 64));
        PoolProject(front_, "block4.layer0.weight0", ew[1], eh[1], ew[2], eh[2], 32, 64, skips_[0], mid_in_);
        mid_w_ = ew[2];
        mid_h_ = eh[2];
        mid_c_ = 64;
        coarse = mid_out_;
    } else {
        // levels 2-4 (C = 64, 128, 256): x is transformed in place, ff and scratch per level
        Level lv[5];
        dx12::GpuBuffer* x[5] = {};
        for (int l = 2; l <= 4; ++l) {
            lv[l].x = x[l] = NewBuffer(bytes(l, channels[l]));
            lv[l].ff = NewBuffer(bytes(l, channels[l]));
            lv[l].sc = NewBuffer(bytes(l, channels[l]));
            if (use_dml_) {
                lv[l].t = NewBuffer(bytes(l, channels[l]));
                lv[l].h1 = NewBuffer(bytes(l, channels[l] * 4));
                lv[l].qkv = NewBuffer(bytes(l, channels[l] * 3));
            }
            skips_[l - 1] = NewBuffer(bytes(l, channels[l]));
        }
        mid_in_ = NewBuffer(bytes(5, 512));
        mid_out_ = NewBuffer(bytes(5, 512));
        mid_w_ = ew[5];
        mid_h_ = eh[5];
        mid_c_ = 512;

        PoolProject(front_, "block4.layer0.weight0", ew[1], eh[1], ew[2], eh[2], 32, 64, skips_[0], x[2]);
        static const int kRegular[5][2] = {{0, 0}, {0, 0}, {5, 8}, {9, 14}, {15, 22}};   // [first, transition]
        for (int l = 2; l <= 4; ++l) {
            for (int b = kRegular[l][0]; b < kRegular[l][1]; ++b) {
                WideBlock(front_, b, ew[l], eh[l], channels[l], lv[l], x[l]);
            }
            const int t = kRegular[l][1];
            WideBlock(front_, t, ew[l], eh[l], channels[l], lv[l], skips_[l - 1]);
            dx12::GpuBuffer* next = l < 4 ? x[l + 1] : mid_in_;
            PoolProject(front_, "block" + std::to_string(t) + ".layer0.weight0", ew[l], eh[l], ew[l + 1], eh[l + 1],
                        channels[l], channels[l + 1], skips_[l - 1], next);
        }

        // back, levels 4 -> 2: up-merge with the level's skip, then the level's blocks in place
        static const int kDecoder[5][2] = {{0, 0}, {0, 0}, {62, 66}, {56, 62}, {48, 56}};   // [transition, end)
        dx12::GpuBuffer* from = mid_out_;
        for (int l = 4; l >= 2; --l) {
            const std::string up = "block" + std::to_string(kDecoder[l][0]) + ".layer0.";
            UpMerge(back_, up + "weight0", up + "sin", ew[l], eh[l], ew[l + 1], eh[l + 1], channels[l + 1],
                    channels[l], from, skips_[l - 1], x[l], "b" + std::to_string(kDecoder[l][0]) + " up-merge");
            for (int b = kDecoder[l][0]; b < kDecoder[l][1]; ++b) {
                WideBlock(back_, b, ew[l], eh[l], channels[l], lv[l], x[l]);
            }
            from = x[l];
        }
        coarse = x[2];

        if (split == HybridSplit::kNative) {
            // the bottleneck: split blocks at 1/32 (in place on MidIn, block 30 into the split
            // skip), block 30's padded pool into 1/64, the ViT, block 39 back up into MidOut and
            // the split blocks 40-47 in place there
            const uint32_t w5 = ew[5], h5 = eh[5];
            const uint32_t w6 = (w5 + 7) / 8 * 4, h6 = (h5 + 7) / 8 * 4;
            const uint64_t t5 = static_cast<uint64_t>(w5) * h5, t6 = static_cast<uint64_t>(w6) * h6;
            Level l5, l6;
            l5.ff = NewBuffer(t5 * 512 * 2);
            l5.sc = NewBuffer(t5 * 512 * 2);
            l5.t = NewBuffer(t5 * 512 * 2);
            l5.h1 = NewBuffer(t5 * 2048 * 2);
            l5.qkv = NewBuffer(t5 * 1536 * 2);
            dx12::GpuBuffer* split_skip = NewBuffer(t5 * 512 * 2);
            l6.x = NewBuffer(t6 * 1024 * 2);
            l6.ff = NewBuffer(t6 * 1024 * 2);
            l6.sc = NewBuffer(t6 * 1024 * 2);
            l6.t = NewBuffer(t6 * 1024 * 2);
            l6.h1 = NewBuffer(t6 * 4096 * 2);
            l6.qkv = NewBuffer(t6 * 3072 * 2);
            l5.x = mid_in_;
            for (int b = 23; b < 30; ++b) SplitBlock(core_, b, w5, h5, l5, mid_in_);
            SplitBlock(core_, 30, w5, h5, l5, split_skip);
            PoolProject(core_, "block30.layer4.weight", w5, h5, w6, h6, 512, 1024, split_skip, l6.x);
            for (int b = 31; b < 39; ++b) VitBlock(core_, b, static_cast<uint32_t>(t6), l6);
            UpMerge(core_, "block39.layer0.conv_weight", "block39.layer0.inp_upsample_sin", w5, h5, w6, h6, 1024, 512,
                    l6.x, split_skip, mid_out_, "b39 up-merge");
            l5.x = mid_out_;
            for (int b = 40; b < 48; ++b) SplitBlock(core_, b, w5, h5, l5, mid_out_);
        }
    }

    // back, levels 1-0: blocks 66-70, 32 channels, and the head
    Block32(back_, 66, ew[1], eh[1], ew[coarse_level], eh[coarse_level], 2, 0, coarse, skips_[0], t32a);
    Block32(back_, 67, ew[1], eh[1], ew[1], eh[1], 0, 0, t32a, nullptr, t32b);
    Block32(back_, 68, ew[1], eh[1], ew[1], eh[1], 0, 0, t32b, nullptr, t32a);
    Block32(back_, 69, ew[1], eh[1], ew[1], eh[1], 0, 0, t32a, nullptr, t32b);
    Block32(back_, 70, ew[0], eh[0], ew[1], eh[1], 3, 1, t32b, full_skip_, &head);

    for (auto& b : buffers_) {
        const uint64_t size = b->byte_size;
        if (!rm.CreateDefaultBuffer(size, *b, true)) {
            if (error) *error = "allocate hybrid activations";
            return false;
        }
    }
    if (!upload.ResetCommandList()) return false;
    for (auto& b : buffers_) {
        dx12::ResourceManager::TransitionBuffer(upload.GetCommandList(), *b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    upload.ExecuteAndWait();
    if (use_dml_ && !gemm_.Finalize(rm, upload)) {
        if (error) *error = gemm_.error();
        return false;
    }
    return true;
}

void HybridEnds::Record(ID3D12GraphicsCommandList* cmd, const std::vector<Op>& ops, uint32_t query_base) {
    const bool profile = profile_heap_ && query_base + ops.size() < profile_capacity_;
    if (profile) cmd->EndQuery(profile_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query_base);
    ID3D12PipelineState* bound = nullptr;
    uint32_t query = query_base;
    for (const Op& op : ops) {
        if (op.gemm >= 0) {
            // DirectML sets its own root signature, pipeline and descriptor heap
            gemm_.Record(cmd, op.gemm);
            if (profile) cmd->EndQuery(profile_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ++query);
            bound = nullptr;
            continue;
        }
        if (!bound) {
            cmd->SetComputeRootSignature(root_sig_.Get());
            cmd->SetComputeRootShaderResourceView(1, weights_.resource->GetGPUVirtualAddress());
            cmd->SetComputeRootShaderResourceView(
                5, (weights16_.resource ? weights16_ : weights_).resource->GetGPUVirtualAddress());
        }
        if (op.pso != bound) {
            cmd->SetPipelineState(op.pso);
            bound = op.pso;
        }
        cmd->SetComputeRoot32BitConstants(0, kConstants, op.constants, 0);
        cmd->SetComputeRootUnorderedAccessView(2, op.a->resource->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(3, (op.b ? op.b : op.a)->resource->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(4, op.out->resource->GetGPUVirtualAddress());
        cmd->Dispatch(op.groups[0], op.groups[1], op.groups[2]);
        dx12::ResourceManager::RecordUAVBarrier(cmd, nullptr);
        if (profile) cmd->EndQuery(profile_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ++query);
    }
    if (profile) {
        cmd->ResolveQueryData(profile_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query_base,
                              static_cast<UINT>(ops.size() + 1), profile_readback_.Get(), query_base * sizeof(uint64_t));
    }
}

void HybridEnds::RecordOps(ID3D12GraphicsCommandList* cmd, const std::vector<Op>& ops, size_t begin, size_t end,
                           ID3D12PipelineState*& bound) {
    for (size_t i = begin; i < end && i < ops.size(); ++i) {
        const Op& op = ops[i];
        if (op.gemm >= 0) {
            gemm_.Record(cmd, op.gemm);   // DirectML sets its own root signature, pipeline and heap
            bound = nullptr;
            continue;
        }
        if (!bound) {
            cmd->SetComputeRootSignature(root_sig_.Get());
            cmd->SetComputeRootShaderResourceView(1, weights_.resource->GetGPUVirtualAddress());
            cmd->SetComputeRootShaderResourceView(
                5, (weights16_.resource ? weights16_ : weights_).resource->GetGPUVirtualAddress());
        }
        if (op.pso != bound) {
            cmd->SetPipelineState(op.pso);
            bound = op.pso;
        }
        cmd->SetComputeRoot32BitConstants(0, kConstants, op.constants, 0);
        cmd->SetComputeRootUnorderedAccessView(2, op.a->resource->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(3, (op.b ? op.b : op.a)->resource->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(4, op.out->resource->GetGPUVirtualAddress());
        cmd->Dispatch(op.groups[0], op.groups[1], op.groups[2]);
        dx12::ResourceManager::RecordUAVBarrier(cmd, nullptr);
    }
}

void HybridEnds::RecordRange(ID3D12GraphicsCommandList* cmd, size_t begin, size_t end) {
    ID3D12PipelineState* bound = nullptr;
    size_t base = 0;
    for (const std::vector<Op>* list : {&front_, &core_, &back_}) {
        const size_t lo = std::max(begin, base), hi = std::min(end, base + list->size());
        if (lo < hi) RecordOps(cmd, *list, lo - base, hi - base, bound);
        base += list->size();
    }
}

void HybridEnds::RecordFront(ID3D12GraphicsCommandList* cmd) { Record(cmd, front_, 0); }

void HybridEnds::RecordCore(ID3D12GraphicsCommandList* cmd) {
    Record(cmd, core_, static_cast<uint32_t>(front_.size() + 1));
}

void HybridEnds::RecordBack(ID3D12GraphicsCommandList* cmd) {
    Record(cmd, back_, static_cast<uint32_t>(front_.size() + core_.size() + 2));
}

bool HybridEnds::EnableProfiling(ID3D12CommandQueue* queue) {
    profile_capacity_ = 1024;
    D3D12_QUERY_HEAP_DESC qh{};
    qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qh.Count = profile_capacity_;
    D3D12_HEAP_PROPERTIES rb{};
    rb.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sizeof(uint64_t) * profile_capacity_;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(device_->CreateQueryHeap(&qh, IID_PPV_ARGS(&profile_heap_))) &&
           SUCCEEDED(device_->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      nullptr, IID_PPV_ARGS(&profile_readback_))) &&
           SUCCEEDED(queue->GetTimestampFrequency(&profile_freq_)) && profile_freq_ > 0;
}

bool HybridEnds::ReadProfile(std::vector<std::pair<std::string, double>>& out) {
    out.clear();
    if (!profile_readback_ || front_.size() + core_.size() + back_.size() + 3 > profile_capacity_) return false;
    void* mapped = nullptr;
    D3D12_RANGE range{0, sizeof(uint64_t) * profile_capacity_};
    if (FAILED(profile_readback_->Map(0, &range, &mapped)) || !mapped) return false;
    const auto* t = static_cast<const uint64_t*>(mapped);
    const double scale = 1000.0 / static_cast<double>(profile_freq_);
    size_t base = 0;
    for (const auto& [name, list] : {std::pair<const char*, const std::vector<Op>*>{"front ", &front_},
                                     {"core ", &core_}, {"back ", &back_}}) {
        for (size_t i = 0; i < list->size(); ++i) {
            out.emplace_back(name + (*list)[i].label, (t[base + i + 1] - t[base + i]) * scale);
        }
        base += list->size() + 1;
    }
    D3D12_RANGE none{0, 0};
    profile_readback_->Unmap(0, &none);
    return true;
}

} // namespace dlss_nr::neural
