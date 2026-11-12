#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dx12/command_queue.h"
#include "dx12/resources.h"
#include "neural/dml_gemm.h"

namespace dlss_nr::neural {

using Microsoft::WRL::ComPtr;

// The hybrid runner's HLSL ends: HNet's window blocks at both ends of the U-Net as compute
// kernels (src/compute/neural_blocks.hlsl for C = 32, neural_blocks_wide.hlsl for C = 64-256)
// around an ONNX middle (tools/hnet/export_onnx.py --part middle | middle-deep).
//
//   shallow  front blocks 0-4  -> middle 5-65  (64ch at 1/4)                 -> back 66-70
//   deep     front blocks 0-22 -> middle 23-47 (512ch at 1/32, padded to 8)  -> back 48-70
//   native   front blocks 0-22 -> core 23-47 here too (needs the DirectML GEMMs) -> back 48-70:
//            no ONNX Runtime at all
//
// The front runs from the features to MidIn() and keeps the encoder skips; the back runs
// from MidOut() and the skips to the head. Both are lists of dispatches built at Allocate.
enum class HybridSplit { kShallow, kDeep, kNative };

class HybridEnds {
public:
    static std::string BlobFileName(bool synthetic);
    static std::string MiddleFileName(HybridSplit split, uint32_t field_w, uint32_t field_h, bool synthetic);
    // the extent of the 512-channel level for a field (block 22 pads 1/16 to a multiple of 8)
    static void DeepExtent(uint32_t field_w, uint32_t field_h, uint32_t& w, uint32_t& h);

    // With `dml`, the wide blocks' projections run as DirectML GEMMs (XMX); without, as HLSL.
    bool Initialize(ID3D12Device* device, IDMLDevice* dml, std::string* error);
    bool UsesDmlGemms() const { return use_dml_; }
    // true: the vendor's E4M3 publishes (NeuralOptions::precise); takes effect at the next Allocate
    void SetPrecise(bool precise) { precise_ = precise; }
    bool LoadWeights(const std::string& path, dx12::ResourceManager& rm, dx12::CommandQueue& upload, std::string* error);
    // activation buffers and dispatch lists for a field; leaves every buffer in UNORDERED_ACCESS
    bool Allocate(HybridSplit split, uint32_t field_w, uint32_t field_h, dx12::GpuBuffer& features,
                  dx12::GpuBuffer& head, dx12::ResourceManager& rm, dx12::CommandQueue& upload, std::string* error);

    void RecordFront(ID3D12GraphicsCommandList* cmd);
    void RecordCore(ID3D12GraphicsCommandList* cmd);    // native only: MidIn() -> MidOut()
    void RecordBack(ID3D12GraphicsCommandList* cmd);
    // The whole graph is one sequence front, core, back of OpCount() dispatches that only the
    // first reads the features of and only the last writes the head of, every buffer in between
    // the graph's own; so it may be cut anywhere and run as parts on consecutive frames
    // (NeuralOptions::interval). Records ops [begin, end) of that sequence, without profiling.
    size_t OpCount() const { return front_.size() + core_.size() + back_.size(); }
    void RecordRange(ID3D12GraphicsCommandList* cmd, size_t begin, size_t end);

    // Per-dispatch GPU timestamps (profiling the bench): once the frame completed,
    // ReadProfile returns (label, ms) for every front and back dispatch.
    bool EnableProfiling(ID3D12CommandQueue* queue);
    void DisableProfiling() {
        profile_heap_.Reset();
        profile_readback_.Reset();
    }
    bool Profiling() const { return profile_heap_ != nullptr; }
    bool ReadProfile(std::vector<std::pair<std::string, double>>& out);

    HybridSplit Split() const { return split_; }
    uint32_t MidWidth() const { return mid_w_; }
    uint32_t MidHeight() const { return mid_h_; }
    uint32_t MidChannels() const { return mid_c_; }
    dx12::GpuBuffer& MidIn() { return *mid_in_; }
    dx12::GpuBuffer& MidOut() { return *mid_out_; }
    dx12::GpuBuffer& FullSkip() { return *full_skip_; }
    // encoder level i's skip (shallow: level 0 only)
    dx12::GpuBuffer* Skip(uint32_t level) { return level < 4 ? skips_[level] : nullptr; }
    uint32_t SkipCount() const { return split_ == HybridSplit::kShallow ? 1u : 4u; }
    // every activation buffer, for state bookkeeping
    std::vector<dx12::GpuBuffer*> Buffers();

private:
    struct Op {
        ID3D12PipelineState* pso = nullptr;
        uint32_t constants[24] = {};
        dx12::GpuBuffer* a = nullptr;
        dx12::GpuBuffer* b = nullptr;
        dx12::GpuBuffer* out = nullptr;
        uint32_t groups[3] = {1, 1, 1};
        std::string label;
        int gemm = -1;                    // a DirectML GEMM instead of an HLSL dispatch
    };

    uint32_t Offset(const std::string& name) const;
    dx12::GpuBuffer* NewBuffer(uint64_t bytes);
    void Record(ID3D12GraphicsCommandList* cmd, const std::vector<Op>& ops, uint32_t query_base);
    // ops[begin, end), no timestamps; `bound` carries the pipeline across calls (nullptr: unbound)
    void RecordOps(ID3D12GraphicsCommandList* cmd, const std::vector<Op>& ops, size_t begin, size_t end,
                   ID3D12PipelineState*& bound);

    // op builders (append to `ops`)
    void Block32(std::vector<Op>& ops, int block, uint32_t w, uint32_t h, uint32_t in_w, uint32_t in_h,
                 uint32_t input_mode, uint32_t output_mode, dx12::GpuBuffer* a, dx12::GpuBuffer* b,
                 dx12::GpuBuffer* out);
    void Pool32(std::vector<Op>& ops, uint32_t in_w, uint32_t in_h, dx12::GpuBuffer* a, dx12::GpuBuffer* out);
    struct Level {                        // a wide level's working buffers
        dx12::GpuBuffer* x = nullptr;     // block input / output (transformed in place)
        dx12::GpuBuffer* ff = nullptr;    // feed-forward output
        dx12::GpuBuffer* sc = nullptr;    // hs, then the attention's o
        dx12::GpuBuffer* t = nullptr;     // GEMM outputs before their residual
        dx12::GpuBuffer* h1 = nullptr;    // expanded hidden (T, 4C)
        dx12::GpuBuffer* qkv = nullptr;   // (T, 3C)
    };
    void WideBlock(std::vector<Op>& ops, int block, uint32_t w, uint32_t h, uint32_t channels, const Level& level,
                   dx12::GpuBuffer* out);
    void WideBlockDml(std::vector<Op>& ops, int block, uint32_t w, uint32_t h, uint32_t channels, const Level& level,
                      dx12::GpuBuffer* out);
    // The vendor publishes (rounds to E4M3) every block's output except the transitions' — 0, 4,
    // 8, 14, 22, 30 feed a pool and a skip that round it themselves — and 70's, which feeds the head.
    static bool Publishes(int block) {
        return block != 0 && block != 4 && block != 8 && block != 14 && block != 22 && block != 30 && block != 70;
    }
    Op Elementwise(ID3D12PipelineState* pso, uint64_t halves, uint32_t channels, uint32_t o_cos, dx12::GpuBuffer* a,
                   dx12::GpuBuffer* b, dx12::GpuBuffer* out, const std::string& label, uint32_t publish = 0);
    Op GemmOp(const DmlGemmSet::Gemm& gemm, const std::string& label);
    uint64_t Offset16(const std::string& name) const;
    void PoolProject(std::vector<Op>& ops, const std::string& weight, uint32_t in_w, uint32_t in_h, uint32_t out_w,
                     uint32_t out_h, uint32_t channels, uint32_t out_channels, dx12::GpuBuffer* a, dx12::GpuBuffer* out);
    void UpMerge(std::vector<Op>& ops, const std::string& weight, const std::string& sine, uint32_t w, uint32_t h,
                 uint32_t coarse_w, uint32_t coarse_h, uint32_t channels, uint32_t out_channels,
                 dx12::GpuBuffer* coarse, dx12::GpuBuffer* skip, dx12::GpuBuffer* out, const std::string& label);
    void SplitBlock(std::vector<Op>& ops, int block, uint32_t w, uint32_t h, const Level& level, dx12::GpuBuffer* out);
    void VitBlock(std::vector<Op>& ops, int block, uint32_t tokens, const Level& level);
    void Gemm(std::vector<Op>& ops, const std::string& label, uint32_t m, uint32_t k, uint32_t n,
              dx12::GpuBuffer* a, const std::string& weight, dx12::GpuBuffer* out, uint32_t batch = 1,
              uint32_t a_row = 0, uint32_t a_batch = 0, uint32_t out_row = 0, uint32_t out_batch = 0);

    ID3D12Device* device_ = nullptr;
    ComPtr<ID3D12RootSignature> root_sig_;
    // both variants of every kernel; pso_*_ below are the active set (Allocate)
    struct PipelineSet {
        ComPtr<ID3D12PipelineState> block32, pool32, ffn, project, attention, pool_project, up_merge, gate,
            residual, attention_qkv, global_attention, quantize;
    };
    PipelineSet sets_[2];
    bool precise_ = true;
    ComPtr<ID3D12PipelineState> pso_block32_;
    ComPtr<ID3D12PipelineState> pso_pool32_;
    ComPtr<ID3D12PipelineState> pso_ffn_;
    ComPtr<ID3D12PipelineState> pso_project_;
    ComPtr<ID3D12PipelineState> pso_attention_;
    ComPtr<ID3D12PipelineState> pso_pool_project_;
    ComPtr<ID3D12PipelineState> pso_up_merge_;
    ComPtr<ID3D12PipelineState> pso_gate_;
    ComPtr<ID3D12PipelineState> pso_residual_;
    ComPtr<ID3D12PipelineState> pso_attention_qkv_;
    ComPtr<ID3D12PipelineState> pso_global_attention_;
    ComPtr<ID3D12PipelineState> pso_quantize_;

    bool use_dml_ = false;
    // the fused blocks FFN in half (dot2add + half2): measured slower on Arc B580 driver 32.0.101.9150
    // (fused32 10.8 -> 12.4 ms at 960x576), so off; kept for other drivers / GPUs
    bool half_ffn_ = false;
    DmlGemmSet gemm_;
    dx12::GpuBuffer weights16_;           // the GEMMs' matrices in half, (N, K) per batch
    std::map<std::string, uint64_t> offsets16_;   // bytes

    dx12::GpuBuffer weights_;
    std::string weights_path_;
    std::map<std::string, uint32_t> offsets_;

    HybridSplit split_ = HybridSplit::kShallow;
    std::vector<std::unique_ptr<dx12::GpuBuffer>> buffers_;
    dx12::GpuBuffer* full_skip_ = nullptr;
    dx12::GpuBuffer* skips_[4] = {};
    dx12::GpuBuffer* mid_in_ = nullptr;
    dx12::GpuBuffer* mid_out_ = nullptr;
    uint32_t mid_w_ = 0, mid_h_ = 0, mid_c_ = 0;
    std::vector<Op> front_;
    std::vector<Op> core_;
    std::vector<Op> back_;

    ComPtr<ID3D12QueryHeap> profile_heap_;
    ComPtr<ID3D12Resource> profile_readback_;
    uint64_t profile_freq_ = 0;
    uint32_t profile_capacity_ = 0;
};

} // namespace dlss_nr::neural
