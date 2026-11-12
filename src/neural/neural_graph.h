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
#include <string>
#include <vector>

#include "dx12/command_queue.h"
#include "dx12/resources.h"
#include "neural/hybrid_ends.h"
#include "neural/neural_field.h"
#include "neural/ort_runtime.h"

namespace dlss_nr::neural {

using Microsoft::WRL::ComPtr;

struct NeuralOptions {
    // where onnxruntime.dll (DirectML build) and DirectML.dll 1.15.x are looked up
    std::vector<std::string> ort_dirs = {"ort", "third_party/onnxruntime/extracted/runtimes/win-x64/native"};
    std::vector<std::string> directml_dirs = {"ort", "third_party/directml/extracted/bin/x64-win"};
    // where hnet_<W>x<H>_<tag>_fp16[_synthetic].onnx are (tools/hnet/export_onnx.py)
    std::vector<std::string> model_dirs = {"models/neural", "work/onnx"};
    std::string model_tag = "nq-exp";   // numerics of the export: nq-exp is the fast one, vk-pwl the closest
    bool synthetic_weights = false;   // pick the *_synthetic.onnx files (plumbing and timing only)

    float render_scale = 1.0f;        // < 1: the network runs on a downscaled frame
    float intensity = 1.0f;           // blend of the model's picture against the frame
    float style = 0.0f;               // normalised style index (profile)
    float tone = 1.0f;                // local tone strength
    float structure = 1.0f;           // local structure strength
    bool temporal = true;             // feed the previous prediction back (channels 7-9 + gate)
    // 2-4 (native graph): the network spread over that many frames — the op sequence cut into
    // parts of about equal GPU time (measured once per field), the features taken on the first —
    // so about 1/interval of the GPU time per frame. Every frame composes the newest complete
    // head over its own picture, moved there along the picture's motion (CSWarpMotion,
    // CSHeadWarp) and faded out where the colours disagree, so the correction follows motion
    // within +-24 field pixels instead of lagging interval-1 frames behind.
    uint32_t interval = 1;
    // the game bridge only (ini capture=N, the graph ignores it): when it changes to N > 0, the next
    // N frames' game picture and shown picture are written raw next to the log (offline analysis)
    uint32_t capture_frames = 0;
    bool warp = true;                 // interval > 1: move the last head along the motion (false: as it is)
    // with the warp: correct the moved head by the colour difference between this frame and the
    // reference at the matched spot along the head's local colour slope (head ~ A colour + b,
    // CSGuideFit): the correction's edges are this frame's, so a motion a little off (cloth,
    // limbs, a block between two motions) no longer shows the old picture's edges (a ghost)
    bool warp_model = true;
    // with the model: a new head's weight against the last one carried onto its picture when it
    // becomes the reference (1: the new head alone). Averages the network's frame-to-frame change
    // (it is not shift invariant) that otherwise arrives at once every `interval` frames.
    float warp_blend = 0.5f;
    // with the warp: a full search on a quarter-resolution luma (+-96 field pixels around no
    // motion, CSWarpPyramid) adds a second centre to the search around the predicted motion, so a
    // camera that reverses, starts or jerks is still found (swing scene, interval 4)
    bool warp_pyramid = true;
    // with the warp: the motion matched frame to frame and summed to the reference's frame
    // (CSStepMotion, CSChain) centres the search instead of last vector x age ratio: a fast camera
    // moved a 3-6 frame old head beyond the search (Spider-Man capture 2026-11-13)
    // Off by default: +0.35-0.65 ms at 1440p for 0.503 -> 0.484 on the capture (interval 4) and
    // nothing at interval 2; it matters without the pyramid (swing 0.476 -> 0.280)
    bool warp_chain = false;
    // with the model: the moved correction's detail fades to the regional model (CSCoarseFit) from
    // this speed of the picture (field pixels a frame) to warp_fade_full (warp_fade_full 0: never).
    // A moved correction is never quite in place under a fast camera: its detail swam (a ghost,
    // motion sickness in the 2026-11-12 Spider-Man test); the regional model takes this frame's colour
    // (sb_off, rs 0.5, interval 4: pans of 1-3 px unchanged; pan 24 jump 0.187 -> 0.148)
    float warp_fade_start = 2.0f;
    float warp_fade_full = 6.0f;
    // ... to no correction at all instead of the regional model (ini warp_fade_to=none)
    bool warp_fade_none = false;
    // true (ini precision=accurate): the vendor's E4M3 roundings, effect corr vs the oracle 0.977
    // with the real weights; false (fast): without them, 0.85, about 20 % less GPU time
    bool precise = true;
    // > 0: CSHeadStabilize averages the head over frames along the picture's motion (CSMotion), a
    // new head weighing this much (0.05 ~ the last 20 frames); 0 off. Against the flicker of
    // precise (a new head moves 27 % for invisible noise) and the shimmer under motion of both
    // precisions (`--stability`). The game's default is 0.05 in both.
    float stabilize = 0.0f;
    // scales the colour change (after motion) counted as noise (2/255) and as a mismatch (8/255):
    // higher averages through more jitter / film grain, lower drops the average sooner
    float stabilize_tolerance = 1.0f;
    // where anything moves, the least weight of a new head (the average dies out within a few
    // frames): a block's translation cannot follow rotation or parallax, and the colour check
    // misses smooth surfaces (the nose smear of the first game test)
    float stabilize_motion = 0.5f;
    // With nq-exp and both hnet-mid_<field>_nq-exp_fp16.onnx and hnet-ends_nq-exp.bin present,
    // run the hybrid graph: the 32-channel ends as fused HLSL kernels around an ONNX middle.
    bool allow_hybrid = true;
    bool deep_hybrid = true;          // prefer the deep split (hnet-mid23_*) when its middle exists
    bool dml_gemms = true;            // the deep split's wide-block projections as DirectML GEMMs (XMX)
    bool native = true;               // with the GEMMs and the blob: the whole graph without ONNX Runtime
};

// The three conditioning scalars of a named profile (standard, natural, cinematic, neutral).
bool ApplyProfile(const std::string& name, NeuralOptions& options);

struct NeuralTimings {
    double pre_ms = 0.0;       // feature pre-pass (GPU)
    double network_ms = 0.0;   // pre end -> post begin on the queue: the ORT/DirectML graph and any gap
    double post_ms = 0.0;      // composition (GPU)
    double total_ms = 0.0;     // pre begin -> post end
};

// The DLSS-NR network on a D3D12 queue: an HLSL pre-pass builds the 16-channel features,
// ONNX Runtime's DirectML EP runs HNet on the same device and queue with our buffers bound
// zero-copy, and an HLSL post-pass composes the head over the frame and keeps the history.
//
// Per frame, on one queue, in this order:
//   copy the frame into GetInputTexture(); RecordPre(list A); execute A;
//   RunNetwork();
//   RecordPost(list B); copy GetOutputTexture() out; execute B.
// With options.interval 2-4 (native graph only) consecutive frames run consecutive parts of the
// network; the caller does not change.
// The native graph's ~570 dispatches are recorded once per extent into reusable command lists,
// one copy per timing slot, which RunNetwork submits. So a timing slot must not be reused
// before that slot's previous frame completed on the GPU (the bridge and the app both wait).
class NeuralGraph {
public:
    static constexpr uint32_t kTimingSlots = 4;
    static constexpr uint32_t kMaxInterval = 4;

    NeuralGraph() = default;
    ~NeuralGraph();

    NeuralGraph(const NeuralGraph&) = delete;
    NeuralGraph& operator=(const NeuralGraph&) = delete;

    bool Initialize(ID3D12Device* device, ID3D12CommandQueue* queue, const NeuralOptions& options);

    // (Re)builds everything extent-dependent — textures, buffers, noise, the ORT session —
    // when the output size, format or render scale changed. The caller must have drained
    // the queue of frames that use the old resources. `format` is the UAV-capable format of
    // the input/output textures (the back buffer's, without _SRGB).
    bool EnsureExtent(uint32_t output_w, uint32_t output_h, DXGI_FORMAT format);

    void RecordPre(ID3D12GraphicsCommandList* cmd, uint32_t timing_slot = 0);
    bool RunNetwork();
    void RecordPost(ID3D12GraphicsCommandList* cmd, uint32_t timing_slot = 0);

    // after the slot's work completed on the GPU
    bool ReadTimings(uint32_t timing_slot, NeuralTimings& out) const;

    bool IsReady() const { return session_ != nullptr || native_active_; }
    bool IsNative() const { return native_active_; }
    bool IsHybrid() const { return hybrid_active_; }

    // Validation of the hybrid ends against tools/hnet/export_hybrid.py --dump: runs the front
    // from DIR/features.f16 and the back from DIR/middle_value.f32 on the upload queue (no
    // ORT) and compares with the dumped tensors. Needs EnsureExtent at the dumped field first.
    bool CheckHybridEnds(const std::string& dir, std::string& report);
    HybridEnds* Hybrid() { return hybrid_active_ ? &hybrid_ : nullptr; }
    // also restarts the network: frames went by unprocessed (F11, a resize), so the next one runs
    // it whole and the parts and the reference head start afresh
    void ResetHistory() {
        history_valid_ = false;
        stable_valid_ = false;
        head_valid_ = false;
        ref_valid_ = false;
        warp_age_valid_ = false;
        capture_valid_ = false;
        chain_prev_ = chain_pending_ = chain_active_ = false;
        phase_ = 0;
    }
    void SetStabilize(float weight) { options_.stabilize = weight; }
    void SetWarp(bool warp) { options_.warp = warp; }
    void SetWarpModel(bool model) { options_.warp_model = model; }
    void SetWarpBlend(float weight) { options_.warp_blend = weight; }
    // a different interval on the same extent (the app's --interval-check); restarts the network
    bool SetInterval(uint32_t interval) {
        options_.interval = interval < 1 ? 1 : (interval > kMaxInterval ? kMaxInterval : interval);
        ResetHistory();
        return !Warping() || part_begin_.size() == options_.interval + 1 || BalanceParts();
    }
    void SetIntensity(float intensity) { options_.intensity = intensity; }
    // the options that take effect on the next frame without new buffers (an ini reloaded while
    // the game runs): intensity, profile, temporal, warp*, stabilize*. Not the interval
    // (SetInterval), the render scale (SetRenderScale, then a new extent) or the precision.
    void SetLiveOptions(const NeuralOptions& o) {
        options_.intensity = o.intensity;
        options_.style = o.style;
        options_.tone = o.tone;
        options_.structure = o.structure;
        options_.temporal = o.temporal;
        options_.warp = o.warp;
        options_.warp_model = o.warp_model;
        options_.warp_blend = o.warp_blend;
        options_.warp_pyramid = o.warp_pyramid;
        options_.warp_chain = o.warp_chain;
        options_.warp_fade_start = o.warp_fade_start;
        options_.warp_fade_full = o.warp_fade_full;
        options_.warp_fade_none = o.warp_fade_none;
        options_.stabilize = o.stabilize;
        options_.stabilize_tolerance = o.stabilize_tolerance;
        options_.stabilize_motion = o.stabilize_motion;
    }
    void SetConditioning(float style, float tone, float structure);
    void SetRenderScale(float scale) { options_.render_scale = scale; }

    dx12::GpuTexture& GetInputTexture() { return input_tex_; }
    dx12::GpuTexture& GetOutputTexture() { return output_tex_; }
    const NeuralExtent& GetExtent() const { return extent_; }
    const NeuralOptions& GetOptions() const { return options_; }
    const std::string& LastError() const { return error_; }
    const std::string& ModelPath() const { return model_path_; }
    // interval > 1: the parts the network is cut into, for the log (empty before the first cut)
    const std::string& PartSummary() const { return part_summary_; }
    double LastSessionCreateMs() const { return session_create_ms_; }
    uint64_t GetGpuMemoryBytes() const { return resources_.GetTotalAllocatedBytes(); }

    static std::string ModelFileName(uint32_t field_w, uint32_t field_h, const std::string& tag, bool synthetic);

private:
    bool CreatePipelines();
    bool CreateSession(const std::string& path);
    std::string FindModel(const std::string& name) const;
    void ReleaseSession();
    void SetConstants(ID3D12GraphicsCommandList* cmd) const;
    bool Fail(const std::string& message);

    ID3D12Device* device_ = nullptr;
    ComPtr<ID3D12CommandQueue> queue_;
    NeuralOptions options_{};
    std::string error_;
    std::string ort_error_;          // why ONNX Runtime is unavailable, when it is

    OrtRuntime ort_;
    ComPtr<IDMLDevice> dml_device_;
    OrtSession* session_ = nullptr;
    OrtIoBinding* binding_ = nullptr;
    OrtMemoryInfo* dml_memory_ = nullptr;
    OrtValue* features_value_ = nullptr;
    OrtValue* head_value_ = nullptr;
    void* features_alloc_ = nullptr;
    void* head_alloc_ = nullptr;
    std::string model_path_;
    double session_create_ms_ = 0.0;

    dx12::CommandQueue upload_queue_;
    dx12::ResourceManager resources_;
    ComPtr<ID3D12RootSignature> root_sig_;
    ComPtr<ID3D12PipelineState> pso_pre_;
    ComPtr<ID3D12PipelineState> pso_post_;
    ComPtr<ID3D12PipelineState> pso_stabilize_;
    ComPtr<ID3D12PipelineState> pso_motion_;
    ComPtr<ID3D12PipelineState> pso_commit_;
    ComPtr<ID3D12PipelineState> pso_field_colour_;
    ComPtr<ID3D12PipelineState> pso_ref_commit_;
    ComPtr<ID3D12PipelineState> pso_warp_motion_;
    ComPtr<ID3D12PipelineState> pso_head_warp_;
    ComPtr<ID3D12PipelineState> pso_guide_fit_;
    ComPtr<ID3D12PipelineState> pso_guide_smooth_;
    ComPtr<ID3D12PipelineState> pso_coarse_sums_;
    ComPtr<ID3D12PipelineState> pso_coarse_fit_;
    ComPtr<ID3D12PipelineState> pso_quarter_cur_;
    ComPtr<ID3D12PipelineState> pso_quarter_ref_;
    ComPtr<ID3D12PipelineState> pso_warp_pyramid_;
    ComPtr<ID3D12PipelineState> pso_step_motion_;
    ComPtr<ID3D12PipelineState> pso_chain_;
    ComPtr<ID3D12PipelineState> pso_keep_luma_;

    NeuralExtent extent_{};
    DXGI_FORMAT format_ = DXGI_FORMAT_UNKNOWN;
    float extent_scale_ = 0.0f;
    dx12::GpuTexture input_tex_;
    dx12::GpuTexture output_tex_;
    dx12::GpuBuffer noise_buf_;
    dx12::GpuBuffer features_buf_;
    dx12::GpuBuffer head_buf_;
    dx12::GpuBuffer history_buf_;
    dx12::GpuBuffer stable_buf_;     // CSMotion/CSHeadStabilize: 2 x 16 B per field pixel + 8 B per 8x8 block
    dx12::GpuBuffer warp_buf_;       // interval > 1: reference head, colour, luma; this frame's colour, luma; motion
    dx12::DescriptorHandle table_;   // t0 input, t1 noise, u0 features, u1 head, u2 history, u3 output,
                                     // u4 stabiliser, u5 warp
    bool history_valid_ = false;
    bool stable_valid_ = false;      // stable_buf_ holds the last frame's head
    bool stable_parity_ = false;     // which half of stable_buf_ holds it
    bool head_valid_ = false;        // a complete prediction exists (interval > 1 needs one)
    bool ref_valid_ = false;         // warp_buf_ holds the last complete head and its colour
    uint32_t warp_age_ = 0;          // frames between the reference's picture and the last frame
    bool warp_age_valid_ = false;    // ... and last frame's motion is there to predict from
    // the motion chain (CSStepMotion, CSChain): last frame's luma kept, the pending sum anchored on
    // a features frame, the active one on the current reference's; the sums' ping-pong half
    bool chain_prev_ = false, chain_pending_ = false, chain_active_ = false, chain_parity_ = false;
    uint32_t chain_flags_ = 0;       // this frame's kChain* bits (RecordPost)
    float warp_predict_ = 0.0f;      // this frame's age / last frame's (CSWarpMotion's prediction)
    float fade_lo_ = 0.0f, fade_hi_ = 0.0f;   // CSHeadWarp: warp_fade_start / _full x this frame's age
    bool capture_now_ = false;       // this frame keeps the head it shows for the next commit's blend
    bool blend_now_ = false;         // this frame's new reference blends with the kept head
    bool capture_valid_ = false;     // a kept head waits for the next commit
    uint32_t phase_ = 0;             // interval > 1: the part this frame runs (0 also takes the features)
    bool run_front_ = true, run_back_ = true;   // this frame takes the features / completes a head
    uint32_t part_ = 0;              // this frame's network list: 0 whole, 1 + phase_ a part
    std::vector<size_t> part_begin_; // interval + 1 op indices into HybridEnds' sequence
    std::string part_summary_;
    bool BalanceParts();             // part_begin_ from one profiled run (equal GPU time)
    bool Warping() const { return native_active_ && options_.interval > 1; }
    uint32_t slot_ = 0;              // this frame's timing slot (RecordPre)

    // The native network, recorded once: [slot][0 whole, 1 + k part k]. A list can't be
    // resubmitted while a previous submission runs, hence one copy per slot.
    struct NetworkList {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        bool valid = false;
    };
    NetworkList network_lists_[kTimingSlots][1 + kMaxInterval];
    bool lists_profiled_ = false;    // recorded with HybridEnds' per-dispatch timestamps
    void InvalidateNetworkLists();

    HybridEnds hybrid_;
    bool hybrid_ready_ = false;      // pipelines created
    bool hybrid_active_ = false;     // this extent runs the hybrid graph
    bool native_active_ = false;     // ... and its middle too: no ORT session

    ComPtr<ID3D12QueryHeap> timestamp_heap_;
    ComPtr<ID3D12Resource> timestamp_readback_;
    uint64_t timestamp_freq_hz_ = 0;
};

} // namespace dlss_nr::neural
