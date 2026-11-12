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

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "neural/neural_graph.h"

namespace dlss_nr::bridge {

using Microsoft::WRL::ComPtr;

struct BridgeTelemetry {
    uint64_t frames_intercepted = 0;
    uint64_t frames_processed = 0;
    uint32_t resize_events = 0;
    uint32_t current_width = 0;
    uint32_t current_height = 0;
    uint32_t inner_width = 0;      // what the network was handed (render_scale)
    uint32_t inner_height = 0;
    uint32_t field_width = 0;      // the padded network field (ONNX model)
    uint32_t field_height = 0;
    DXGI_FORMAT backbuffer_format = DXGI_FORMAT_UNKNOWN;
    double last_gpu_inference_ms = 0.0;
    double avg_gpu_inference_ms = 0.0;
    double avg_network_ms = 0.0;
    double avg_cpu_record_ms = 0.0;
    double last_session_create_ms = 0.0;
    bool inference_enabled = true;
    bool device_healthy = true;
};

// The DLSS-NR network on a game's DXGI swapchain (dxgi.dll proxy / OptiScaler ASI plugin).
// At every Present: the back buffer is copied into the NeuralGraph's input, the pre-pass
// and the ORT/DirectML network run on the game's own queue, and the composed frame is
// copied back before the real Present.
class D3D12GameBridge {
public:
    D3D12GameBridge() = default;
    ~D3D12GameBridge();

    D3D12GameBridge(const D3D12GameBridge&) = delete;
    D3D12GameBridge& operator=(const D3D12GameBridge&) = delete;

    // Legacy entry point kept for the proxy/test: only the log path is still used.
    void SetShaderAndModelDirs(const std::string& shader_dir,
                               const std::string& weights_path,
                               const std::string& log_path = "dlss_nr_dx12_bridge.log");
    void SetNeuralOptions(const neural::NeuralOptions& options);
    // dlss_nr_neural.ini re-read with `read` whenever it is saved while the game runs (checked
    // twice a second): the options that can change live are applied at once and logged
    void SetOptionsReloader(const std::string& ini_path, std::function<neural::NeuralOptions()> read);

    // Called when IDXGIFactory2::CreateSwapChainForHwnd or CreateSwapChain is intercepted
    void RegisterGameCommandQueue(IUnknown* candidate_queue);

    // Called immediately before IDXGISwapChain::Present / Present1
    bool OnPresentPre(IDXGISwapChain* swapchain);

    // Called before and after IDXGISwapChain::ResizeBuffers
    void OnResizeBuffersPre(IDXGISwapChain* swapchain);
    void OnResizeBuffersPost(IDXGISwapChain* swapchain, uint32_t new_width, uint32_t new_height);

    // Explicit cleanup when called before process exit
    void Shutdown();

    void SetInferenceEnabled(bool enabled) { inference_enabled_ = enabled; }
    // Build a new extent (weights, buffers, GEMMs or the ORT session: 0.2-5 s) on a worker
    // thread while frames go out unprocessed, instead of stalling Present. On by default.
    void SetAsyncSetup(bool async) { async_setup_ = async; }
    bool IsInferenceEnabled() const { return inference_enabled_; }

    BridgeTelemetry GetTelemetry() const;
    bool WriteTelemetryReportMarkdown(const std::string& path) const;
    void LogEvent(const std::string& msg);

private:
    bool EnsureInitialized(IDXGISwapChain* swapchain);
    bool EnsureNeural();
    void WaitForAllFramesInFlight();
    void ReadSlotTimings(uint32_t slot);
    void FinishExtent(bool ok, const std::string& key, DXGI_FORMAT format);
    void WaitExtentJob();
    void ReloadOptionsIfChanged();
    void StartCapture(uint32_t frames);
    void RecordCapture(uint32_t slot, const dx12::GpuTexture& in_tex, const dx12::GpuTexture& out_tex);
    void CollectCapture(uint32_t slot);

    static constexpr uint32_t kFramesInFlight = 3;

    mutable std::mutex mutex_;
    bool initialized_ = false;
    bool init_failed_ = false;
    bool inference_enabled_ = true;
    bool f11_was_down_ = false;      // the hotkey's own edge detection (see OnPresentPre)
    std::string ini_path_;
    std::function<neural::NeuralOptions()> read_options_;
    FILETIME ini_time_{};
    bool ini_time_valid_ = false;
    uint64_t ini_checked_ms_ = 0;

    std::string log_path_ = "dlss_nr_dx12_bridge.log";
    neural::NeuralOptions options_{};

    ComPtr<ID3D12Device> game_device_;
    ComPtr<ID3D12CommandQueue> game_queue_;
    IDXGISwapChain* last_swapchain_ = nullptr;

    // bound to game_queue_: rebuilt when the swapchain's queue changes
    std::unique_ptr<neural::NeuralGraph> neural_;
    ID3D12CommandQueue* neural_queue_ = nullptr;
    bool neural_failed_ = false;
    std::string failed_extent_;      // the extent whose model is missing, logged once
    bool async_setup_ = true;
    std::future<bool> extent_job_;   // EnsureExtent on a worker thread; neural_ is its alone meanwhile
    std::string extent_job_key_;
    DXGI_FORMAT extent_job_format_ = DXGI_FORMAT_UNKNOWN;

    // two command lists' worth per frame slot: copy-in + pre-pass, then post-pass + copy-out
    ComPtr<ID3D12CommandAllocator> frame_allocators_[kFramesInFlight][2];
    uint64_t frame_fence_values_[kFramesInFlight] = {};
    bool slot_has_timing_[kFramesInFlight] = {};
    ComPtr<ID3D12GraphicsCommandList> bridge_cmd_list_;
    ComPtr<ID3D12Fence> bridge_fence_;
    HANDLE bridge_fence_event_ = nullptr;
    uint64_t next_fence_value_ = 0;

    BridgeTelemetry telemetry_{};

    // capture (ini capture=N): the game's picture (before the filter) and the shown one, raw RGBA8
    // rows, read back per slot once its fence passed and written by a thread of their own so the
    // game's frame rate (and so its motion a frame) stays as it was
    struct CaptureJob {
        std::string path;
        std::vector<uint8_t> data;
    };
    uint32_t capture_left_ = 0;
    uint32_t capture_index_ = 0;
    std::string capture_dir_;
    ComPtr<ID3D12Resource> capture_rb_[kFramesInFlight][2];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT capture_fp_{};
    uint32_t capture_w_ = 0, capture_h_ = 0;
    int64_t capture_pending_[kFramesInFlight] = {-1, -1, -1};
    std::mutex capture_mutex_;
    std::condition_variable capture_cv_;
    std::deque<CaptureJob> capture_jobs_;
    bool capture_writer_ = false;
};

} // namespace dlss_nr::bridge
