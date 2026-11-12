#include "bridge/d3d12_bridge.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>

namespace dlss_nr::bridge {

namespace {

DXGI_FORMAT ToUAVCompatibleFormat(DXGI_FORMAT fmt) {
    switch (fmt) {
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        default:
            return fmt;
    }
}

bool SupportsTypedUAVStore(ID3D12Device* device, DXGI_FORMAT fmt) {
    if (!device || fmt == DXGI_FORMAT_UNKNOWN) return false;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT supp{};
    supp.Format = fmt;
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &supp, sizeof(supp)))) {
        return false;
    }
    return (supp.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0 &&
           (supp.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0;
}

const char* FormatToShortName(DXGI_FORMAT fmt) {
    switch (fmt) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return "R8G8B8A8_UNORM_SRGB";
        case DXGI_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return "B8G8R8A8_UNORM_SRGB";
        case DXGI_FORMAT_R10G10B10A2_UNORM: return "R10G10B10A2_UNORM";
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return "R32G32B32A32_FLOAT";
        default: return "DXGI_FORMAT_OTHER";
    }
}

void Transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    cmd->ResourceBarrier(1, &b);
}

} // namespace

D3D12GameBridge::~D3D12GameBridge() {
    Shutdown();
}

void D3D12GameBridge::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return;
    WaitExtentJob();
    WaitForAllFramesInFlight();
    neural_.reset();
    if (bridge_fence_event_) {
        CloseHandle(bridge_fence_event_);
        bridge_fence_event_ = nullptr;
    }
    bridge_fence_.Reset();
    bridge_cmd_list_.Reset();
    for (auto& slot : frame_allocators_) {
        slot[0].Reset();
        slot[1].Reset();
    }
    game_queue_.Reset();
    game_device_.Reset();
    initialized_ = false;
}

void D3D12GameBridge::SetShaderAndModelDirs(const std::string& /*shader_dir*/,
                                            const std::string& /*weights_path*/,
                                            const std::string& log_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    log_path_ = log_path;
}

void D3D12GameBridge::SetNeuralOptions(const neural::NeuralOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    options_ = options;
}

void D3D12GameBridge::SetOptionsReloader(const std::string& ini_path, std::function<neural::NeuralOptions()> read) {
    std::lock_guard<std::mutex> lock(mutex_);
    ini_path_ = ini_path;
    read_options_ = std::move(read);
    WIN32_FILE_ATTRIBUTE_DATA attributes{};
    ini_time_valid_ = GetFileAttributesExA(ini_path_.c_str(), GetFileExInfoStandard, &attributes) != 0;
    if (ini_time_valid_) ini_time_ = attributes.ftLastWriteTime;
}

// The ini saved while the game runs: for A/B in one scene without the hotkeys (a keyboard without
// F keys cannot reach F11 with Shift or Ctrl). Precision, model and synthetic need a restart.
void D3D12GameBridge::ReloadOptionsIfChanged() {
    if (!read_options_ || ini_path_.empty() || extent_job_.valid()) return;
    const uint64_t now = GetTickCount64();
    if (now - ini_checked_ms_ < 500) return;
    ini_checked_ms_ = now;
    WIN32_FILE_ATTRIBUTE_DATA attributes{};
    if (!GetFileAttributesExA(ini_path_.c_str(), GetFileExInfoStandard, &attributes)) return;
    if (ini_time_valid_ && CompareFileTime(&attributes.ftLastWriteTime, &ini_time_) == 0) return;
    ini_time_ = attributes.ftLastWriteTime;
    ini_time_valid_ = true;
    neural::NeuralOptions next;
    try {
        next = read_options_();
    } catch (...) {
        LogEvent("dlss_nr_neural.ini changed but has an unreadable value; the options stay as they were.");
        return;
    }
    const neural::NeuralOptions old = options_;
    std::ostringstream oss;
    oss << "dlss_nr_neural.ini reloaded:";
    bool changed = false;
    auto note = [&](const char* name, auto before, auto after) {
        if (before == after) return;
        oss << " " << name << " " << before << " -> " << after;
        changed = true;
    };
    note("render_scale", old.render_scale, next.render_scale);
    note("intensity", old.intensity, next.intensity);
    note("style", old.style, next.style);
    note("tone", old.tone, next.tone);
    note("structure", old.structure, next.structure);
    note("temporal", old.temporal, next.temporal);
    note("interval", old.interval, next.interval);
    note("warp", old.warp, next.warp);
    note("warp_model", old.warp_model, next.warp_model);
    note("warp_blend", old.warp_blend, next.warp_blend);
    note("warp_pyramid", old.warp_pyramid, next.warp_pyramid);
    note("warp_chain", old.warp_chain, next.warp_chain);
    note("warp_fade_start", old.warp_fade_start, next.warp_fade_start);
    note("warp_fade_full", old.warp_fade_full, next.warp_fade_full);
    note("warp_fade_none", old.warp_fade_none, next.warp_fade_none);
    note("stabilize", old.stabilize, next.stabilize);
    note("stabilize_tolerance", old.stabilize_tolerance, next.stabilize_tolerance);
    note("stabilize_motion", old.stabilize_motion, next.stabilize_motion);
    const bool restart = old.precise != next.precise || old.model_tag != next.model_tag ||
                         old.synthetic_weights != next.synthetic_weights;
    if (restart) oss << " (precision / model_tag / synthetic changed: they take effect after a restart)";
    if (next.capture_frames != old.capture_frames) {   // a capture starts when the number changes to N > 0
        options_.capture_frames = next.capture_frames;
        if (next.capture_frames > 0) StartCapture(next.capture_frames);
    }
    if (!changed && !restart) return;
    next.precise = old.precise;
    next.model_tag = old.model_tag;
    next.synthetic_weights = old.synthetic_weights;
    options_ = next;
    if (neural_ && changed) {
        WaitForAllFramesInFlight();   // the network's lists and buffers may still be in use
        neural_->SetLiveOptions(next);
        neural_->SetRenderScale(next.render_scale);   // a new extent on the next frame if it changed
        if (next.interval != neural_->GetOptions().interval) {
            const bool ok = neural_->SetInterval(next.interval);   // re-measures the parts: a short hitch
            if (!ok) oss << " (interval failed: " << neural_->LastError() << ")";
            else if (next.interval > 1) oss << "; " << neural_->PartSummary();
        }
        neural_->ResetHistory();
    }
    LogEvent(oss.str());
    Beep(1000, 80);
}

void D3D12GameBridge::LogEvent(const std::string& msg) {
    if (log_path_.empty()) return;
    std::ofstream out(log_path_, std::ios::out | std::ios::app);
    if (out.is_open()) {
        out << "[DLSS-NR-DX12-Bridge] " << msg << "\n";
    }
}

void D3D12GameBridge::RegisterGameCommandQueue(IUnknown* candidate_queue) {
    if (!candidate_queue) return;
    std::lock_guard<std::mutex> lock(mutex_);
    ComPtr<ID3D12CommandQueue> q;
    if (SUCCEEDED(candidate_queue->QueryInterface(IID_PPV_ARGS(&q))) && q) {
        D3D12_COMMAND_QUEUE_DESC desc = q->GetDesc();
        if (desc.Type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
            game_queue_ = q;
            LogEvent("Captured game DIRECT ID3D12CommandQueue.");
        }
    }
}

bool D3D12GameBridge::EnsureInitialized(IDXGISwapChain* swapchain) {
    if (initialized_) return true;
    if (init_failed_ || !swapchain) return false;

    ComPtr<ID3D12Resource> backbuffer0;
    if (FAILED(swapchain->GetBuffer(0, IID_PPV_ARGS(&backbuffer0))) || !backbuffer0) {
        // Non-D3D12 swapchain (e.g. D3D11 splash/video swapchain); ignore without failing permanently.
        return false;
    }
    if (FAILED(backbuffer0->GetDevice(IID_PPV_ARGS(&game_device_))) || !game_device_) {
        return false;
    }

    // The swapchain of a D3D12 game is created on its DIRECT queue; without one captured at
    // creation there is nothing safe to submit to.
    if (!game_queue_) {
        if (SUCCEEDED(swapchain->QueryInterface(IID_PPV_ARGS(&game_queue_))) && game_queue_) {
            LogEvent("Captured game DIRECT ID3D12CommandQueue via swapchain->QueryInterface.");
        } else if (SUCCEEDED(swapchain->GetDevice(IID_PPV_ARGS(&game_queue_))) && game_queue_) {
            LogEvent("Captured game DIRECT ID3D12CommandQueue via swapchain->GetDevice.");
        } else {
            LogEvent("No game ID3D12CommandQueue captured; the bridge stays inactive.");
            init_failed_ = true;
            return false;
        }
    }

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        for (auto& alloc : frame_allocators_[i]) {
            if (FAILED(game_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)))) {
                init_failed_ = true;
                return false;
            }
        }
        frame_fence_values_[i] = 0;
        slot_has_timing_[i] = false;
    }
    if (FAILED(game_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, frame_allocators_[0][0].Get(),
                                               nullptr, IID_PPV_ARGS(&bridge_cmd_list_)))) {
        init_failed_ = true;
        return false;
    }
    bridge_cmd_list_->Close();
    if (FAILED(game_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&bridge_fence_)))) {
        init_failed_ = true;
        return false;
    }
    bridge_fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    next_fence_value_ = 0;

    initialized_ = true;
    LogEvent("Initialized DLSS-NR neural bridge on the game ID3D12Device.");
    return true;
}

void D3D12GameBridge::WaitExtentJob() {
    if (extent_job_.valid()) FinishExtent(extent_job_.get(), extent_job_key_, extent_job_format_);
}

void D3D12GameBridge::FinishExtent(bool ok, const std::string& key, DXGI_FORMAT format) {
    if (!ok) {
        failed_extent_ = key;
        LogEvent("Neural pass disabled for " + key + ": " + neural_->LastError());
        return;
    }
    failed_extent_.clear();
    const neural::NeuralExtent& e = neural_->GetExtent();
    telemetry_.inner_width = e.inner_w;
    telemetry_.inner_height = e.inner_h;
    telemetry_.field_width = e.field_w;
    telemetry_.field_height = e.field_h;
    telemetry_.last_session_create_ms = neural_->LastSessionCreateMs();
    std::ostringstream oss;
    oss << "Neural extent " << e.output_w << "x" << e.output_h << " (" << FormatToShortName(format) << "): inner "
        << e.inner_w << "x" << e.inner_h << ", field " << e.field_w << "x" << e.field_h << ", "
        << (neural_->IsNative() ? "native" : neural_->IsHybrid() ? "hybrid" : "onnx") << " model "
        << neural_->ModelPath() << ", setup " << std::fixed << std::setprecision(0) << neural_->LastSessionCreateMs()
        << " ms";
    LogEvent(oss.str());
    if (neural_->GetOptions().interval > 1 && !neural_->PartSummary().empty()) LogEvent(neural_->PartSummary());
}

bool D3D12GameBridge::EnsureNeural() {
    if (neural_failed_) return false;
    if (neural_ && neural_queue_ == game_queue_.Get()) return true;
    WaitExtentJob();
    // the DML EP is bound to one queue at session creation: a new queue needs a new graph
    WaitForAllFramesInFlight();
    neural_ = std::make_unique<neural::NeuralGraph>();
    neural_queue_ = game_queue_.Get();
    if (!neural_->Initialize(game_device_.Get(), game_queue_.Get(), options_)) {
        LogEvent("NeuralGraph initialization failed: " + neural_->LastError());
        neural_.reset();
        neural_failed_ = true;
        return false;
    }
    std::ostringstream oss;
    oss << "NeuralGraph ready (ONNX Runtime DirectML on the game queue): model tag " << options_.model_tag
        << (options_.synthetic_weights ? " (synthetic)" : "") << ", render scale " << options_.render_scale
        << ", intensity " << options_.intensity << ", interval " << options_.interval << ", warp "
        << (options_.warp ? (options_.warp_model ? "model" : "head") : "off") << ", warp blend "
        << options_.warp_blend << ", warp pyramid " << (options_.warp_pyramid ? 1 : 0) << ", warp chain " << (options_.warp_chain ? 1 : 0) << ", warp fade "
        << options_.warp_fade_start << ".." << options_.warp_fade_full << (options_.warp_fade_none ? " to none" : " to regional") << ", precision " << (options_.precise ? "accurate" : "fast") << ", stabilize "
        << options_.stabilize << ", temporal " << (options_.temporal ? 1 : 0);
    LogEvent(oss.str());
    return true;
}

void D3D12GameBridge::WaitForAllFramesInFlight() {
    if (!initialized_ || !game_queue_ || !bridge_fence_ || !bridge_fence_event_) return;
    ++next_fence_value_;
    if (SUCCEEDED(game_queue_->Signal(bridge_fence_.Get(), next_fence_value_))) {
        if (bridge_fence_->GetCompletedValue() < next_fence_value_) {
            bridge_fence_->SetEventOnCompletion(next_fence_value_, bridge_fence_event_);
            WaitForSingleObject(bridge_fence_event_, INFINITE);
        }
    }
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        frame_fence_values_[i] = next_fence_value_;
    }
}

void D3D12GameBridge::OnResizeBuffersPre(IDXGISwapChain* /*swapchain*/) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return;
    WaitForAllFramesInFlight();
    ++telemetry_.resize_events;
    LogEvent("OnResizeBuffersPre: drained in-flight GPU work before swapchain resize.");
}

void D3D12GameBridge::OnResizeBuffersPost(IDXGISwapChain* /*swapchain*/,
                                          uint32_t new_width,
                                          uint32_t new_height) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return;
    telemetry_.current_width = new_width;
    telemetry_.current_height = new_height;
    if (neural_ && !extent_job_.valid()) neural_->ResetHistory();
    std::ostringstream oss;
    oss << "OnResizeBuffersPost: new resolution " << new_width << "x" << new_height;
    LogEvent(oss.str());
}

void D3D12GameBridge::ReadSlotTimings(uint32_t slot) {
    if (!neural_ || !slot_has_timing_[slot]) return;
    neural::NeuralTimings t;
    if (!neural_->ReadTimings(slot, t)) return;
    telemetry_.last_gpu_inference_ms = t.total_ms;
    auto smooth = [](double& avg, double v) { avg = avg == 0.0 ? v : 0.9 * avg + 0.1 * v; };
    smooth(telemetry_.avg_gpu_inference_ms, t.total_ms);
    smooth(telemetry_.avg_network_ms, t.network_ms);
}

bool D3D12GameBridge::OnPresentPre(IDXGISwapChain* swapchain) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++telemetry_.frames_intercepted;

    // Hotkeys: F11 toggles the neural pass in real time; for A/B in one scene, Shift+F11 cycles
    // the interval 1 -> 2 -> 4 and Ctrl+F11 the warp model -> head -> off (the native graph only).
    // A press is the key going down between two presents, from the key's current state: the
    // "pressed since the last call" bit is shared by every caller in the system (the game,
    // OptiScaler, overlays), so someone else can take it — in the last Spider-Man test F11
    // worked but no Shift/Ctrl+F11 was ever logged. The bit still catches a tap shorter than a
    // frame.
    const SHORT f11 = GetAsyncKeyState(VK_F11);
    const bool f11_down = (f11 & 0x8000) != 0;
    const bool f11_pressed = !f11_was_down_ && (f11_down || (f11 & 1) != 0);
    f11_was_down_ = f11_down;
    if (f11_pressed) {
        auto held = [](int a, int b, int c) {
            return ((GetAsyncKeyState(a) | GetAsyncKeyState(b) | GetAsyncKeyState(c)) & 0x8000) != 0;
        };
        const bool shift = held(VK_SHIFT, VK_LSHIFT, VK_RSHIFT);
        const bool ctrl = held(VK_CONTROL, VK_LCONTROL, VK_RCONTROL);
        if ((shift || ctrl) && neural_ && !extent_job_.valid()) {
            WaitForAllFramesInFlight();   // the network's lists and buffers may still be in use
            std::ostringstream oss;
            if (shift) {
                const uint32_t now = neural_->GetOptions().interval;
                const uint32_t next = now == 1 ? 2u : (now == 2 ? 4u : 1u);
                const bool ok = neural_->SetInterval(next);   // re-measures the parts: a short hitch
                oss << "Hotkey Shift+F11: interval " << next << (ok ? "" : " (failed: " + neural_->LastError() + ")");
                if (ok && next > 1) oss << "; " << neural_->PartSummary();
                Beep(500 + 200 * next, 120);
            } else {
                // model (the default) -> head (the old warp) -> off -> model
                const auto& o = neural_->GetOptions();
                const int mode = !o.warp ? 0 : (o.warp_model ? 2 : 1);
                const int next = mode == 2 ? 1 : (mode == 1 ? 0 : 2);
                neural_->SetWarp(next != 0);
                neural_->SetWarpModel(next == 2);
                neural_->ResetHistory();
                oss << "Hotkey Ctrl+F11: warp " << (next == 2 ? "model" : next == 1 ? "head" : "OFF");
                Beep(next == 2 ? 1200 : next == 1 ? 800 : 300, 120);
            }
            LogEvent(oss.str());
            return true;
        }
        inference_enabled_ = !inference_enabled_;
        telemetry_.inference_enabled = inference_enabled_;
        // the history is stale once frames went past unprocessed (a pending build resets it anyway)
        if (neural_ && !extent_job_.valid()) neural_->ResetHistory();
        Beep(inference_enabled_ ? 880 : 440, 120);
        LogEvent(inference_enabled_ ? "Hotkey F11: DLSS-NR neural pass ENABLED"
                                    : "Hotkey F11: DLSS-NR neural pass DISABLED");
    }
    ReloadOptionsIfChanged();
    if (!inference_enabled_) return true;
    if (!EnsureInitialized(swapchain)) return false;

    if (swapchain != last_swapchain_) {
        ComPtr<ID3D12CommandQueue> sc_queue;
        if (SUCCEEDED(swapchain->QueryInterface(IID_PPV_ARGS(&sc_queue))) && sc_queue &&
            game_queue_.Get() != sc_queue.Get()) {
            WaitForAllFramesInFlight();
            game_queue_ = sc_queue;
            LogEvent("Updated active ID3D12CommandQueue for new IDXGISwapChain.");
        }
        last_swapchain_ = swapchain;
    }
    if (!EnsureNeural()) return false;
    if (extent_job_.valid()) {
        // a build is running: neural_ belongs to it until it is done; the frame goes out as is
        if (extent_job_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return true;
        FinishExtent(extent_job_.get(), extent_job_key_, extent_job_format_);
    }

    ComPtr<IDXGISwapChain3> sc3;
    if (FAILED(swapchain->QueryInterface(IID_PPV_ARGS(&sc3))) || !sc3) return false;
    ComPtr<ID3D12Resource> backbuffer;
    if (FAILED(sc3->GetBuffer(sc3->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&backbuffer))) || !backbuffer) {
        return false;
    }

    const D3D12_RESOURCE_DESC bb_desc = backbuffer->GetDesc();
    const uint32_t width = static_cast<uint32_t>(bb_desc.Width);
    const uint32_t height = bb_desc.Height;
    const DXGI_FORMAT uav_fmt = ToUAVCompatibleFormat(bb_desc.Format);
    telemetry_.current_width = width;
    telemetry_.current_height = height;
    telemetry_.backbuffer_format = bb_desc.Format;
    if (!SupportsTypedUAVStore(game_device_.Get(), uav_fmt)) return false;

    // A new extent replaces textures, buffers and the ORT session: drain first, and do not
    // retry a missing model every frame.
    const neural::NeuralExtent& current = neural_->GetExtent();
    const neural::NeuralExtent wanted = neural::ComputeExtent(width, height, options_.render_scale);
    std::ostringstream extent_key;
    extent_key << width << "x" << height << "/" << static_cast<int>(uav_fmt);
    if (!(current == wanted) || neural_->GetInputTexture().format != uav_fmt || !neural_->IsReady()) {
        if (failed_extent_ == extent_key.str()) return true;
        WaitForAllFramesInFlight();   // the old textures and buffers may still be in use
        for (bool& has : slot_has_timing_) has = false;
        if (async_setup_) {
            neural::NeuralGraph* graph = neural_.get();
            extent_job_ = std::async(std::launch::async, [graph, width, height, uav_fmt] {
                return graph->EnsureExtent(width, height, uav_fmt);
            });
            extent_job_key_ = extent_key.str();
            extent_job_format_ = bb_desc.Format;
            LogEvent("Building the neural pass for " + extent_key.str() + " in the background.");
            return true;
        }
        FinishExtent(neural_->EnsureExtent(width, height, uav_fmt), extent_key.str(), bb_desc.Format);
        if (!failed_extent_.empty()) return true;
    }

    const uint32_t slot = static_cast<uint32_t>(telemetry_.frames_processed % kFramesInFlight);
    if (frame_fence_values_[slot] != 0 && bridge_fence_->GetCompletedValue() < frame_fence_values_[slot]) {
        bridge_fence_->SetEventOnCompletion(frame_fence_values_[slot], bridge_fence_event_);
        WaitForSingleObject(bridge_fence_event_, INFINITE);
    }
    ReadSlotTimings(slot);
    CollectCapture(slot);

    auto t_cpu0 = std::chrono::high_resolution_clock::now();
    auto& in_tex = neural_->GetInputTexture();
    auto& out_tex = neural_->GetOutputTexture();

    // A: back buffer -> input texture, pre-pass. The back buffer stays COPY_SOURCE until B.
    if (FAILED(frame_allocators_[slot][0]->Reset()) ||
        FAILED(bridge_cmd_list_->Reset(frame_allocators_[slot][0].Get(), nullptr))) {
        return false;
    }
    Transition(bridge_cmd_list_.Get(), backbuffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    dx12::ResourceManager::TransitionTexture(bridge_cmd_list_.Get(), in_tex, D3D12_RESOURCE_STATE_COPY_DEST);
    bridge_cmd_list_->CopyResource(in_tex.resource.Get(), backbuffer.Get());
    neural_->RecordPre(bridge_cmd_list_.Get(), slot);
    bridge_cmd_list_->Close();
    ID3D12CommandList* list_a[] = {bridge_cmd_list_.Get()};
    game_queue_->ExecuteCommandLists(1, list_a);

    // the network: ORT submits its DirectML work to the same queue, after A
    const bool network_ok = neural_->RunNetwork();
    if (!network_ok) LogEvent("RunNetwork failed: " + neural_->LastError());

    // B: post-pass (skipped if the network failed: the frame goes out unprocessed), copy back
    if (FAILED(frame_allocators_[slot][1]->Reset()) ||
        FAILED(bridge_cmd_list_->Reset(frame_allocators_[slot][1].Get(), nullptr))) {
        return false;
    }
    if (network_ok) {
        neural_->RecordPost(bridge_cmd_list_.Get(), slot);
        dx12::ResourceManager::TransitionTexture(bridge_cmd_list_.Get(), out_tex, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(bridge_cmd_list_.Get(), backbuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        bridge_cmd_list_->CopyResource(backbuffer.Get(), out_tex.resource.Get());
        Transition(bridge_cmd_list_.Get(), backbuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        if (capture_left_ > 0) RecordCapture(slot, in_tex, out_tex);
    } else {
        Transition(bridge_cmd_list_.Get(), backbuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
    }
    bridge_cmd_list_->Close();
    ID3D12CommandList* list_b[] = {bridge_cmd_list_.Get()};
    game_queue_->ExecuteCommandLists(1, list_b);

    ++next_fence_value_;
    game_queue_->Signal(bridge_fence_.Get(), next_fence_value_);
    frame_fence_values_[slot] = next_fence_value_;
    slot_has_timing_[slot] = network_ok;

    const double cpu_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_cpu0).count();
    telemetry_.avg_cpu_record_ms =
        telemetry_.avg_cpu_record_ms == 0.0 ? cpu_ms : 0.9 * telemetry_.avg_cpu_record_ms + 0.1 * cpu_ms;

    telemetry_.device_healthy = SUCCEEDED(game_device_->GetDeviceRemovedReason());
    ++telemetry_.frames_processed;
    if (telemetry_.frames_processed == 1 || (telemetry_.frames_processed % 300) == 0) {
        std::ostringstream oss;
        oss << "Frame " << telemetry_.frames_processed << " at " << width << "x" << height << " (inner "
            << telemetry_.inner_width << "x" << telemetry_.inner_height << "), GPU neural pass avg " << std::fixed
            << std::setprecision(2) << telemetry_.avg_gpu_inference_ms << " ms (network "
            << telemetry_.avg_network_ms << " ms), CPU " << telemetry_.avg_cpu_record_ms << " ms";
        LogEvent(oss.str());
    }
    return telemetry_.device_healthy;
}

// ini capture=N: a folder next to the log (dlss_nr_capture/<date-time>) for the next N frames
void D3D12GameBridge::StartCapture(uint32_t frames) {
    std::string dir = log_path_;
    const size_t cut = dir.find_last_of("/\\");
    dir = cut == std::string::npos ? std::string(".") : dir.substr(0, cut);
    SYSTEMTIME st{};
    GetLocalTime(&st);
    char stamp[32];
    std::snprintf(stamp, sizeof(stamp), "%04u%02u%02u-%02u%02u%02u", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute,
                  st.wSecond);
    CreateDirectoryA((dir + "/dlss_nr_capture").c_str(), nullptr);
    capture_dir_ = dir + "/dlss_nr_capture/" + stamp;
    if (!CreateDirectoryA(capture_dir_.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
        LogEvent("Capture: cannot create " + capture_dir_);
        return;
    }
    capture_left_ = frames;
    capture_index_ = 0;
    LogEvent("Capture: the next " + std::to_string(frames) + " frames -> " + capture_dir_);
    if (!capture_writer_) {   // one writer for the bridge's life (the bridge is never destroyed)
        capture_writer_ = true;
        std::thread([this] {
            for (;;) {
                CaptureJob job;
                {
                    std::unique_lock<std::mutex> lock(capture_mutex_);
                    capture_cv_.wait(lock, [this] { return !capture_jobs_.empty(); });
                    job = std::move(capture_jobs_.front());
                    capture_jobs_.pop_front();
                }
                std::ofstream out(job.path, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char*>(job.data.data()), static_cast<std::streamsize>(job.data.size()));
            }
        }).detach();
    }
}

// list B, after the copy back: this frame's game picture (the input texture) and the shown one
// (the output texture) into the slot's readback buffers; read when the slot comes round again
void D3D12GameBridge::RecordCapture(uint32_t slot, const dx12::GpuTexture& in_tex, const dx12::GpuTexture& out_tex) {
    const D3D12_RESOURCE_DESC desc = in_tex.resource->GetDesc();
    if (capture_w_ != in_tex.width || capture_h_ != in_tex.height) {
        for (auto& pair : capture_rb_) {
            for (auto& rb : pair) rb.Reset();
        }
        for (auto& p : capture_pending_) p = -1;
        capture_w_ = in_tex.width;
        capture_h_ = in_tex.height;
        UINT64 bytes = 0;
        game_device_->GetCopyableFootprints(&desc, 0, 1, 0, &capture_fp_, nullptr, nullptr, &bytes);
    }
    if (!capture_rb_[slot][0]) {
        UINT64 bytes = 0;
        game_device_->GetCopyableFootprints(&desc, 0, 1, 0, &capture_fp_, nullptr, nullptr, &bytes);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buf{};
        buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buf.Width = bytes;
        buf.Height = 1;
        buf.DepthOrArraySize = 1;
        buf.MipLevels = 1;
        buf.SampleDesc.Count = 1;
        buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        for (auto& rb : capture_rb_[slot]) {
            if (FAILED(game_device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COPY_DEST,
                                                             nullptr, IID_PPV_ARGS(&rb)))) {
                LogEvent("Capture: readback buffer failed; capture stopped.");
                capture_left_ = 0;
                return;
            }
        }
    }
    auto& input = const_cast<dx12::GpuTexture&>(in_tex);
    dx12::ResourceManager::TransitionTexture(bridge_cmd_list_.Get(), input, D3D12_RESOURCE_STATE_COPY_SOURCE);
    const dx12::GpuTexture* sources[2] = {&in_tex, &out_tex};   // out_tex is COPY_SOURCE already
    for (int k = 0; k < 2; ++k) {
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = capture_rb_[slot][k].Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = capture_fp_;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = sources[k]->resource.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;
        bridge_cmd_list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    capture_pending_[slot] = capture_index_++;
    --capture_left_;
}

// the slot's fence passed: its captured pair (if any) to the writer thread as tight RGBA8 rows,
// in_NNNN.raw (the game's picture) and out_NNNN.raw (shown), and meta.txt with the first
void D3D12GameBridge::CollectCapture(uint32_t slot) {
    if (capture_pending_[slot] < 0) return;
    const int64_t index = capture_pending_[slot];
    capture_pending_[slot] = -1;
    const size_t row = static_cast<size_t>(capture_w_) * 4;
    char name[32];
    for (int k = 0; k < 2; ++k) {
        void* mapped = nullptr;
        const D3D12_RANGE range{0, static_cast<SIZE_T>(capture_fp_.Offset + static_cast<UINT64>(capture_fp_.Footprint.RowPitch) * capture_h_)};
        if (FAILED(capture_rb_[slot][k]->Map(0, &range, &mapped)) || !mapped) continue;
        CaptureJob job;
        job.data.resize(row * capture_h_);
        for (uint32_t y = 0; y < capture_h_; ++y) {
            std::memcpy(job.data.data() + y * row,
                        static_cast<const uint8_t*>(mapped) + capture_fp_.Offset + static_cast<size_t>(y) * capture_fp_.Footprint.RowPitch,
                        row);
        }
        const D3D12_RANGE none{0, 0};
        capture_rb_[slot][k]->Unmap(0, &none);
        std::snprintf(name, sizeof(name), "/%s_%04lld.raw", k == 0 ? "in" : "out", static_cast<long long>(index));
        job.path = capture_dir_ + name;
        {
            std::lock_guard<std::mutex> lock(capture_mutex_);
            capture_jobs_.push_back(std::move(job));
        }
        capture_cv_.notify_one();
    }
    if (index == 0) {
        std::ofstream meta(capture_dir_ + "/meta.txt");
        const neural::NeuralOptions& o = neural_ ? neural_->GetOptions() : options_;
        meta << "width " << capture_w_ << "\nheight " << capture_h_ << "\nformat "
             << static_cast<int>(capture_fp_.Footprint.Format) << "\nrender_scale " << options_.render_scale
             << "\ninterval " << o.interval << "\nwarp " << o.warp << "\nwarp_chain " << o.warp_chain << "\nwarp_model " << o.warp_model << "\nwarp_blend "
             << o.warp_blend << "\nwarp_pyramid " << o.warp_pyramid << "\nwarp_fade " << o.warp_fade_start << " "
             << o.warp_fade_full << "\nprecision " << (o.precise ? "accurate" : "fast") << "\nintensity " << o.intensity
             << "\nprofile style " << o.style << " tone " << o.tone << " structure " << o.structure << "\n";
    }
    if (capture_left_ == 0) {
        bool waiting = false;
        for (int64_t p : capture_pending_) waiting = waiting || p >= 0;
        if (!waiting) {
            LogEvent("Capture: " + std::to_string(capture_index_) + " frames written to " + capture_dir_);
            Beep(1500, 150);
        }
    }
}

BridgeTelemetry D3D12GameBridge::GetTelemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return telemetry_;
}

bool D3D12GameBridge::WriteTelemetryReportMarkdown(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;

    out << "# DXGI / D3D12 Game Bridge Report (N3: DLSS-NR network)\n\n";
    out << "## 1. Per-frame pipeline\n\n";
    out << "| Step | API | Queue |\n| :--- | :--- | :--- |\n";
    out << "| Swapchain interception (`Present` / `ResizeBuffers`) | DXGI (`IDXGISwapChain3`) | — |\n";
    out << "| Back buffer copy + feature pre-pass (`CSNeuralPre`) | D3D12 compute | game DIRECT queue |\n";
    out << "| HNet, 71 blocks (ONNX FP16) | ONNX Runtime DirectML EP on the game's `ID3D12Device` | game DIRECT queue |\n";
    out << "| Composition + temporal gate + history (`CSNeuralPost`) + copy back | D3D12 compute | game DIRECT queue |\n\n";
    out << "## 2. Bridge runtime telemetry\n\n";
    out << "| Metric | Value |\n| :--- | :--- |\n";
    out << "| Frames intercepted (`Present`) | " << telemetry_.frames_intercepted << " |\n";
    out << "| Frames processed by the neural pass | " << telemetry_.frames_processed << " |\n";
    out << "| Swapchain resize events | " << telemetry_.resize_events << " |\n";
    out << "| Final output resolution | `" << telemetry_.current_width << "x" << telemetry_.current_height << "` |\n";
    out << "| Network input (render scale " << options_.render_scale << ") | `" << telemetry_.inner_width << "x"
        << telemetry_.inner_height << "`, field `" << telemetry_.field_width << "x" << telemetry_.field_height << "` |\n";
    out << "| Back buffer format | `" << FormatToShortName(telemetry_.backbuffer_format) << "` |\n";
    out << "| Model | `" << options_.model_tag << "`" << (options_.synthetic_weights ? " (synthetic weights)" : "") << " |\n";
    out << "| Rolling average GPU time, whole neural pass | `" << std::fixed << std::setprecision(2)
        << telemetry_.avg_gpu_inference_ms << " ms` |\n";
    out << "| Rolling average network span | `" << telemetry_.avg_network_ms << " ms` |\n";
    out << "| Rolling average CPU recording + ORT submit | `" << telemetry_.avg_cpu_record_ms << " ms` |\n";
    out << "| Last ORT session creation | `" << std::setprecision(0) << telemetry_.last_session_create_ms << " ms` |\n";
    out << "| D3D12 device health (`GetDeviceRemovedReason`) | `"
        << (telemetry_.device_healthy ? "Healthy (S_OK)" : "Device Removed!") << "` |\n";
    return true;
}

} // namespace dlss_nr::bridge
