#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "bridge/d3d12_bridge.h"
#include "dx12/device.h"

namespace {

LRESULT CALLBACK HostWndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wparam, lparam);
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    std::string shader_dir = "src/compute";
    std::string output_dir = "docs";
    uint32_t total_frames = 60;
    bool show_window = false;
    // 1280x720 and 1920x1080 at 0.16 both land on the 320x320 field, whose synthetic models
    // tools/hnet/export_onnx.py --field 320x320 --weights synthetic produces
    float render_scale = 0.16f;
    std::string model_tag = "nq-exp";
    bool synthetic = true;
    uint32_t interval = 1;      // --interval 2: the network spread over two frames
    bool async_setup = false;   // --async: the game's mode, where the first frames go out unprocessed
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;   // --bgra: the format most D3D12 games present
    // --reload: an ini saved a quarter of the way in (interval -> 2, warp_blend -> 0.35) must be
    // picked up and logged by the bridge (the game's way to A/B without hotkeys)
    bool reload_test = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--shader-dir" && i + 1 < argc) {
            shader_dir = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            total_frames = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--show-window") {
            show_window = true;
        } else if (arg == "--render-scale" && i + 1 < argc) {
            render_scale = std::stof(argv[++i]);
        } else if (arg == "--tag" && i + 1 < argc) {
            model_tag = argv[++i];
        } else if (arg == "--interval" && i + 1 < argc) {
            interval = std::min<uint32_t>(std::max<uint32_t>(static_cast<uint32_t>(std::stoul(argv[++i])), 1u), 4u);
        } else if (arg == "--async") {
            async_setup = true;
        } else if (arg == "--bgra") {
            format = DXGI_FORMAT_B8G8R8A8_UNORM;
        } else if (arg == "--real-weights") {
            synthetic = false;
        } else if (arg == "--reload") {
            reload_test = true;
        }
    }

    std::cout << "=== [M6 / M7] DLSS-NR Native DX12 Game Bridge & Swapchain Interception Test ===\n";

    dlss_nr::dx12::Device device;
    if (!device.Initialize(false, true)) {
        std::cerr << "[FAIL] Failed to initialize D3D12 device.\n";
        return 1;
    }

    ID3D12Device* d3d_dev = device.GetD3D12Device();
    IDXGIFactory6* factory = device.GetDXGIFactory();

    // Create game DIRECT command queue
    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> game_queue;
    if (FAILED(d3d_dev->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&game_queue)))) {
        std::cerr << "[FAIL] CreateCommandQueue failed.\n";
        return 1;
    }

    // Create Win32 game window
    WNDCLASSEXA wc{};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = HostWndProc;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "DLSS_NR_DX12_GAME_HOST";
    RegisterClassExA(&wc);

    uint32_t width = 1280;
    uint32_t height = 720;
    HWND hwnd = CreateWindowExA(
        0, wc.lpszClassName, "DLSS-NR Native DX12 Game Simulator",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        static_cast<int>(width), static_cast<int>(height),
        nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) {
        std::cerr << "[FAIL] CreateWindowExA failed.\n";
        return 1;
    }
    if (show_window) {
        ShowWindow(hwnd, SW_SHOW);
    }

    // Create 3-buffer Flip-Model SwapChain (3 frames in flight)
    static constexpr uint32_t kBufferCount = 3;
    DXGI_SWAP_CHAIN_DESC1 sc_desc{};
    sc_desc.Width = width;
    sc_desc.Height = height;
    sc_desc.Format = format;
    sc_desc.SampleDesc.Count = 1;
    sc_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc_desc.BufferCount = kBufferCount;
    sc_desc.Scaling = DXGI_SCALING_STRETCH;
    sc_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    Microsoft::WRL::ComPtr<IDXGISwapChain1> sc1;
    if (FAILED(factory->CreateSwapChainForHwnd(
            game_queue.Get(), hwnd, &sc_desc, nullptr, nullptr, &sc1))) {
        std::cerr << "[FAIL] CreateSwapChainForHwnd failed.\n";
        return 1;
    }
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapchain;
    sc1.As(&swapchain);

    // Initialize D3D12GameBridge and register game command queue
    dlss_nr::bridge::D3D12GameBridge bridge;
    bridge.SetShaderAndModelDirs(shader_dir, "", output_dir + "/dlss_nr_dx12_bridge.log");
    dlss_nr::neural::NeuralOptions neural_options;
    neural_options.render_scale = render_scale;
    neural_options.model_tag = model_tag;
    neural_options.synthetic_weights = synthetic;
    neural_options.interval = interval;
    bridge.SetNeuralOptions(neural_options);
    bridge.SetAsyncSetup(async_setup);
    const std::string log_path = output_dir + "/dlss_nr_dx12_bridge.log";
    std::streamoff log_start = 0;
    {
        std::ifstream existing(log_path, std::ios::binary | std::ios::ate);
        if (existing) log_start = existing.tellg();
    }
    char ini_full[MAX_PATH] = {};
    GetFullPathNameA((output_dir + "/reload_test.ini").c_str(), MAX_PATH, ini_full, nullptr);
    const std::string ini = ini_full;
    if (reload_test) {
        WritePrivateProfileStringA("neural", "interval", std::to_string(interval).c_str(), ini.c_str());
        WritePrivateProfileStringA("neural", "warp_blend", "0.5", ini.c_str());
        WritePrivateProfileStringA("neural", "capture", "0", ini.c_str());
        bridge.SetOptionsReloader(ini, [neural_options, ini] {
            dlss_nr::neural::NeuralOptions o = neural_options;
            char buf[64] = {};
            GetPrivateProfileStringA("neural", "interval", "1", buf, sizeof(buf), ini.c_str());
            o.interval = static_cast<uint32_t>(std::stoul(buf));
            GetPrivateProfileStringA("neural", "warp_blend", "0.5", buf, sizeof(buf), ini.c_str());
            o.warp_blend = std::stof(buf);
            GetPrivateProfileStringA("neural", "capture", "0", buf, sizeof(buf), ini.c_str());
            o.capture_frames = static_cast<uint32_t>(std::stoul(buf));
            return o;
        });
    }
    bridge.RegisterGameCommandQueue(game_queue.Get());

    // Create RTV descriptor heap and per-frame allocators for the simulated game
    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.NumDescriptors = kBufferCount;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtv_heap;
    d3d_dev->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap));
    UINT rtv_inc = d3d_dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    Microsoft::WRL::ComPtr<ID3D12Resource> backbuffers[kBufferCount];
    auto acquire_backbuffers = [&]() {
        D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        for (uint32_t i = 0; i < kBufferCount; ++i) {
            swapchain->GetBuffer(i, IID_PPV_ARGS(&backbuffers[i]));
            d3d_dev->CreateRenderTargetView(backbuffers[i].Get(), nullptr, rtv_handle);
            rtv_handle.ptr += rtv_inc;
        }
    };
    acquire_backbuffers();

    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> game_alloc;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> game_cmd;
    d3d_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&game_alloc));
    d3d_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, game_alloc.Get(), nullptr, IID_PPV_ARGS(&game_cmd));
    game_cmd->Close();

    Microsoft::WRL::ComPtr<ID3D12Fence> game_fence;
    d3d_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&game_fence));
    HANDLE fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    uint64_t fence_val = 0;

    auto wait_game_gpu = [&]() {
        ++fence_val;
        game_queue->Signal(game_fence.Get(), fence_val);
        if (game_fence->GetCompletedValue() < fence_val) {
            game_fence->SetEventOnCompletion(fence_val, fence_event);
            WaitForSingleObject(fence_event, INFINITE);
        }
    };

    for (uint32_t frame = 0; frame < total_frames; ++frame) {
        MSG msg{};
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }

        if (reload_test && frame == total_frames / 4) {
            WritePrivateProfileStringA("neural", "interval", "2", ini.c_str());
            WritePrivateProfileStringA("neural", "warp_blend", "0.35", ini.c_str());
            WritePrivateProfileStringA("neural", "capture", "12", ini.c_str());   // the next 12 frames captured
            Sleep(600);   // past the bridge's check period
        }

        // At midpoint frame, simulate dynamic game resolution change (720p -> 1080p)
        if (frame == total_frames / 2) {
            wait_game_gpu();
            for (auto& bb : backbuffers) bb.Reset();

            bridge.OnResizeBuffersPre(swapchain.Get());
            width = 1920;
            height = 1080;
            if (FAILED(swapchain->ResizeBuffers(kBufferCount, width, height, format, 0))) {
                std::cerr << "[FAIL] Swapchain ResizeBuffers to 1920x1080 failed.\n";
                return 1;
            }
            bridge.OnResizeBuffersPost(swapchain.Get(), width, height);
            acquire_backbuffers();
            std::cout << "[BRIDGE] Dynamic Swapchain Resize 1280x720 -> 1920x1080 at frame " << frame << " succeeded.\n";
        }

        uint32_t bb_idx = swapchain->GetCurrentBackBufferIndex();
        game_alloc->Reset();
        game_cmd->Reset(game_alloc.Get(), nullptr);

        // Transition backbuffer PRESENT -> RENDER_TARGET and render animated game frame
        D3D12_RESOURCE_BARRIER to_rt{};
        to_rt.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        to_rt.Transition.pResource = backbuffers[bb_idx].Get();
        to_rt.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        to_rt.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        to_rt.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        game_cmd->ResourceBarrier(1, &to_rt);

        D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        rtv_handle.ptr += static_cast<SIZE_T>(bb_idx) * rtv_inc;

        float phase = static_cast<float>(frame) / static_cast<float>(total_frames);
        const float clear_color[4] = {0.15f + 0.5f * phase, 0.25f, 0.65f - 0.4f * phase, 1.0f};
        game_cmd->ClearRenderTargetView(rtv_handle, clear_color, 0, nullptr);

        // Transition RENDER_TARGET -> PRESENT before calling Present
        D3D12_RESOURCE_BARRIER to_present{};
        to_present.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        to_present.Transition.pResource = backbuffers[bb_idx].Get();
        to_present.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        to_present.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        to_present.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        game_cmd->ResourceBarrier(1, &to_present);

        game_cmd->Close();
        ID3D12CommandList* lists[] = {game_cmd.Get()};
        game_queue->ExecuteCommandLists(1, lists);

        // Intercept Present with Native DX12 Game Bridge
        if (!bridge.OnPresentPre(swapchain.Get())) {
            std::cerr << "[FAIL] bridge.OnPresentPre failed at frame " << frame << "\n";
            return 1;
        }

        if (FAILED(swapchain->Present(0, 0))) {
            std::cerr << "[FAIL] swapchain->Present failed at frame " << frame << "\n";
            return 1;
        }

        wait_game_gpu();
        if (async_setup) Sleep(16);   // a game-like ~60 fps, so the background builds can land
    }

    auto tel = bridge.GetTelemetry();
    bridge.WriteTelemetryReportMarkdown(output_dir + "/bridge-report.md");
    bridge.Shutdown();

    CloseHandle(fence_event);
    DestroyWindow(hwnd);

    std::cout << "[BRIDGE] Telemetry Summary:\n"
              << "  Frames Intercepted        : " << tel.frames_intercepted << "\n"
              << "  Frames Processed on DX12  : " << tel.frames_processed << "\n"
              << "  Swapchain Resizes Handled : " << tel.resize_events << "\n"
              << "  Final Active Resolution   : " << tel.current_width << "x" << tel.current_height << "\n"
              << "  Avg GPU Inference Latency : " << std::fixed << std::setprecision(3) << tel.avg_gpu_inference_ms << " ms\n"
              << "  Network Input / Field     : " << tel.inner_width << "x" << tel.inner_height << " / "
              << tel.field_width << "x" << tel.field_height << "\n"
              << "  Avg Network Span          : " << tel.avg_network_ms << " ms\n"
              << "  Avg CPU Recording Overhead: " << tel.avg_cpu_record_ms << " ms\n"
              << "  Device Health Status      : " << (tel.device_healthy ? "HEALTHY (S_OK)" : "FAILED") << "\n";

    // async: frames go out unprocessed while each extent builds, but most must be processed
    const bool frames_ok = async_setup ? tel.frames_processed >= total_frames / 2 : tel.frames_processed == total_frames;
    if (!tel.device_healthy || !frames_ok || tel.resize_events != 1) {
        std::cerr << "[FAIL] Bridge telemetry validation failed.\n";
        return 1;
    }
    if (reload_test) {
        std::ifstream log(log_path, std::ios::binary);
        log.seekg(log_start);
        std::stringstream text;
        text << log.rdbuf();
        const std::string wanted = "dlss_nr_neural.ini reloaded: interval " + std::to_string(interval) + " -> 2 warp_blend 0.5 -> 0.35";
        if (text.str().find(wanted) == std::string::npos) {
            std::cerr << "[FAIL] the saved ini was not reloaded (no \"" << wanted << "\" in the log).\n";
            return 1;
        }
        std::cout << "[BRIDGE] ini reload picked up: " << wanted << "\n";
        // ... and capture=12: 12 pairs of raw RGBA8 frames (720p: the resize comes later) in the
        // folder the log names, the game's picture being the clear colour of its frame
        const std::string done = "Capture: 12 frames written to ";
        const size_t at = text.str().find(done);
        if (at == std::string::npos) {
            std::cerr << "[FAIL] no \"" << done << "\" in the log.\n";
            return 1;
        }
        const size_t end = text.str().find_first_of("\r\n", at);
        const std::string dir = text.str().substr(at + done.size(), end - at - done.size());
        const size_t bytes = 1280u * 720u * 4u;
        for (int k = 0; k < 12; ++k) {
            for (const char* kind : {"in", "out"}) {
                char name[32];
                std::snprintf(name, sizeof(name), "/%s_%04d.raw", kind, k);
                std::vector<uint8_t> data;
                for (int tries = 0; tries < 40 && data.size() != bytes; ++tries) {   // the writer thread
                    std::ifstream f(dir + name, std::ios::binary);
                    data.assign(std::istreambuf_iterator<char>(f), {});
                    if (data.size() != bytes) Sleep(50);
                }
                if (data.size() != bytes) {
                    std::cerr << "[FAIL] capture " << dir << name << ": " << data.size() << " bytes, want " << bytes << "\n";
                    return 1;
                }
                // the clear colour about a quarter in: ~(0.27, 0.25, 0.55) x 255
                if (std::string(kind) == "in" && (data[2] < 100 || data[1] < 50 || data[1] > 80)) {
                    std::cerr << "[FAIL] capture " << name << " is not the game's frame (" << int(data[0]) << ", "
                              << int(data[1]) << ", " << int(data[2]) << ")\n";
                    return 1;
                }
            }
        }
        std::cout << "[BRIDGE] capture: 12 frame pairs in " << dir << "\n";
    }

    std::cout << "=== [M6 / M7] Game Bridge & Swapchain Interception Test PASSED ===\n";
    return 0;
}
