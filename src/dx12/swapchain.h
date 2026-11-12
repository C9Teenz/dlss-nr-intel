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

#include "dx12/command_queue.h"
#include "dx12/resources.h"

namespace dlss_nr::dx12 {

using Microsoft::WRL::ComPtr;

class Swapchain {
public:
    Swapchain() = default;
    ~Swapchain();

    bool InitializeWindowed(IDXGIFactory4* factory,
                            ID3D12CommandQueue* queue,
                            uint32_t width,
                            uint32_t height,
                            const char* window_title = "DLSS-NR Native DX12 PoC");

    bool PresentFromTexture(CommandQueue& queue, GpuTexture& src_rgba8_tex);

    static bool SaveRGBA32FloatToPPM(const std::string& path,
                                     const float* rgba_pixels,
                                     uint32_t width,
                                     uint32_t height);

    static bool SaveRGBA32FloatToBMP(const std::string& path,
                                     const float* rgba_pixels,
                                     uint32_t width,
                                     uint32_t height);

private:
    static constexpr uint32_t kBufferCount = 2;
    HWND hwnd_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    ComPtr<IDXGISwapChain3> swapchain_;
    ComPtr<ID3D12Resource> backbuffers_[kBufferCount];
};

} // namespace dlss_nr::dx12
