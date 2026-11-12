#include "dx12/swapchain.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

namespace dlss_nr::dx12 {

namespace {

LRESULT CALLBACK SimpleWndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wparam, lparam);
}

uint8_t FloatToUNorm8(float v) {
    float clamped = std::clamp(v, 0.0f, 1.0f);
    return static_cast<uint8_t>(std::lround(clamped * 255.0f));
}

} // namespace

Swapchain::~Swapchain() {
    for (auto& bb : backbuffers_) {
        bb.Reset();
    }
    swapchain_.Reset();
    if (hwnd_) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
}

bool Swapchain::InitializeWindowed(IDXGIFactory4* factory,
                                   ID3D12CommandQueue* queue,
                                   uint32_t width,
                                   uint32_t height,
                                   const char* window_title) {
    if (!factory || !queue || width == 0 || height == 0) return false;
    width_ = width;
    height_ = height;

    WNDCLASSEXA wc{};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = SimpleWndProc;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "DLSS_NR_DX12_POC_WND";
    RegisterClassExA(&wc);

    RECT rect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    hwnd_ = CreateWindowExA(
        0,
        wc.lpszClassName,
        window_title,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        rect.right - rect.left,
        rect.bottom - rect.top,
        nullptr,
        nullptr,
        wc.hInstance,
        nullptr);

    if (!hwnd_) return false;

    DXGI_SWAP_CHAIN_DESC1 sc_desc{};
    sc_desc.Width = width;
    sc_desc.Height = height;
    sc_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc_desc.Stereo = FALSE;
    sc_desc.SampleDesc.Count = 1;
    sc_desc.SampleDesc.Quality = 0;
    sc_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc_desc.BufferCount = kBufferCount;
    sc_desc.Scaling = DXGI_SCALING_STRETCH;
    sc_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sc_desc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    sc_desc.Flags = 0;

    ComPtr<IDXGISwapChain1> sc1;
    if (FAILED(factory->CreateSwapChainForHwnd(queue, hwnd_, &sc_desc, nullptr, nullptr, &sc1))) {
        return false;
    }

    if (FAILED(sc1.As(&swapchain_))) {
        return false;
    }

    for (uint32_t i = 0; i < kBufferCount; ++i) {
        if (FAILED(swapchain_->GetBuffer(i, IID_PPV_ARGS(&backbuffers_[i])))) {
            return false;
        }
    }

    return true;
}

bool Swapchain::PresentFromTexture(CommandQueue& queue, GpuTexture& src_rgba8_tex) {
    if (!swapchain_) return false;
    uint32_t frame_idx = swapchain_->GetCurrentBackBufferIndex();
    ID3D12Resource* backbuffer = backbuffers_[frame_idx].Get();
    if (!backbuffer) return false;

    if (!queue.ResetCommandList()) return false;
    auto* cmd = queue.GetCommandList();

    ResourceManager::TransitionTexture(cmd, src_rgba8_tex, D3D12_RESOURCE_STATE_COPY_SOURCE);

    D3D12_RESOURCE_BARRIER bb_to_copy{};
    bb_to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bb_to_copy.Transition.pResource = backbuffer;
    bb_to_copy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    bb_to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    bb_to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    cmd->ResourceBarrier(1, &bb_to_copy);

    cmd->CopyResource(backbuffer, src_rgba8_tex.resource.Get());

    D3D12_RESOURCE_BARRIER bb_to_present{};
    bb_to_present.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bb_to_present.Transition.pResource = backbuffer;
    bb_to_present.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    bb_to_present.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    bb_to_present.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    cmd->ResourceBarrier(1, &bb_to_present);

    queue.ExecuteAndWait();
    return SUCCEEDED(swapchain_->Present(0, 0));
}

bool Swapchain::SaveRGBA32FloatToPPM(const std::string& path,
                                     const float* rgba_pixels,
                                     uint32_t width,
                                     uint32_t height) {
    if (!rgba_pixels || width == 0 || height == 0) return false;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;

    out << "P6\n" << width << " " << height << "\n255\n";
    std::vector<uint8_t> row(static_cast<size_t>(width) * 3);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            row[x * 3 + 0] = FloatToUNorm8(rgba_pixels[idx + 0]);
            row[x * 3 + 1] = FloatToUNorm8(rgba_pixels[idx + 1]);
            row[x * 3 + 2] = FloatToUNorm8(rgba_pixels[idx + 2]);
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
    }
    return true;
}

bool Swapchain::SaveRGBA32FloatToBMP(const std::string& path,
                                     const float* rgba_pixels,
                                     uint32_t width,
                                     uint32_t height) {
    if (!rgba_pixels || width == 0 || height == 0) return false;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;

    const uint32_t row_stride = (width * 3 + 3) & ~3u;
    const uint32_t pixel_data_size = row_stride * height;
    const uint32_t file_size = 54 + pixel_data_size;

    uint8_t header[54] = {};
    header[0] = 'B';
    header[1] = 'M';
    *reinterpret_cast<uint32_t*>(&header[2]) = file_size;
    *reinterpret_cast<uint32_t*>(&header[10]) = 54;
    *reinterpret_cast<uint32_t*>(&header[14]) = 40;
    *reinterpret_cast<int32_t*>(&header[18]) = static_cast<int32_t>(width);
    *reinterpret_cast<int32_t*>(&header[22]) = -static_cast<int32_t>(height); // top-down
    *reinterpret_cast<uint16_t*>(&header[26]) = 1;
    *reinterpret_cast<uint16_t*>(&header[28]) = 24;
    *reinterpret_cast<uint32_t*>(&header[34]) = pixel_data_size;

    out.write(reinterpret_cast<const char*>(header), sizeof(header));

    std::vector<uint8_t> row(row_stride, 0);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            row[x * 3 + 0] = FloatToUNorm8(rgba_pixels[idx + 2]); // B
            row[x * 3 + 1] = FloatToUNorm8(rgba_pixels[idx + 1]); // G
            row[x * 3 + 2] = FloatToUNorm8(rgba_pixels[idx + 0]); // R
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
    }
    return true;
}

} // namespace dlss_nr::dx12
