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
#include <vector>

#include "dx12/command_queue.h"

namespace dlss_nr::dx12 {

using Microsoft::WRL::ComPtr;

struct DescriptorHandle {
    uint32_t index = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
};

struct GpuBuffer {
    ComPtr<ID3D12Resource> resource;
    uint64_t byte_size = 0;
    D3D12_HEAP_TYPE heap_type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_STATES current_state = D3D12_RESOURCE_STATE_COMMON;
};

struct GpuTexture {
    ComPtr<ID3D12Resource> resource;
    uint32_t width = 0;
    uint32_t height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    D3D12_RESOURCE_STATES current_state = D3D12_RESOURCE_STATE_COMMON;
};

class ResourceManager {
public:
    ResourceManager() = default;
    ~ResourceManager() = default;

    bool Initialize(ID3D12Device* device, uint32_t max_descriptors = 1024);

    ID3D12DescriptorHeap* GetDescriptorHeap() const { return srv_uav_heap_.Get(); }
    DescriptorHandle AllocateDescriptor();
    void ResetDescriptors() { next_descriptor_index_ = 0; }

    bool CreateDefaultBuffer(uint64_t byte_size, GpuBuffer& out_buf, bool allow_uav = true);
    bool CreateUploadBuffer(uint64_t byte_size, GpuBuffer& out_buf);
    bool CreateReadbackBuffer(uint64_t byte_size, GpuBuffer& out_buf);

    bool CreateTexture2D(uint32_t width,
                         uint32_t height,
                         DXGI_FORMAT format,
                         GpuTexture& out_tex,
                         bool allow_uav = true);

    DescriptorHandle CreateBufferSRV(const GpuBuffer& buf, uint32_t num_elements, uint32_t stride_bytes);
    DescriptorHandle CreateBufferUAV(const GpuBuffer& buf, uint32_t num_elements, uint32_t stride_bytes);
    DescriptorHandle CreateTexture2DSRV(const GpuTexture& tex);
    DescriptorHandle CreateTexture2DUAV(const GpuTexture& tex);

    static void TransitionBuffer(ID3D12GraphicsCommandList* cmd,
                                 GpuBuffer& buf,
                                 D3D12_RESOURCE_STATES new_state);
    static void TransitionTexture(ID3D12GraphicsCommandList* cmd,
                                  GpuTexture& tex,
                                  D3D12_RESOURCE_STATES new_state);
    static void RecordUAVBarrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource = nullptr);

    bool UploadBufferData(CommandQueue& queue,
                          GpuBuffer& dst_default,
                          GpuBuffer& staging_upload,
                          const void* src_data,
                          uint64_t byte_size,
                          double* out_gpu_transfer_ms = nullptr);

    bool ReadbackBufferData(CommandQueue& queue,
                            GpuBuffer& src_default,
                            GpuBuffer& staging_readback,
                            void* dst_data,
                            uint64_t byte_size,
                            double* out_gpu_transfer_ms = nullptr);

    bool UploadTexture2DData(CommandQueue& queue,
                             GpuTexture& dst_tex,
                             GpuBuffer& staging_upload,
                             const void* src_pixels,
                             uint32_t src_row_pitch_bytes,
                             uint32_t bytes_per_pixel,
                             double* out_gpu_transfer_ms = nullptr);

    bool ReadbackTexture2DData(CommandQueue& queue,
                               GpuTexture& src_tex,
                               GpuBuffer& staging_readback,
                               void* dst_pixels,
                               uint32_t dst_row_pitch_bytes,
                               uint32_t bytes_per_pixel,
                               double* out_gpu_transfer_ms = nullptr);

    uint64_t GetTotalAllocatedBytes() const { return total_allocated_bytes_; }
    uint64_t GetPeakAllocatedBytes() const { return peak_allocated_bytes_; }

private:
    void TrackAllocation(uint64_t bytes);

    ID3D12Device* device_ = nullptr;
    ComPtr<ID3D12DescriptorHeap> srv_uav_heap_;
    uint32_t descriptor_size_ = 0;
    uint32_t max_descriptors_ = 0;
    uint32_t next_descriptor_index_ = 0;

    uint64_t total_allocated_bytes_ = 0;
    uint64_t peak_allocated_bytes_ = 0;
};

} // namespace dlss_nr::dx12
