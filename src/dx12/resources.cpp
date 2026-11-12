#include "dx12/resources.h"

#include <cstring>
#include <iostream>

namespace dlss_nr::dx12 {

namespace {

uint32_t AlignUp(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

bool ResourceManager::Initialize(ID3D12Device* device, uint32_t max_descriptors) {
    if (!device) return false;
    device_ = device;
    max_descriptors_ = max_descriptors;
    next_descriptor_index_ = 0;

    D3D12_DESCRIPTOR_HEAP_DESC hdesc{};
    hdesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hdesc.NumDescriptors = max_descriptors;
    hdesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hdesc.NodeMask = 0;

    if (FAILED(device_->CreateDescriptorHeap(&hdesc, IID_PPV_ARGS(&srv_uav_heap_)))) {
        std::cerr << "[ResourceManager] CreateDescriptorHeap failed\n";
        return false;
    }

    descriptor_size_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

void ResourceManager::TrackAllocation(uint64_t bytes) {
    total_allocated_bytes_ += bytes;
    if (total_allocated_bytes_ > peak_allocated_bytes_) {
        peak_allocated_bytes_ = total_allocated_bytes_;
    }
}

DescriptorHandle ResourceManager::AllocateDescriptor() {
    DescriptorHandle handle{};
    if (next_descriptor_index_ >= max_descriptors_) {
        next_descriptor_index_ = 0;
    }
    handle.index = next_descriptor_index_++;
    handle.cpu = srv_uav_heap_->GetCPUDescriptorHandleForHeapStart();
    handle.cpu.ptr += static_cast<SIZE_T>(handle.index) * descriptor_size_;
    handle.gpu = srv_uav_heap_->GetGPUDescriptorHandleForHeapStart();
    handle.gpu.ptr += static_cast<UINT64>(handle.index) * descriptor_size_;
    return handle;
}

bool ResourceManager::CreateDefaultBuffer(uint64_t byte_size, GpuBuffer& out_buf, bool allow_uav) {
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC rdesc{};
    rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rdesc.Alignment = 0;
    rdesc.Width = byte_size;
    rdesc.Height = 1;
    rdesc.DepthOrArraySize = 1;
    rdesc.MipLevels = 1;
    rdesc.Format = DXGI_FORMAT_UNKNOWN;
    rdesc.SampleDesc.Count = 1;
    rdesc.SampleDesc.Quality = 0;
    rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rdesc.Flags = allow_uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;

    out_buf.byte_size = byte_size;
    out_buf.heap_type = D3D12_HEAP_TYPE_DEFAULT;
    out_buf.current_state = D3D12_RESOURCE_STATE_COMMON;

    if (FAILED(device_->CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &rdesc,
            out_buf.current_state,
            nullptr,
            IID_PPV_ARGS(&out_buf.resource)))) {
        return false;
    }

    TrackAllocation(byte_size);
    return true;
}

bool ResourceManager::CreateUploadBuffer(uint64_t byte_size, GpuBuffer& out_buf) {
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC rdesc{};
    rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rdesc.Alignment = 0;
    rdesc.Width = byte_size;
    rdesc.Height = 1;
    rdesc.DepthOrArraySize = 1;
    rdesc.MipLevels = 1;
    rdesc.Format = DXGI_FORMAT_UNKNOWN;
    rdesc.SampleDesc.Count = 1;
    rdesc.SampleDesc.Quality = 0;
    rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rdesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    out_buf.byte_size = byte_size;
    out_buf.heap_type = D3D12_HEAP_TYPE_UPLOAD;
    out_buf.current_state = D3D12_RESOURCE_STATE_GENERIC_READ;

    if (FAILED(device_->CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &rdesc,
            out_buf.current_state,
            nullptr,
            IID_PPV_ARGS(&out_buf.resource)))) {
        return false;
    }

    TrackAllocation(byte_size);
    return true;
}

bool ResourceManager::CreateReadbackBuffer(uint64_t byte_size, GpuBuffer& out_buf) {
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_READBACK;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC rdesc{};
    rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rdesc.Alignment = 0;
    rdesc.Width = byte_size;
    rdesc.Height = 1;
    rdesc.DepthOrArraySize = 1;
    rdesc.MipLevels = 1;
    rdesc.Format = DXGI_FORMAT_UNKNOWN;
    rdesc.SampleDesc.Count = 1;
    rdesc.SampleDesc.Quality = 0;
    rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rdesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    out_buf.byte_size = byte_size;
    out_buf.heap_type = D3D12_HEAP_TYPE_READBACK;
    out_buf.current_state = D3D12_RESOURCE_STATE_COPY_DEST;

    if (FAILED(device_->CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &rdesc,
            out_buf.current_state,
            nullptr,
            IID_PPV_ARGS(&out_buf.resource)))) {
        return false;
    }

    TrackAllocation(byte_size);
    return true;
}

bool ResourceManager::CreateTexture2D(uint32_t width,
                                      uint32_t height,
                                      DXGI_FORMAT format,
                                      GpuTexture& out_tex,
                                      bool allow_uav) {
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC rdesc{};
    rdesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rdesc.Alignment = 0;
    rdesc.Width = width;
    rdesc.Height = height;
    rdesc.DepthOrArraySize = 1;
    rdesc.MipLevels = 1;
    rdesc.Format = format;
    rdesc.SampleDesc.Count = 1;
    rdesc.SampleDesc.Quality = 0;
    rdesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rdesc.Flags = allow_uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;

    out_tex.width = width;
    out_tex.height = height;
    out_tex.format = format;
    out_tex.current_state = D3D12_RESOURCE_STATE_COMMON;

    if (FAILED(device_->CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &rdesc,
            out_tex.current_state,
            nullptr,
            IID_PPV_ARGS(&out_tex.resource)))) {
        return false;
    }

    D3D12_RESOURCE_ALLOCATION_INFO alloc_info = device_->GetResourceAllocationInfo(0, 1, &rdesc);
    TrackAllocation(alloc_info.SizeInBytes);
    return true;
}

DescriptorHandle ResourceManager::CreateBufferSRV(const GpuBuffer& buf,
                                                  uint32_t num_elements,
                                                  uint32_t stride_bytes) {
    DescriptorHandle handle = AllocateDescriptor();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = 0;
    srv.Buffer.NumElements = num_elements;
    srv.Buffer.StructureByteStride = stride_bytes;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    device_->CreateShaderResourceView(buf.resource.Get(), &srv, handle.cpu);
    return handle;
}

DescriptorHandle ResourceManager::CreateBufferUAV(const GpuBuffer& buf,
                                                  uint32_t num_elements,
                                                  uint32_t stride_bytes) {
    DescriptorHandle handle = AllocateDescriptor();
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = 0;
    uav.Buffer.NumElements = num_elements;
    uav.Buffer.StructureByteStride = stride_bytes;
    uav.Buffer.CounterOffsetInBytes = 0;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    device_->CreateUnorderedAccessView(buf.resource.Get(), nullptr, &uav, handle.cpu);
    return handle;
}

DescriptorHandle ResourceManager::CreateTexture2DSRV(const GpuTexture& tex) {
    DescriptorHandle handle = AllocateDescriptor();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = tex.format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MostDetailedMip = 0;
    srv.Texture2D.MipLevels = 1;
    srv.Texture2D.PlaneSlice = 0;
    srv.Texture2D.ResourceMinLODClamp = 0.0f;
    device_->CreateShaderResourceView(tex.resource.Get(), &srv, handle.cpu);
    return handle;
}

DescriptorHandle ResourceManager::CreateTexture2DUAV(const GpuTexture& tex) {
    DescriptorHandle handle = AllocateDescriptor();
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = tex.format;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Texture2D.MipSlice = 0;
    uav.Texture2D.PlaneSlice = 0;
    device_->CreateUnorderedAccessView(tex.resource.Get(), nullptr, &uav, handle.cpu);
    return handle;
}

void ResourceManager::TransitionBuffer(ID3D12GraphicsCommandList* cmd,
                                       GpuBuffer& buf,
                                       D3D12_RESOURCE_STATES new_state) {
    if (buf.current_state == new_state) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = buf.resource.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = buf.current_state;
    barrier.Transition.StateAfter = new_state;
    cmd->ResourceBarrier(1, &barrier);
    buf.current_state = new_state;
}

void ResourceManager::TransitionTexture(ID3D12GraphicsCommandList* cmd,
                                        GpuTexture& tex,
                                        D3D12_RESOURCE_STATES new_state) {
    if (tex.current_state == new_state) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = tex.resource.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = tex.current_state;
    barrier.Transition.StateAfter = new_state;
    cmd->ResourceBarrier(1, &barrier);
    tex.current_state = new_state;
}

void ResourceManager::RecordUAVBarrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.UAV.pResource = resource;
    cmd->ResourceBarrier(1, &barrier);
}

bool ResourceManager::UploadBufferData(CommandQueue& queue,
                                       GpuBuffer& dst_default,
                                       GpuBuffer& staging_upload,
                                       const void* src_data,
                                       uint64_t byte_size,
                                       double* out_gpu_transfer_ms) {
    if (!src_data || byte_size > dst_default.byte_size || byte_size > staging_upload.byte_size) {
        return false;
    }

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, 0};
    if (FAILED(staging_upload.resource->Map(0, &read_range, &mapped)) || !mapped) {
        return false;
    }
    std::memcpy(mapped, src_data, static_cast<size_t>(byte_size));
    staging_upload.resource->Unmap(0, nullptr);

    if (!queue.ResetCommandList()) return false;
    auto* cmd = queue.GetCommandList();

    TransitionBuffer(cmd, dst_default, D3D12_RESOURCE_STATE_COPY_DEST);
    queue.WriteTimestampBegin(0);
    cmd->CopyBufferRegion(dst_default.resource.Get(), 0, staging_upload.resource.Get(), 0, byte_size);
    queue.WriteTimestampEnd(0);
    TransitionBuffer(cmd, dst_default, D3D12_RESOURCE_STATE_COMMON);
    queue.ResolveTimestamps(1);

    queue.ExecuteAndWait();
    if (out_gpu_transfer_ms) {
        *out_gpu_transfer_ms = queue.GetTimestampPairDurationMs(0);
    }
    return true;
}

bool ResourceManager::ReadbackBufferData(CommandQueue& queue,
                                         GpuBuffer& src_default,
                                         GpuBuffer& staging_readback,
                                         void* dst_data,
                                         uint64_t byte_size,
                                         double* out_gpu_transfer_ms) {
    if (!dst_data || byte_size > src_default.byte_size || byte_size > staging_readback.byte_size) {
        return false;
    }

    if (!queue.ResetCommandList()) return false;
    auto* cmd = queue.GetCommandList();

    TransitionBuffer(cmd, src_default, D3D12_RESOURCE_STATE_COPY_SOURCE);
    queue.WriteTimestampBegin(0);
    cmd->CopyBufferRegion(staging_readback.resource.Get(), 0, src_default.resource.Get(), 0, byte_size);
    queue.WriteTimestampEnd(0);
    TransitionBuffer(cmd, src_default, D3D12_RESOURCE_STATE_COMMON);
    queue.ResolveTimestamps(1);

    queue.ExecuteAndWait();
    if (out_gpu_transfer_ms) {
        *out_gpu_transfer_ms = queue.GetTimestampPairDurationMs(0);
    }

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(byte_size)};
    if (FAILED(staging_readback.resource->Map(0, &read_range, &mapped)) || !mapped) {
        return false;
    }
    std::memcpy(dst_data, mapped, static_cast<size_t>(byte_size));
    D3D12_RANGE write_range{0, 0};
    staging_readback.resource->Unmap(0, &write_range);
    return true;
}

bool ResourceManager::UploadTexture2DData(CommandQueue& queue,
                                          GpuTexture& dst_tex,
                                          GpuBuffer& staging_upload,
                                          const void* src_pixels,
                                          uint32_t src_row_pitch_bytes,
                                          uint32_t bytes_per_pixel,
                                          double* out_gpu_transfer_ms) {
    const uint32_t aligned_pitch = AlignUp(dst_tex.width * bytes_per_pixel, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    const uint64_t required_bytes = static_cast<uint64_t>(aligned_pitch) * dst_tex.height;
    if (staging_upload.byte_size < required_bytes) return false;

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, 0};
    if (FAILED(staging_upload.resource->Map(0, &read_range, &mapped)) || !mapped) {
        return false;
    }

    auto* dst_bytes = static_cast<uint8_t*>(mapped);
    const auto* src_bytes = static_cast<const uint8_t*>(src_pixels);
    const uint32_t copy_width_bytes = dst_tex.width * bytes_per_pixel;
    for (uint32_t y = 0; y < dst_tex.height; ++y) {
        std::memcpy(dst_bytes + static_cast<size_t>(y) * aligned_pitch,
                    src_bytes + static_cast<size_t>(y) * src_row_pitch_bytes,
                    copy_width_bytes);
    }
    staging_upload.resource->Unmap(0, nullptr);

    if (!queue.ResetCommandList()) return false;
    auto* cmd = queue.GetCommandList();

    TransitionTexture(cmd, dst_tex, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_TEXTURE_COPY_LOCATION dst_loc{};
    dst_loc.pResource = dst_tex.resource.Get();
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst_loc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource = staging_upload.resource.Get();
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_loc.PlacedFootprint.Offset = 0;
    src_loc.PlacedFootprint.Footprint.Format = dst_tex.format;
    src_loc.PlacedFootprint.Footprint.Width = dst_tex.width;
    src_loc.PlacedFootprint.Footprint.Height = dst_tex.height;
    src_loc.PlacedFootprint.Footprint.Depth = 1;
    src_loc.PlacedFootprint.Footprint.RowPitch = aligned_pitch;

    queue.WriteTimestampBegin(0);
    cmd->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
    queue.WriteTimestampEnd(0);
    TransitionTexture(cmd, dst_tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    queue.ResolveTimestamps(1);

    queue.ExecuteAndWait();
    if (out_gpu_transfer_ms) {
        *out_gpu_transfer_ms = queue.GetTimestampPairDurationMs(0);
    }
    return true;
}

bool ResourceManager::ReadbackTexture2DData(CommandQueue& queue,
                                            GpuTexture& src_tex,
                                            GpuBuffer& staging_readback,
                                            void* dst_pixels,
                                            uint32_t dst_row_pitch_bytes,
                                            uint32_t bytes_per_pixel,
                                            double* out_gpu_transfer_ms) {
    const uint32_t aligned_pitch = AlignUp(src_tex.width * bytes_per_pixel, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    const uint64_t required_bytes = static_cast<uint64_t>(aligned_pitch) * src_tex.height;
    if (staging_readback.byte_size < required_bytes) return false;

    if (!queue.ResetCommandList()) return false;
    auto* cmd = queue.GetCommandList();

    TransitionTexture(cmd, src_tex, D3D12_RESOURCE_STATE_COPY_SOURCE);

    D3D12_TEXTURE_COPY_LOCATION dst_loc{};
    dst_loc.pResource = staging_readback.resource.Get();
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_loc.PlacedFootprint.Offset = 0;
    dst_loc.PlacedFootprint.Footprint.Format = src_tex.format;
    dst_loc.PlacedFootprint.Footprint.Width = src_tex.width;
    dst_loc.PlacedFootprint.Footprint.Height = src_tex.height;
    dst_loc.PlacedFootprint.Footprint.Depth = 1;
    dst_loc.PlacedFootprint.Footprint.RowPitch = aligned_pitch;

    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource = src_tex.resource.Get();
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;

    queue.WriteTimestampBegin(0);
    cmd->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
    queue.WriteTimestampEnd(0);
    TransitionTexture(cmd, src_tex, D3D12_RESOURCE_STATE_COMMON);
    queue.ResolveTimestamps(1);

    queue.ExecuteAndWait();
    if (out_gpu_transfer_ms) {
        *out_gpu_transfer_ms = queue.GetTimestampPairDurationMs(0);
    }

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(required_bytes)};
    if (FAILED(staging_readback.resource->Map(0, &read_range, &mapped)) || !mapped) {
        return false;
    }

    const auto* src_bytes = static_cast<const uint8_t*>(mapped);
    auto* dst_bytes = static_cast<uint8_t*>(dst_pixels);
    const uint32_t copy_width_bytes = src_tex.width * bytes_per_pixel;
    for (uint32_t y = 0; y < src_tex.height; ++y) {
        std::memcpy(dst_bytes + static_cast<size_t>(y) * dst_row_pitch_bytes,
                    src_bytes + static_cast<size_t>(y) * aligned_pitch,
                    copy_width_bytes);
    }
    D3D12_RANGE write_range{0, 0};
    staging_readback.resource->Unmap(0, &write_range);
    return true;
}

} // namespace dlss_nr::dx12
