#include "dx12/command_queue.h"

#include <chrono>
#include <iostream>

namespace dlss_nr::dx12 {

CommandQueue::~CommandQueue() {
    if (fence_event_) {
        CloseHandle(fence_event_);
        fence_event_ = nullptr;
    }
}

bool CommandQueue::Initialize(ID3D12Device* device,
                              D3D12_COMMAND_LIST_TYPE type,
                              uint32_t max_timestamp_pairs) {
    if (!device) return false;
    max_timestamp_pairs_ = max_timestamp_pairs;

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = type;
    qdesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    qdesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    qdesc.NodeMask = 0;

    if (FAILED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue_)))) {
        std::cerr << "[CommandQueue] CreateCommandQueue failed\n";
        return false;
    }

    if (FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator_)))) {
        std::cerr << "[CommandQueue] CreateCommandAllocator failed\n";
        return false;
    }

    if (FAILED(device->CreateCommandList(0, type, allocator_.Get(), nullptr, IID_PPV_ARGS(&cmd_list_)))) {
        std::cerr << "[CommandQueue] CreateCommandList failed\n";
        return false;
    }
    cmd_list_->Close();

    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) {
        std::cerr << "[CommandQueue] CreateFence failed\n";
        return false;
    }
    fence_value_ = 0;

    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event_) {
        std::cerr << "[CommandQueue] CreateEventW failed\n";
        return false;
    }

    if (FAILED(queue_->GetTimestampFrequency(&timestamp_freq_hz_)) || timestamp_freq_hz_ == 0) {
        timestamp_freq_hz_ = 10000000ULL;
    }

    if (max_timestamp_pairs_ > 0) {
        D3D12_QUERY_HEAP_DESC qh_desc{};
        qh_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qh_desc.Count = max_timestamp_pairs_ * 2;
        qh_desc.NodeMask = 0;
        if (FAILED(device->CreateQueryHeap(&qh_desc, IID_PPV_ARGS(&timestamp_heap_)))) {
            std::cerr << "[CommandQueue] CreateQueryHeap failed\n";
            return false;
        }

        D3D12_HEAP_PROPERTIES heap_props{};
        heap_props.Type = D3D12_HEAP_TYPE_READBACK;
        heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_props.CreationNodeMask = 1;
        heap_props.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC rdesc{};
        rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rdesc.Alignment = 0;
        rdesc.Width = static_cast<UINT64>(max_timestamp_pairs_) * 2 * sizeof(uint64_t);
        rdesc.Height = 1;
        rdesc.DepthOrArraySize = 1;
        rdesc.MipLevels = 1;
        rdesc.Format = DXGI_FORMAT_UNKNOWN;
        rdesc.SampleDesc.Count = 1;
        rdesc.SampleDesc.Quality = 0;
        rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rdesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        if (FAILED(device->CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &rdesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&timestamp_readback_)))) {
            std::cerr << "[CommandQueue] CreateCommittedResource for timestamp readback failed\n";
            return false;
        }
    }

    return true;
}

bool CommandQueue::ResetCommandList() {
    if (FAILED(allocator_->Reset())) return false;
    if (FAILED(cmd_list_->Reset(allocator_.Get(), nullptr))) return false;
    return true;
}

void CommandQueue::WriteTimestampBegin(uint32_t pair_index) {
    if (!timestamp_heap_ || pair_index >= max_timestamp_pairs_) return;
    cmd_list_->EndQuery(timestamp_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, pair_index * 2);
}

void CommandQueue::WriteTimestampEnd(uint32_t pair_index) {
    if (!timestamp_heap_ || pair_index >= max_timestamp_pairs_) return;
    cmd_list_->EndQuery(timestamp_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, pair_index * 2 + 1);
}

void CommandQueue::ResolveTimestamps(uint32_t pair_count) {
    if (!timestamp_heap_ || !timestamp_readback_ || pair_count == 0) return;
    if (pair_count > max_timestamp_pairs_) pair_count = max_timestamp_pairs_;
    cmd_list_->ResolveQueryData(
        timestamp_heap_.Get(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        0,
        pair_count * 2,
        timestamp_readback_.Get(),
        0);
}

SubmissionTiming CommandQueue::ExecuteAndWait() {
    SubmissionTiming timing{};
    auto t0 = std::chrono::high_resolution_clock::now();

    cmd_list_->Close();
    ID3D12CommandList* lists[] = {cmd_list_.Get()};
    queue_->ExecuteCommandLists(1, lists);

    auto t1 = std::chrono::high_resolution_clock::now();
    WaitForIdle();
    auto t2 = std::chrono::high_resolution_clock::now();

    timing.cpu_submit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing.cpu_wait_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    return timing;
}

void CommandQueue::Execute() {
    cmd_list_->Close();
    ID3D12CommandList* lists[] = {cmd_list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
}

void CommandQueue::WaitForIdle() {
    ++fence_value_;
    queue_->Signal(fence_.Get(), fence_value_);
    if (fence_->GetCompletedValue() < fence_value_) {
        fence_->SetEventOnCompletion(fence_value_, fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }
}

double CommandQueue::GetTimestampPairDurationMs(uint32_t pair_index) const {
    if (!timestamp_readback_ || pair_index >= max_timestamp_pairs_ || timestamp_freq_hz_ == 0) {
        return 0.0;
    }

    D3D12_RANGE read_range{};
    read_range.Begin = static_cast<SIZE_T>(pair_index) * 2 * sizeof(uint64_t);
    read_range.End = read_range.Begin + 2 * sizeof(uint64_t);

    void* mapped = nullptr;
    if (FAILED(timestamp_readback_->Map(0, &read_range, &mapped)) || !mapped) {
        return 0.0;
    }

    const auto* timestamps = reinterpret_cast<const uint64_t*>(mapped);
    uint64_t start = timestamps[pair_index * 2];
    uint64_t end = timestamps[pair_index * 2 + 1];

    D3D12_RANGE write_range{0, 0};
    timestamp_readback_->Unmap(0, &write_range);

    if (end <= start) return 0.0;
    double ticks = static_cast<double>(end - start);
    return (ticks * 1000.0) / static_cast<double>(timestamp_freq_hz_);
}

} // namespace dlss_nr::dx12
