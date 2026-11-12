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

namespace dlss_nr::dx12 {

using Microsoft::WRL::ComPtr;

struct SubmissionTiming {
    double cpu_submit_ms = 0.0;
    double cpu_wait_ms = 0.0;
};

class CommandQueue {
public:
    CommandQueue() = default;
    ~CommandQueue();

    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;

    bool Initialize(ID3D12Device* device,
                    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT,
                    uint32_t max_timestamp_pairs = 256);

    bool ResetCommandList();
    void WriteTimestampBegin(uint32_t pair_index);
    void WriteTimestampEnd(uint32_t pair_index);
    void ResolveTimestamps(uint32_t pair_count);

    SubmissionTiming ExecuteAndWait();
    // Close and submit the command list without waiting (pair with WaitForIdle).
    void Execute();
    void WaitForIdle();

    double GetTimestampPairDurationMs(uint32_t pair_index) const;
    uint64_t GetTimestampFrequencyHz() const { return timestamp_freq_hz_; }

    ID3D12CommandQueue* GetQueue() const { return queue_.Get(); }
    ID3D12GraphicsCommandList* GetCommandList() const { return cmd_list_.Get(); }
    ID3D12CommandAllocator* GetAllocator() const { return allocator_.Get(); }

private:
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> cmd_list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_ = nullptr;
    uint64_t fence_value_ = 0;

    ComPtr<ID3D12QueryHeap> timestamp_heap_;
    ComPtr<ID3D12Resource> timestamp_readback_;
    uint32_t max_timestamp_pairs_ = 0;
    uint64_t timestamp_freq_hz_ = 0;
};

} // namespace dlss_nr::dx12
