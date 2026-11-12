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

#include "DirectML.h"
#include "dx12/command_queue.h"
#include "dx12/resources.h"

namespace dlss_nr::neural {

using Microsoft::WRL::ComPtr;

// Half-precision GEMMs as individual DirectML operators recorded into our own command list,
// where the driver runs them on the XMX units (MetaCommands). No ONNX Runtime in between:
// one dispatch per GEMM, interleaved with the hybrid's HLSL kernels.
//
//   out[b] (M x N) = A[b] (M x K) @ B[b]^T,   B stored (N x K) — the (out, in) layout the
//   weight blob already uses — for b in [0, batch), every operand with its own strides.
class DmlGemmSet {
public:
    struct Gemm {
        uint32_t batch = 1, m = 0, k = 0, n = 0;
        // strides in halves: rows and batches; 0 means packed
        uint32_t a_row = 0, a_batch = 0, b_row = 0, b_batch = 0, out_row = 0, out_batch = 0;
        dx12::GpuBuffer* a = nullptr;
        uint64_t a_offset = 0;            // bytes, multiple of 16
        dx12::GpuBuffer* b = nullptr;
        uint64_t b_offset = 0;
        dx12::GpuBuffer* out = nullptr;
        uint64_t out_offset = 0;
    };

    bool Initialize(ID3D12Device* device, IDMLDevice* dml);
    void Clear();
    // returns the GEMM's index. GEMMs of the same shape and strides share one compiled operator
    // (and its persistent resource), each with its own binding table: the native graph's 308
    // GEMMs are 20 distinct operators.
    int Add(const Gemm& gemm);
    // compiles the distinct operators (false with error() when DirectML rejects one), then the
    // descriptor heap, persistent and temporary resources, binding tables, and
    // the operator initialisation (executed and waited on `upload`)
    bool Finalize(dx12::ResourceManager& rm, dx12::CommandQueue& upload);
    void Record(ID3D12GraphicsCommandList* cmd, int index);
    size_t Count() const { return ops_.size(); }
    size_t CompiledCount() const { return compiled_.size(); }
    const std::string& error() const { return error_; }

private:
    struct Compiled {
        std::vector<UINT> key;            // sizes and strides of A, B and the output
        std::string label;                // batch x M x K x N, for errors
        ComPtr<IDMLCompiledOperator> op;
        dx12::GpuBuffer persistent;
        DML_BINDING_PROPERTIES props{};
    };
    bool Compile(Compiled& c) const;
    struct Op {
        Gemm gemm;
        size_t compiled = 0;              // index into compiled_
        ComPtr<IDMLBindingTable> table;
        uint64_t a_bytes = 0, b_bytes = 0, out_bytes = 0;
    };

    ID3D12Device* device_ = nullptr;
    ComPtr<IDMLDevice> dml_;
    ComPtr<IDMLCommandRecorder> recorder_;
    ComPtr<ID3D12DescriptorHeap> heap_;
    dx12::GpuBuffer temporary_;
    std::vector<Compiled> compiled_;
    std::vector<Op> ops_;
    std::string error_;
};

} // namespace dlss_nr::neural
