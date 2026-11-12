#include "neural/dml_gemm.h"

#include <algorithm>

#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IDMLOperator, 0x26caae7a, 0x3081, 0x4633, 0x95, 0x81, 0x22, 0x6f, 0xbe, 0x57, 0x69, 0x5d)
__CRT_UUID_DECL(IDMLCompiledOperator, 0x6b15e56a, 0xbf5c, 0x4902, 0x92, 0xd8, 0xda, 0x3a, 0x65, 0x0a, 0xfe, 0xa4)
__CRT_UUID_DECL(IDMLOperatorInitializer, 0x427c1113, 0x435c, 0x469c, 0x86, 0x76, 0x4d, 0x5d, 0xd0, 0x72, 0xf8, 0x13)
__CRT_UUID_DECL(IDMLCommandRecorder, 0xe6857a76, 0x2e3e, 0x4fdd, 0xbf, 0xf4, 0x5d, 0x2b, 0xa1, 0x0f, 0xb4, 0x53)
__CRT_UUID_DECL(IDMLBindingTable, 0x29c687dc, 0xde74, 0x4e3b, 0xab, 0x00, 0x11, 0x68, 0xf2, 0xfc, 0x3c, 0xfc)
#endif

namespace dlss_nr::neural {

namespace {

// IDMLDispatchable::GetBindingProperties returns a struct by value, which the MinGW/clang
// COM ABI and MSVC disagree on: call the vtable slot with the hidden result pointer.
DML_BINDING_PROPERTIES BindingProperties(IDMLDispatchable* d) {
    DML_BINDING_PROPERTIES props{};
#if defined(_MSC_VER) && !defined(__MINGW32__)
    props = d->GetBindingProperties();
#else
    using Fn = DML_BINDING_PROPERTIES*(STDMETHODCALLTYPE*)(IDMLDispatchable*, DML_BINDING_PROPERTIES*);
    auto fn = reinterpret_cast<Fn>((*reinterpret_cast<void***>(d))[8]);
    fn(d, &props);
#endif
    return props;
}

// bytes a strided 4D half tensor spans, rounded up to 4 (DMLCalcBufferTensorSize)
uint64_t TensorBytes(const UINT sizes[4], const UINT strides[4]) {
    uint64_t last = 0;
    for (int i = 0; i < 4; ++i) last += static_cast<uint64_t>(sizes[i] - 1) * strides[i];
    return ((last + 1) * 2 + 3) & ~3ull;
}

} // namespace

bool DmlGemmSet::Initialize(ID3D12Device* device, IDMLDevice* dml) {
    device_ = device;
    dml_ = dml;
    if (!dml_ || FAILED(dml_->CreateCommandRecorder(IID_PPV_ARGS(&recorder_)))) {
        error_ = "DirectML command recorder";
        return false;
    }
    return true;
}

void DmlGemmSet::Clear() {
    ops_.clear();
    compiled_.clear();
    heap_.Reset();
    temporary_ = {};
}

int DmlGemmSet::Add(const Gemm& g) {
    Op op;
    op.gemm = g;
    const UINT a_row = g.a_row ? g.a_row : g.k, b_row = g.b_row ? g.b_row : g.k, o_row = g.out_row ? g.out_row : g.n;
    const UINT a_sizes[4] = {1, g.batch, g.m, g.k};
    const UINT a_strides[4] = {g.batch * (g.a_batch ? g.a_batch : g.m * a_row), g.a_batch ? g.a_batch : g.m * a_row,
                               a_row, 1};
    const UINT b_sizes[4] = {1, g.batch, g.n, g.k};      // stored (N, K), used transposed
    const UINT b_strides[4] = {g.batch * (g.b_batch ? g.b_batch : g.n * b_row), g.b_batch ? g.b_batch : g.n * b_row,
                               b_row, 1};
    const UINT o_sizes[4] = {1, g.batch, g.m, g.n};
    const UINT o_strides[4] = {g.batch * (g.out_batch ? g.out_batch : g.m * o_row),
                               g.out_batch ? g.out_batch : g.m * o_row, o_row, 1};
    op.a_bytes = TensorBytes(a_sizes, a_strides);
    op.b_bytes = TensorBytes(b_sizes, b_strides);
    op.out_bytes = TensorBytes(o_sizes, o_strides);

    std::vector<UINT> key;
    for (const UINT* part : {a_sizes, a_strides, b_sizes, b_strides, o_sizes, o_strides}) {
        key.insert(key.end(), part, part + 4);
    }
    for (size_t i = 0; i < compiled_.size(); ++i) {
        if (compiled_[i].key == key) {
            op.compiled = i;
            ops_.push_back(std::move(op));
            return static_cast<int>(ops_.size() - 1);
        }
    }

    // compiled in Finalize, all distinct shapes at once
    Compiled compiled;
    compiled.key = std::move(key);
    compiled.label = std::to_string(g.batch) + "x" + std::to_string(g.m) + "x" + std::to_string(g.k) + "x" +
                     std::to_string(g.n);
    op.compiled = compiled_.size();
    compiled_.push_back(std::move(compiled));
    ops_.push_back(std::move(op));
    return static_cast<int>(ops_.size() - 1);
}

bool DmlGemmSet::Compile(Compiled& c) const {
    const UINT* k = c.key.data();   // a sizes, a strides, b sizes, b strides, out sizes, out strides
    DML_BUFFER_TENSOR_DESC a{DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, k, k + 4, TensorBytes(k, k + 4), 16};
    DML_BUFFER_TENSOR_DESC b{DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, k + 8, k + 12,
                             TensorBytes(k + 8, k + 12), 16};
    DML_BUFFER_TENSOR_DESC o{DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, k + 16, k + 20,
                             TensorBytes(k + 16, k + 20), 16};
    DML_TENSOR_DESC ta{DML_TENSOR_TYPE_BUFFER, &a}, tb{DML_TENSOR_TYPE_BUFFER, &b}, to{DML_TENSOR_TYPE_BUFFER, &o};
    DML_GEMM_OPERATOR_DESC gemm{};
    gemm.ATensor = &ta;
    gemm.BTensor = &tb;
    gemm.OutputTensor = &to;
    gemm.TransA = DML_MATRIX_TRANSFORM_NONE;
    gemm.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;
    gemm.Alpha = 1.0f;
    gemm.Beta = 0.0f;
    DML_OPERATOR_DESC desc{DML_OPERATOR_GEMM, &gemm};
    ComPtr<IDMLOperator> dml_op;
    return SUCCEEDED(dml_->CreateOperator(&desc, IID_PPV_ARGS(&dml_op))) &&
           SUCCEEDED(dml_->CompileOperator(dml_op.Get(), DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION,
                                           IID_PPV_ARGS(&c.op)));
}

bool DmlGemmSet::Finalize(dx12::ResourceManager& rm, dx12::CommandQueue& upload) {
    if (ops_.empty()) return true;
    // One shape after another (5-20 ms each, ~130 ms for the native graph's 20). The docs call
    // IDMLDevice thread-safe, but compiling on 4 or 16 threads crashed (abort / access violation,
    // DirectML 1.15.4, Arc B580 driver 32.0.101.9150, 2026-11-04): do not parallelise this.
    for (auto& c : compiled_) {
        if (!Compile(c)) {
            error_ = "DirectML GEMM " + c.label + " rejected";
            return false;
        }
    }
    std::vector<IDMLCompiledOperator*> compiled;
    for (auto& c : compiled_) compiled.push_back(c.op.Get());
    ComPtr<IDMLOperatorInitializer> init;
    if (FAILED(dml_->CreateOperatorInitializer(static_cast<UINT>(compiled.size()), compiled.data(),
                                               IID_PPV_ARGS(&init)))) {
        error_ = "DirectML operator initializer";
        return false;
    }
    const DML_BINDING_PROPERTIES init_props = BindingProperties(init.Get());
    UINT descriptors = init_props.RequiredDescriptorCount;
    uint64_t temp = init_props.TemporaryResourceSize;
    for (auto& c : compiled_) {
        c.props = BindingProperties(c.op.Get());
        temp = std::max(temp, c.props.TemporaryResourceSize);
        if (c.props.PersistentResourceSize > 0 &&
            !rm.CreateDefaultBuffer(c.props.PersistentResourceSize, c.persistent, true)) {
            error_ = "DirectML persistent resource";
            return false;
        }
    }
    for (auto& op : ops_) descriptors += compiled_[op.compiled].props.RequiredDescriptorCount;
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = std::max(1u, descriptors);
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap_))) ||
        (temp > 0 && !rm.CreateDefaultBuffer(temp, temporary_, true))) {
        error_ = "DirectML descriptor heap / temporary resource";
        return false;
    }
    const UINT increment = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap_->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap_->GetGPUDescriptorHandleForHeapStart();

    // initialise every operator at once: persistent resources are the initializer's outputs
    DML_BINDING_TABLE_DESC td{init.Get(), cpu, gpu, init_props.RequiredDescriptorCount};
    ComPtr<IDMLBindingTable> init_table;
    if (FAILED(dml_->CreateBindingTable(&td, IID_PPV_ARGS(&init_table)))) {
        error_ = "DirectML initializer binding table";
        return false;
    }
    std::vector<DML_BUFFER_BINDING> persistent(compiled_.size());
    std::vector<DML_BINDING_DESC> persistent_desc(compiled_.size());
    for (size_t i = 0; i < compiled_.size(); ++i) {
        if (compiled_[i].persistent.resource) {
            persistent[i] = {compiled_[i].persistent.resource.Get(), 0, compiled_[i].persistent.byte_size};
            persistent_desc[i] = {DML_BINDING_TYPE_BUFFER, &persistent[i]};
        } else {
            persistent_desc[i] = {DML_BINDING_TYPE_NONE, nullptr};
        }
    }
    init_table->BindOutputs(static_cast<UINT>(persistent_desc.size()), persistent_desc.data());
    DML_BUFFER_BINDING temp_binding{temporary_.resource.Get(), 0, temporary_.byte_size};
    DML_BINDING_DESC temp_desc{DML_BINDING_TYPE_BUFFER, &temp_binding};
    if (init_props.TemporaryResourceSize > 0) init_table->BindTemporaryResource(&temp_desc);

    if (!upload.ResetCommandList()) return false;
    ID3D12GraphicsCommandList* cmd = upload.GetCommandList();
    if (temporary_.resource) dx12::ResourceManager::TransitionBuffer(cmd, temporary_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    for (auto& c : compiled_) {
        if (c.persistent.resource) {
            dx12::ResourceManager::TransitionBuffer(cmd, c.persistent, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    cmd->SetDescriptorHeaps(1, heaps);
    recorder_->RecordDispatch(cmd, init.Get(), init_table.Get());
    upload.ExecuteAndWait();

    // the execution binding tables, one per GEMM, after the initializer's range
    cpu.ptr += static_cast<SIZE_T>(init_props.RequiredDescriptorCount) * increment;
    gpu.ptr += static_cast<UINT64>(init_props.RequiredDescriptorCount) * increment;
    for (size_t i = 0; i < ops_.size(); ++i) {
        Op& op = ops_[i];
        Compiled& c = compiled_[op.compiled];
        DML_BINDING_TABLE_DESC d{c.op.Get(), cpu, gpu, c.props.RequiredDescriptorCount};
        if (FAILED(dml_->CreateBindingTable(&d, IID_PPV_ARGS(&op.table)))) {
            error_ = "DirectML binding table";
            return false;
        }
        DML_BUFFER_BINDING a{op.gemm.a->resource.Get(), op.gemm.a_offset, op.a_bytes};
        DML_BUFFER_BINDING b{op.gemm.b->resource.Get(), op.gemm.b_offset, op.b_bytes};
        DML_BUFFER_BINDING o{op.gemm.out->resource.Get(), op.gemm.out_offset, op.out_bytes};
        DML_BINDING_DESC inputs[3] = {{DML_BINDING_TYPE_BUFFER, &a}, {DML_BINDING_TYPE_BUFFER, &b},
                                      {DML_BINDING_TYPE_NONE, nullptr}};
        DML_BINDING_DESC output{DML_BINDING_TYPE_BUFFER, &o};
        op.table->BindInputs(3, inputs);
        op.table->BindOutputs(1, &output);
        if (c.persistent.resource) {
            DML_BUFFER_BINDING pb{c.persistent.resource.Get(), 0, c.persistent.byte_size};
            DML_BINDING_DESC pd{DML_BINDING_TYPE_BUFFER, &pb};
            op.table->BindPersistentResource(&pd);
        }
        if (c.props.TemporaryResourceSize > 0) {
            DML_BUFFER_BINDING tb{temporary_.resource.Get(), 0, c.props.TemporaryResourceSize};
            DML_BINDING_DESC tdesc{DML_BINDING_TYPE_BUFFER, &tb};
            op.table->BindTemporaryResource(&tdesc);
        }
        cpu.ptr += static_cast<SIZE_T>(c.props.RequiredDescriptorCount) * increment;
        gpu.ptr += static_cast<UINT64>(c.props.RequiredDescriptorCount) * increment;
    }
    return true;
}

void DmlGemmSet::Record(ID3D12GraphicsCommandList* cmd, int index) {
    Op& op = ops_[static_cast<size_t>(index)];
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    cmd->SetDescriptorHeaps(1, heaps);
    recorder_->RecordDispatch(cmd, compiled_[op.compiled].op.Get(), op.table.Get());
    dx12::ResourceManager::RecordUAVBarrier(cmd, nullptr);
}

} // namespace dlss_nr::neural
