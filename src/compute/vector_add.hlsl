cbuffer VectorParams : register(b0) {
    uint g_num_elements;
    float g_alpha;
    float g_beta;
    uint g_pad;
};

StructuredBuffer<float> InputA : register(t0);
StructuredBuffer<float> InputB : register(t1);
RWStructuredBuffer<float> OutputC : register(u0);

[numthreads(256, 1, 1)]
void CSVectorAdd(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx < g_num_elements) {
        OutputC[idx] = InputA[idx] + InputB[idx];
    }
}

[numthreads(256, 1, 1)]
void CSVectorMul(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx < g_num_elements) {
        OutputC[idx] = InputA[idx] * InputB[idx];
    }
}

[numthreads(256, 1, 1)]
void CSVectorChainedAddMul(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx < g_num_elements) {
        OutputC[idx] = mad(InputA[idx], g_alpha, InputB[idx] * g_beta);
    }
}
