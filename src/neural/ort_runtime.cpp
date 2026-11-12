#include "neural/ort_runtime.h"

#include <cstdio>
#include <filesystem>

__CRT_UUID_DECL(IDMLDevice, 0x6dbd6437, 0x96fd, 0x423f, 0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f)

namespace dlss_nr::neural {

namespace {

namespace fs = std::filesystem;

fs::path ThisModuleDirectory() {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&ThisModuleDirectory), &self);
    wchar_t buffer[MAX_PATH] = {};
    GetModuleFileNameW(self, buffer, MAX_PATH);
    return fs::path(buffer).parent_path();
}

// Every existing `<dir>/<file>` for the candidate dirs, as given and module-relative.
std::vector<fs::path> Candidates(const std::vector<std::string>& dirs, const wchar_t* file) {
    std::vector<fs::path> out;
    const fs::path module_dir = ThisModuleDirectory();
    for (const auto& dir : dirs) {
        fs::path p(dir);
        out.push_back(p / file);
        if (p.is_relative()) out.push_back(module_dir / p / file);
    }
    return out;
}

HMODULE LoadFirst(const std::vector<fs::path>& candidates, std::string& loaded_path) {
    for (const auto& path : candidates) {
        std::error_code ec;
        if (!fs::is_regular_file(path, ec)) continue;
        HMODULE module = LoadLibraryExW(path.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (module) {
            loaded_path = path.string();
            return module;
        }
    }
    return nullptr;
}

} // namespace

OrtRuntime::~OrtRuntime() {
    if (api_ && env_) api_->ReleaseEnv(env_);
    env_ = nullptr;
    // The modules stay loaded for the process lifetime: the DML EP keeps device objects
    // alive past session release, and unloading under a live game is not worth the risk.
}

bool OrtRuntime::Check(OrtStatus* status, const char* what, std::string* error) const {
    if (!status) return true;
    if (error) {
        *error = std::string(what) + ": " + api_->GetErrorMessage(status);
    }
    api_->ReleaseStatus(status);
    return false;
}

bool OrtRuntime::Load(const std::vector<std::string>& ort_dirs,
                      const std::vector<std::string>& directml_dirs,
                      std::string* error, std::string* ort_error) {
    if (api_) return true;

    if (!directml_module_) directml_module_ = LoadFirst(Candidates(directml_dirs, L"DirectML.dll"), directml_path_);
    if (!directml_module_) {
        // System32 carries an older DirectML; ORT 1.24 needs 1.15.x, so say so rather than
        // fail obscurely inside the EP.
        if (error) *error = "DirectML.dll (1.15.x) not found in the configured directories";
        return false;
    }
    // from here on a failure only means "no ONNX Runtime": DirectML is up
    error = ort_error;
    ort_module_ = LoadFirst(Candidates(ort_dirs, L"onnxruntime.dll"), ort_path_);
    if (!ort_module_) {
        if (error) *error = "onnxruntime.dll (DirectML build) not found in the configured directories";
        return true;
    }

    using PFN_OrtGetApiBase = const OrtApiBase*(ORT_API_CALL*)();
    auto get_api_base = reinterpret_cast<PFN_OrtGetApiBase>(GetProcAddress(ort_module_, "OrtGetApiBase"));
    if (!get_api_base) {
        if (error) *error = "OrtGetApiBase export missing in " + ort_path_;
        return true;
    }
    const OrtApiBase* base = get_api_base();
    api_ = base ? base->GetApi(ORT_API_VERSION) : nullptr;
    if (!api_) {
        if (error) {
            *error = "onnxruntime " + std::string(base ? base->GetVersionString() : "?") +
                     " does not provide API version " + std::to_string(ORT_API_VERSION);
        }
        return true;
    }

    const void* dml = nullptr;
    if (!Check(api_->GetExecutionProviderApi("DML", ORT_API_VERSION, &dml), "GetExecutionProviderApi(DML)", error) ||
        !dml) {
        api_ = nullptr;
        return true;
    }
    dml_api_ = static_cast<const OrtDmlApi*>(dml);

    if (!Check(api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dlss_nr_neural", &env_), "CreateEnv", error)) {
        api_ = nullptr;
    }
    return true;
}

bool OrtRuntime::CreateDmlDevice(ID3D12Device* device, ComPtr<IDMLDevice>& out, std::string* error) const {
    using PFN_DMLCreateDevice1 = HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, DML_FEATURE_LEVEL, REFIID, void**);
    auto create = reinterpret_cast<PFN_DMLCreateDevice1>(GetProcAddress(directml_module_, "DMLCreateDevice1"));
    if (!create) {
        if (error) *error = "DMLCreateDevice1 export missing in " + directml_path_;
        return false;
    }
    HRESULT hr = create(device, DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_5_0,
                        __uuidof(IDMLDevice), reinterpret_cast<void**>(out.ReleaseAndGetAddressOf()));
    if (FAILED(hr)) {
        if (error) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "DMLCreateDevice1 failed (HRESULT=0x%08lX)", static_cast<unsigned long>(hr));
            *error = buf;
        }
        return false;
    }
    return true;
}

} // namespace dlss_nr::neural
