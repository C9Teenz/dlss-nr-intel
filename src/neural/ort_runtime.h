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

#include <string>
#include <vector>

#include "DirectML.h"
#include "onnxruntime_c_api.h"
#include "dml_provider_factory.h"

namespace dlss_nr::neural {

using Microsoft::WRL::ComPtr;

// ONNX Runtime + DirectML, loaded at run time from a directory we name: the game's own
// process must not need them on its PATH, and the PoC builds without an import library.
// DirectML.dll is loaded first so onnxruntime.dll binds to that copy (1.15.x).
class OrtRuntime {
public:
    OrtRuntime() = default;
    ~OrtRuntime();

    OrtRuntime(const OrtRuntime&) = delete;
    OrtRuntime& operator=(const OrtRuntime&) = delete;

    // Each candidate directory is tried in order for onnxruntime.dll and, separately, for
    // DirectML.dll. Relative directories resolve against the process's current directory
    // and against the directory of the module holding this code. DirectML is required;
    // ONNX Runtime is optional (the native graph does without it): true when DirectML
    // loaded, and IsLoaded() tells whether ORT did too (`ort_error` says why not).
    bool Load(const std::vector<std::string>& ort_dirs,
              const std::vector<std::string>& directml_dirs,
              std::string* error, std::string* ort_error = nullptr);

    bool IsLoaded() const { return api_ != nullptr; }
    const OrtApi* Api() const { return api_; }
    const OrtDmlApi* Dml() const { return dml_api_; }
    OrtEnv* Env() const { return env_; }
    const std::string& OrtPath() const { return ort_path_; }
    const std::string& DirectMLPath() const { return directml_path_; }

    // A DirectML device on `device` (the game's), as the DML EP wants it.
    bool CreateDmlDevice(ID3D12Device* device, ComPtr<IDMLDevice>& out, std::string* error) const;

    // OrtStatus* -> false + message (and releases the status). nullptr -> true.
    bool Check(OrtStatus* status, const char* what, std::string* error) const;

private:
    HMODULE ort_module_ = nullptr;
    HMODULE directml_module_ = nullptr;
    const OrtApi* api_ = nullptr;
    const OrtDmlApi* dml_api_ = nullptr;
    OrtEnv* env_ = nullptr;
    std::string ort_path_;
    std::string directml_path_;
};

} // namespace dlss_nr::neural
