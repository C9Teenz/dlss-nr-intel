#include "bridge/d3d12_bridge.h"

#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

namespace {

HMODULE g_this_module = nullptr;
HMODULE g_real_dxgi = nullptr;
std::mutex g_hook_mutex;
std::atomic<bool> g_asi_thread_started{false};
std::atomic<uint64_t> g_last_create_sc_tick{0};
std::atomic<bool> g_swapchain_hooks_active{false};
IDXGISwapChain* g_captured_swapchain = nullptr;
dlss_nr::bridge::D3D12GameBridge* g_bridge = nullptr;

using PFN_CreateDXGIFactory = HRESULT(WINAPI*)(REFIID, void**);
using PFN_CreateDXGIFactory1 = HRESULT(WINAPI*)(REFIID, void**);
using PFN_CreateDXGIFactory2 = HRESULT(WINAPI*)(UINT, REFIID, void**);
using PFN_DXGIGetDebugInterface1 = HRESULT(WINAPI*)(UINT, REFIID, void**);
using PFN_DeclareAdapterRemovalSupport = HRESULT(WINAPI*)();
using PFN_GenericPassthrough = uint64_t(WINAPI*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using PFN_D3D12CreateDevice = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);

using PFN_SwapChain_Present = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
using PFN_SwapChain_ResizeBuffers = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);
using PFN_SwapChain1_Present1 = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);

using PFN_Factory_CreateSwapChain = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
using PFN_Factory2_CreateSwapChainForHwnd = HRESULT(STDMETHODCALLTYPE*)(
    IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);

template <typename FnType>
struct VtableOrigMap {
    struct Entry {
        void** vtable = nullptr;
        FnType orig = nullptr;
    };
    static constexpr size_t kMaxEntries = 16;
    Entry entries[kMaxEntries] = {};
    size_t count = 0;

    void Set(void** vtbl, FnType fn) {
        for (size_t i = 0; i < count; ++i) {
            if (entries[i].vtable == vtbl) {
                entries[i].orig = fn;
                return;
            }
        }
        if (count < kMaxEntries) {
            entries[count].vtable = vtbl;
            entries[count].orig = fn;
            ++count;
        }
    }

    FnType Get(void** vtbl) const {
        for (size_t i = 0; i < count; ++i) {
            if (entries[i].vtable == vtbl) {
                return entries[i].orig;
            }
        }
        return (count > 0) ? entries[0].orig : nullptr;
    }

    void RestoreAll(size_t slot_index) {
        for (size_t i = 0; i < count; ++i) {
            void** vtbl = entries[i].vtable;
            FnType orig = entries[i].orig;
            if (vtbl && orig && vtbl[slot_index] != reinterpret_cast<void*>(orig)) {
                DWORD old_protect = 0;
                if (VirtualProtect(&vtbl[slot_index], sizeof(void*), PAGE_EXECUTE_READWRITE, &old_protect)) {
                    vtbl[slot_index] = reinterpret_cast<void*>(orig);
                    VirtualProtect(&vtbl[slot_index], sizeof(void*), old_protect, &old_protect);
                }
            }
        }
    }
};

VtableOrigMap<PFN_SwapChain_Present> g_present_map;
VtableOrigMap<PFN_SwapChain_ResizeBuffers> g_resize_buffers_map;
VtableOrigMap<PFN_SwapChain1_Present1> g_present1_map;
VtableOrigMap<PFN_Factory_CreateSwapChain> g_create_swapchain_map;
VtableOrigMap<PFN_Factory2_CreateSwapChainForHwnd> g_create_swapchain_hwnd_map;

thread_local bool t_in_present = false;
thread_local bool t_in_resize = false;
thread_local bool t_in_create_sc = false;

std::string ToLowerAscii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

// Walk PE Export Directory directly in memory to bypass any third-party GetProcAddress hooks
FARPROC GetExportAddressFromPEHeader(HMODULE hmod, const char* func_name) {
    if (!hmod || !func_name) return nullptr;
    const auto* base = reinterpret_cast<const uint8_t*>(hmod);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return GetProcAddress(hmod, func_name);

    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return GetProcAddress(hmod, func_name);

    const auto& exp_dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (exp_dir.VirtualAddress == 0 || exp_dir.Size == 0) return GetProcAddress(hmod, func_name);

    const auto* exp = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + exp_dir.VirtualAddress);
    const auto* names = reinterpret_cast<const uint32_t*>(base + exp->AddressOfNames);
    const auto* ords = reinterpret_cast<const uint16_t*>(base + exp->AddressOfNameOrdinals);
    const auto* funcs = reinterpret_cast<const uint32_t*>(base + exp->AddressOfFunctions);

    for (uint32_t i = 0; i < exp->NumberOfNames; ++i) {
        const char* name = reinterpret_cast<const char*>(base + names[i]);
        if (std::strcmp(name, func_name) == 0) {
            uint16_t ord_idx = ords[i];
            uint32_t fn_rva = funcs[ord_idx];
            if (fn_rva >= exp_dir.VirtualAddress && fn_rva < exp_dir.VirtualAddress + exp_dir.Size) {
                return GetProcAddress(hmod, func_name);
            }
            return reinterpret_cast<FARPROC>(const_cast<uint8_t*>(base + fn_rva));
        }
    }
    return GetProcAddress(hmod, func_name);
}

// Locate C:\Windows\System32\dxgi.dll in process memory using Toolhelp32 (bypasses LoadLibrary/Ntdll hooks)
HMODULE FindSystem32DxgiInMemory() {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (snap == INVALID_HANDLE_VALUE) return nullptr;

    MODULEENTRY32 me{};
    me.dwSize = sizeof(MODULEENTRY32);
    HMODULE found = nullptr;
    if (Module32First(snap, &me)) {
        do {
            if (me.hModule == g_this_module) continue;
            std::string mod_name = ToLowerAscii(me.szModule);
            std::string exe_path = ToLowerAscii(me.szExePath);
            if (mod_name == "dxgi.dll" &&
                (exe_path.find("system32\\dxgi.dll") != std::string::npos ||
                 exe_path.find("syswow64\\dxgi.dll") != std::string::npos)) {
                found = me.hModule;
                break;
            }
        } while (Module32Next(snap, &me));
    }
    CloseHandle(snap);
    return found;
}

std::string GetThisDllPath() {
    HMODULE hmod = g_this_module;
    if (!hmod) {
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&GetThisDllPath),
            &hmod);
    }
    if (hmod) {
        char buf[MAX_PATH] = {};
        if (GetModuleFileNameA(hmod, buf, MAX_PATH) > 0) {
            return std::string(buf);
        }
    }
    return "";
}

std::string GetThisDllDirectory() {
    std::string full_path = GetThisDllPath();
    if (!full_path.empty()) {
        std::filesystem::path p(full_path);
        std::filesystem::path parent = p.parent_path();
        if (ToLowerAscii(parent.filename().string()) == "plugins") {
            return parent.parent_path().string();
        }
        return parent.string();
    }
    return ".";
}

// dlss_nr_neural.ini next to the game executable (or this DLL), section [neural]:
//   render_scale = 0.5        network input as a fraction of the back buffer
//   intensity    = 1.0        blend of the model's picture against the frame
//   profile      = standard   standard | natural | cinematic | neutral
//   temporal     = 1          feed the previous prediction back
//   interval     = 1          2-4: the network spread over that many frames (about 1/N of the GPU
//                              time), its last head moved along the picture's motion
//   warp         = 1          interval > 1: move the last head along the motion (0: show it as it is)
//   warp_model   = 1          ... move its colour model and apply it to this frame's colour (0: the head)
//   warp_blend   = 0.5        ... a new head's weight against the last one carried onto its picture
//                              (1: the new head alone; lower calms the network's own shimmer more)
//   warp_pyramid = 1          ... also search the motion coarsely over +-96 field pixels (fast or
//                              reversing camera motion)
//   warp_chain   = 0          ... follow the motion frame to frame up to the reference's frame (fast camera)
//   warp_fade_start = 2, warp_fade_full = 6  ... the moved correction's detail fades to the
//                              regional model from / to this speed (field pixels a frame; full 0: never)
//   capture      = 0          changed to N > 0 while the game runs: the next N frames (game picture
//                              and shown picture, raw) into dlss_nr_capture/<date-time> next to the log
//   precision    = fast       accurate: the vendor's E4M3 roundings (closer, but flickers without a
//                              working temporal path; fast stays the game default for now)
//   stabilize    = 0.05       weight of a new head in the average along the picture's motion; 0 off
//   stabilize_tolerance = 1.0 colour change still averaged through (x 2/255 .. 8/255)
//   stabilize_motion = 0.5    least weight of a new head where anything moves (1 = no averaging)
//   model_tag    = nq-exp     numerics of the ONNX export (nq-exp, vk-pwl)
//   synthetic    = 0          use *_synthetic.onnx (plumbing only, no real weights)
// Models are looked up in <dir>/models/neural, onnxruntime.dll and DirectML.dll in <dir>/ort.
dlss_nr::neural::NeuralOptions ReadNeuralOptions(const std::string& dir) {
    dlss_nr::neural::NeuralOptions options;
    const std::string ini = dir + "/dlss_nr_neural.ini";
    char buf[256] = {};
    auto read = [&](const char* key, const char* fallback) {
        GetPrivateProfileStringA("neural", key, fallback, buf, sizeof(buf), ini.c_str());
        return std::string(buf);
    };
    options.render_scale = std::stof(read("render_scale", "0.5"));
    options.intensity = std::stof(read("intensity", "1.0"));
    if (!dlss_nr::neural::ApplyProfile(read("profile", "standard"), options)) {
        dlss_nr::neural::ApplyProfile("standard", options);
    }
    options.temporal = read("temporal", "1") != "0";
    options.interval = std::clamp<uint32_t>(static_cast<uint32_t>(std::stoul(read("interval", "1"))), 1u,
                                            dlss_nr::neural::NeuralGraph::kMaxInterval);
    options.warp = read("warp", "1") != "0";
    options.warp_model = read("warp_model", "1") != "0";
    options.warp_blend = std::clamp(std::stof(read("warp_blend", "0.5")), 0.05f, 1.0f);
    options.warp_pyramid = read("warp_pyramid", "1") != "0";
    options.warp_chain = read("warp_chain", "0") != "0";
    options.warp_fade_none = read("warp_fade_to", "regional") == "none";
    options.warp_fade_start = std::max(0.0f, std::stof(read("warp_fade_start", "2")));
    options.warp_fade_full = std::max(0.0f, std::stof(read("warp_fade_full", "6")));
    options.precise = read("precision", "fast") == "accurate";
    options.stabilize = std::stof(read("stabilize", "0.05"));
    options.stabilize_tolerance = std::stof(read("stabilize_tolerance", "1.0"));
    options.stabilize_motion = std::stof(read("stabilize_motion", "0.5"));
    options.model_tag = read("model_tag", options.model_tag.c_str());
    options.synthetic_weights = read("synthetic", "0") == "1";
    options.capture_frames = std::min<uint32_t>(static_cast<uint32_t>(std::stoul(read("capture", "0"))), 240u);
    options.model_dirs = {dir + "/models/neural"};
    options.ort_dirs = {dir + "/ort", dir};
    options.directml_dirs = {dir + "/ort", dir};
    return options;
}

void EnsureBridgeCreated() {
    if (g_bridge) return;
    std::string dll_dir = GetThisDllDirectory();
    g_bridge = new dlss_nr::bridge::D3D12GameBridge();
    std::string log_path = dll_dir + "/dlss_nr_dx12_bridge.log";
    g_bridge->SetShaderAndModelDirs(dll_dir + "/compute", dll_dir + "/models", log_path);
    g_bridge->SetOptionsReloader(dll_dir + "/dlss_nr_neural.ini", [dll_dir] { return ReadNeuralOptions(dll_dir); });
    try {
        g_bridge->SetNeuralOptions(ReadNeuralOptions(dll_dir));
    } catch (...) {
        g_bridge->LogEvent("dlss_nr_neural.ini has an unreadable value; using defaults.");
        dlss_nr::neural::NeuralOptions options;
        options.render_scale = 0.5f;
        options.model_dirs = {dll_dir + "/models/neural"};
        options.ort_dirs = {dll_dir + "/ort", dll_dir};
        options.directml_dirs = {dll_dir + "/ort", dll_dir};
        g_bridge->SetNeuralOptions(options);
    }
}

void EnsureRealDxgiLoaded() {
    EnsureBridgeCreated();
    if (g_real_dxgi) return;

    g_real_dxgi = FindSystem32DxgiInMemory();
    if (!g_real_dxgi) {
        char sys_dir[MAX_PATH] = {};
        GetSystemDirectoryA(sys_dir, MAX_PATH);
        std::string path = std::string(sys_dir) + "\\dxgi.dll";
        HMODULE loaded = LoadLibraryA(path.c_str());
        // Verify LoadLibrary didn't get redirected back to this module
        if (loaded && loaded != g_this_module) {
            HMODULE sys_mod = FindSystem32DxgiInMemory();
            g_real_dxgi = sys_mod ? sys_mod : loaded;
        }
    }
}

uint64_t ForwardGeneric(const char* name,
                        uint64_t a0, uint64_t a1, uint64_t a2,
                        uint64_t a3, uint64_t a4, uint64_t a5) {
    EnsureRealDxgiLoaded();
    if (!g_real_dxgi) return 0;
    auto fn = reinterpret_cast<PFN_GenericPassthrough>(GetExportAddressFromPEHeader(g_real_dxgi, name));
    return fn ? fn(a0, a1, a2, a3, a4, a5) : 0;
}

template <typename FnType>
void PatchVtableSlot(void** vtable, size_t index, FnType hook_fn, VtableOrigMap<FnType>& map) {
    if (!vtable || vtable[index] == reinterpret_cast<void*>(hook_fn)) return;
    DWORD old_protect = 0;
    if (VirtualProtect(&vtable[index], sizeof(void*), PAGE_EXECUTE_READWRITE, &old_protect)) {
        map.Set(vtable, reinterpret_cast<FnType>(vtable[index]));
        vtable[index] = reinterpret_cast<void*>(hook_fn);
        VirtualProtect(&vtable[index], sizeof(void*), old_protect, &old_protect);
    }
}

HRESULT STDMETHODCALLTYPE Hook_SwapChain_Present(IDXGISwapChain* This, UINT SyncInterval, UINT Flags) {
    void** vtbl = *reinterpret_cast<void***>(This);
    auto orig_fn = g_present_map.Get(vtbl);
    if (!orig_fn) return E_FAIL;

    if (!t_in_present) {
        t_in_present = true;
        if (g_bridge && (Flags & DXGI_PRESENT_TEST) == 0) {
            g_bridge->OnPresentPre(This);
        }
        HRESULT hr = orig_fn(This, SyncInterval, Flags);
        t_in_present = false;
        return hr;
    }
    return orig_fn(This, SyncInterval, Flags);
}

HRESULT STDMETHODCALLTYPE Hook_SwapChain1_Present1(IDXGISwapChain1* This,
                                                   UINT SyncInterval,
                                                   UINT PresentFlags,
                                                   const DXGI_PRESENT_PARAMETERS* pPresentParameters) {
    void** vtbl = *reinterpret_cast<void***>(This);
    auto orig_fn = g_present1_map.Get(vtbl);
    if (!orig_fn) return E_FAIL;

    if (!t_in_present) {
        t_in_present = true;
        if (g_bridge && (PresentFlags & DXGI_PRESENT_TEST) == 0) {
            g_bridge->OnPresentPre(This);
        }
        HRESULT hr = orig_fn(This, SyncInterval, PresentFlags, pPresentParameters);
        t_in_present = false;
        return hr;
    }
    return orig_fn(This, SyncInterval, PresentFlags, pPresentParameters);
}

HRESULT STDMETHODCALLTYPE Hook_SwapChain_ResizeBuffers(IDXGISwapChain* This,
                                                       UINT BufferCount,
                                                       UINT Width,
                                                       UINT Height,
                                                       DXGI_FORMAT NewFormat,
                                                       UINT SwapChainFlags) {
    void** vtbl = *reinterpret_cast<void***>(This);
    auto orig_fn = g_resize_buffers_map.Get(vtbl);
    if (!orig_fn) return E_FAIL;

    if (!t_in_resize) {
        t_in_resize = true;
        if (g_bridge) g_bridge->OnResizeBuffersPre(This);
        HRESULT hr = orig_fn(This, BufferCount, Width, Height, NewFormat, SwapChainFlags);
        if (SUCCEEDED(hr) && g_bridge) {
            g_bridge->OnResizeBuffersPost(This, Width, Height);
        }
        t_in_resize = false;
        return hr;
    }
    return orig_fn(This, BufferCount, Width, Height, NewFormat, SwapChainFlags);
}

bool IsGameExecutable() {
    char exe_buf[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, exe_buf, MAX_PATH) == 0) {
        return true;
    }
    std::string exe_lower = ToLowerAscii(exe_buf);
    if (exe_lower.find("crs-handler") != std::string::npos ||
        exe_lower.find("crs-video") != std::string::npos ||
        exe_lower.find("crashreport") != std::string::npos ||
        exe_lower.find("unitycrashhandler") != std::string::npos) {
        return false;
    }
    return true;
}

void RestoreAllSwapChainHooksLocked() {
    g_present_map.RestoreAll(8);
    g_resize_buffers_map.RestoreAll(13);
    g_present1_map.RestoreAll(22);
}

void InstallSwapChainHooks(IDXGISwapChain* swapchain) {
    if (!swapchain) return;
    std::lock_guard<std::mutex> lock(g_hook_mutex);
    void** vtbl = *reinterpret_cast<void***>(swapchain);
    PatchVtableSlot(vtbl, 8, &Hook_SwapChain_Present, g_present_map);
    PatchVtableSlot(vtbl, 13, &Hook_SwapChain_ResizeBuffers, g_resize_buffers_map);

    Microsoft::WRL::ComPtr<IDXGISwapChain1> sc1;
    if (SUCCEEDED(swapchain->QueryInterface(IID_PPV_ARGS(&sc1))) && sc1) {
        void** vtbl1 = *reinterpret_cast<void***>(sc1.Get());
        PatchVtableSlot(vtbl1, 22, &Hook_SwapChain1_Present1, g_present1_map);
    }
}

bool IsD3D12CommandQueue(IUnknown* pDevice) {
    if (!pDevice) return false;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> q;
    return SUCCEEDED(pDevice->QueryInterface(IID_PPV_ARGS(&q))) && q != nullptr;
}

HRESULT STDMETHODCALLTYPE Hook_Factory_CreateSwapChain(IDXGIFactory* This,
                                                       IUnknown* pDevice,
                                                       DXGI_SWAP_CHAIN_DESC* pDesc,
                                                       IDXGISwapChain** ppSwapChain) {
    void** vtbl = *reinterpret_cast<void***>(This);
    auto orig_fn = g_create_swapchain_map.Get(vtbl);
    if (!orig_fn) return E_FAIL;

    if (!t_in_create_sc) {
        t_in_create_sc = true;
        bool is_d3d12 = IsD3D12CommandQueue(pDevice);
        if (is_d3d12) {
            {
                std::lock_guard<std::mutex> lock(g_hook_mutex);
                RestoreAllSwapChainHooksLocked();
                g_swapchain_hooks_active.store(false);
                g_last_create_sc_tick.store(GetTickCount64());
            }
            if (g_bridge) {
                g_bridge->RegisterGameCommandQueue(pDevice);
            }
        }
        HRESULT hr = orig_fn(This, pDevice, pDesc, ppSwapChain);
        if (SUCCEEDED(hr) && is_d3d12 && ppSwapChain && *ppSwapChain) {
            std::lock_guard<std::mutex> lock(g_hook_mutex);
            g_captured_swapchain = *ppSwapChain;
            g_last_create_sc_tick.store(GetTickCount64());
            g_swapchain_hooks_active.store(false);
        }
        t_in_create_sc = false;
        return hr;
    }
    return orig_fn(This, pDevice, pDesc, ppSwapChain);
}

HRESULT STDMETHODCALLTYPE Hook_Factory2_CreateSwapChainForHwnd(
    IDXGIFactory2* This,
    IUnknown* pDevice,
    HWND hWnd,
    const DXGI_SWAP_CHAIN_DESC1* pDesc,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFullscreenDesc,
    IDXGIOutput* pRestrictToOutput,
    IDXGISwapChain1** ppSwapChain) {
    void** vtbl = *reinterpret_cast<void***>(This);
    auto orig_fn = g_create_swapchain_hwnd_map.Get(vtbl);
    if (!orig_fn) return E_FAIL;

    if (!t_in_create_sc) {
        t_in_create_sc = true;
        bool is_d3d12 = IsD3D12CommandQueue(pDevice);
        if (is_d3d12) {
            {
                std::lock_guard<std::mutex> lock(g_hook_mutex);
                RestoreAllSwapChainHooksLocked();
                g_swapchain_hooks_active.store(false);
                g_last_create_sc_tick.store(GetTickCount64());
            }
            if (g_bridge) {
                g_bridge->RegisterGameCommandQueue(pDevice);
            }
        }
        HRESULT hr = orig_fn(This, pDevice, hWnd, pDesc, pFullscreenDesc, pRestrictToOutput, ppSwapChain);
        if (SUCCEEDED(hr) && is_d3d12 && ppSwapChain && *ppSwapChain) {
            std::lock_guard<std::mutex> lock(g_hook_mutex);
            g_captured_swapchain = *ppSwapChain;
            g_last_create_sc_tick.store(GetTickCount64());
            g_swapchain_hooks_active.store(false);
        }
        t_in_create_sc = false;
        return hr;
    }
    return orig_fn(This, pDevice, hWnd, pDesc, pFullscreenDesc, pRestrictToOutput, ppSwapChain);
}

void InstallFactoryHooks(IUnknown* factory_unk) {
    if (!factory_unk) return;
    std::lock_guard<std::mutex> lock(g_hook_mutex);

    Microsoft::WRL::ComPtr<IDXGIFactory> f0;
    if (SUCCEEDED(factory_unk->QueryInterface(IID_PPV_ARGS(&f0))) && f0) {
        void** vtbl = *reinterpret_cast<void***>(f0.Get());
        PatchVtableSlot(vtbl, 10, &Hook_Factory_CreateSwapChain, g_create_swapchain_map);
    }

    Microsoft::WRL::ComPtr<IDXGIFactory2> f2;
    if (SUCCEEDED(factory_unk->QueryInterface(IID_PPV_ARGS(&f2))) && f2) {
        void** vtbl2 = *reinterpret_cast<void***>(f2.Get());
        PatchVtableSlot(vtbl2, 15, &Hook_Factory2_CreateSwapChainForHwnd, g_create_swapchain_hwnd_map);
    }
}

bool InstallRealDxgiFactoryHooks() {
    EnsureRealDxgiLoaded();
    if (!g_real_dxgi) return false;
    auto fn_create_fac = reinterpret_cast<PFN_CreateDXGIFactory1>(
        GetExportAddressFromPEHeader(g_real_dxgi, "CreateDXGIFactory1"));
    if (!fn_create_fac) return false;

    Microsoft::WRL::ComPtr<IDXGIFactory2> real_factory2;
    if (FAILED(fn_create_fac(IID_PPV_ARGS(&real_factory2))) || !real_factory2) return false;

    InstallFactoryHooks(real_factory2.Get());
    return true;
}

// Create a temporary 16x16 dummy D3D12 swapchain on System32\dxgi.dll to patch the shared CDXGISwapChain vtable in .rdata
bool HookSystemSwapChainVtableViaDummy() {
    if (!g_real_dxgi) return false;

    HMODULE d3d12_mod = GetModuleHandleA("d3d12.dll");
    if (!d3d12_mod) d3d12_mod = LoadLibraryA("d3d12.dll");
    if (!d3d12_mod) return false;

    auto fn_create_dev = reinterpret_cast<PFN_D3D12CreateDevice>(GetProcAddress(d3d12_mod, "D3D12CreateDevice"));
    auto fn_create_fac = reinterpret_cast<PFN_CreateDXGIFactory1>(
        GetExportAddressFromPEHeader(g_real_dxgi, "CreateDXGIFactory1"));
    if (!fn_create_dev || !fn_create_fac) return false;

    Microsoft::WRL::ComPtr<IDXGIFactory2> real_factory2;
    if (FAILED(fn_create_fac(IID_PPV_ARGS(&real_factory2))) || !real_factory2) return false;

    InstallFactoryHooks(real_factory2.Get());

    Microsoft::WRL::ComPtr<ID3D12Device> dummy_dev;
    if (FAILED(fn_create_dev(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dummy_dev))) || !dummy_dev) {
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> dummy_q;
    if (FAILED(dummy_dev->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&dummy_q))) || !dummy_q) {
        return false;
    }

    WNDCLASSEXA wc{};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "DLSS_NR_DUMMY_SC_WND";
    RegisterClassExA(&wc);

    HWND hwnd = CreateWindowExA(
        0, wc.lpszClassName, "", WS_POPUP, 0, 0, 16, 16,
        nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) return false;

    DXGI_SWAP_CHAIN_DESC1 sc_desc{};
    sc_desc.Width = 16;
    sc_desc.Height = 16;
    sc_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc_desc.SampleDesc.Count = 1;
    sc_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc_desc.BufferCount = 2;
    sc_desc.Scaling = DXGI_SCALING_STRETCH;
    sc_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    // Call the original unhooked CreateSwapChainForHwnd on System32\dxgi.dll's factory
    void** fac_vtbl = *reinterpret_cast<void***>(real_factory2.Get());
    auto orig_create_sc = g_create_swapchain_hwnd_map.Get(fac_vtbl);
    if (!orig_create_sc) {
        orig_create_sc = reinterpret_cast<PFN_Factory2_CreateSwapChainForHwnd>(fac_vtbl[15]);
    }

    bool hooked_sc = false;
    if (orig_create_sc) {
        t_in_create_sc = true;
        Microsoft::WRL::ComPtr<IDXGISwapChain1> dummy_sc;
        HRESULT hr = orig_create_sc(real_factory2.Get(), dummy_q.Get(), hwnd, &sc_desc, nullptr, nullptr, &dummy_sc);
        t_in_create_sc = false;
        if (SUCCEEDED(hr) && dummy_sc) {
            InstallSwapChainHooks(dummy_sc.Get());
            hooked_sc = true;
        }
    }

    DestroyWindow(hwnd);
    return hooked_sc;
}

DWORD WINAPI AsiHookWorkerThread(LPVOID) {
    Sleep(50);
    EnsureRealDxgiLoaded();
    InstallRealDxgiFactoryHooks();

    if (g_bridge) {
        std::ostringstream oss;
        oss << "ASI Worker Thread started. Real System32 dxgi.dll = 0x"
            << std::hex << reinterpret_cast<uintptr_t>(g_real_dxgi);
        g_bridge->LogEvent(oss.str());
    }

    const uint64_t thread_start_tick = GetTickCount64();
    bool dummy_fallback_attempted = false;
    bool logged_interception = false;

    while (true) {
        EnsureRealDxgiLoaded();
        const uint64_t now = GetTickCount64();
        const uint64_t last_create = g_last_create_sc_tick.load();

        IDXGISwapChain* captured_sc = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_hook_mutex);
            captured_sc = g_captured_swapchain;
        }

        if (captured_sc != nullptr) {
            if (!g_swapchain_hooks_active.load() && (now - last_create >= 1000)) {
                InstallSwapChainHooks(captured_sc);
                g_swapchain_hooks_active.store(true);
                if (g_bridge) {
                    g_bridge->LogEvent("Hooked game IDXGISwapChain Present/Present1/ResizeBuffers vtable after swapchain creation settled.");
                }
            }
        } else if (!dummy_fallback_attempted &&
                   GetModuleHandleA("D3D12Core.dll") != nullptr &&
                   (now - thread_start_tick >= 4000) &&
                   g_bridge && g_bridge->GetTelemetry().frames_intercepted == 0) {
            dummy_fallback_attempted = true;
            // The fallback creates a D3D12 device of its own on this thread. Not while the game is
            // still inside CreateSwapChain (last_create != 0: our factory hook saw the call, the
            // swapchain arrives when it returns), and never next to Streamline: in Marvel's
            // Spider-Man Remastered (sl.interposer + OptiScaler with XeFG) that D3D12CreateDevice
            // bounced between Streamline's and OptiScaler's hooks until the stack overflowed and
            // the game crashed at start (2026-11-05, crs dumps).
            if (last_create != 0) {
                g_bridge->LogEvent("Dummy swapchain fallback skipped: the game's CreateSwapChain is in progress.");
            } else if (GetModuleHandleA("sl.interposer.dll") != nullptr) {
                g_bridge->LogEvent("Dummy swapchain fallback skipped: Streamline (sl.interposer.dll) is loaded.");
            } else if (HookSystemSwapChainVtableViaDummy()) {
                g_swapchain_hooks_active.store(true);
                if (g_bridge) {
                    g_bridge->LogEvent("Hooked System32 CDXGISwapChain Present/Present1/ResizeBuffers vtable via dummy swapchain fallback.");
                }
            }
        }

        if (!logged_interception && g_bridge && g_bridge->GetTelemetry().frames_intercepted > 0) {
            logged_interception = true;
            g_bridge->LogEvent("ASI Worker Thread confirmed active SwapChain Present interception!");
        }

        Sleep(250);
    }
    return 0;
}

void StartAsiBackgroundThreadIfNeeded() {
    if (!IsGameExecutable()) return;
    bool expected = false;
    if (g_asi_thread_started.compare_exchange_strong(expected, true)) {
        HANDLE h = CreateThread(nullptr, 0, &AsiHookWorkerThread, nullptr, 0, nullptr);
        if (h) CloseHandle(h);
    }
}

} // namespace

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID /*lpReserved*/) {
    if (ul_reason_for_call == DLL_PROCESS_ATTACH) {
        g_this_module = hModule;
        DisableThreadLibraryCalls(hModule);
        if (IsGameExecutable()) {
            std::string path = ToLowerAscii(GetThisDllPath());
            if (path.size() >= 4 && path.substr(path.size() - 4) == ".asi") {
                StartAsiBackgroundThreadIfNeeded();
            }
        }
    }
    return TRUE;
}

extern "C" {

// @1
uint64_t WINAPI Proxy_ApplyCompatResolutionQuirking(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("ApplyCompatResolutionQuirking", a0, a1, a2, a3, a4, a5);
}
// @2
uint64_t WINAPI Proxy_CompatString(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("CompatString", a0, a1, a2, a3, a4, a5);
}
// @3
uint64_t WINAPI Proxy_CompatValue(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("CompatValue", a0, a1, a2, a3, a4, a5);
}
// @4
uint64_t WINAPI Proxy_DXGIDumpJournal(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("DXGIDumpJournal", a0, a1, a2, a3, a4, a5);
}
// @5
uint64_t WINAPI Proxy_PIXBeginCapture(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("PIXBeginCapture", a0, a1, a2, a3, a4, a5);
}
// @6
uint64_t WINAPI Proxy_PIXEndCapture(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("PIXEndCapture", a0, a1, a2, a3, a4, a5);
}
// @7
uint64_t WINAPI Proxy_PIXGetCaptureState(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("PIXGetCaptureState", a0, a1, a2, a3, a4, a5);
}
// @8
uint64_t WINAPI Proxy_SetAppCompatStringPointer(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("SetAppCompatStringPointer", a0, a1, a2, a3, a4, a5);
}
// @9
uint64_t WINAPI Proxy_UpdateHMDEmulationStatus(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("UpdateHMDEmulationStatus", a0, a1, a2, a3, a4, a5);
}
// @10
HRESULT WINAPI Proxy_CreateDXGIFactory(REFIID riid, void** ppFactory) {
    EnsureRealDxgiLoaded();
    if (!g_real_dxgi) return E_FAIL;
    auto fn = reinterpret_cast<PFN_CreateDXGIFactory>(GetExportAddressFromPEHeader(g_real_dxgi, "CreateDXGIFactory"));
    if (!fn) return E_FAIL;
    HRESULT hr = fn(riid, ppFactory);
    if (SUCCEEDED(hr) && ppFactory && *ppFactory && IsGameExecutable()) {
        InstallFactoryHooks(static_cast<IUnknown*>(*ppFactory));
        StartAsiBackgroundThreadIfNeeded();
    }
    return hr;
}
// @11
HRESULT WINAPI Proxy_CreateDXGIFactory1(REFIID riid, void** ppFactory) {
    EnsureRealDxgiLoaded();
    if (!g_real_dxgi) return E_FAIL;
    auto fn = reinterpret_cast<PFN_CreateDXGIFactory1>(GetExportAddressFromPEHeader(g_real_dxgi, "CreateDXGIFactory1"));
    if (!fn) return E_FAIL;
    HRESULT hr = fn(riid, ppFactory);
    if (SUCCEEDED(hr) && ppFactory && *ppFactory && IsGameExecutable()) {
        InstallFactoryHooks(static_cast<IUnknown*>(*ppFactory));
        StartAsiBackgroundThreadIfNeeded();
    }
    return hr;
}
// @12
HRESULT WINAPI Proxy_CreateDXGIFactory2(UINT Flags, REFIID riid, void** ppFactory) {
    EnsureRealDxgiLoaded();
    if (!g_real_dxgi) return E_FAIL;
    auto fn = reinterpret_cast<PFN_CreateDXGIFactory2>(GetExportAddressFromPEHeader(g_real_dxgi, "CreateDXGIFactory2"));
    if (!fn) return E_FAIL;
    HRESULT hr = fn(Flags, riid, ppFactory);
    if (SUCCEEDED(hr) && ppFactory && *ppFactory && IsGameExecutable()) {
        InstallFactoryHooks(static_cast<IUnknown*>(*ppFactory));
        StartAsiBackgroundThreadIfNeeded();
    }
    return hr;
}
// @13
uint64_t WINAPI Proxy_DXGID3D10CreateDevice(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("DXGID3D10CreateDevice", a0, a1, a2, a3, a4, a5);
}
// @14
uint64_t WINAPI Proxy_DXGID3D10CreateLayeredDevice(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("DXGID3D10CreateLayeredDevice", a0, a1, a2, a3, a4, a5);
}
// @15
uint64_t WINAPI Proxy_DXGID3D10GetLayeredDeviceSize(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("DXGID3D10GetLayeredDeviceSize", a0, a1, a2, a3, a4, a5);
}
// @16
uint64_t WINAPI Proxy_DXGID3D10RegisterLayers(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("DXGID3D10RegisterLayers", a0, a1, a2, a3, a4, a5);
}
// @17
HRESULT WINAPI Proxy_DXGIDeclareAdapterRemovalSupport() {
    EnsureRealDxgiLoaded();
    if (!g_real_dxgi) return S_OK;
    auto fn = reinterpret_cast<PFN_DeclareAdapterRemovalSupport>(
        GetExportAddressFromPEHeader(g_real_dxgi, "DXGIDeclareAdapterRemovalSupport"));
    return fn ? fn() : S_OK;
}
// @18
uint64_t WINAPI Proxy_DXGIDisableVBlankVirtualization(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("DXGIDisableVBlankVirtualization", a0, a1, a2, a3, a4, a5);
}
// @19
HRESULT WINAPI Proxy_DXGIGetDebugInterface1(UINT Flags, REFIID riid, void** pDebug) {
    EnsureRealDxgiLoaded();
    if (!g_real_dxgi) return E_FAIL;
    auto fn = reinterpret_cast<PFN_DXGIGetDebugInterface1>(
        GetExportAddressFromPEHeader(g_real_dxgi, "DXGIGetDebugInterface1"));
    return fn ? fn(Flags, riid, pDebug) : E_NOINTERFACE;
}
// @20
uint64_t WINAPI Proxy_DXGIReportAdapterConfiguration(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    return ForwardGeneric("DXGIReportAdapterConfiguration", a0, a1, a2, a3, a4, a5);
}

// Entrypoint for OptiScaler ASI plugin loader (plugins\DLSS_NR_DX12.asi)
void WINAPI Proxy_InitializeASI() {
    if (!IsGameExecutable()) return;
    EnsureRealDxgiLoaded();
    InstallRealDxgiFactoryHooks();
    if (g_bridge) {
        g_bridge->LogEvent("InitializeASI called by OptiScaler.");
    }
    StartAsiBackgroundThreadIfNeeded();
}

} // extern "C"
