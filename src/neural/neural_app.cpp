// `dlss_nr_dx12_poc --neural ...`: the DLSS-NR network standalone on the D3D12 device.
//
//   --golden STEM        validate against the numpy oracle: reads STEM_color.pfm,
//                        STEM_output.pfm and STEM_history_output.pfm (tools/hnet/oracle.py),
//                        runs two frames (the second with the first as history) and
//                        requires PSNR >= --min-psnr (35 dB) on both
//   --bench WxH,...      GPU timing per output size on a synthetic frame
//   --frames N           frames per bench size (default 30)
//   --render-scale S     network on the output at scale S (default 1)
//   --interval N         the native network over N = 2-4 frames (NeuralOptions::interval)
//   --precision fast     without the E4M3 roundings (NeuralOptions::precise)
//   --interval-check STEM  moving scenes on STEM_color.pfm: --interval N (default 4) with and
//                        without the warp against interval 1 (the head of the frame itself)
//   --interval-scenes I,J  only these scenes (their numbers, as in --interval-dump's file names)
//   --interval-dump DIR  the corrections of the compared frames as PFMs (+ 0.5)
//   --warp-model 0|1     the warp moves the head's colour model (1, default) or the head (0)
//   --warp-blend W       with the model: a new reference's weight against the last one (1 = off)
//   --warp-pyramid 0|1   the warp's coarse search over +-96 field pixels (1, default)
//   --warp-chain 0|1     the warp's search centred on the motion summed frame to frame (0, default)
//   --warp-fade-to none|regional  what the speed fade goes to (regional, default)
//   --warp-fade S[,F]    the moved detail fades to the regional model from S to F field px a frame
//                        (default 2,6; F = 4 S when left out; 1,0 = off)
//   --profile NAME       standard | natural | cinematic | neutral
//   --synthetic          use the *_synthetic.onnx models (no real weights needed)
//   --model-dir DIR      extra directory with hnet_*.onnx (repeatable)
//   --tag TAG            numerics tag of the models (default nq-exp)
//   --no-hybrid          run the whole graph in ONNX even where the hybrid files exist
//   --hybrid-check DIR   compare the hybrid's HLSL ends with tools/hnet/export_hybrid.py --dump
//                        (DIR = work/hybrid/<W>x<H>[_synthetic], the output size is taken from it)
//   --output-dir DIR     where neural-report.md and output PFMs go (default docs)

#include "neural/neural_app.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "dx12/command_queue.h"
#include "dx12/device.h"
#include "dx12/resources.h"
#include "neural/neural_graph.h"

namespace dlss_nr::neural {

namespace {

struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> rgba;   // top-down, 4 floats per pixel
};

bool ReadPfm(const std::string& path, Image& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string magic;
    double scale = 0;
    in >> magic >> out.width >> out.height >> scale;
    in.get();
    if (magic != "PF" || out.width == 0 || out.height == 0 || scale >= 0) return false;   // little-endian RGB only
    std::vector<float> rgb(static_cast<size_t>(out.width) * out.height * 3);
    in.read(reinterpret_cast<char*>(rgb.data()), rgb.size() * sizeof(float));
    if (!in) return false;
    out.rgba.assign(static_cast<size_t>(out.width) * out.height * 4, 1.0f);
    for (uint32_t y = 0; y < out.height; ++y) {
        const float* row = rgb.data() + static_cast<size_t>(out.height - 1 - y) * out.width * 3;   // PFM is bottom-up
        for (uint32_t x = 0; x < out.width; ++x) {
            for (int c = 0; c < 3; ++c) out.rgba[(static_cast<size_t>(y) * out.width + x) * 4 + c] = row[x * 3 + c];
        }
    }
    return true;
}

bool WritePfm(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "PF\n" << img.width << " " << img.height << "\n-1.0\n";
    std::vector<float> row(static_cast<size_t>(img.width) * 3);
    for (uint32_t y = img.height; y-- > 0;) {
        for (uint32_t x = 0; x < img.width; ++x) {
            for (int c = 0; c < 3; ++c) row[x * 3 + c] = img.rgba[(static_cast<size_t>(y) * img.width + x) * 4 + c];
        }
        out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    return static_cast<bool>(out);
}

// How the change a frame makes (got - input) matches the oracle's (want - input), over RGB:
// correlation, gain (projection onto the oracle's change) and the RMS error relative to it.
struct Effect {
    double corr = 0.0, gain = 0.0, error = 0.0;
};
constexpr double kMinEffectCorr = 0.95;       // precision accurate: real weights 0.977 (PyTorch: 0.977)
constexpr double kMinEffectCorrFast = 0.80;   // precision fast: 0.85

Effect CompareEffect(const Image& got, const Image& want, const Image& input) {
    Effect e;
    if (got.rgba.size() != want.rgba.size() || want.rgba.size() != input.rgba.size()) return e;
    double sg = 0, sw = 0, sgw = 0, sgg = 0, sww = 0, sdd = 0, n = 0;
    for (size_t i = 0; i < want.rgba.size(); ++i) {
        if (i % 4 == 3) continue;
        const double g = static_cast<double>(got.rgba[i]) - input.rgba[i];
        const double w = static_cast<double>(want.rgba[i]) - input.rgba[i];
        sg += g; sw += w; sgw += g * w; sgg += g * g; sww += w * w; sdd += (g - w) * (g - w); n += 1;
    }
    const double cov = sgw / n - (sg / n) * (sw / n);
    const double var = std::sqrt(std::max(0.0, sgg / n - (sg / n) * (sg / n)) * std::max(0.0, sww / n - (sw / n) * (sw / n)));
    e.corr = var > 0 ? cov / var : 0.0;
    e.gain = sww > 0 ? sgw / sww : 0.0;
    e.error = sww > 0 ? std::sqrt(sdd / sww) : 0.0;
    return e;
}

double Psnr(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height) return -1.0;
    double mse = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < a.rgba.size(); ++i) {
        if (i % 4 == 3) continue;
        double d = static_cast<double>(a.rgba[i]) - b.rgba[i];
        mse += d * d;
        ++n;
    }
    mse /= static_cast<double>(n);
    return mse <= 0.0 ? 200.0 : 10.0 * std::log10(1.0 / mse);
}

Image TestImage(uint32_t w, uint32_t h) {
    Image img{w, h, std::vector<float>(static_cast<size_t>(w) * h * 4, 1.0f)};
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            float u = static_cast<float>(x) / std::max(1u, w - 1), v = static_cast<float>(y) / std::max(1u, h - 1);
            float* p = &img.rgba[(static_cast<size_t>(y) * w + x) * 4];
            p[0] = 0.15f + 0.7f * u;
            p[1] = 0.2f + 0.6f * v;
            p[2] = 0.5f + 0.4f * std::sin(6.28f * (u + v));
            if (((x / 16) + (y / 16)) % 2 == 0) {
                p[0] *= 0.9f; p[1] *= 0.9f; p[2] *= 0.9f;
            }
        }
    }
    return img;
}

// Pre and post must be separate submissions around the ORT run, which submits its own work
// to the same queue in between. Frames go back to back with up to kSlots in flight, as in a
// game: waiting for an idle GPU every frame lets its clocks drop and measures the ramp.
class FrameRunner {
public:
    static constexpr uint32_t kSlots = 3;

    bool Initialize(ID3D12Device* device, dx12::CommandQueue& queue) {
        queue_ = &queue;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) return false;
        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        for (uint32_t i = 0; i < kSlots * 2; ++i) {
            if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc_[i]))) ||
                FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc_[i].Get(), nullptr,
                                                 IID_PPV_ARGS(&list_[i])))) {
                return false;
            }
            list_[i]->Close();
        }
        return true;
    }

    ~FrameRunner() {
        if (event_) CloseHandle(event_);
    }

    // submits one frame; waits only for the frame that last used this slot
    bool Submit(NeuralGraph& graph, uint32_t frame) {
        const uint32_t slot = frame % kSlots;
        WaitFor(slot_fence_[slot]);
        ID3D12CommandQueue* q = queue_->GetQueue();
        ID3D12GraphicsCommandList* pre = list_[slot * 2].Get();
        ID3D12GraphicsCommandList* post = list_[slot * 2 + 1].Get();
        alloc_[slot * 2]->Reset();
        pre->Reset(alloc_[slot * 2].Get(), nullptr);
        graph.RecordPre(pre, slot);
        pre->Close();
        ID3D12CommandList* a[] = {pre};
        q->ExecuteCommandLists(1, a);
        if (!graph.RunNetwork()) return false;
        alloc_[slot * 2 + 1]->Reset();
        post->Reset(alloc_[slot * 2 + 1].Get(), nullptr);
        graph.RecordPost(post, slot);
        post->Close();
        ID3D12CommandList* b[] = {post};
        q->ExecuteCommandLists(1, b);
        q->Signal(fence_.Get(), ++fence_value_);
        slot_fence_[slot] = fence_value_;
        return true;
    }

    void Drain() { WaitFor(fence_value_); }

    // after Drain(): the GPU timings of the frame that last used `frame`'s slot
    static uint32_t Slot(uint32_t frame) { return frame % kSlots; }

private:
    void WaitFor(uint64_t value) {
        if (value == 0 || fence_->GetCompletedValue() >= value) return;
        fence_->SetEventOnCompletion(value, event_);
        WaitForSingleObject(event_, INFINITE);
    }

    dx12::CommandQueue* queue_ = nullptr;
    ComPtr<ID3D12Fence> fence_;
    HANDLE event_ = nullptr;
    uint64_t fence_value_ = 0;
    uint64_t slot_fence_[kSlots] = {};
    ComPtr<ID3D12CommandAllocator> alloc_[kSlots * 2];
    ComPtr<ID3D12GraphicsCommandList> list_[kSlots * 2];
};

bool Upload(dx12::ResourceManager& rm, dx12::CommandQueue& queue, dx12::GpuTexture& tex, const Image& img) {
    dx12::GpuBuffer staging;
    const uint32_t pitch = img.width * 16;
    const uint64_t size = static_cast<uint64_t>(((pitch + 255) / 256) * 256) * img.height;
    return rm.CreateUploadBuffer(size, staging) &&
           rm.UploadTexture2DData(queue, tex, staging, img.rgba.data(), pitch, 16);
}

bool Download(dx12::ResourceManager& rm, dx12::CommandQueue& queue, dx12::GpuTexture& tex, Image& img) {
    img.width = tex.width;
    img.height = tex.height;
    img.rgba.assign(static_cast<size_t>(tex.width) * tex.height * 4, 0.0f);
    dx12::GpuBuffer staging;
    const uint32_t pitch = tex.width * 16;
    const uint64_t size = static_cast<uint64_t>(((pitch + 255) / 256) * 256) * tex.height;
    return rm.CreateReadbackBuffer(size, staging) &&
           rm.ReadbackTexture2DData(queue, tex, staging, img.rgba.data(), pitch, 16);
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::vector<std::pair<uint32_t, uint32_t>> ParseSizes(const std::string& text) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        unsigned w = 0, h = 0;
        if (std::sscanf(item.c_str(), "%ux%u", &w, &h) == 2 && w && h) out.emplace_back(w, h);
    }
    return out;
}

} // namespace

int RunNeuralMode(int argc, char** argv) {
    NeuralOptions options;
    std::string golden, output_dir = "docs", bench, hybrid_check, stability, interval_check, interval_dump, interval_scenes,
        interval_sequence;
    uint32_t sequence_first = 0, sequence_last = 0xFFFFFFFFu;
    uint32_t frames = 30;
    bool profile_ends = false;
    double min_psnr = 35.0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "--golden") golden = next();
        else if (arg == "--stability") stability = next();
        else if (arg == "--interval-check") interval_check = next();
        else if (arg == "--interval-sequence") interval_sequence = next();
        else if (arg == "--interval-frames") {   // A,B: the captured frames A..B only (memory at 1440p)
            const std::string v = next();
            const size_t comma = v.find(',');
            sequence_first = static_cast<uint32_t>(std::stoul(v.substr(0, comma)));
            sequence_last = comma == std::string::npos ? sequence_first + 47 : static_cast<uint32_t>(std::stoul(v.substr(comma + 1)));
        }
        else if (arg == "--interval-dump") interval_dump = next();
        else if (arg == "--interval-scenes") interval_scenes = "," + next() + ",";
        else if (arg == "--bench") bench = next();
        else if (arg == "--frames") frames = static_cast<uint32_t>(std::stoul(next()));
        else if (arg == "--render-scale") options.render_scale = std::stof(next());
        else if (arg == "--interval") options.interval = std::clamp<uint32_t>(static_cast<uint32_t>(std::stoul(next())), 1u, NeuralGraph::kMaxInterval);
        else if (arg == "--precision") options.precise = next() != "fast";
        else if (arg == "--stabilize") options.stabilize = std::stof(next());
        else if (arg == "--warp-model") options.warp_model = next() != "0";
        else if (arg == "--warp-blend") options.warp_blend = std::stof(next());
        else if (arg == "--warp-pyramid") options.warp_pyramid = next() != "0";
        else if (arg == "--warp-chain") options.warp_chain = next() != "0";
        else if (arg == "--warp-fade-to") options.warp_fade_none = next() == "none";
        else if (arg == "--warp-fade") {   // START,FULL field pixels a frame (FULL 0: off)
            const std::string v = next();
            const size_t comma = v.find(',');
            options.warp_fade_start = std::stof(v.substr(0, comma));
            options.warp_fade_full = comma == std::string::npos ? options.warp_fade_start * 4.0f : std::stof(v.substr(comma + 1));
        }
        else if (arg == "--stabilize-tolerance") options.stabilize_tolerance = std::stof(next());
        else if (arg == "--stabilize-motion") options.stabilize_motion = std::stof(next());
        else if (arg == "--intensity") options.intensity = std::stof(next());
        else if (arg == "--profile") {
            if (!ApplyProfile(next(), options)) { std::cerr << "[FATAL] unknown profile\n"; return 2; }
        } else if (arg == "--synthetic") options.synthetic_weights = true;
        else if (arg == "--model-dir") options.model_dirs.insert(options.model_dirs.begin(), next());
        else if (arg == "--ort-dir") options.ort_dirs.insert(options.ort_dirs.begin(), next());
        else if (arg == "--directml-dir") options.directml_dirs.insert(options.directml_dirs.begin(), next());
        else if (arg == "--tag") options.model_tag = next();
        else if (arg == "--output-dir") output_dir = next();
        else if (arg == "--min-psnr") min_psnr = std::stod(next());
        else if (arg == "--no-temporal") options.temporal = false;
        else if (arg == "--no-hybrid") options.allow_hybrid = false;
        else if (arg == "--shallow") options.deep_hybrid = false;
        else if (arg == "--no-dml-gemm") options.dml_gemms = false;
        else if (arg == "--no-native") options.native = false;
        else if (arg == "--profile-ends") profile_ends = true;
        else if (arg == "--hybrid-check") hybrid_check = next();
    }
    if (golden.empty() && bench.empty() && hybrid_check.empty() && stability.empty() &&
        interval_check.empty()) bench = "320x320,960x540,1920x1080";
    // a reference dumped for the shallow split must be checked on the shallow split
    if (hybrid_check.find("shallow") != std::string::npos) {
        options.deep_hybrid = false;
        options.native = false;
    }

    std::cout << "=== DLSS-NR neural graph (ONNX Runtime DirectML on the D3D12 device) ===\n";
    dx12::Device device;
    if (!device.Initialize(false, false) && !device.Initialize(false, true)) {
        std::cerr << "[FATAL] no D3D12 adapter\n";
        return 1;
    }
    std::cout << "  GPU: " << device.GetCapabilityReport().selected_adapter.description << "\n";
    dx12::CommandQueue queue;
    dx12::ResourceManager rm;
    FrameRunner runner;
    if (!queue.Initialize(device.GetD3D12Device(), D3D12_COMMAND_LIST_TYPE_DIRECT, 4) ||
        !rm.Initialize(device.GetD3D12Device(), 16) || !runner.Initialize(device.GetD3D12Device(), queue)) {
        std::cerr << "[FATAL] queue/resources\n";
        return 1;
    }
    NeuralGraph graph;
    if (!graph.Initialize(device.GetD3D12Device(), queue.GetQueue(), options)) {
        std::cerr << "[FATAL] NeuralGraph: " << graph.LastError() << "\n";
        return 1;
    }

    std::ostringstream report;
    report << "# Neural graph report (N2)\n\n"
           << "GPU: " << device.GetCapabilityReport().selected_adapter.description
           << ", driver " << device.GetCapabilityReport().driver_version << "  \n"
           << "Models: `" << options.model_tag << "`" << (options.allow_hybrid ? " (hybrid where available)" : "")
           << (options.synthetic_weights ? " (synthetic weights)" : "") << ", render scale "
           << options.render_scale << "\n\n";
    int status = 0;
    constexpr DXGI_FORMAT kFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;

    if (!hybrid_check.empty()) {
        unsigned w = 0, h = 0;
        const std::string leaf = hybrid_check.substr(hybrid_check.find_last_of("/\\") + 1);
        if (std::sscanf(leaf.c_str(), "%ux%u", &w, &h) != 2 || !graph.EnsureExtent(w, h, kFormat)) {
            std::cerr << "[FATAL] hybrid check setup: " << graph.LastError() << "\n";
            return 1;
        }
        std::string check;
        const bool pass = graph.IsHybrid() && graph.CheckHybridEnds(hybrid_check, check);
        std::cout << "  hybrid ends vs PyTorch (" << hybrid_check << "), model " << graph.ModelPath() << ":\n"
                  << check << (graph.IsHybrid() ? "" : "  not hybrid: " + graph.LastError() + "\n")
                  << "  -> " << (pass ? "PASS" : "FAIL") << "\n";
        report << "## Hybrid ends vs PyTorch\n\n```\n" << check << "```\n\n";
        if (!pass) status = 4;
    }

    if (!golden.empty()) {
        Image color, want1, want2;
        if (!ReadPfm(golden + "_color.pfm", color) || !ReadPfm(golden + "_output.pfm", want1) ||
            !ReadPfm(golden + "_history_output.pfm", want2)) {
            std::cerr << "[FATAL] cannot read " << golden << "_{color,output,history_output}.pfm\n";
            return 1;
        }
        if (!graph.EnsureExtent(color.width, color.height, kFormat)) {
            std::cerr << "[FATAL] " << graph.LastError() << "\n";
            return 1;
        }
        std::cout << "  model " << graph.ModelPath() << " (session " << std::fixed << std::setprecision(0)
                  << graph.LastSessionCreateMs() << " ms)\n";
        Image got1, got2;
        auto frame = [&](uint32_t index, Image& out) {
            if (!runner.Submit(graph, index)) return false;
            runner.Drain();
            return Download(rm, queue, graph.GetOutputTexture(), out);
        };
        bool ok = Upload(rm, queue, graph.GetInputTexture(), color) && frame(0, got1) && frame(1, got2);
        if (!ok) {
            std::cerr << "[FATAL] frame failed: " << graph.LastError() << "\n";
            return 1;
        }
        // without history (--no-temporal) the second frame repeats the first
        const Image& want_second = graph.GetOptions().temporal ? want2 : want1;
        const double p0 = Psnr(color, want1), p1 = Psnr(got1, want1), p2 = Psnr(got2, want_second);
        // With the real weights the network changes the picture only a little (input vs oracle
        // ~39 dB), so PSNR says little: the gate is how well the change itself is reproduced.
        // Frame 2's history path is a known open issue with the real weights (PROGRESS-neural.md):
        // reported, not gated.
        const Effect e1 = CompareEffect(got1, want1, color), e2 = CompareEffect(got2, want_second, color);
        const double min_effect = graph.GetOptions().precise ? kMinEffectCorr : kMinEffectCorrFast;
        const bool pass = p1 >= min_psnr && p2 >= min_psnr && e1.corr >= min_effect;
        std::cout << std::setprecision(2) << "  PSNR frame 1 vs oracle: " << p1 << " dB, frame 2"
                  << (graph.GetOptions().temporal ? " (history)" : " (no history)") << ": " << p2
                  << " dB  (input vs oracle " << p0 << " dB)\n"
                  << std::setprecision(4) << "  effect frame 1: corr " << e1.corr << ", gain " << e1.gain
                  << ", error/effect " << e1.error << "  (gate corr >= " << min_effect << ")\n"
                  << "  effect frame 2: corr " << e2.corr << ", gain " << e2.gain << ", error/effect " << e2.error
                  << "  (not gated)  -> " << (pass ? "PASS" : "FAIL") << "\n";
        if (graph.GetOptions().interval > 1) {
            // Interval N: frame 2 takes features and runs part 0 and composes frame 1's head moved
            // to it, the next N-1 frames run the other parts. Over two whole cycles on a still
            // picture without history every frame must give frame 1's output exactly: the parts
            // did not disturb each other's buffers and the warp of a still picture is the identity.
            const uint32_t n = graph.GetOptions().interval;
            float worst = 0.0f;
            for (uint32_t k = 2; k < 2 + 2 * n; ++k) {
                Image later;
                if (!frame(k, later)) {
                    std::cerr << "[FATAL] frame failed: " << graph.LastError() << "\n";
                    return 1;
                }
                for (size_t i = 0; i < got1.rgba.size(); ++i) worst = std::max(worst, std::abs(later.rgba[i] - got1.rgba[i]));
            }
            const bool exact = worst == 0.0f;
            std::cout << std::fixed << std::setprecision(4) << "  interval " << n << ", frames 3-" << (2 + 2 * n)
                      << " vs frame 1: max |d| " << worst
                      << (graph.GetOptions().temporal ? "  (history on: not compared)\n"
                                                      : (exact ? "  -> PASS\n" : "  -> FAIL\n"));
            if (!graph.GetOptions().temporal && !exact) status = 3;
        }
        WritePfm(output_dir + "/neural_frame1.pfm", got1);
        WritePfm(output_dir + "/neural_frame2.pfm", got2);
        report << "## Validation against the numpy oracle\n\n| frame | PSNR vs oracle | gate |\n| --- | --- | --- |\n"
               << std::fixed << std::setprecision(2)
               << "| 1 (no history) | " << p1 << " dB | >= " << min_psnr << " |\n"
               << "| 2 (history) | " << p2 << " dB | >= " << min_psnr << " |\n"
               << "| input itself | " << p0 << " dB | (how much the network changes) |\n\n"
               << std::setprecision(4) << "Effect (output - input against the oracle's): frame 1 corr " << e1.corr
               << ", gain " << e1.gain << ", error/effect " << e1.error << " (gate corr >= " << min_effect
               << "); frame 2 corr " << e2.corr << ", gain " << e2.gain << ", error/effect " << e2.error
               << " (not gated)\n\n"
               << "Golden: `" << golden << "`, verdict **" << (pass ? "PASS" : "FAIL") << "**\n\n";
        if (!pass) status = 3;
    }

    if (!stability.empty()) {
        // Frame-to-frame stability of the correction (output - input) on STEM_color.pfm, every frame
        // under fresh noise of +-0.5/255 (what a game's dithering and jitter look like), with the
        // stabiliser off and at options.stabilize:
        //   still  the picture standing still: flicker (RMS change of the correction between frames
        //          over its RMS) and the bias of the average (last frame against the mean of the
        //          unstabilised corrections)
        //   pan    the picture moving 1 pixel a frame: flicker with the frames aligned (shimmer)
        //   jitter the still picture resampled at a random offset of up to +-0.25 pixel a frame (the
        //          residual jitter of a TAA or upscaled image, e.g. a game's photo mode): flicker
        //   cut    a different picture (the mirror image) from the middle on: the first frame after
        //          the cut against the unstabilised one — nothing of the old picture may be dragged
        Image color;
        if (!ReadPfm(stability + "_color.pfm", color) || !graph.EnsureExtent(color.width, color.height, kFormat)) {
            std::cerr << "[FATAL] stability setup: " << graph.LastError() << "\n";
            return 1;
        }
        enum Scene { kStill, kJitter, kPan, kCut };
        const uint32_t frames = 24, warm = 8, cut = 12, w = color.width, h = color.height;
        const float weight = options.stabilize > 0.0f ? options.stabilize : 0.05f;
        auto run_sequence = [&](Scene scene, float stabilize, std::vector<std::vector<float>>& corrections) {
            graph.SetStabilize(stabilize);
            graph.ResetHistory();
            corrections.assign(frames, {});
            uint32_t state = 12345u;
            auto uniform = [&state]() {   // [-0.5, 0.5)
                state = state * 1664525u + 1013904223u;
                return static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
            };
            for (uint32_t f = 0; f < frames; ++f) {
                Image in = color, out;
                const float jx = scene == kJitter ? 0.5f * uniform() : 0.0f, jy = scene == kJitter ? 0.5f * uniform() : 0.0f;
                for (uint32_t y = 0; y < h; ++y) {
                    for (uint32_t x = 0; x < w; ++x) {
                        uint32_t sx = x;
                        if (scene == kPan) sx = (x + w - f % w) % w;      // content moves right
                        if (scene == kCut && f >= cut) sx = w - 1 - x;    // the mirror image
                        // bilinear at (sx + jx, y + jy), clamped (jx = jy = 0 outside the jitter scene)
                        const float fx = std::min(std::max(sx + jx, 0.0f), static_cast<float>(w - 1));
                        const float fy = std::min(std::max(y + jy, 0.0f), static_cast<float>(h - 1));
                        const uint32_t x0 = static_cast<uint32_t>(fx), y0 = static_cast<uint32_t>(fy);
                        const uint32_t x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
                        const float ax = fx - x0, ay = fy - y0;
                        for (uint32_t c = 0; c < 3; ++c) {
                            auto at = [&](uint32_t px, uint32_t py) { return color.rgba[(py * w + px) * 4 + c]; };
                            const float v = (1 - ay) * ((1 - ax) * at(x0, y0) + ax * at(x1, y0)) +
                                            ay * ((1 - ax) * at(x0, y1) + ax * at(x1, y1)) + uniform() / 255.0f;
                            in.rgba[(y * w + x) * 4 + c] = std::min(1.0f, std::max(0.0f, v));
                        }
                    }
                }
                if (!Upload(rm, queue, graph.GetInputTexture(), in) || !runner.Submit(graph, f)) return false;
                runner.Drain();
                if (!Download(rm, queue, graph.GetOutputTexture(), out)) return false;
                corrections[f].resize(static_cast<size_t>(w) * h * 3);
                for (size_t i = 0, j = 0; i < in.rgba.size(); ++i) {
                    if (i % 4 != 3) corrections[f][j++] = out.rgba[i] - in.rgba[i];
                }
            }
            return true;
        };
        auto rms = [](const std::vector<float>& a) {
            double s = 0;
            for (float v : a) s += static_cast<double>(v) * v;
            return std::sqrt(s / static_cast<double>(a.size()));
        };
        auto relative = [&](const std::vector<float>& a, const std::vector<float>& b) {   // rms(a - b) / rms(b)
            std::vector<float> d(a.size());
            for (size_t i = 0; i < d.size(); ++i) d[i] = a[i] - b[i];
            return rms(d) / rms(b);
        };
        auto flicker = [&](const std::vector<std::vector<float>>& seq, bool pan) {
            double total = 0;
            for (uint32_t f = warm; f < frames; ++f) {
                std::vector<float> d(seq[f].size(), 0.0f);
                for (uint32_t y = 0; y < h; ++y) {
                    for (uint32_t x = 2; x + 2 < w; ++x) {   // the pan's wrapped edge left out
                        const uint32_t px = pan ? x - 1 : x;
                        for (uint32_t c = 0; c < 3; ++c) {
                            d[(y * w + x) * 3 + c] = seq[f][(y * w + x) * 3 + c] - seq[f - 1][(y * w + px) * 3 + c];
                        }
                    }
                }
                total += rms(d) / rms(seq[f]);
            }
            return total / (frames - warm);
        };
        // gates (real weights, accurate, stabilize 0.05, motion 0.5: still 0.55 -> 0.03, jitter
        // 0.37 -> 0.05, pan 0.39 -> 0.22 — on purpose: where anything moves the average must die
        // out fast, or it smears — cut ~0.02)
        bool stable_ok = true;
        std::cout << std::fixed << std::setprecision(3);
        for (const Scene scene : {kStill, kJitter, kPan, kCut}) {
            std::vector<std::vector<float>> off, on;
            if (!run_sequence(scene, 0.0f, off) || !run_sequence(scene, weight, on)) {
                std::cerr << "[FATAL] stability frame failed: " << graph.LastError() << "\n";
                return 1;
            }
            if (scene == kStill) {
                std::vector<float> mean(off[0].size(), 0.0f);
                for (uint32_t f = warm; f < frames; ++f) {
                    for (size_t i = 0; i < mean.size(); ++i) mean[i] += off[f][i] / (frames - warm);
                }
                const double calm = flicker(on, false), bias = relative(on[frames - 1], mean);
                const bool ok = calm <= 0.1;
                stable_ok = stable_ok && ok;
                std::cout << "  still + noise : flicker off " << flicker(off, false) << ", on (stabilize " << weight
                          << ") " << calm << "  (gate <= 0.1); the average against the mean of the off frames "
                          << bias << (ok ? "" : "  FAIL") << "\n";
            } else if (scene == kJitter) {
                const double calm = flicker(on, false);
                const bool ok = calm <= 0.2;
                stable_ok = stable_ok && ok;
                std::cout << "  jitter 0.25 px: flicker off " << flicker(off, false) << ", on " << calm
                          << "  (gate <= 0.2)" << (ok ? "" : "  FAIL") << "\n";
            } else if (scene == kPan) {
                // relative: how shift-variant the network is depends on the weights (synthetic 0.79)
                const double calm = flicker(on, true), raw = flicker(off, true);
                const bool ok = calm <= 0.7 * raw;
                stable_ok = stable_ok && ok;
                std::cout << "  pan 1 px/frame: flicker off " << raw << ", on " << calm << "  (gate <= 0.7 x off)"
                          << (ok ? "" : "  FAIL") << "\n";
            } else {
                const double dragged = relative(on[cut], off[cut]), next = relative(on[cut + 1], off[cut + 1]);
                const bool ok = dragged <= 0.2;
                stable_ok = stable_ok && ok;
                std::cout << "  cut           : first frame after it, on vs off " << dragged << " (next " << next
                          << ")  (gate <= 0.2)" << (ok ? "" : "  FAIL") << "\n";
            }
        }
        std::cout << "  stabiliser -> " << (stable_ok ? "PASS" : "FAIL") << "\n";
        if (!stable_ok) status = 3;
    }

    if (!interval_check.empty() || !interval_sequence.empty()) {
        // The network over N frames against the head of the frame itself (interval 1), on moving
        // scenes: per frame (after the first cycles) rms(correction_N - correction_1) / rms(correction_1)
        // (0 is the ideal) and the gain <c_N, c_1> / <c_1, c_1> (how much of the correction is kept;
        // the warp fades it where the colours disagree). Warp on and off (off = the last head as it is).
        //   pan 1 / 3 / 8   the picture moving 1 / 3 / 8 pixels a frame
        // and the pulse: the mean change of the correction's strength (rms) between frames over its
        // mean — a head arriving every N frames and fading in between shows here (interval 1: ~0)
        //   object          the middle third moving 2 px a frame over the still picture (boundaries,
        //                   disocclusion)
        //   wobble, 2       non-rigid motion (cloth, limbs): a pan plus a moving sinusoidal deformation
        //   rotate          0.6 degrees a frame about the centre
        //   pan fast        a fast camera turn: 12 field pixels a frame at any render scale (24 px
        //                   at 0.5; the shown head is 36-72 field pixels behind at interval 4). In
        //                   picture pixels at render scale 1 on a 320 px field it entered 45 % new
        //                   content a cycle, nothing like a game at 1344 field pixels
        //   swing           the camera turned back and forth (offset 96 sin(2 pi f / 20): up to
        //                   30 px a frame, reversing; the prediction is wrong at every turn)
        //   follow          the camera following a character: the background panning 16 px a frame
        //                   behind a still middle region whose content deforms like limbs (5 px at a
        //                   wavelength of 48 px, 0.7 rad a frame)
        //   cut             the mirror image from frame 12 on: frames 12-15 (nothing may be dragged)
        //   zoom            2 % a frame into the centre (moving forward: the motion grows to the
        //                   edges and differs across a block, like rotate)
        //   capture         --interval-sequence DIR: frames the game bridge captured (ini capture=N:
        //                   in_NNNN.raw, RGBA8 rows, size in meta.txt), the only scene then
        std::vector<Image> sequence;
        if (!interval_sequence.empty()) {
            std::ifstream meta(interval_sequence + "/meta.txt");
            std::string key;
            uint32_t sw = 0, sh = 0;
            while (meta >> key) {
                if (key == "width") meta >> sw;
                else if (key == "height") meta >> sh;
            }
            for (uint32_t i = sequence_first; sw && sh && i <= sequence_last; ++i) {
                char name[32];
                std::snprintf(name, sizeof(name), "/in_%04u.raw", i);
                std::ifstream f(interval_sequence + name, std::ios::binary);
                if (!f) break;
                std::vector<uint8_t> raw(static_cast<size_t>(sw) * sh * 4);
                if (!f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()))) break;
                Image img;
                img.width = sw;
                img.height = sh;
                img.rgba.resize(raw.size());
                for (size_t k = 0; k < raw.size(); ++k) img.rgba[k] = (k % 4 == 3) ? 1.0f : raw[k] / 255.0f;
                sequence.push_back(std::move(img));
            }
            if (sequence.size() < 12) {
                std::cerr << "[FATAL] " << interval_sequence << ": " << sequence.size() << " frames (meta.txt " << sw << "x" << sh
                          << "), at least 12 needed\n";
                return 1;
            }
            std::cout << "  sequence " << interval_sequence << ": " << sequence.size() << " frames " << sw << "x" << sh << "\n";
        }
        Image color;
        if (!sequence.empty()) color = sequence[0];
        if ((sequence.empty() && !ReadPfm(interval_check + "_color.pfm", color)) ||
            !graph.EnsureExtent(color.width, color.height, kFormat)) {
            std::cerr << "[FATAL] interval check setup: " << graph.LastError() << "\n";
            return 1;
        }
        const uint32_t n = options.interval > 1 ? options.interval : 4u;
        const uint32_t frames = sequence.empty() ? 24u : static_cast<uint32_t>(sequence.size());
        const uint32_t warm = 2 * n + 1, cut = 12, w = color.width, h = color.height;
        enum Scene { kPan1, kPan3, kPan8, kObject, kWobble, kWobble2, kRotate, kShake, kWhip, kPan24, kSwing, kFollow, kCut, kZoom,
                     kCapture };
        const uint32_t fast = std::max(1u, static_cast<uint32_t>(std::lround(12.0 * w / graph.GetExtent().inner_w)));
        auto speed_of = [fast](Scene scene) {
            return scene == kPan1 ? 1u : scene == kPan3 ? 3u : scene == kPan8 ? 8u : scene == kPan24 ? fast : 0u;
        };
        // shake: the picture at a random offset of up to +-16 x +-10 px every frame (a camera shake,
        // an impact: the motion is nothing like the last frame's); whip: a pan accelerating by
        // 1 px/frame each frame (the prediction from the last motion always short)
        auto shake_offset = [](uint32_t f, int& ox, int& oy) {
            uint32_t s = 2654435761u * (f + 1);
            s ^= s >> 13; s *= 1274126177u; s ^= s >> 16;
            ox = static_cast<int>(s % 33) - 16;
            oy = static_cast<int>((s >> 8) % 21) - 10;
        };
        // the picture bilinear at (fx, fy), x wrapped like the pans, y clamped
        auto sample = [&](float fx, float fy, uint32_t c) {
            fx = std::fmod(std::fmod(fx, static_cast<float>(w)) + w, static_cast<float>(w));
            fy = std::min(std::max(fy, 0.0f), static_cast<float>(h - 1));
            const uint32_t x0 = std::min(static_cast<uint32_t>(fx), w - 1), y0 = static_cast<uint32_t>(fy);
            const uint32_t x1 = (x0 + 1) % w, y1 = std::min(y0 + 1, h - 1);
            const float ax = fx - x0, ay = fy - y0;
            auto at = [&](uint32_t px, uint32_t py) { return color.rgba[(py * w + px) * 4 + c]; };
            return (1 - ay) * ((1 - ax) * at(x0, y0) + ax * at(x1, y0)) + ay * ((1 - ax) * at(x0, y1) + ax * at(x1, y1));
        };
        auto picture = [&](Scene scene, uint32_t f) {
            if (scene == kCapture) return sequence[f];
            Image in = color;
            if (scene == kFollow) {
                const uint32_t x_lo = w * 3 / 8, x_hi = w * 5 / 8, y_lo = h / 4, y_hi = h * 3 / 4;
                const float k = 6.2831853f / 48.0f, phase = 0.7f * f;
                for (uint32_t y = 0; y < h; ++y) {
                    for (uint32_t x = 0; x < w; ++x) {
                        const bool body = x >= x_lo && x < x_hi && y >= y_lo && y < y_hi;
                        for (uint32_t c = 0; c < 4; ++c) {
                            in.rgba[(y * w + x) * 4 + c] =
                                body ? sample(x - 5.0f * std::sin(k * y + phase), y - 5.0f * std::sin(k * x + phase), c)
                                     : color.rgba[(y * w + (x + w - (16 * f) % w) % w) * 4 + c];
                        }
                    }
                }
                return in;
            }
            if (scene == kShake || scene == kWhip || scene == kSwing) {
                int ox = 0, oy = 0;
                if (scene == kShake) shake_offset(f, ox, oy);
                else if (scene == kSwing) ox = static_cast<int>(std::lround(96.0 * std::sin(6.2831853 * f / 20.0)));
                else ox = static_cast<int>(f * (f + 1) / 2);
                for (uint32_t y = 0; y < h; ++y) {
                    const uint32_t sy = static_cast<uint32_t>(std::clamp(static_cast<int>(y) - oy, 0, static_cast<int>(h) - 1));
                    for (uint32_t x = 0; x < w; ++x) {
                        const uint32_t sx = static_cast<uint32_t>(((static_cast<int>(x) - ox) % static_cast<int>(w) + static_cast<int>(w)) % static_cast<int>(w));
                        for (uint32_t c = 0; c < 4; ++c) in.rgba[(y * w + x) * 4 + c] = color.rgba[(sy * w + sx) * 4 + c];
                    }
                }
                return in;
            }
            if (scene == kWobble || scene == kWobble2 || scene == kRotate || scene == kZoom) {
                // the source position of each pixel, bilinear (clamped):
                //   wobble  a non-rigid motion (cloth, limbs): 2 px/frame to the right plus a 3 px
                //           sinusoidal deformation (wavelength 96 px) whose phase moves 0.35 rad a
                //           frame, so the displacement varies by ~1.6 px across an 8x8 block
                //  wobble 2 the same, 5 px at a wavelength of 48 px: ~5 px across a block
                //   rotate  0.6 degrees a frame about the centre (~3 px/frame at the edges)
                const float cx = 0.5f * (w - 1), cy = 0.5f * (h - 1);
                const float k = 6.2831853f / (scene == kWobble2 ? 48.0f : 96.0f), amp = scene == kWobble2 ? 5.0f : 3.0f;
                const float angle = 0.6f * 3.14159265f / 180.0f * f, ca = std::cos(angle), sa = std::sin(angle);
                for (uint32_t y = 0; y < h; ++y) {
                    for (uint32_t x = 0; x < w; ++x) {
                        float fx, fy;
                        if (scene == kZoom) {
                            const float s = std::pow(1.02f, -static_cast<float>(f));
                            fx = cx + s * (x - cx);
                            fy = cy + s * (y - cy);
                        } else if (scene != kRotate) {
                            fx = x - 2.0f * f - amp * std::sin(k * y + 0.35f * f);
                            fy = y - amp * std::sin(k * x + 0.35f * f);
                        } else {
                            fx = cx + ca * (x - cx) + sa * (y - cy);
                            fy = cy - sa * (x - cx) + ca * (y - cy);
                        }
                        fx = std::fmod(std::fmod(fx, static_cast<float>(w)) + w, static_cast<float>(w));   // wrap like the pans
                        fy = std::min(std::max(fy, 0.0f), static_cast<float>(h - 1));
                        const uint32_t x0 = std::min(static_cast<uint32_t>(fx), w - 1), y0 = static_cast<uint32_t>(fy);
                        const uint32_t x1 = (x0 + 1) % w, y1 = std::min(y0 + 1, h - 1);
                        const float ax = fx - x0, ay = fy - y0;
                        for (uint32_t c = 0; c < 4; ++c) {
                            auto at = [&](uint32_t px, uint32_t py) { return color.rgba[(py * w + px) * 4 + c]; };
                            in.rgba[(y * w + x) * 4 + c] = (1 - ay) * ((1 - ax) * at(x0, y0) + ax * at(x1, y0)) +
                                                           ay * ((1 - ax) * at(x0, y1) + ax * at(x1, y1));
                        }
                    }
                }
                return in;
            }
            for (uint32_t y = 0; y < h; ++y) {
                for (uint32_t x = 0; x < w; ++x) {
                    uint32_t sx = x;
                    if (speed_of(scene) > 0) sx = (x + w - (speed_of(scene) * f) % w) % w;
                    if (scene == kObject) {   // a w/3 square from the middle, drawn moved 2f to the right
                        const uint32_t side = w / 3, ox = side + 2 * f;
                        if (x >= ox && x < ox + side && y >= h / 3 && y < h / 3 + side) sx = x - 2 * f;
                    }
                    if (scene == kCut && f >= cut) sx = w - 1 - x;
                    for (uint32_t c = 0; c < 4; ++c) in.rgba[(y * w + x) * 4 + c] = color.rgba[(y * w + sx) * 4 + c];
                }
            }
            return in;
        };
        auto run = [&](Scene scene, uint32_t interval, bool warp, std::vector<std::vector<float>>& corr) {
            graph.SetWarp(warp);
            if (!graph.SetInterval(interval)) return false;
            corr.assign(frames, {});
            for (uint32_t f = 0; f < frames; ++f) {
                Image in = picture(scene, f), out;
                if (!Upload(rm, queue, graph.GetInputTexture(), in) || !runner.Submit(graph, f)) return false;
                runner.Drain();
                if (!Download(rm, queue, graph.GetOutputTexture(), out)) return false;
                corr[f].resize(static_cast<size_t>(w) * h * 3);
                for (size_t i = 0, j = 0; i < in.rgba.size(); ++i) {
                    if (i % 4 != 3) corr[f][j++] = out.rgba[i] - in.rgba[i];
                }
            }
            return true;
        };
        auto compare = [&](const std::vector<float>& got, const std::vector<float>& want, double& error, double& gain) {
            double dd = 0, ww = 0, gw = 0;
            for (size_t i = 0; i < want.size(); ++i) {
                const double d = static_cast<double>(got[i]) - want[i];
                dd += d * d;
                ww += static_cast<double>(want[i]) * want[i];
                gw += static_cast<double>(got[i]) * want[i];
            }
            error += std::sqrt(dd / std::max(ww, 1e-20));
            gain += gw / std::max(ww, 1e-20);
        };
        std::cout << std::fixed << std::setprecision(3) << "  interval " << n
                  << " against interval 1: error rms(c_N - c_1)/rms(c_1), gain (1 = all of the correction kept)\n";
        const std::string fast_name = "pan " + std::to_string(fast) + " px/frame";
        const char* names[] = {"pan 1 px/frame", "pan 3 px/frame", "pan 8 px/frame", "object 2 px/frame",
                               "wobble (non-rigid)", "wobble 2 (stronger)", "rotate 0.6 deg/frame", "shake +-16 px",
                               "whip (accelerating)", fast_name.c_str(), "swing (+-96 px)", "follow (character)",
                               "cut (frames 12-15)", "zoom 2 %/frame", "captured sequence"};
        bool warp_ok = true;
        auto pulse = [&](const std::vector<std::vector<float>>& seq, uint32_t first, uint32_t last) {
            auto r = [](const std::vector<float>& a) {
                double t = 0;
                for (float v : a) t += static_cast<double>(v) * v;
                return std::sqrt(t / a.size());
            };
            double change = 0, level = 0;
            for (uint32_t f = first; f < last; ++f) {
                level += r(seq[f]);
                if (f > first) change += std::abs(r(seq[f]) - r(seq[f - 1]));
            }
            return (change / (last - first - 1)) / std::max(level / (last - first), 1e-12);
        };
        for (const Scene scene : {kPan1, kPan3, kPan8, kObject, kWobble, kWobble2, kRotate, kShake, kWhip, kPan24, kSwing,
                                  kFollow, kCut, kZoom, kCapture}) {
            if ((scene == kCapture) != !sequence.empty()) continue;   // the captured sequence alone, or not at all
            if (!interval_scenes.empty() &&
                interval_scenes.find("," + std::to_string(static_cast<int>(scene)) + ",") == std::string::npos) {
                continue;
            }
            std::vector<std::vector<float>> ref, on, off;
            bool jump_ok = true;
            if (!run(scene, 1, true, ref) || !run(scene, n, true, on) || !run(scene, n, false, off)) {
                std::cerr << "[FATAL] interval check frame failed: " << graph.LastError() << "\n";
                return 1;
            }
            if (scene == kPan1) std::cout << "  " << graph.PartSummary() << "\n";
            const uint32_t first = scene == kCut ? cut : warm, last = scene == kCut ? cut + 4 : frames;
            double e_on = 0, g_on = 0, e_off = 0, g_off = 0;
            for (uint32_t f = first; f < last; ++f) {
                compare(on[f], ref[f], e_on, g_on);
                compare(off[f], ref[f], e_off, g_off);
            }
            const double k = 1.0 / (last - first);
            if (scene == kCapture) {   // per frame: where along the capture it goes wrong
                auto rms = [](const std::vector<float>& a) {
                    double t = 0;
                    for (float v : a) t += static_cast<double>(v) * v;
                    return std::sqrt(t / a.size());
                };
                std::cout << "  frame   rms c_1   rms c_N   error  gain    without: error  gain\n";
                for (uint32_t f = first; f < last; ++f) {
                    double e1 = 0, g1 = 0, e2 = 0, g2 = 0;
                    compare(on[f], ref[f], e1, g1);
                    compare(off[f], ref[f], e2, g2);
                    std::cout << "  " << std::setw(5) << f + sequence_first << "   " << std::setprecision(4) << rms(ref[f]) << "    "
                              << rms(on[f]) << "    " << std::setprecision(3) << e1 << "  " << g1 << "            " << e2 << "  "
                              << g2 << "\n";
                }
            }
            std::cout << "  " << std::left << std::setw(20) << names[scene] << std::right << " warp: error " << e_on * k
                      << ", gain " << g_on * k << "   without: error " << e_off * k << ", gain " << g_off * k;
            if (scene != kCut) {
                std::cout << "   pulse " << pulse(on, first, last) << " / " << pulse(off, first, last) << " (interval 1: "
                          << pulse(ref, first, last) << ")";
            }
            // overshoot: the share of values beyond anything the network itself gave that frame
            // (|c| > 1.25 max |c_1| + 1/255): a correction pushed out of range shows as a white or
            // black blotch (a moved head is always in range; its slope correction need not be)
            auto overshoot = [&](const std::vector<std::vector<float>>& seq) {
                double share = 0;
                for (uint32_t f = first; f < last; ++f) {
                    float m = 0;
                    for (float v : ref[f]) m = std::max(m, std::abs(v));
                    const float limit = 1.25f * m + 1.0f / 255.0f;
                    size_t count = 0;
                    for (float v : seq[f]) count += std::abs(v) > limit ? 1 : 0;
                    share += static_cast<double>(count) / seq[f].size() / (last - first);
                }
                return 100.0 * share;
            };
            const double over_on = overshoot(on);
            // holes: the share of pixels with a clear correction at interval 1 (> 2/255) that show
            // less than a quarter of it (a block whose motion failed and was faded out: a box
            // brighter or darker than its neighbours, the "white boxes" of the 2026-11-11 game test)
            auto holes = [&](const std::vector<std::vector<float>>& seq) {
                double share = 0;
                for (uint32_t f = first; f < last; ++f) {
                    size_t count = 0, strong = 0;
                    for (size_t i = 0; i + 2 < seq[f].size(); i += 3) {
                        const float r = std::abs(ref[f][i]) + std::abs(ref[f][i + 1]) + std::abs(ref[f][i + 2]);
                        const float g = std::abs(seq[f][i]) + std::abs(seq[f][i + 1]) + std::abs(seq[f][i + 2]);
                        if (r > 6.0f / 255.0f) {
                            ++strong;
                            count += g < 0.25f * r ? 1 : 0;
                        }
                    }
                    share += strong ? static_cast<double>(count) / strong / (last - first) : 0.0;
                }
                return 100.0 * share;
            };
            std::cout << std::setprecision(2) << "   holes " << holes(on) << " %" << std::setprecision(3);
            // false detail: the correction's detail (minus a 17x17 box mean) that interval 1 does
            // not have there, rms(d_N - k d_1) / rms(d_1) with k = <d_N, d_1> / <d_1, d_1> (also
            // shown: "kept"). A moved correction slightly off puts its detail in the wrong place
            // (a ghost, swimming under a fast camera); one faded to the smooth regional model loses
            // detail (kept < 1) but places none falsely. The error above counts both alike.
            auto detail_of = [&](const std::vector<float>& c) {
                const int r = 8;
                std::vector<float> tmp(c.size()), out(c.size());
                for (uint32_t y = 0; y < h; ++y) {   // horizontal box mean, clamped
                    for (uint32_t ch = 0; ch < 3; ++ch) {
                        double s = 0;
                        int n = 0;
                        for (int x = -r; x <= r; ++x) {
                            if (x >= 0 && x < static_cast<int>(w)) { s += c[(y * w + x) * 3 + ch]; ++n; }
                        }
                        for (uint32_t x = 0; x < w; ++x) {
                            tmp[(y * w + x) * 3 + ch] = static_cast<float>(s / n);
                            const int add = static_cast<int>(x) + r + 1, sub = static_cast<int>(x) - r;
                            if (add < static_cast<int>(w)) { s += c[(y * w + add) * 3 + ch]; ++n; }
                            if (sub >= 0) { s -= c[(y * w + sub) * 3 + ch]; --n; }
                        }
                    }
                }
                for (uint32_t x = 0; x < w; ++x) {   // vertical
                    for (uint32_t ch = 0; ch < 3; ++ch) {
                        double s = 0;
                        int n = 0;
                        for (int y = -r; y <= r; ++y) {
                            if (y >= 0 && y < static_cast<int>(h)) { s += tmp[(y * w + x) * 3 + ch]; ++n; }
                        }
                        for (uint32_t y = 0; y < h; ++y) {
                            const size_t i = (static_cast<size_t>(y) * w + x) * 3 + ch;
                            out[i] = c[i] - static_cast<float>(s / n);
                            const int add = static_cast<int>(y) + r + 1, sub = static_cast<int>(y) - r;
                            if (add < static_cast<int>(h)) { s += tmp[(add * w + x) * 3 + ch]; ++n; }
                            if (sub >= 0) { s -= tmp[(sub * w + x) * 3 + ch]; --n; }
                        }
                    }
                }
                return out;
            };
            double false_detail = 0, kept = 0;
            for (uint32_t f = first; f < last; ++f) {
                const std::vector<float> dn = detail_of(on[f]), d1 = detail_of(ref[f]);
                double nn = 0, n1 = 0, s11 = 0;
                for (size_t i = 0; i < d1.size(); ++i) {
                    n1 += static_cast<double>(dn[i]) * d1[i];
                    s11 += static_cast<double>(d1[i]) * d1[i];
                    nn += static_cast<double>(dn[i]) * dn[i];
                }
                const double k1 = n1 / std::max(s11, 1e-20);
                // |d_N - k d_1|^2 = |d_N|^2 - 2k <d_N, d_1> + k^2 |d_1|^2
                false_detail += std::sqrt(std::max(nn - 2 * k1 * n1 + k1 * k1 * s11, 0.0) / std::max(s11, 1e-20)) / (last - first);
                kept += k1 / (last - first);
            }
            std::cout << "   false detail " << false_detail << " (kept " << kept << ")";
            if (!interval_dump.empty()) {
                // DIR/<scene>_<frame>_{in,on,off,ref}.pfm for the frames compared (corrections + 0.5)
                for (uint32_t f = first; f < last; ++f) {
                    const std::string stem = interval_dump + "/" + std::to_string(static_cast<int>(scene)) + "_" + std::to_string(f);
                    WritePfm(stem + "_in.pfm", picture(scene, f));
                    for (const auto& [suffix, seq] : {std::pair<const char*, const std::vector<std::vector<float>>*>{"_on", &on},
                                                      {"_off", &off}, {"_ref", &ref}}) {
                        Image img;
                        img.width = w;
                        img.height = h;
                        img.rgba.resize(static_cast<size_t>(w) * h * 4, 1.0f);
                        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                            for (int c = 0; c < 3; ++c) img.rgba[i * 4 + c] = (*seq)[f][i * 3 + c] + 0.5f;
                        }
                        WritePfm(stem + suffix + ".pfm", img);
                    }
                }
            }
            std::cout << std::setprecision(4) << "   overshoot " << over_on << " % / " << overshoot(off) << " %"
                      << std::setprecision(3);
            if (speed_of(scene) > 0) {
                // the jump: the correction's change between consecutive frames with the motion taken
                // out, rms(c[f](x) - c[f-1](x - speed)) / rms(c[f]), mean and the largest (a new head
                // every n frames changes it at once where interval 1 changes it a little every frame)
                const uint32_t speed = speed_of(scene);
                auto jump = [&](const std::vector<std::vector<float>>& seq, double& mean, double& peak) {
                    mean = 0;
                    peak = 0;
                    for (uint32_t f = first + 1; f < last; ++f) {
                        double dd = 0, cc = 0;
                        for (uint32_t y = 0; y < h; ++y) {
                            for (uint32_t x = speed + 2; x + 2 < w; ++x) {
                                for (uint32_t c = 0; c < 3; ++c) {
                                    const double a = seq[f][(y * w + x) * 3 + c], b = seq[f - 1][(y * w + x - speed) * 3 + c];
                                    dd += (a - b) * (a - b);
                                    cc += a * a;
                                }
                            }
                        }
                        const double j = std::sqrt(dd / std::max(cc, 1e-20));
                        mean += j / (last - first - 1);
                        peak = std::max(peak, j);
                    }
                };
                double jm_on, jp_on, jm_ref, jp_ref;
                jump(on, jm_on, jp_on);
                jump(ref, jm_ref, jp_ref);
                std::cout << "   jump " << jm_on << " (max " << jp_on << "), interval 1 " << jm_ref << " (max " << jp_ref << ")";
                // a new head every n frames must not jump more than the network does every frame at
                // interval 1 (game frame, pans: the head warp alone 0.16-0.20 against 0.12-0.19;
                // with the colour slope and the blend of references 0.11-0.13)
                const NeuralOptions& o = graph.GetOptions();
                jump_ok = !o.warp_model || o.warp_blend >= 1.0f || jp_on <= 1.1 * jp_ref;
            }
            if (scene == kRotate || scene == kZoom) {
                // the jump along the known motion: where frame f's pixel x was in frame f - 1
                // (rotate: one step of the rotation; zoom: (x - c) / 1.02 + c), the correction
                // there bilinear; rms(c[f](x) - c[f-1](y)) / rms(c[f]) away from the borders. A
                // correction that stays on the content is ~0 (interval 1 shows the network's own
                // change); one that swims over it is large
                const float cx = 0.5f * (w - 1), cy = 0.5f * (h - 1);
                const float a = 0.6f * 3.14159265f / 180.0f, ca = std::cos(a), sa = std::sin(a);
                auto jump_mapped = [&](const std::vector<std::vector<float>>& seq, double& mean, double& peak) {
                    mean = 0;
                    peak = 0;
                    for (uint32_t f = first + 1; f < last; ++f) {
                        double dd = 0, cc = 0;
                        for (uint32_t y = 8; y + 8 < h; ++y) {
                            for (uint32_t x = 8; x + 8 < w; ++x) {
                                float px, py;
                                if (scene == kZoom) {
                                    px = cx + (x - cx) / 1.02f;
                                    py = cy + (y - cy) / 1.02f;
                                } else {
                                    px = cx + ca * (x - cx) + sa * (y - cy);
                                    py = cy - sa * (x - cx) + ca * (y - cy);
                                }
                                const uint32_t x0 = static_cast<uint32_t>(px), y0 = static_cast<uint32_t>(py);
                                const float ax = px - x0, ay = py - y0;
                                for (uint32_t c = 0; c < 3; ++c) {
                                    auto at = [&](uint32_t qx, uint32_t qy) { return seq[f - 1][(qy * w + qx) * 3 + c]; };
                                    const double b = (1 - ay) * ((1 - ax) * at(x0, y0) + ax * at(x0 + 1, y0)) +
                                                     ay * ((1 - ax) * at(x0, y0 + 1) + ax * at(x0 + 1, y0 + 1));
                                    const double v = seq[f][(y * w + x) * 3 + c];
                                    dd += (v - b) * (v - b);
                                    cc += v * v;
                                }
                            }
                        }
                        const double j = std::sqrt(dd / std::max(cc, 1e-20));
                        mean += j / (last - first - 1);
                        peak = std::max(peak, j);
                    }
                };
                double jm_on, jp_on, jm_ref, jp_ref;
                jump_mapped(on, jm_on, jp_on);
                jump_mapped(ref, jm_ref, jp_ref);
                std::cout << "   jump " << jm_on << " (max " << jp_on << "), interval 1 " << jm_ref << " (max " << jp_ref << ")";
            }
            if (speed_of(scene) > 0) {
                // a perfect warp: the interval-1 correction of the frame the shown head was computed
                // from (heads complete on frames n, 2n, ... from frames 1, n + 1, ...), moved exactly;
                // what is left is the network's own shift variance
                const uint32_t speed = speed_of(scene);
                double e_ideal = 0, g_ideal = 0;
                for (uint32_t f = first; f < last; ++f) {
                    const uint32_t src = f < n ? 0 : f - (n - 1) - (f - n) % n;
                    const uint32_t shift = (f - src) * speed;
                    std::vector<float> moved(ref[f].size(), 0.0f), want(ref[f].size(), 0.0f);
                    for (uint32_t y = 0; y < h; ++y) {
                        for (uint32_t x = shift + 2; x + 2 < w; ++x) {
                            for (uint32_t c = 0; c < 3; ++c) {
                                moved[(y * w + x) * 3 + c] = ref[src][(y * w + x - shift) * 3 + c];
                                want[(y * w + x) * 3 + c] = ref[f][(y * w + x) * 3 + c];
                            }
                        }
                    }
                    compare(moved, want, e_ideal, g_ideal);
                }
                std::cout << "   perfect warp: error " << e_ideal * k << ", gain " << g_ideal * k;
            }
            // gate: the warp must beat the unmoved head on every scene (meaningful with the real
            // weights only: the synthetic network's correction hardly depends on the picture)
            // (not on a captured sequence: what it shows is the question, not a known motion)
            const bool ok = scene == kCapture || (e_on < e_off && jump_ok);
            warp_ok = warp_ok && ok;
            std::cout << (ok ? "" : (jump_ok ? "  FAIL" : "  FAIL (jump)")) << "\n";
        }
        std::cout << "  warp -> " << (warp_ok ? "PASS" : "FAIL") << "\n";
        if (!warp_ok) status = 3;
        graph.SetWarp(options.warp);
        graph.SetInterval(options.interval);
    }

    if (!bench.empty()) {
        report << "## GPU timing\n\n| output | inner | field | pre | network | post | total (GPU) | wall per frame | session create |\n"
               << "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
        for (auto [w, h] : ParseSizes(bench)) {
            if (!graph.EnsureExtent(w, h, kFormat)) {
                std::cout << "  " << w << "x" << h << ": SKIP (" << graph.LastError() << ")\n";
                report << "| " << w << "x" << h << " | - | - | skipped: " << graph.LastError() << " |||||\n";
                continue;
            }
            const NeuralExtent& e = graph.GetExtent();
            std::cout << "  " << w << "x" << h << ": "
                      << (graph.IsNative() ? "native" : graph.IsHybrid() ? "hybrid" : "onnx") << " "
                      << graph.ModelPath() << "\n";
            if (!Upload(rm, queue, graph.GetInputTexture(), TestImage(w, h))) {
                std::cerr << "[FATAL] upload\n";
                return 1;
            }
            std::vector<double> pre, net, post, total, wall;
            for (uint32_t f = 0; f < 3; ++f) {                     // the first runs build the DML graph
                if (!runner.Submit(graph, f)) {
                    std::cerr << "[FATAL] frame failed: " << graph.LastError() << "\n";
                    return 1;
                }
            }
            runner.Drain();
            // throughput: frames back to back; GPU timings sampled from the slot ring as it
            // comes round (a slot's timestamps are complete once Submit waited for it)
            auto started = std::chrono::high_resolution_clock::now();
            for (uint32_t f = 3; f < frames + 3; ++f) {
                if (!runner.Submit(graph, f)) {
                    std::cerr << "[FATAL] frame failed: " << graph.LastError() << "\n";
                    return 1;
                }
                NeuralTimings t;
                if (f >= 3 + FrameRunner::kSlots && graph.ReadTimings(FrameRunner::Slot(f + 1), t)) {
                    pre.push_back(t.pre_ms);
                    net.push_back(t.network_ms);
                    post.push_back(t.post_ms);
                    total.push_back(t.total_ms);
                }
            }
            runner.Drain();
            if (profile_ends && graph.Hybrid()) {
                // one more frame with per-dispatch timestamps, summed by label
                graph.Hybrid()->EnableProfiling(queue.GetQueue());
                runner.Submit(graph, 0);
                runner.Drain();
                std::vector<std::pair<std::string, double>> ops;
                if (graph.Hybrid()->ReadProfile(ops)) {
                    std::map<std::string, double> by_kind, by_part;
                    double sum = 0;
                    for (const auto& [label, ms] : ops) {
                        by_part[label.substr(0, label.find(' '))] += ms;   // front | core | back
                        // "front b15 C256 ffn" -> "C256 ffn"; "front b0 fused32" -> "fused32 b0"
                        std::string kind = label.substr(label.find(' ') + 1);
                        const size_t space = kind.find(' ');
                        if (kind[0] == 'b' && space != std::string::npos) kind = kind.substr(space + 1);
                        by_kind[kind] += ms;
                        sum += ms;
                    }
                    std::cout << "    per-dispatch profile (" << ops.size() << " dispatches, " << sum << " ms):\n";
                    std::cout << "    parts:";
                    for (const auto& [part, ms] : by_part) std::cout << " " << part << " " << ms << " ms";
                    std::cout << "\n";
                    for (const auto& [kind, ms] : by_kind) std::cout << "      " << kind << ": " << ms << " ms\n";
                }
            }
            wall.push_back(std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() - started).count() / frames);
            std::cout << std::fixed << std::setprecision(2) << "  " << w << "x" << h << " (inner " << e.inner_w << "x"
                      << e.inner_h << ", field " << e.field_w << "x" << e.field_h << "): pre " << Median(pre)
                      << " ms, network " << Median(net) << " ms, post " << Median(post) << " ms, total "
                      << Median(total) << " ms (min " << *std::min_element(total.begin(), total.end()) << ", max "
                      << *std::max_element(total.begin(), total.end()) << "), wall/frame " << Median(wall) << " ms\n";
            report << std::fixed << std::setprecision(2) << "| " << w << "x" << h << " | " << e.inner_w << "x"
                   << e.inner_h << " | " << e.field_w << "x" << e.field_h << " | " << Median(pre) << " ms | "
                   << Median(net) << " ms | " << Median(post) << " ms | " << Median(total) << " ms | "
                   << Median(wall) << " ms | " << std::setprecision(0) << graph.LastSessionCreateMs() << " ms |\n";
        }
        report << "\nFrames run back to back (" << FrameRunner::kSlots << " in flight) for " << frames
               << " frames after 3 warm-up frames; GPU columns are medians of the timestamps, `wall per frame` is the "
               << "throughput. `network` is the span on the queue between the pre-pass end and the post-pass begin: "
               << "the ORT/DirectML graph plus any submission gap.\n";
    }

    std::ofstream(output_dir + "/neural-report.md") << report.str();
    std::cout << "  report: " << output_dir << "/neural-report.md\n";
    return status;
}

} // namespace dlss_nr::neural
