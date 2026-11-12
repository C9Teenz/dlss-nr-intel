#pragma once

#include <cstdint>
#include <vector>

namespace dlss_nr::neural {

// Extents of one neural pass.
//   output: the texture we are handed (swapchain back buffer)
//   inner:  the frame the network is given (output at render_scale)
//   field:  the padded network field the ONNX graph was exported for
struct NeuralExtent {
    uint32_t output_w = 0;
    uint32_t output_h = 0;
    uint32_t inner_w = 0;
    uint32_t inner_h = 0;
    uint32_t field_w = 0;
    uint32_t field_h = 0;

    bool operator==(const NeuralExtent&) const = default;
};

constexpr uint32_t kVendorMinimumExtent = 320;
constexpr uint32_t kGraphMinimumExtent = 128;

// The vendor's field for a frame (tools/hnet/frame.py network_geometry): each side aligned
// to the graph's own reductions and at least `minimum`, one alignment wider when both sides
// are four alignments (a field that pools to a padding-free bottleneck runs 25-30 % weaker),
// then rounded up to 64.
void NetworkField(uint32_t width, uint32_t height, uint32_t& field_w, uint32_t& field_h,
                  uint32_t minimum = kVendorMinimumExtent);

// The frame handed to the network for an output at `scale`: of the frames at least the
// scale's own size (aspect kept), the one on the cheapest field, and of those the largest
// (nr_frame.render_extent). scale >= 1 is the output itself.
NeuralExtent ComputeExtent(uint32_t output_w, uint32_t output_h, float scale,
                           uint32_t minimum = kVendorMinimumExtent);

// The three deterministic Gaussian noise channels of the features (MLX-DLSS
// features.deterministic_noise), FP16 bits, packed 4 halves per field pixel (n0 n1 n2 0).
std::vector<uint16_t> DeterministicNoise(uint32_t field_w, uint32_t field_h, uint32_t frame_index = 0);

uint16_t FloatToHalf(float value);   // round to nearest even, overflow to inf
float HalfToFloat(uint16_t bits);

} // namespace dlss_nr::neural
