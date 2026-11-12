#include "neural/neural_field.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dlss_nr::neural {

namespace {

uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

uint32_t FieldAlignment(uint32_t extent) {
    uint32_t reductions = 0;
    uint32_t size = extent;
    for (int level = 0; level < 6; ++level) {
        uint32_t half = CeilDiv((size + 1) / 2, 4) * 4;
        reductions += half < size ? 1u : 0u;
        reductions += (level == 0 && half % 8 != 0) ? 1u : 0u;
        size = half;
    }
    return 1u << reductions;
}

uint64_t FieldCost(uint32_t w, uint32_t h, uint32_t minimum) {
    uint32_t fw = 0, fh = 0;
    NetworkField(w, h, fw, fh, minimum);
    return static_cast<uint64_t>(fw) * fh;
}

uint32_t DynamicShiftMix(uint32_t value) {
    uint32_t shift = (value >> 28) + 4u;
    uint32_t mixed = value ^ (value >> shift);
    return mixed * 0x108EF2D9u;
}

float Uniform24(uint32_t value) {
    uint32_t mixed = DynamicShiftMix(value);
    uint32_t bits = (mixed >> 30) ^ (mixed >> 8);
    return static_cast<float>(bits + 1u) * 5.960464477539063e-8f;
}

} // namespace

uint16_t FloatToHalf(float value) {
    uint32_t f;
    std::memcpy(&f, &value, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    uint32_t abs = f & 0x7FFFFFFFu;
    if (abs >= 0x7F800000u) {                       // inf / nan
        return static_cast<uint16_t>(sign | 0x7C00u | (abs > 0x7F800000u ? 0x200u : 0u));
    }
    if (abs >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);   // rounds to inf
    if (abs < 0x38800000u) {                        // half subnormal (or zero)
        if (abs < 0x33000000u) return static_cast<uint16_t>(sign);
        uint32_t exponent = abs >> 23;
        uint32_t mantissa = (abs & 0x7FFFFFu) | 0x800000u;
        uint32_t shift = 126u - exponent;           // value / 2^-24 = mantissa * 2^(exponent - 126)
        uint32_t rounded = mantissa >> shift;
        uint32_t remainder = mantissa & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rebased = abs - 0x38000000u;           // exponent 127 -> 15
    uint32_t rounded = rebased >> 13;
    uint32_t remainder = rebased & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) ++rounded;
    return static_cast<uint16_t>(sign | rounded);
}

float HalfToFloat(uint16_t bits) {
    uint32_t sign = (bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1Fu;
    uint32_t mantissa = bits & 0x3FFu;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign;
        } else {
            float value = std::ldexp(static_cast<float>(mantissa), -24);
            std::memcpy(&f, &value, 4);
            f |= sign;
        }
    } else if (exponent == 31) {
        f = sign | 0x7F800000u | (mantissa << 13);
    } else {
        f = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

void NetworkField(uint32_t width, uint32_t height, uint32_t& field_w, uint32_t& field_h, uint32_t minimum) {
    const uint32_t floor = std::max(kGraphMinimumExtent, minimum);
    const uint32_t aw = FieldAlignment(width);
    const uint32_t ah = FieldAlignment(height);
    uint32_t fw = std::max(floor, CeilDiv(width, aw) * aw);
    uint32_t fh = std::max(floor, CeilDiv(height, ah) * ah);
    if (fw % (4 * aw) == 0 && fh % (4 * ah) == 0) fw += aw;
    field_w = CeilDiv(fw, 64) * 64;
    field_h = CeilDiv(fh, 64) * 64;
}

NeuralExtent ComputeExtent(uint32_t output_w, uint32_t output_h, float scale, uint32_t minimum) {
    NeuralExtent e{};
    e.output_w = output_w;
    e.output_h = output_h;
    uint32_t best_w = output_w, best_h = output_h;
    if (scale < 1.0f) {
        best_w = std::max<uint32_t>(64, static_cast<uint32_t>(std::lround(output_w * scale)));
        best_h = std::max<uint32_t>(64, static_cast<uint32_t>(std::lround(output_h * scale)));
        uint64_t cheapest = FieldCost(best_w, best_h, minimum);
        for (uint32_t w = best_w + 1; w <= output_w; ++w) {
            uint32_t h = std::max<uint32_t>(
                64, static_cast<uint32_t>(std::lround(static_cast<double>(output_h) * w / output_w)));
            uint64_t price = FieldCost(w, h, minimum);
            if (price <= cheapest) {
                best_w = w;
                best_h = h;
                cheapest = price;
            }
        }
        best_w = std::min(best_w, output_w);
        best_h = std::min(best_h, output_h);
    }
    e.inner_w = best_w;
    e.inner_h = best_h;
    NetworkField(best_w, best_h, e.field_w, e.field_h, minimum);
    return e;
}

std::vector<uint16_t> DeterministicNoise(uint32_t field_w, uint32_t field_h, uint32_t frame_index) {
    std::vector<uint16_t> out(static_cast<size_t>(field_w) * field_h * 4, 0);
    const uint32_t frame_term = static_cast<uint32_t>(frame_index * 0x9E3779B9ull);
    const float tau = 6.2831854820251465f;
    for (uint32_t y = 0; y < field_h; ++y) {
        for (uint32_t x = 0; x < field_w; ++x) {
            uint32_t seed = y * 0xD8163841u;
            seed ^= x * 0x8DA6B343u;
            seed ^= frame_term;
            seed ^= 0x243F6A88u;
            uint32_t multiplied = DynamicShiftMix(seed);
            uint32_t mixed = multiplied ^ (multiplied >> 22);
            float radius_a_uniform = Uniform24(mixed * 0xCAA5B80Du + 0x21DD796Bu);
            float angle_b_uniform = Uniform24(mixed * 0x83232C31u + 0x3463E0ACu);
            float radius_b_uniform = Uniform24(mixed * 0x2C9277B5u + 0xAC564B05u);
            float angle_a_uniform = Uniform24(mixed * 0xFA6DC5F9u + 0x4712A88Eu);
            float radius_a = std::sqrt(-2.0f * std::log(radius_a_uniform));
            float radius_b = std::sqrt(-2.0f * std::log(radius_b_uniform));
            float angle_a = tau * angle_a_uniform;
            float angle_b = tau * angle_b_uniform;
            size_t i = (static_cast<size_t>(y) * field_w + x) * 4;
            out[i + 0] = FloatToHalf(radius_b * std::cos(angle_a));
            out[i + 1] = FloatToHalf(radius_b * std::sin(angle_a));
            out[i + 2] = FloatToHalf(radius_a * std::cos(angle_b));
        }
    }
    return out;
}

} // namespace dlss_nr::neural
