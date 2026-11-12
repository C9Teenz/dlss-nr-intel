// DLSS-NR neural rendering: the passes around the network (NeuralGraph, src/neural).
//
//   CSNeuralPre   frame (+ history) -> the 16-channel FP16 features on the padded field
//   CSNeuralPost  head + frame + history -> the composed frame and the next history
//   CSMotion, CSHeadStabilize  (optional, before the post) a new head averaged with the last
//                    ones along the picture's motion (not a CPU-reference pass: our addition)
//
// The CPU reference for both is tools/hnet/frame.py (make_features, compose); the channel
// contract is docs/ARCHITECTURE.md of uzbekunknown/dlss-nr-on-intel, section 2.
//
// Extents: the *output* is the texture we are handed (src_w x src_h); the network sees the
// *inner* frame (inner_w x inner_h, the output at render_scale, box-filtered down), mirrored
// out to the *field* (field_w x field_h). With render_scale 1 inner == output.
//
// Buffers are FP16 packed two to a uint (ByteAddressBuffer): features 16 halves (32 bytes)
// per field pixel, head 4 halves (8 bytes) per field pixel, noise 4 halves per field pixel
// (3 used), history RGBA16F (8 bytes) per output pixel.

cbuffer NeuralConstants : register(b0) {
    uint field_w;
    uint field_h;
    uint inner_w;
    uint inner_h;
    uint src_w;
    uint src_h;
    uint flags;          // bit 0: history valid, bit 1: the stabiliser's previous head valid,
                         // bit 2: its parity, bit 3: the warp moves the colour model (kWarpModel),
                         // bit 4: CSHeadWarp keeps this frame's head (kWarpCapture), bit 5: one is kept
    float style;
    float tone;
    float structure;
    float intensity;
    float blend_scale;   // half(0.73974609375)
    float stabilize;     // CSHeadStabilize: the weight of a new head where the picture stands still
    float tolerance;     // CSHeadStabilize: scales the colour changes counted as still / as a mismatch
    float motion_weight; // CSHeadStabilize: the least weight of a new head where anything moves
    float warp_predict;  // CSWarpMotion: this frame's age of the reference over last frame's (0: no prediction)
    float warp_blend;    // CSRefCommit with kCaptureValid: the weight of the new head against the carried one
    float fade_lo;       // CSHeadWarp with the model: the motion since the reference (field pixels) where the
    float fade_hi;       // moved detail starts / ends fading to the regional model (0: never; speed x age)
};

static const uint kHistoryValid = 1u;
static const uint kStableValid = 2u;
static const uint kStableParity = 4u;   // which half of g_stable holds the previous frame
static const uint kWarpModel = 8u;      // CSHeadWarp moves the reference's colour model, not its head
static const uint kWarpCapture = 16u;   // ... and keeps what it shows (the frame the next head comes from)
static const uint kCaptureValid = 32u;  // CSRefCommit blends the new head with that kept one
static const uint kWarpPyramid = 64u;   // CSWarpMotion also searches around CSWarpPyramid's vector
static const uint kChain = 128u;        // CSWarpMotion centres on the summed steps (CSStepMotion, CSChain)
static const uint kChainPrev = 256u;    // last frame's luma is there (else the step is 0)
static const uint kChainFront = 512u;   // this frame takes the next head's features: the pending sum restarts
static const uint kChainBack = 1024u;   // this frame completes a head: the pending sum becomes the active one
static const uint kChainParity = 2048u; // CSChain reads the sums' half 1 and writes half 0 (else the reverse)
static const uint kFadeNone = 4096u;    // CSHeadWarp's speed fade goes to no correction, not the regional model

Texture2D<float4> g_color : register(t0);
ByteAddressBuffer g_noise : register(t1);
RWByteAddressBuffer g_features : register(u0);   // pre: features (field)
RWTexture2D<float4> g_output : register(u3);     // post: composed frame (output extent)
RWByteAddressBuffer g_head : register(u1);
RWByteAddressBuffer g_history : register(u2);
// stabiliser: two halves of 16 bytes per field pixel — head (4 halves), colour (3 + 1) — the
// previous frame's in one, this frame's written to the other; then a float2 motion per 8x8 block
RWByteAddressBuffer g_stable : register(u4);

float h(float x) { return f16tof32(f32tof16(x)); }

uint pack2(float a, float b) { return f32tof16(a) | (f32tof16(b) << 16); }

float2 unpack2(uint v) { return float2(f16tof32(v & 0xFFFFu), f16tof32(v >> 16)); }

// the vendor's extension: one reflection, then clamped at the origin
uint mirror(uint i, uint extent) {
    return i < extent ? i : (uint)max(0, 2 * (int)extent - 2 - (int)i);
}

float3 scaled_colour(float3 v) {
    return float3(h(h(h(v.x) - 0.5) * 0.125), h(h(h(v.y) - 0.5) * 0.125), h(h(h(v.z) - 0.5) * 0.125));
}

// The inner pixel (x, y) from an output-extent source: the output itself at render_scale 1,
// otherwise the mean of the output pixels its footprint covers (up to 4x4 taps).
float3 inner_colour(uint x, uint y) {
    if (inner_w == src_w && inner_h == src_h) return g_color.Load(int3(x, y, 0)).rgb;
    uint x0 = x * src_w / inner_w, x1 = max(x0 + 1, min(src_w, ((x + 1) * src_w + inner_w - 1) / inner_w));
    uint y0 = y * src_h / inner_h, y1 = max(y0 + 1, min(src_h, ((y + 1) * src_h + inner_h - 1) / inner_h));
    uint sx = max(1u, (x1 - x0 + 3) / 4), sy = max(1u, (y1 - y0 + 3) / 4);
    float3 total = 0;
    float count = 0;
    for (uint yy = y0; yy < y1; yy += sy)
        for (uint xx = x0; xx < x1; xx += sx) {
            total += g_color.Load(int3(xx, yy, 0)).rgb;
            count += 1;
        }
    return total / count;
}

float3 history_at(uint x, uint y) {
    uint2 v = g_history.Load2((y * src_w + x) * 8u);
    return float3(unpack2(v.x), unpack2(v.y).x);
}

float3 inner_history(uint x, uint y) {
    if (inner_w == src_w && inner_h == src_h) return history_at(x, y);
    uint x0 = x * src_w / inner_w, x1 = max(x0 + 1, min(src_w, ((x + 1) * src_w + inner_w - 1) / inner_w));
    uint y0 = y * src_h / inner_h, y1 = max(y0 + 1, min(src_h, ((y + 1) * src_h + inner_h - 1) / inner_h));
    uint sx = max(1u, (x1 - x0 + 3) / 4), sy = max(1u, (y1 - y0 + 3) / 4);
    float3 total = 0;
    float count = 0;
    for (uint yy = y0; yy < y1; yy += sy)
        for (uint xx = x0; xx < x1; xx += sx) {
            total += history_at(xx, yy);
            count += 1;
        }
    return total / count;
}

[numthreads(8, 8, 1)]
void CSNeuralPre(uint3 id : SV_DispatchThreadID) {
    if (id.x >= field_w || id.y >= field_h) return;
    uint x = mirror(id.x, inner_w);
    uint y = mirror(id.y, inner_h);
    uint pixel = id.y * field_w + id.x;

    float3 colour = scaled_colour(saturate(inner_colour(x, y)));
    float3 previous = (flags & kHistoryValid) != 0 ? scaled_colour(saturate(inner_history(x, y))) : colour;
    uint2 noise = g_noise.Load2(pixel * 8u);           // n0 n1 | n2 0

    uint4 a, b;
    a.x = noise.x;                                      // 0, 1   noise
    a.y = (noise.y & 0xFFFFu) | (f32tof16(1.0) << 16);  // 2 noise, 3 constant 1
    a.z = pack2(colour.x, colour.y);                    // 4, 5   frame
    a.w = pack2(colour.z, previous.x);                  // 6 frame, 7 history
    b.x = pack2(previous.y, previous.z);                // 8, 9   history
    b.y = pack2(style, tone);                           // 10, 11
    b.z = pack2(structure, -1.0);                       // 12, 13 skin structure off
    b.w = pack2(-1.0, 0.0);                             // 14 automatic mask off, 15 unused
    g_features.Store4(pixel * 32u, a);
    g_features.Store4(pixel * 32u + 16u, b);
}

float4 head_at(uint x, uint y) {
    uint2 v = g_head.Load2((y * field_w + x) * 8u);
    return float4(unpack2(v.x), unpack2(v.y));
}

// the head at an output pixel: direct at render_scale 1, bilinear over the inner grid otherwise
float4 head_for_output(uint x, uint y) {
    if (inner_w == src_w && inner_h == src_h) return head_at(x, y);
    float fx = clamp((x + 0.5) * inner_w / (float)src_w - 0.5, 0.0, inner_w - 1.0);
    float fy = clamp((y + 0.5) * inner_h / (float)src_h - 0.5, 0.0, inner_h - 1.0);
    uint x0 = (uint)fx, y0 = (uint)fy;
    uint x1 = min(x0 + 1, inner_w - 1), y1 = min(y0 + 1, inner_h - 1);
    float wx = fx - x0, wy = fy - y0;
    float4 top = lerp(head_at(x0, y0), head_at(x1, y0), wx);
    float4 bottom = lerp(head_at(x0, y1), head_at(x1, y1), wx);
    return lerp(top, bottom, wy);
}

// The stabiliser. With the vendor's E4M3 roundings (precision accurate) the head jumps between
// frames that differ only by noise — 27 % of its RMS for ±0.5/255 against 0.7 % without them —
// and flickers; the network also shimmers under motion (it is not shift invariant), in both
// precisions. The vendor's own temporal path is not reproduced yet. So: an exponential average
// of the head per field pixel, taken along the picture's motion (CSMotion, block matching on the
// frames the heads were computed from, features 4-6): where the moved previous frame matches
// this one, a new head only gets weight `stabilize`; where it does not (a disocclusion, a cut, a
// failed match) the new head replaces the average outright, so nothing is dragged. Only the
// low-resolution correction is averaged, never the frame itself.
static const float kStillBelow = 2.0 / 255.0;    // colour change still counted as noise
static const float kMovingAbove = 8.0 / 255.0;   // ... and as a mismatch: the new head alone
static const int kSearch = 6;                    // motion searched within +-6 field pixels a frame
static const float kMotionPenalty = 0.003;       // SAD per pixel of displacement (prefers short)
static const float kStillBias = 0.01;            // SAD margin within which "no motion" wins
// the anti-ghosting box widened by its own size on each side: 0.5 / 1 / 2 kept a 1 px/frame
// pan's shimmer at 0.17 / 0.13 / 0.09 (accurate; 0.05 unclamped), the cut's drag 0.026-0.034
static const float kClampMargin = 1.0;

uint stable_base(bool previous) {
    const bool upper = ((flags & kStableParity) != 0) == previous;
    return upper ? field_w * field_h * 16u : 0u;
}
uint motion_base() { return field_w * field_h * 32u; }

float3 colour_at(uint x, uint y) {   // this frame's scaled colour (features 4-6)
    const uint4 f = g_features.Load4((y * field_w + x) * 32u);
    return float3(unpack2(f.z), unpack2(f.w).x);
}

float3 previous_colour_at(int x, int y) {
    const uint px = (uint)clamp(x, 0, (int)field_w - 1), py = (uint)clamp(y, 0, (int)field_h - 1);
    const uint2 v = g_stable.Load2(stable_base(true) + (py * field_w + px) * 16u + 8u);
    return float3(unpack2(v.x), unpack2(v.y).x);
}

float luma(float3 c) { return dot(c, float3(0.25, 0.5, 0.25)); }

static const uint kTile = 8 + 2 * kSearch;
groupshared float gs_cur[64];
groupshared float gs_prev[kTile * kTile];
groupshared float gs_sad[(2 * kSearch + 1) * (2 * kSearch + 1)];

// One 8x8 block a group: the displacement d minimising SAD(this frame, previous frame at p + d),
// refined to a fraction of a pixel by a parabola through the neighbouring SADs.
[numthreads(8, 8, 1)]
void CSMotion(uint3 group : SV_GroupID, uint3 tid : SV_GroupThreadID, uint index : SV_GroupIndex) {
    const int2 base = int2(group.xy) * 8;
    const uint cx = min((uint)base.x + tid.x, field_w - 1), cy = min((uint)base.y + tid.y, field_h - 1);
    gs_cur[tid.y * 8 + tid.x] = luma(colour_at(cx, cy));
    for (uint i = index; i < kTile * kTile; i += 64) {
        gs_prev[i] = luma(previous_colour_at(base.x - kSearch + (int)(i % kTile), base.y - kSearch + (int)(i / kTile)));
    }
    GroupMemoryBarrierWithGroupSync();
    const uint side = 2 * kSearch + 1;
    for (uint c = index; c < side * side; c += 64) {
        const int dx = (int)(c % side) - kSearch, dy = (int)(c / side) - kSearch;
        const uint origin = (uint)((kSearch + dy) * (int)kTile + kSearch + dx);   // block (0, 0) displaced
        float sad = 0;
        [unroll] for (uint y = 0; y < 8; ++y) {
            [unroll] for (uint x = 0; x < 8; ++x) sad += abs(gs_cur[y * 8 + x] - gs_prev[origin + y * kTile + x]);
        }
        gs_sad[c] = sad + kMotionPenalty * (abs(dx) + abs(dy));
    }
    GroupMemoryBarrierWithGroupSync();
    if (index != 0) return;
    // (the best SAD kept in a register: a groupshared element picked in the loop is a pointer phi
    // the DXIL validator rejects)
    const uint none = kSearch * side + kSearch;   // no motion
    const float still = gs_sad[none];
    uint best = none;
    float best_sad = still;
    for (uint k = 0; k < side * side; ++k) {
        const float v = gs_sad[k];
        if (v < best_sad) {
            best_sad = v;
            best = k;
        }
    }
    if (still <= best_sad + kStillBias) best = none;
    const int bx = (int)(best % side), by = (int)(best / side);
    float2 d = float2(bx - kSearch, by - kSearch);
    // a block that did not move stays exactly in place: a fraction fitted to the noise would
    // resample the average a little off every frame (measured: still-picture flicker 0.034 -> 0.052)
    const bool moved = best != none;
    if (moved && bx > 0 && bx < (int)side - 1) {
        const float l = gs_sad[best - 1], m = gs_sad[best], r = gs_sad[best + 1];
        const float denom = l - 2 * m + r;
        if (denom > 1e-6) d.x += clamp(0.5 * (l - r) / denom, -0.5, 0.5);
    }
    if (moved && by > 0 && by < (int)side - 1) {
        const float u = gs_sad[best - side], m = gs_sad[best], b = gs_sad[best + side];
        const float denom = u - 2 * m + b;
        if (denom > 1e-6) d.y += clamp(0.5 * (u - b) / denom, -0.5, 0.5);
    }
    const uint blocks_x = (field_w + 7) / 8;
    g_stable.Store2(motion_base() + (group.y * blocks_x + group.x) * 8u, asuint(d));
}

// the previous frame's head and colour, bilinear at a fractional field position (clamped)
void previous_at(float2 q, out float4 head, out float3 colour) {
    q = clamp(q, 0.0, float2(field_w - 1, field_h - 1));
    const uint x0 = (uint)q.x, y0 = (uint)q.y;
    const uint x1 = min(x0 + 1, field_w - 1), y1 = min(y0 + 1, field_h - 1);
    const float2 w = q - float2(x0, y0);
    head = 0;
    colour = 0;
    const uint xs[2] = {x0, x1}, ys[2] = {y0, y1};
    [unroll] for (uint j = 0; j < 2; ++j) {
        [unroll] for (uint i = 0; i < 2; ++i) {
            const float k = (i ? w.x : 1 - w.x) * (j ? w.y : 1 - w.y);
            const uint4 v = g_stable.Load4(stable_base(true) + (ys[j] * field_w + xs[i]) * 16u);
            head += k * float4(unpack2(v.x), unpack2(v.y));
            colour += k * float3(unpack2(v.z), unpack2(v.w).x);
        }
    }
}

[numthreads(8, 8, 1)]
void CSHeadStabilize(uint3 id : SV_DispatchThreadID) {
    if (id.x >= field_w || id.y >= field_h) return;
    const uint pixel = id.y * field_w + id.x;
    float4 head = head_at(id.x, id.y);
    const float3 colour = colour_at(id.x, id.y);
    if ((flags & kStableValid) != 0) {
        const uint blocks_x = (field_w + 7) / 8;
        const float2 motion = asfloat(g_stable.Load2(motion_base() + ((id.y / 8) * blocks_x + id.x / 8) * 8u));
        float4 previous_head;
        float3 previous;
        previous_at(float2(id.xy) + motion, previous_head, previous);
        const float3 d = abs(colour - previous) * 8.0;   // scaled_colour is (c - 0.5) / 8
        const float change = max(d.x, max(d.y, d.z));
        const float weight = lerp(stabilize, 1.0, saturate((change - kStillBelow * tolerance) /
                                                           ((kMovingAbove - kStillBelow) * tolerance)));
        // TAA's anti-ghosting clamp, on the correction: the average may only hold values the new
        // head shows around this pixel (3x3). Noise lies inside that range and is still averaged
        // away; what was carried in from another place or object (a wrong motion at a boundary,
        // parallax) lies outside it and is cut, even where the colours happen to match.
        float4 lo = head, hi = head;
        [unroll] for (int dy = -1; dy <= 1; ++dy) {
            [unroll] for (int dx = -1; dx <= 1; ++dx) {
                const float4 n = head_at((uint)clamp((int)id.x + dx, 0, (int)field_w - 1),
                                         (uint)clamp((int)id.y + dy, 0, (int)field_h - 1));
                lo = min(lo, n);
                hi = max(hi, n);
            }
        }
        // Motion: this block's or a neighbour's (a boundary between motions, a rotation). A block
        // translation cannot follow a turning face — in the first game test (photo-mode orbit)
        // the averaged correction smeared a dark patch along the nose whatever the tolerance,
        // since the skin's colour barely changes. So where anything moves the new head weighs at
        // least `motion_weight` (0.5: what was carried in halves every frame), and the anti-ghost
        // clamp applies. Only a still block keeps the full average (and no clamp: with precision
        // accurate the head's noise is correlated over whole attention windows, so a clamp there
        // would also cut the averaging it is for — still-picture flicker 0.03 -> 0.20).
        bool moving = false;
        const int bx = (int)(id.x / 8), by = (int)(id.y / 8);
        const int nbx = (int)blocks_x, nby = (int)((field_h + 7) / 8);
        [unroll] for (int ny = -1; ny <= 1; ++ny) {
            [unroll] for (int nx = -1; nx <= 1; ++nx) {
                const int qx = clamp(bx + nx, 0, nbx - 1), qy = clamp(by + ny, 0, nby - 1);
                const float2 m = asfloat(g_stable.Load2(motion_base() + (uint)(qy * nbx + qx) * 8u));
                moving = moving || any(m != 0.0);
            }
        }
        const float4 margin = (hi - lo) * kClampMargin;
        if (moving) previous_head = clamp(previous_head, lo - margin, hi + margin);
        head = lerp(previous_head, head, moving ? max(weight, motion_weight) : weight);
    }
    g_stable.Store4(stable_base(false) + pixel * 16u, uint4(pack2(head.x, head.y), pack2(head.z, head.w),
                                                            pack2(colour.x, colour.y), pack2(colour.z, 0.0)));
}

// The stabilised head into the head buffer the post reads (CSHeadStabilize reads its neighbours'
// new heads there, so it cannot write it in place).
[numthreads(8, 8, 1)]
void CSHeadCommit(uint3 id : SV_DispatchThreadID) {
    if (id.x >= field_w || id.y >= field_h) return;
    const uint pixel = id.y * field_w + id.x;
    g_head.Store2(pixel * 8u, g_stable.Load2(stable_base(false) + pixel * 16u));
}

// ---- interval > 1: the network spread over N frames, its head moved along the picture's motion ----
// The head that completes on frame t was computed from frame t-N+1's picture. On every frame the
// last complete head (the "reference", with the colour it was computed from) is moved to this
// frame: CSWarpMotion block-matches this frame's luma against the reference luma (8x8 blocks,
// +-kWarpSearch field pixels, coarse to fine), CSHeadWarp picks per pixel the best of its block's and the four
// neighbouring blocks' vectors (and none) by a 3x3 colour match, samples the reference head there
// and fades it out where even the best match differs in colour (a disocclusion, motion beyond the
// search, a cut): there the game's own pixel shows, never a correction dragged from elsewhere.
// With kWarpModel (the default) the moved head is corrected by the colour difference along the
// reference's local colour slope (CSGuideFit), and a new reference is blended with the last one
// carried onto its picture (CSRefCommit, kWarpCapture / kCaptureValid).
//   g_warp: [0, F*8) reference head (4 halves) | [F*8, F*16) reference colour (3 halves + 0)
//           | [F*16, F*24) this frame's colour | [F*24, F*28) reference luma (float)
//           | [F*28, F*32) this frame's luma | then a float2 motion per 8x8 block   (F = field pixels)
//           | (16-aligned) two colour slopes of 36 B per 2x2 cell | the kept head, 8 B per field
//           pixel | a count
RWByteAddressBuffer g_warp : register(u5);

// field pixels between this frame and the reference: at interval 4 the shown head is 3-6 frames
// old, so +-24 follows 4-8 field pixels a frame (8-16 screen pixels at render_scale 0.5) away from
// the predicted motion; the motion pyramid (CSWarpPyramid) covers +-96 around no motion
static const int kWarpSearch = 24;
static const float kWarpBelow = 4.0 / 255.0;      // colour change still taken as the same surface
static const float kWarpAbove = 16.0 / 255.0;     // ... and as another one: no correction at all

uint warp_ref_head(uint pixel) { return pixel * 8u; }
uint warp_ref_colour(uint pixel) { return field_w * field_h * 8u + pixel * 8u; }
uint warp_cur_colour(uint pixel) { return field_w * field_h * 16u + pixel * 8u; }
uint warp_ref_luma(uint pixel) { return field_w * field_h * 24u + pixel * 4u; }
uint warp_cur_luma(uint pixel) { return field_w * field_h * 28u + pixel * 4u; }
uint warp_motion_base() { return field_w * field_h * 32u; }

uint field_pixel(int x, int y) {
    return (uint)clamp(y, 0, (int)field_h - 1) * field_w + (uint)clamp(x, 0, (int)field_w - 1);
}

float3 warp_colour(uint offset) {
    const uint2 v = g_warp.Load2(offset);
    return float3(unpack2(v.x), unpack2(v.y).x);
}

// the colour model's and the carried head's places in g_warp (see CSGuideFit, CSRefCommit)
static const uint kGuideCell = 2;
static const int kGuideRadius = 2;                // cells: a 5x5-cell (10x10 field pixel) window
static const float kGuideEps = 1e-3 / 64.0;       // 1e-3 in [0, 1] colour units; scaled_colour is / 8
static const uint kGuideTile = 8 + 2 * kGuideRadius;
// CSHeadWarp's colour check with the slope: the reference's range over the 10x10 field pixels
// around the spot (precomputed per 2x2 cell by CSGuideFit; the head alone: a 3x3 box). The slope
// handles colours anywhere in that range, so a cloth or limb moving a little off its block's
// vector keeps its correction (wobble scenes, a 7x7 box: gain 0.947 / 0.852 -> 0.964 / 0.900 at
// interval 4) where the 3x3 box faded it out and back in.

uint guide_w() { return (field_w + kGuideCell - 1) / kGuideCell; }
uint guide_h() { return (field_h + kGuideCell - 1) / kGuideCell; }
uint guide_fit_base() {
    const uint blocks = ((field_w + 7) / 8) * ((field_h + 7) / 8);
    return (warp_motion_base() + blocks * 8u + 15u) & ~15u;
}
uint guide_final_base() { return guide_fit_base() + guide_w() * guide_h() * 36u; }
// per cell, over the same 5x5 cells as its fit: the reference's colour and head range (12 halves:
// colour lo, colour hi, head lo, head hi), CSHeadWarp's colour check and head clamp
uint guide_range_base() { return guide_final_base() + guide_w() * guide_h() * 36u; }
// the head CSHeadWarp showed on the frame the next reference comes from (4 halves: rgb, confidence)
uint warp_capture(uint pixel) { return guide_range_base() + guide_w() * guide_h() * 24u + pixel * 8u; }
// ... and how many of its pixels failed the strict match (a uint, zeroed by CSFieldColour)
uint warp_capture_misses() { return guide_range_base() + guide_w() * guide_h() * 24u + field_w * field_h * 8u; }
// The regional model (CSCoarseSums, CSCoarseFit): head.rgb ~ A colour + b per 16x16 field pixel
// cell over the 5x5 cells around it (80x80), with that window's colour range. Where the moved
// reference does not match this frame (motion beyond the search or the prediction, a blurred or
// turning thing, new content at the screen's edge) the correction falls back to it instead of to
// nothing: faded to nothing, a moving area showed the game's own, brighter picture until the next
// head caught up (the "white, like a delay" of the 2026-11-11 test). On a game frame it keeps the
// correction's error at 0.37 / 0.42 / 0.57 for a position 0 / 24 / 50 px off (nothing: 1.0), and
// on new content (a cut) 0.89 (scratch probe).
static const uint kCoarseCell = 16;
static const int kCoarseRadius = 2;
uint coarse_w() { return (field_w + kCoarseCell - 1) / kCoarseCell; }
uint coarse_h() { return (field_h + kCoarseCell - 1) / kCoarseCell; }
uint coarse_sums_base() { return (warp_capture_misses() + 4u + 15u) & ~15u; }   // 28 floats a cell
uint coarse_fit_base() { return coarse_sums_base() + coarse_w() * coarse_h() * 112u; }   // 18 floats a cell
// a new picture rather than a moved one where more than this share of it missed: no blend
static const float kCutShare = 0.25;

// this frame's scaled colour on the field (what CSNeuralPre writes into features 4-6)
[numthreads(8, 8, 1)]
void CSFieldColour(uint3 id : SV_DispatchThreadID) {
    if (id.x >= field_w || id.y >= field_h) return;
    const float3 c = scaled_colour(saturate(inner_colour(mirror(id.x, inner_w), mirror(id.y, inner_h))));
    const uint pixel = id.y * field_w + id.x;
    g_warp.Store2(warp_cur_colour(pixel), uint2(pack2(c.x, c.y), pack2(c.z, 0.0)));
    g_warp.Store(warp_cur_luma(pixel), asuint(luma(c)));
    if ((flags & kWarpCapture) != 0 && pixel == 0) g_warp.Store(warp_capture_misses(), 0u);
}

// a new complete head becomes the reference, with the colour of the frame it was computed from
// (features 4-6: the pre-pass ran on that frame)
[numthreads(8, 8, 1)]
void CSRefCommit(uint3 id : SV_DispatchThreadID) {
    if (id.x >= field_w || id.y >= field_h) return;
    const uint pixel = id.y * field_w + id.x;
    const uint4 f = g_features.Load4(pixel * 32u);
    uint2 committed = g_head.Load2(pixel * 8u);
    if ((flags & kCaptureValid) != 0) {
        // The network is not shift invariant: between frames that only moved, its head changes by
        // 10-15 % (RMS), and at interval N that change arrives at once every N frames, a pulse. So
        // the new head is averaged with the last reference as shown on the frame the new head was
        // computed from (kWarpCapture: moved there along its colour slope, so with that
        // picture's edges), weight warp_blend for the new head; where the old one was faded out
        // (a disocclusion, a cut) the new head alone. At 0.5 (interval 4, a game frame) the
        // largest frame-to-frame jump of a pan fell from 0.16-0.20 to 0.11-0.12 and the error
        // against interval 1 fell too (pan 8 0.222 -> 0.206); 0.35 / 0.25 calm little more and
        // lag an object or a rotation more; clamping the carried head to the new one's 3x3 range
        // (TAA's anti-ghost box) only cost (no edges to cut: the carried head has this picture's).
        // A cut or a sweeping change is told by its share of misses at the capture: a picture
        // with similar colours in similar places (the cut scene's mirror image of a face) passes
        // the per-pixel check often enough to carry the old correction into the new reference.
        const float4 fresh = float4(unpack2(committed.x), unpack2(committed.y));
        const uint2 kv = g_warp.Load2(warp_capture(pixel));
        const float4 kept = float4(unpack2(kv.x), unpack2(kv.y));   // rgb, confidence
        const bool cut = g_warp.Load(warp_capture_misses()) > (uint)(kCutShare * field_w * field_h);
        if (!cut) {
            const float3 rgb = lerp(fresh.rgb, kept.rgb, (1.0 - warp_blend) * saturate(kept.w));
            committed = uint2(pack2(rgb.x, rgb.y), pack2(rgb.z, fresh.w));
        }
    }
    g_warp.Store2(warp_ref_head(pixel), committed);
    g_warp.Store2(warp_ref_colour(pixel), uint2(f.z, f.w & 0xFFFFu));
    g_warp.Store(warp_ref_luma(pixel), asuint(luma(float3(unpack2(f.z), unpack2(f.w).x))));
}

// ---- the reference's colour slope (kWarpModel) ----
// Moving the head itself drags its edges along wherever the motion is a little off (a non-rigid
// cloth, a turning limb, a block between two motions): the old picture's edges show next to this
// one's, a ghost. But the head is mostly a local function of the colour: per pixel
// head.rgb ~ A * colour + b over a 10x10 field window explains ~87 % of it (RMS, the real weights
// on a game frame; the rest is texture the network adds), and A is smooth. So the moved head is
// corrected by the colour difference between this frame and the reference at the matched spot:
//     shown = head_ref(q) + A(q) * (colour(p) - colour_ref(q)),   q = p + motion
// (= the moved model applied to this frame's colour plus the moved residual, up to interpolation).
// Where q is a little off, colour_ref(q) differs from colour(p) by the edge it is off by and the
// slope moves the head across it: the correction's edges are this frame's. A 2 / 4 / 8 px motion
// error left 0.14 / 0.18 / 0.30 of the correction wrong against 0.27 / 0.36 / 0.50 for the moved
// head alone (scratch probe on a game frame, PROGRESS-neural.md 2026-11-11). Where it matches
// (a still picture) the term is exactly 0. It does not carry over to new content (a cut: worse
// than nothing), so the colour check still fades it out there.
// A by the fast guided filter (He & Sun 2015) on 2x2 cells of the field: the cells' mean colour
// and head, their 5x5-cell statistics, a regularised 3x3 solve per cell, those slopes averaged
// over 5x5 cells again, and bilinear between the cells' centres where it is used.
//   after the motion in g_warp: [G, G + C*36) the fit per cell (A row-major, guide x out) |
//   [.., + C*36) the averaged fit | the kept head (8 B per field pixel) | its count of misses

float3x3 load_slope(uint base, uint cell) {
    const uint offset = base + cell * 36u;
    return float3x3(asfloat(g_warp.Load3(offset)), asfloat(g_warp.Load3(offset + 12u)), asfloat(g_warp.Load3(offset + 24u)));
}

void store_slope(uint base, uint cell, float3x3 a) {
    const uint offset = base + cell * 36u;
    g_warp.Store3(offset, asuint(a[0]));
    g_warp.Store3(offset + 12u, asuint(a[1]));
    g_warp.Store3(offset + 24u, asuint(a[2]));
}

// the averaged slope bilinear between the cells' centres at a fractional field position (clamped);
// a[i][k]: guide (colour) channel i -> head channel k
float3x3 slope_at(float2 q) {
    const float2 g = clamp((q + 0.5) / kGuideCell - 0.5, 0.0, float2(guide_w() - 1, guide_h() - 1));
    const uint x0 = (uint)g.x, y0 = (uint)g.y;
    const uint x1 = min(x0 + 1, guide_w() - 1), y1 = min(y0 + 1, guide_h() - 1);
    const float2 w = g - float2(x0, y0);
    float3x3 a = 0;
    const uint xs[2] = {x0, x1}, ys[2] = {y0, y1};
    [unroll] for (uint j = 0; j < 2; ++j) {
        [unroll] for (uint i = 0; i < 2; ++i) {
            a += ((i ? w.x : 1 - w.x) * (j ? w.y : 1 - w.y)) * load_slope(guide_final_base(), ys[j] * guide_w() + xs[i]);
        }
    }
    return a;
}

groupshared float3 gg_colour[kGuideTile * kGuideTile];
groupshared float3 gg_head[kGuideTile * kGuideTile];
groupshared float3 gg_colour_lo[kGuideTile * kGuideTile];
groupshared float3 gg_colour_hi[kGuideTile * kGuideTile];
groupshared float3 gg_head_lo[kGuideTile * kGuideTile];
groupshared float3 gg_head_hi[kGuideTile * kGuideTile];

uint3 pack6(float3 a, float3 b) { return uint3(pack2(a.x, a.y), pack2(a.z, b.x), pack2(b.y, b.z)); }
void unpack6(uint3 v, out float3 a, out float3 b) {
    const float2 p = unpack2(v.x), q = unpack2(v.y), r = unpack2(v.z);
    a = float3(p, q.x);
    b = float3(q.y, r);
}

// One cell a thread: the least-squares slope of head.rgb ~ A colour + b over the 5x5 cells around
// it (each cell the mean of its 2x2 field pixels), regularised by kGuideEps (a flat window: A -> 0)
[numthreads(8, 8, 1)]
void CSGuideFit(uint3 group : SV_GroupID, uint3 tid : SV_GroupThreadID, uint index : SV_GroupIndex) {
    const int2 base = int2(group.xy) * 8 - kGuideRadius;
    for (uint i = index; i < kGuideTile * kGuideTile; i += 64) {
        const int gx = clamp(base.x + (int)(i % kGuideTile), 0, (int)guide_w() - 1);
        const int gy = clamp(base.y + (int)(i / kGuideTile), 0, (int)guide_h() - 1);
        float3 c = 0, hd = 0, clo = 1e9, chi = -1e9, hlo = 1e9, hhi = -1e9;
        [unroll] for (uint j = 0; j < kGuideCell * kGuideCell; ++j) {
            const uint pixel = field_pixel(gx * (int)kGuideCell + (int)(j % kGuideCell), gy * (int)kGuideCell + (int)(j / kGuideCell));
            const float3 pc = warp_colour(warp_ref_colour(pixel));
            const uint2 hv = g_warp.Load2(warp_ref_head(pixel));
            const float3 ph = float3(unpack2(hv.x), unpack2(hv.y).x);
            c += pc;
            hd += ph;
            clo = min(clo, pc);
            chi = max(chi, pc);
            hlo = min(hlo, ph);
            hhi = max(hhi, ph);
        }
        gg_colour[i] = c / (kGuideCell * kGuideCell);
        gg_head[i] = hd / (kGuideCell * kGuideCell);
        gg_colour_lo[i] = clo;
        gg_colour_hi[i] = chi;
        gg_head_lo[i] = hlo;
        gg_head_hi[i] = hhi;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint cx = group.x * 8 + tid.x, cy = group.y * 8 + tid.y;
    if (cx >= guide_w() || cy >= guide_h()) return;
    float3 mc = 0, mh = 0, clo = 1e9, chi = -1e9, hlo = 1e9, hhi = -1e9;
    float3x3 cc = 0, ch = 0;   // sums of colour colour^T and colour head^T
    for (uint y = 0; y < 2 * kGuideRadius + 1; ++y) {
        for (uint x = 0; x < 2 * kGuideRadius + 1; ++x) {
            const uint t = (tid.y + y) * kGuideTile + tid.x + x;
            const float3 c = gg_colour[t], hd = gg_head[t];
            mc += c;
            mh += hd;
            cc += float3x3(c.x * c, c.y * c, c.z * c);
            ch += float3x3(c.x * hd, c.y * hd, c.z * hd);
            clo = min(clo, gg_colour_lo[t]);
            chi = max(chi, gg_colour_hi[t]);
            hlo = min(hlo, gg_head_lo[t]);
            hhi = max(hhi, gg_head_hi[t]);
        }
    }
    const uint range = guide_range_base() + (cy * guide_w() + cx) * 24u;
    g_warp.Store3(range, pack6(clo, chi));
    g_warp.Store3(range + 12u, pack6(hlo, hhi));
    const float n = (2 * kGuideRadius + 1) * (2 * kGuideRadius + 1);
    mc /= n;
    mh /= n;
    const float3x3 cov = cc / n - float3x3(mc.x * mc, mc.y * mc, mc.z * mc) + float3x3(kGuideEps, 0, 0, 0, kGuideEps, 0, 0, 0, kGuideEps);
    const float3x3 cross = ch / n - float3x3(mc.x * mh, mc.y * mh, mc.z * mh);
    // the symmetric 3x3 inverse by its adjugate
    const float s00 = cov[0][0], s01 = cov[0][1], s02 = cov[0][2], s11 = cov[1][1], s12 = cov[1][2], s22 = cov[2][2];
    const float i00 = s11 * s22 - s12 * s12, i01 = s02 * s12 - s01 * s22, i02 = s01 * s12 - s02 * s11;
    const float i11 = s00 * s22 - s02 * s02, i12 = s01 * s02 - s00 * s12, i22 = s00 * s11 - s01 * s01;
    const float det = s00 * i00 + s01 * i01 + s02 * i02;
    const float3x3 inv = float3x3(i00, i01, i02, i01, i11, i12, i02, i12, i22) / det;
    store_slope(guide_fit_base(), cy * guide_w() + cx, mul(inv, cross));
}

// One cell a thread: the slopes averaged over the 5x5 cells around it (the guided filter's second
// mean, which makes them smooth across the cells)
[numthreads(8, 8, 1)]
void CSGuideSmooth(uint3 id : SV_DispatchThreadID) {
    if (id.x >= guide_w() || id.y >= guide_h()) return;
    float3x3 total = 0;
    for (int y = -kGuideRadius; y <= kGuideRadius; ++y) {
        for (int x = -kGuideRadius; x <= kGuideRadius; ++x) {
            const uint gx = (uint)clamp((int)id.x + x, 0, (int)guide_w() - 1);
            const uint gy = (uint)clamp((int)id.y + y, 0, (int)guide_h() - 1);
            total += load_slope(guide_fit_base(), gy * guide_w() + gx);
        }
    }
    store_slope(guide_final_base(), id.y * guide_w() + id.x, total / ((2 * kGuideRadius + 1) * (2 * kGuideRadius + 1)));
}

// One 16x16 coarse cell a group of 64 (4 pixels a thread): the sums of colour (3), head (3),
// colour colour^T (6), colour head^T (9) and the count, then the colour's min (3) and max (3)
static const uint kCoarseValues = 28;
groupshared float gc_part[64 * kCoarseValues];

[numthreads(64, 1, 1)]
void CSCoarseSums(uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    float v[kCoarseValues];
    [unroll] for (uint k = 0; k < 22; ++k) v[k] = 0;
    [unroll] for (uint k = 22; k < 25; ++k) v[k] = 1e9;
    [unroll] for (uint k = 25; k < 28; ++k) v[k] = -1e9;
    [unroll] for (uint r = 0; r < 4; ++r) {
        const uint x = group.x * kCoarseCell + index % 16, y = group.y * kCoarseCell + (index / 16) * 4 + r;
        if (x >= field_w || y >= field_h) continue;
        const uint pixel = y * field_w + x;
        const float3 c = warp_colour(warp_ref_colour(pixel));
        const uint2 hv = g_warp.Load2(warp_ref_head(pixel));
        const float3 h = float3(unpack2(hv.x), unpack2(hv.y).x);
        v[0] += c.x; v[1] += c.y; v[2] += c.z;
        v[3] += h.x; v[4] += h.y; v[5] += h.z;
        v[6] += c.x * c.x; v[7] += c.x * c.y; v[8] += c.x * c.z; v[9] += c.y * c.y; v[10] += c.y * c.z; v[11] += c.z * c.z;
        [unroll] for (uint i = 0; i < 3; ++i) {
            v[12 + i * 3] += c[i] * h.x;
            v[13 + i * 3] += c[i] * h.y;
            v[14 + i * 3] += c[i] * h.z;
        }
        v[21] += 1;
        v[22] = min(v[22], c.x); v[23] = min(v[23], c.y); v[24] = min(v[24], c.z);
        v[25] = max(v[25], c.x); v[26] = max(v[26], c.y); v[27] = max(v[27], c.z);
    }
    [unroll] for (uint k = 0; k < kCoarseValues; ++k) gc_part[index * kCoarseValues + k] = v[k];
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (index < stride) {
            const uint a = index * kCoarseValues, b = (index + stride) * kCoarseValues;
            [unroll] for (uint k = 0; k < 22; ++k) gc_part[a + k] += gc_part[b + k];
            [unroll] for (uint k = 22; k < 25; ++k) gc_part[a + k] = min(gc_part[a + k], gc_part[b + k]);
            [unroll] for (uint k = 25; k < 28; ++k) gc_part[a + k] = max(gc_part[a + k], gc_part[b + k]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (index < kCoarseValues) {
        g_warp.Store(coarse_sums_base() + ((group.y * coarse_w() + group.x) * kCoarseValues + index) * 4u, asuint(gc_part[index]));
    }
}

// One coarse cell a thread: the least-squares head.rgb ~ A colour + b over the 5x5 cells around
// it, and their colour range (18 floats: A row-major, b, colour lo, colour hi)
[numthreads(8, 8, 1)]
void CSCoarseFit(uint3 id : SV_DispatchThreadID) {
    if (id.x >= coarse_w() || id.y >= coarse_h()) return;
    float s[kCoarseValues];
    [unroll] for (uint k = 0; k < 22; ++k) s[k] = 0;
    [unroll] for (uint k = 22; k < 25; ++k) s[k] = 1e9;
    [unroll] for (uint k = 25; k < 28; ++k) s[k] = -1e9;
    for (int y = -kCoarseRadius; y <= kCoarseRadius; ++y) {
        for (int x = -kCoarseRadius; x <= kCoarseRadius; ++x) {
            const int cx = (int)id.x + x, cy = (int)id.y + y;
            if (cx < 0 || cy < 0 || cx >= (int)coarse_w() || cy >= (int)coarse_h()) continue;
            const uint base = coarse_sums_base() + ((uint)cy * coarse_w() + (uint)cx) * kCoarseValues * 4u;
            [unroll] for (uint k = 0; k < 22; ++k) s[k] += asfloat(g_warp.Load(base + k * 4u));
            [unroll] for (uint k = 22; k < 25; ++k) s[k] = min(s[k], asfloat(g_warp.Load(base + k * 4u)));
            [unroll] for (uint k = 25; k < 28; ++k) s[k] = max(s[k], asfloat(g_warp.Load(base + k * 4u)));
        }
    }
    const float n = max(s[21], 1.0);
    const float3 mc = float3(s[0], s[1], s[2]) / n, mh = float3(s[3], s[4], s[5]) / n;
    const float3x3 cc = float3x3(s[6], s[7], s[8], s[7], s[9], s[10], s[8], s[10], s[11]) / n;
    const float3x3 chs = float3x3(s[12], s[13], s[14], s[15], s[16], s[17], s[18], s[19], s[20]) / n;
    const float3x3 cov = cc - float3x3(mc.x * mc, mc.y * mc, mc.z * mc) + float3x3(kGuideEps, 0, 0, 0, kGuideEps, 0, 0, 0, kGuideEps);
    const float3x3 cross = chs - float3x3(mc.x * mh, mc.y * mh, mc.z * mh);
    const float s00 = cov[0][0], s01 = cov[0][1], s02 = cov[0][2], s11 = cov[1][1], s12 = cov[1][2], s22 = cov[2][2];
    const float i00 = s11 * s22 - s12 * s12, i01 = s02 * s12 - s01 * s22, i02 = s01 * s12 - s02 * s11;
    const float i11 = s00 * s22 - s02 * s02, i12 = s01 * s02 - s00 * s12, i22 = s00 * s11 - s01 * s01;
    const float det = s00 * i00 + s01 * i01 + s02 * i02;
    const float3x3 a = mul(float3x3(i00, i01, i02, i01, i11, i12, i02, i12, i22) / det, cross);
    const float3 b = mh - mul(mc, a);
    const uint dst = coarse_fit_base() + (id.y * coarse_w() + id.x) * 72u;
    g_warp.Store3(dst, asuint(a[0]));
    g_warp.Store3(dst + 12u, asuint(a[1]));
    g_warp.Store3(dst + 24u, asuint(a[2]));
    g_warp.Store3(dst + 36u, asuint(b));
    g_warp.Store3(dst + 48u, asuint(float3(s[22], s[23], s[24])));
    g_warp.Store3(dst + 60u, asuint(float3(s[25], s[26], s[27])));
}

// the regional model at a field position (bilinear between the cells' centres) applied to a
// colour held inside the cells' colour range (no extrapolation)
float3 coarse_at(float2 q, float3 colour) {
    const float2 g = clamp((q + 0.5) / kCoarseCell - 0.5, 0.0, float2(coarse_w() - 1, coarse_h() - 1));
    const uint x0 = (uint)g.x, y0 = (uint)g.y;
    const uint x1 = min(x0 + 1, coarse_w() - 1), y1 = min(y0 + 1, coarse_h() - 1);
    const float2 w = g - float2(x0, y0);
    float3x3 a = 0;
    float3 b = 0, lo = 0, hi = 0;
    const uint xs[2] = {x0, x1}, ys[2] = {y0, y1};
    [unroll] for (uint j = 0; j < 2; ++j) {
        [unroll] for (uint i = 0; i < 2; ++i) {
            const float k = (i ? w.x : 1 - w.x) * (j ? w.y : 1 - w.y);
            const uint at = coarse_fit_base() + (ys[j] * coarse_w() + xs[i]) * 72u;
            a += k * float3x3(asfloat(g_warp.Load3(at)), asfloat(g_warp.Load3(at + 12u)), asfloat(g_warp.Load3(at + 24u)));
            b += k * asfloat(g_warp.Load3(at + 36u));
            lo += k * asfloat(g_warp.Load3(at + 48u));
            hi += k * asfloat(g_warp.Load3(at + 60u));
        }
    }
    return mul(clamp(colour, lo, hi), a) + b;
}

// ---- the motion pyramid (kWarpPyramid) ----
// CSWarpMotion searches +-24 field pixels around a prediction (the block's last vector scaled by
// the age). A camera that reverses, starts, stops or jerks is far from that prediction, and the
// block took the best wrong match in range: a whole 8x8 block of correction from another place
// (swing scene, interval 4: error 0.41 against 0.24 for a perfect warp; 0.59-0.66 around the
// turns). So a full search on a quarter-resolution luma covers +-96 field pixels around no motion
// for every 32x32 region, and CSWarpMotion searches around that vector too.
//   after the regional model in g_warp: this frame's quarter luma | the reference's (float each)
//   | a float2 (field pixels) per pyramid block of 8x8 quarter pixels
static const uint kQuarter = 4;
static const int kPyramidSearch = 24;   // quarter pixels: +-96 field pixels
static const uint kPyramidTile = 8 + 2 * kPyramidSearch;
static const uint kPyramidSide = 2 * kPyramidSearch + 1;
uint quarter_w() { return (field_w + kQuarter - 1) / kQuarter; }
uint quarter_h() { return (field_h + kQuarter - 1) / kQuarter; }
uint quarter_cur_base() { return coarse_fit_base() + coarse_w() * coarse_h() * 72u; }
uint quarter_ref_base() { return quarter_cur_base() + quarter_w() * quarter_h() * 4u; }
uint pyramid_base() { return quarter_ref_base() + quarter_w() * quarter_h() * 4u; }
uint pyramid_w() { return (quarter_w() + 7) / 8; }
uint quarter_pixel(int x, int y) {
    return (uint)clamp(y, 0, (int)quarter_h() - 1) * quarter_w() + (uint)clamp(x, 0, (int)quarter_w() - 1);
}

// the mean luma of 4x4 field pixels (this frame's, or the reference's when it changes)
void quarter_luma(uint2 id, bool reference) {
    if (id.x >= quarter_w() || id.y >= quarter_h()) return;
    float sum = 0;
    [unroll] for (uint j = 0; j < kQuarter; ++j) {
        [unroll] for (uint i = 0; i < kQuarter; ++i) {
            const uint pixel = field_pixel((int)(id.x * kQuarter + i), (int)(id.y * kQuarter + j));
            sum += asfloat(g_warp.Load(reference ? warp_ref_luma(pixel) : warp_cur_luma(pixel)));
        }
    }
    const uint base = reference ? quarter_ref_base() : quarter_cur_base();
    g_warp.Store(base + (id.y * quarter_w() + id.x) * 4u, asuint(sum / (kQuarter * kQuarter)));
}

[numthreads(8, 8, 1)]
void CSQuarterCur(uint3 id : SV_DispatchThreadID) { quarter_luma(id.xy, false); }

[numthreads(8, 8, 1)]
void CSQuarterRef(uint3 id : SV_DispatchThreadID) { quarter_luma(id.xy, true); }

groupshared float gp_cur[64];
groupshared float gp_ref[kPyramidTile * kPyramidTile];
groupshared float gp_best[64];
groupshared uint gp_index[64];
groupshared float gp_fine[9];

// SAD of the pyramid block against the reference displaced by (dx, dy) quarter pixels, with a
// small penalty per quarter pixel (a flat region: no motion)
float pyramid_sad(int dx, int dy) {
    const uint origin = (uint)(dy + kPyramidSearch) * kPyramidTile + (uint)(dx + kPyramidSearch);
    float sad = 0;
    [unroll] for (uint y = 0; y < 8; ++y) {
        [unroll] for (uint x = 0; x < 8; ++x) sad += abs(gp_cur[y * 8 + x] - gp_ref[origin + y * kPyramidTile + x]);
    }
    return sad + kMotionPenalty * kQuarter * (abs(dx) + abs(dy));
}

// One pyramid block (8x8 quarter pixels) a group: every second displacement within
// +-kPyramidSearch (25 x 25; the 4x4 box makes a texture's SAD minimum wider than that), then the
// 3 x 3 around the best. With every displacement (49 x 49) and +-4 in CSWarpMotion the pyramid
// cost 0.43 ms at 1344x768, now 0.14 ms; the scenes' errors moved by 0.015 at most (swing 0.260
// -> 0.255, cut 0.580 -> 0.594).
static const uint kPyramidGrid = kPyramidSearch + 1;   // grid points a side, step 2
[numthreads(64, 1, 1)]
void CSWarpPyramid(uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    const int2 base = int2(group.xy) * 8;
    gp_cur[index] = asfloat(g_warp.Load(quarter_cur_base() + quarter_pixel(base.x + (int)(index % 8), base.y + (int)(index / 8)) * 4u));
    for (uint i = index; i < kPyramidTile * kPyramidTile; i += 64) {
        const int x = base.x - kPyramidSearch + (int)(i % kPyramidTile);
        const int y = base.y - kPyramidSearch + (int)(i / kPyramidTile);
        gp_ref[i] = asfloat(g_warp.Load(quarter_ref_base() + quarter_pixel(x, y) * 4u));
    }
    GroupMemoryBarrierWithGroupSync();
    float best = 1e30;
    uint best_k = 0;
    for (uint k = index; k < kPyramidGrid * kPyramidGrid; k += 64) {
        const float sad = pyramid_sad(2 * (int)(k % kPyramidGrid) - kPyramidSearch, 2 * (int)(k / kPyramidGrid) - kPyramidSearch);
        if (sad < best) {
            best = sad;
            best_k = k;
        }
    }
    gp_best[index] = best;
    gp_index[index] = best_k;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (index < stride) {
            const float other = gp_best[index + stride];
            if (other < gp_best[index] || (other == gp_best[index] && gp_index[index + stride] < gp_index[index])) {
                gp_best[index] = other;
                gp_index[index] = gp_index[index + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const int gx = 2 * (int)(gp_index[0] % kPyramidGrid) - kPyramidSearch;
    const int gy = 2 * (int)(gp_index[0] / kPyramidGrid) - kPyramidSearch;
    if (index < 9) {
        const int dx = gx + (int)(index % 3) - 1, dy = gy + (int)(index / 3) - 1;
        gp_fine[index] = max(abs(dx), abs(dy)) <= kPyramidSearch ? pyramid_sad(dx, dy) : 1e30;
    }
    GroupMemoryBarrierWithGroupSync();
    if (index == 0) {
        uint b = 4;
        for (uint k = 0; k < 9; ++k) {
            if (gp_fine[k] < gp_fine[b]) b = k;
        }
        const float2 d = float2(gx + (int)(b % 3) - 1, gy + (int)(b / 3) - 1) * kQuarter;
        g_warp.Store2(pyramid_base() + (group.y * pyramid_w() + group.x) * 8u, asuint(d));
    }
}

// coarse to fine: every fourth displacement within +-kWarpSearch (13 x 13), then the 5 x 5 around
// the best (a texture's SAD minimum is wider than 2 field pixels at this scale); about as cheap as
// a full +-8 search (a full +-12 one cost 0.45 ms more at 1344x768)
static const int kCoarseStep = 4;
static const int kFine = kCoarseStep / 2;
static const int kWarpReach = kWarpSearch + kFine;
static const uint kWarpTile = 8 + 2 * kWarpReach;
static const uint kCoarseSide = 2 * kWarpSearch / kCoarseStep + 1;
static const uint kFineSide = 2 * kFine + 1;
groupshared float gw_cur[64];
groupshared float gw_ref[kWarpTile * kWarpTile];
groupshared float gw_coarse[kCoarseSide * kCoarseSide];
groupshared float gw_fine[kFineSide * kFineSide];
groupshared int gw_centre_x;
groupshared int gw_centre_y;

groupshared float gw_zero[64];
// the second centre (kWarpPyramid): +-kPyramidFine field pixels around the pyramid's vector,
// which is a multiple of 4 (one quarter pixel: the true motion lies within +-2 of it)
static const int kPyramidFine = 2;
static const uint kPyramidFineTile = 8 + 2 * kPyramidFine;
static const uint kPyramidFineSide = 2 * kPyramidFine + 1;
groupshared float gw_pyr[kPyramidFineTile * kPyramidFineTile];
groupshared float gw_pyr_sad[kPyramidFineSide * kPyramidFineSide];

// The motion chain (kChain*): every frame each 8x8 block is matched against last frame's luma
// (CSStepMotion: one frame of motion, small however long the head is shown), and the steps are
// summed along the way (CSChain) into the displacement to the frame the reference came from. With
// interval 4 the shown head is 3-6 frames old: a fast camera (Spider-Man capture 2026-11-13,
// 20-25 field pixels a frame) moved it 60-150 field pixels, beyond the search around the old
// prediction (last vector x age ratio), and blocks took the correction of other places (error
// 0.85-1.41 against interval 1, worse than no correction). Two sums: the active one (to the
// current reference's frame) and the pending one (to the frame whose features the next head is
// computed from; reset on that frame, made active when the head completes). Ping-ponged
// (kChainParity: read that half, write the other), because a sum is read at the moved position.
//   after the pyramid in g_warp: last frame's luma (a float per field pixel) | the step per block
//   (float2) | two halves of { active, pending } per block (float2 each)
uint pyramid_h() { return (quarter_h() + 7) / 8; }
uint chain_prev_luma(uint pixel) { return pyramid_base() + pyramid_w() * pyramid_h() * 8u + pixel * 4u; }
uint chain_blocks() { return ((field_w + 7) / 8) * ((field_h + 7) / 8); }
uint chain_step(uint block) { return chain_prev_luma(field_w * field_h) + block * 8u; }
uint chain_sum(uint block, uint half_index, uint which) {
    return chain_step(chain_blocks()) + ((half_index * 2u + which) * chain_blocks() + block) * 8u;
}

// SAD of the block against the other luma displaced by (dx, dy) from the search centre, plus the
// penalty for the distance from that centre (the prediction)
float warp_sad(int dx, int dy) {
    const uint origin = (uint)((kWarpReach + dy) * (int)kWarpTile + kWarpReach + dx);
    float sad = 0;
    [unroll] for (uint y = 0; y < 8; ++y) {
        [unroll] for (uint x = 0; x < 8; ++x) sad += abs(gw_cur[y * 8 + x] - gw_ref[origin + y * kWarpTile + x]);
    }
    return sad + kMotionPenalty * (abs(dx) + abs(dy));
}

// the same around the pyramid's vector
float pyramid_fine_sad(int dx, int dy) {
    const uint origin = (uint)((kPyramidFine + dy) * (int)kPyramidFineTile + kPyramidFine + dx);
    float sad = 0;
    [unroll] for (uint y = 0; y < 8; ++y) {
        [unroll] for (uint x = 0; x < 8; ++x) sad += abs(gw_cur[y * 8 + x] - gw_pyr[origin + y * kPyramidFineTile + x]);
    }
    return sad + kMotionPenalty * (abs(dx) + abs(dy));
}

// the sub-pixel offset of a SAD minimum m between its neighbours l and r (a parabola)
float parabola(float l, float m, float r) {
    const float denom = l - 2 * m + r;
    return denom > 1e-6 ? clamp(0.5 * (l - r) / denom, -0.5, 0.5) : 0.0;
}

float other_luma(int x, int y, bool prev) {
    const uint pixel = field_pixel(x, y);
    return asfloat(g_warp.Load(prev ? chain_prev_luma(pixel) : warp_ref_luma(pixel)));
}

// One 8x8 block a group (8x8 threads): this frame's luma against the reference's (prev: last
// frame's) around the centre c0, every kCoarseStep-th displacement within +-kWarpSearch, then
// +-kFine around the best with a parabola. Thread 0 gets the vector, its SAD and the SAD of no
// motion (still, without a penalty); gw_cur stays loaded for the caller.
void block_search(int2 base, int2 c0, bool prev, uint2 tid, uint index, out float2 found, out float best_sad, out float still) {
    gw_cur[tid.y * 8 + tid.x] = asfloat(g_warp.Load(warp_cur_luma(field_pixel(base.x + (int)tid.x, base.y + (int)tid.y))));
    gw_zero[tid.y * 8 + tid.x] = other_luma(base.x + (int)tid.x, base.y + (int)tid.y, prev);
    for (uint i = index; i < kWarpTile * kWarpTile; i += 64) {
        gw_ref[i] = other_luma(base.x + c0.x - kWarpReach + (int)(i % kWarpTile), base.y + c0.y - kWarpReach + (int)(i / kWarpTile), prev);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint c = index; c < kCoarseSide * kCoarseSide; c += 64) {
        gw_coarse[c] = warp_sad(kCoarseStep * (int)(c % kCoarseSide) - kWarpSearch, kCoarseStep * (int)(c / kCoarseSide) - kWarpSearch);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint middle = (kCoarseSide / 2) * kCoarseSide + kCoarseSide / 2;   // the prediction itself
    if (index == 0) {
        uint best = middle;
        float sad = gw_coarse[middle];
        for (uint k = 0; k < kCoarseSide * kCoarseSide; ++k) {
            if (gw_coarse[k] < sad) {
                sad = gw_coarse[k];
                best = k;
            }
        }
        gw_centre_x = kCoarseStep * (int)(best % kCoarseSide) - kWarpSearch;
        gw_centre_y = kCoarseStep * (int)(best / kCoarseSide) - kWarpSearch;
    }
    GroupMemoryBarrierWithGroupSync();
    const int cx = gw_centre_x, cy = gw_centre_y;
    if (index < kFineSide * kFineSide) {
        gw_fine[index] = warp_sad(cx + (int)(index % kFineSide) - kFine, cy + (int)(index / kFineSide) - kFine);
    }
    GroupMemoryBarrierWithGroupSync();
    found = 0;
    best_sad = 1e30;
    still = 0;
    if (index == 0) {
        const uint centre = kFine * kFineSide + kFine;
        uint best = centre;
        best_sad = gw_fine[centre];
        for (uint k = 0; k < kFineSide * kFineSide; ++k) {
            if (gw_fine[k] < best_sad) {
                best_sad = gw_fine[k];
                best = k;
            }
        }
        const int bx = (int)(best % kFineSide), by = (int)(best / kFineSide);
        found = float2(c0.x + cx + bx - kFine, c0.y + cy + by - kFine);
        if (bx > 0 && bx < (int)kFineSide - 1) found.x += parabola(gw_fine[best - 1], gw_fine[best], gw_fine[best + 1]);
        if (by > 0 && by < (int)kFineSide - 1) found.y += parabola(gw_fine[best - kFineSide], gw_fine[best], gw_fine[best + kFineSide]);
        [unroll] for (uint k = 0; k < 64; ++k) still += abs(gw_cur[k] - gw_zero[k]);
    }
}

// A tie goes to the prior: without a prediction a block that matches in place within the margin
// stays exactly in place (see CSMotion); with one, "no motion" has to win clearly. A flat block
// (sky, a plain dress) matches every shift equally, and choosing "still" there left the
// correction behind while the camera turned (interval 2, pan 8 px: error 0.33 vs 0.15 ideal).
float2 settle(float2 found, float best_sad, float still, bool predicted) {
    return (predicted ? still + kStillBias >= best_sad : still > best_sad + kStillBias) ? found : float2(0, 0);
}

// One frame of motion per block (this frame against last frame's luma), centred on the block's
// last step; 0 without last frame's luma
[numthreads(8, 8, 1)]
void CSStepMotion(uint3 group : SV_GroupID, uint3 tid : SV_GroupThreadID, uint index : SV_GroupIndex) {
    const uint block = group.y * ((field_w + 7) / 8) + group.x;
    const bool valid = (flags & kChainPrev) != 0;
    const int2 limit = int2(field_w, field_h);
    const int2 c0 = valid ? clamp(int2(round(asfloat(g_warp.Load2(chain_step(block))))), -limit, limit) : int2(0, 0);
    float2 found;
    float best_sad, still;
    block_search(int2(group.xy) * 8, c0, true, tid.xy, index, found, best_sad, still);
    if (index == 0) g_warp.Store2(chain_step(block), asuint(valid ? settle(found, best_sad, still, any(c0 != int2(0, 0))) : float2(0, 0)));
}

// a sum (active 0, pending 1) of last frame, bilinear between the blocks' centres at field position q
float2 chain_sum_at(float2 q, uint half_index, uint which) {
    const uint bw = (field_w + 7) / 8, bh = (field_h + 7) / 8;
    const float2 g = clamp((q - 3.5) / 8.0, 0.0, float2(bw - 1, bh - 1));
    const uint x0 = (uint)g.x, y0 = (uint)g.y, x1 = min(x0 + 1, bw - 1), y1 = min(y0 + 1, bh - 1);
    const float2 w = g - float2(x0, y0);
    const float2 a = asfloat(g_warp.Load2(chain_sum(y0 * bw + x0, half_index, which)));
    const float2 b = asfloat(g_warp.Load2(chain_sum(y0 * bw + x1, half_index, which)));
    const float2 c = asfloat(g_warp.Load2(chain_sum(y1 * bw + x0, half_index, which)));
    const float2 d = asfloat(g_warp.Load2(chain_sum(y1 * bw + x1, half_index, which)));
    return lerp(lerp(a, b, w.x), lerp(c, d, w.x), w.y);
}

// the sums one frame on: sum(p) = step(p) + last sum(p + step(p)); the pending one starts at 0 on
// the frame that takes the next head's features (kChainFront), and becomes the active one on the
// frame that completes it (kChainBack: the new reference is that features frame)
[numthreads(8, 8, 1)]
void CSChain(uint3 id : SV_DispatchThreadID) {
    const uint bw = (field_w + 7) / 8, bh = (field_h + 7) / 8;
    if (id.x >= bw || id.y >= bh) return;
    const uint block = id.y * bw + id.x;
    const uint from = (flags & kChainParity) != 0 ? 1u : 0u, to = 1u - from;
    const float2 s = (flags & kChainPrev) != 0 ? asfloat(g_warp.Load2(chain_step(block))) : float2(0, 0);
    const float2 q = float2(id.xy) * 8.0 + 3.5 + s;
    const float2 pending = (flags & kChainFront) != 0 ? float2(0, 0) : s + chain_sum_at(q, from, 1);
    const float2 active = (flags & kChainBack) != 0 ? pending : s + chain_sum_at(q, from, 0);
    g_warp.Store2(chain_sum(block, to, 0), asuint(active));
    g_warp.Store2(chain_sum(block, to, 1), asuint(pending));
}

// this frame's luma kept for the next frame's step
[numthreads(8, 8, 1)]
void CSKeepLuma(uint3 id : SV_DispatchThreadID) {
    if (id.x >= field_w || id.y >= field_h) return;
    const uint pixel = id.y * field_w + id.x;
    g_warp.Store(chain_prev_luma(pixel), g_warp.Load(warp_cur_luma(pixel)));
}

// One 8x8 block a group: d minimising SAD(this frame at p, reference at p + d) (as CSMotion).
// The search is centred on a prediction: with the chain (kChain) the summed steps to the
// reference's frame; without it last frame's vector of this block scaled by age now / age then
// (warp_predict; across a new reference the same scaling carries the old one's vector over).
// "No motion" is tested on its own as well (a stop).
[numthreads(8, 8, 1)]
void CSWarpMotion(uint3 group : SV_GroupID, uint3 tid : SV_GroupThreadID, uint index : SV_GroupIndex) {
    const int2 base = int2(group.xy) * 8;
    const uint block = group.y * ((field_w + 7) / 8) + group.x;
    const uint slot = warp_motion_base() + block * 8u;
    const float2 last = asfloat(g_warp.Load2(slot));   // only this group reads or writes its block's slot
    const int2 limit = int2(field_w, field_h);
    int2 c0 = warp_predict > 0.0 ? clamp(int2(round(last * warp_predict)), -limit, limit) : int2(0, 0);
    if ((flags & kChain) != 0) {   // the sums CSChain just wrote (into the half it wrote: to)
        c0 = clamp(int2(round(asfloat(g_warp.Load2(chain_sum(block, (flags & kChainParity) != 0 ? 0u : 1u, 0))))), -limit, limit);
    }
    const bool pyramid = (flags & kWarpPyramid) != 0;
    // the pyramid's vector of the 32x32 region this block lies in (CSWarpPyramid)
    const int2 c1 = pyramid ? int2(asfloat(g_warp.Load2(pyramid_base() + ((group.y / 4) * pyramid_w() + group.x / 4) * 8u)))
                            : int2(0, 0);
    if (pyramid) {
        for (uint i = index; i < kPyramidFineTile * kPyramidFineTile; i += 64) {
            const int x = base.x + c1.x - kPyramidFine + (int)(i % kPyramidFineTile);
            const int y = base.y + c1.y - kPyramidFine + (int)(i / kPyramidFineTile);
            gw_pyr[i] = asfloat(g_warp.Load(warp_ref_luma(field_pixel(x, y))));
        }
    }
    float2 found;
    float best_sad, still;
    block_search(base, c0, false, tid.xy, index, found, best_sad, still);   // its barriers cover gw_pyr
    if (pyramid) {
        for (uint c = index; c < kPyramidFineSide * kPyramidFineSide; c += 64) {
            gw_pyr_sad[c] = pyramid_fine_sad((int)(c % kPyramidFineSide) - kPyramidFine, (int)(c / kPyramidFineSide) - kPyramidFine);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (index != 0) return;
    if (pyramid) {
        // the pyramid's candidate replaces the predicted one only where it matches clearly better
        // (a flat block matches both alike: the prediction stays, see settle)
        const uint pc = kPyramidFine * kPyramidFineSide + kPyramidFine;
        uint pb = pc;
        float pb_sad = gw_pyr_sad[pc];
        for (uint k = 0; k < kPyramidFineSide * kPyramidFineSide; ++k) {
            if (gw_pyr_sad[k] < pb_sad) {
                pb_sad = gw_pyr_sad[k];
                pb = k;
            }
        }
        if (pb_sad + kStillBias < best_sad) {
            best_sad = pb_sad;
            const int px = (int)(pb % kPyramidFineSide), py = (int)(pb / kPyramidFineSide);
            found = float2(c1.x + px - kPyramidFine, c1.y + py - kPyramidFine);
            if (px > 0 && px < (int)kPyramidFineSide - 1) found.x += parabola(gw_pyr_sad[pb - 1], gw_pyr_sad[pb], gw_pyr_sad[pb + 1]);
            if (py > 0 && py < (int)kPyramidFineSide - 1) {
                found.y += parabola(gw_pyr_sad[pb - kPyramidFineSide], gw_pyr_sad[pb], gw_pyr_sad[pb + kPyramidFineSide]);
            }
        }
    }
    g_warp.Store2(slot, asuint(settle(found, best_sad, still, any(c0 != int2(0, 0)))));
}

// the reference head and colour, bilinear at a fractional field position (clamped)
void reference_at(float2 q, out float4 head, out float3 colour) {
    q = clamp(q, 0.0, float2(field_w - 1, field_h - 1));
    const uint x0 = (uint)q.x, y0 = (uint)q.y;
    const uint x1 = min(x0 + 1, field_w - 1), y1 = min(y0 + 1, field_h - 1);
    const float2 w = q - float2(x0, y0);
    head = 0;
    colour = 0;
    const uint xs[2] = {x0, x1}, ys[2] = {y0, y1};
    [unroll] for (uint j = 0; j < 2; ++j) {
        [unroll] for (uint i = 0; i < 2; ++i) {
            const float k = (i ? w.x : 1 - w.x) * (j ? w.y : 1 - w.y);
            const uint pixel = ys[j] * field_w + xs[i];
            const uint2 hv = g_warp.Load2(warp_ref_head(pixel));
            head += k * float4(unpack2(hv.x), unpack2(hv.y));
            colour += k * warp_colour(warp_ref_colour(pixel));
        }
    }
}

// 1 where the colour lies inside the reference's box, fading to 0 from kWarpBelow to kWarpAbove outside
float colour_confidence(float3 colour, float3 lo, float3 hi) {
    const float3 d = max(max(lo - colour, colour - hi), 0.0) * 8.0;   // scaled_colour: (c - 0.5) / 8
    return 1.0 - saturate((max(d.x, max(d.y, d.z)) - kWarpBelow) / (kWarpAbove - kWarpBelow));
}

// The length of the motion since the reference, bilinear between the centres of the 8x8 blocks
// (smooth: a fade by block would draw the blocks)
float motion_length_at(int px, int py, uint blocks_x, uint blocks_y) {
    const float2 b = (float2(px, py) - 3.5) / 8.0;
    const int2 b0 = int2(floor(b));
    const float2 w = b - float2(b0);
    float len = 0;
    [unroll] for (int j = 0; j < 2; ++j) {
        [unroll] for (int i = 0; i < 2; ++i) {
            const int qx = clamp(b0.x + i, 0, (int)blocks_x - 1), qy = clamp(b0.y + j, 0, (int)blocks_y - 1);
            const float2 v = asfloat(g_warp.Load2(warp_motion_base() + (uint)(qy * (int)blocks_x + qx) * 8u));
            len += (i ? w.x : 1 - w.x) * (j ? w.y : 1 - w.y) * length(v);
        }
    }
    return len;
}

[numthreads(8, 8, 1)]
void CSHeadWarp(uint3 id : SV_DispatchThreadID) {
    if (id.x >= field_w || id.y >= field_h) return;
    const int px = (int)id.x, py = (int)id.y;
    const uint blocks_x = (field_w + 7) / 8, blocks_y = (field_h + 7) / 8;
    const int bx = px / 8, by = py / 8;
    // this frame's luma around the pixel, once
    float cur[9];
    [unroll] for (int j = 0; j < 3; ++j) {
        [unroll] for (int i = 0; i < 3; ++i) cur[j * 3 + i] = asfloat(g_warp.Load(warp_cur_luma(field_pixel(px + i - 1, py + j - 1))));
    }
    // Candidates: the vectors of this block and its eight neighbours (a boundary between motions),
    // their median, and no motion. The penalty is the distance from the median: a block whose own
    // search went wrong (fast or changing motion beyond the prediction, a flat or blurred block)
    // is an outlier among its neighbours, and anchored on its own vector a flat pixel kept it and
    // the whole block showed the wrong place or was faded out — a box brighter or darker than its
    // surroundings (the "white boxes" of the 2026-11-11 Spider-Man test; whip scene: 5 % holes).
    float2 v[9];
    [unroll] for (int k = 0; k < 9; ++k) {
        const int qx = clamp(bx + k % 3 - 1, 0, (int)blocks_x - 1);
        const int qy = clamp(by + k / 3 - 1, 0, (int)blocks_y - 1);
        v[k] = asfloat(g_warp.Load2(warp_motion_base() + (uint)(qy * (int)blocks_x + qx) * 8u));
    }
    float2 median = 0;   // per component: the value with 4 below and 4 above (ties counted as both)
    [unroll] for (int a = 0; a < 9; ++a) {
        int below_x = 0, above_x = 0, below_y = 0, above_y = 0;
        [unroll] for (int b = 0; b < 9; ++b) {
            below_x += v[b].x < v[a].x ? 1 : 0;
            above_x += v[b].x > v[a].x ? 1 : 0;
            below_y += v[b].y < v[a].y ? 1 : 0;
            above_y += v[b].y > v[a].y ? 1 : 0;
        }
        if (below_x <= 4 && above_x <= 4) median.x = v[a].x;
        if (below_y <= 4 && above_y <= 4) median.y = v[a].y;
    }
    float2 best = v[4];
    float best_cost = 1e30;
    float2 tested[11];
    [unroll] for (int c = 0; c < 11; ++c) {
        const float2 m = c < 9 ? v[(c + 4) % 9] : (c == 9 ? median : float2(0, 0));   // own first
        tested[c] = m;
        // a vector already tried costs the same again: skipped (under a uniform motion all eleven
        // are one, and the eleven 3x3 fetches were 0.27 ms at 1344x768)
        bool seen = false;
        [unroll] for (int k = 0; k < c; ++k) seen = seen || all(tested[k] == m);
        if (seen) continue;
        const int2 r = int2(round(m));
        float cost = 0;
        [unroll] for (int j = 0; j < 3; ++j) {
            [unroll] for (int i = 0; i < 3; ++i) {
                cost += abs(cur[j * 3 + i] - asfloat(g_warp.Load(warp_ref_luma(field_pixel(px + i - 1 + r.x, py + j - 1 + r.y)))));
            }
        }
        cost += kMotionPenalty * (abs(m.x - median.x) + abs(m.y - median.y));
        if (cost < best_cost) {
            best_cost = cost;
            best = m;
        }
    }
    float4 head;
    float3 reference;
    reference_at(float2(px, py) + best, head, reference);
    const float3 cur_colour = warp_colour(warp_cur_colour(id.y * field_w + id.x));
    if ((flags & kWarpModel) != 0) {
        // the moved head across the colour difference along the reference's local slope (see
        // CSGuideFit): the correction's edges follow this frame's
        head.rgb += mul(cur_colour - reference, slope_at(float2(px, py) + best));
    }
    // The colour check against the reference's 3x3 neighbourhood there (TAA's box, not one pixel;
    // with the model the 10x10 range of CSGuideFit's cells):
    // this frame's colour inside the range the reference shows around the spot is the same surface.
    // Against the one bilinear sample, a sub-pixel jitter or a highlight moving a little on a shiny
    // suit already crossed 4/255 at every edge, and the correction came and went every frame
    // (Spider-Man, still camera: the suit's luma jumped 4x more than without the filter; the
    // jitter scene flickered 0.133 at interval 4 against 0.045 at interval 1).
    const int2 r0 = int2(round(float2(px, py) + best));
    const bool model = (flags & kWarpModel) != 0;
    float3 lo = reference, hi = reference, lo3 = reference, hi3 = reference;
    if (model || (flags & kWarpCapture) != 0) {   // the strict 3x3 box (the head alone: its only box)
        [unroll] for (int j = -1; j <= 1; ++j) {
            [unroll] for (int i = -1; i <= 1; ++i) {
                const float3 n = warp_colour(warp_ref_colour(field_pixel(r0.x + i, r0.y + j)));
                lo3 = min(lo3, n);
                hi3 = max(hi3, n);
            }
        }
    }
    if (model) {
        // the 5x5-cell (10x10 field pixel) range of the cell the moved pixel lands in (CSGuideFit):
        // the colour check, and a clamp of the head to what the network gave around there — the
        // slope can never push the correction beyond it (a white or black blotch)
        const uint cell = (uint)clamp(r0.y / (int)kGuideCell, 0, (int)guide_h() - 1) * guide_w() +
                          (uint)clamp(r0.x / (int)kGuideCell, 0, (int)guide_w() - 1);
        const uint range = guide_range_base() + cell * 24u;
        float3 clo, chi, hlo, hhi;
        unpack6(g_warp.Load3(range), clo, chi);
        unpack6(g_warp.Load3(range + 12u), hlo, hhi);
        lo = min(clo, reference);
        hi = max(chi, reference);
        head.rgb = clamp(head.rgb, hlo, hhi);
    } else {
        lo = lo3;
        hi = hi3;
    }
    const float confidence = colour_confidence(cur_colour, lo, hi);
    if ((flags & kWarpCapture) != 0) {
        // the blend into the next reference trusts only the strict 3x3 match: after a cut much of
        // the new picture still falls inside some old 7x7 range, and carried at half weight into the
        // new reference that kept the wrong correction for another N frames (interval 2: cut error
        // 0.43 -> 0.58); the misses are counted for CSRefCommit's cut test
        const float strict = colour_confidence(cur_colour, lo3, hi3);
        g_warp.Store2(warp_capture(id.y * field_w + id.x), uint2(pack2(head.x, head.y), pack2(head.z, strict)));
        const uint misses = WaveActiveCountBits(strict < 0.5);
        if (WaveIsFirstLane() && misses != 0) g_warp.InterlockedAdd(warp_capture_misses(), misses);
    }
    // A moved correction is never quite in place, and the further it moved the less: under a fast
    // camera its detail swam over the scene (Spider-Man, interval 2 and 4: a ghost, motion
    // sickness). Its detail fades to the regional model with the motion's speed (fade_lo / fade_hi
    // are that speed times the reference's age, so a steady pan fades the same at every age); the
    // model is smooth and taken on this frame's colour, so the tone stays and nothing swims.
    float detail = 1.0;
    if (model && fade_hi > 0.0) {
        detail = 1.0 - saturate((motion_length_at(px, py, blocks_x, blocks_y) - fade_lo) / max(fade_hi - fade_lo, 1e-3));
    }
    // where the moved reference does not match: the regional model on this colour (the head alone:
    // nothing)
    // kFadeNone: under a fast camera nothing from an older frame fits (Spider-Man capture, frames
    // 28-37: the moved head, the regional model alike error ~0.98 against interval 1: the game's
    // picture loses half its detail to motion blur and the network corrects it differently), so the
    // correction fades out altogether instead of showing patches from the wrong place
    if (model && (flags & kFadeNone) != 0) {
        head.rgb = lerp(coarse_at(float2(px, py) + best, cur_colour), head.rgb, confidence) * detail;
    } else {
        head.rgb = model ? lerp(coarse_at(float2(px, py) + best, cur_colour), head.rgb, confidence * detail) : head.rgb * confidence;
    }
    g_head.Store2((id.y * field_w + id.x) * 8u, uint2(pack2(head.x, head.y), pack2(head.z, head.w)));
}

[numthreads(8, 8, 1)]
void CSNeuralPost(uint3 id : SV_DispatchThreadID) {
    if (id.x >= src_w || id.y >= src_h) return;
    float4 source = g_color.Load(int3(id.xy, 0));
    float3 colour = saturate(source.rgb);
    float4 head = head_for_output(id.x, id.y);

    float3 predicted = saturate(colour + float3(h(head.x), h(head.y), h(head.z)) * 0.25);
    if ((flags & kHistoryValid) != 0) {
        float alpha = saturate(blend_scale / (1.0 + exp(-h(head.w))));
        predicted += alpha * (history_at(id.x, id.y) - predicted);
    }
    float3 result = saturate(colour + intensity * (predicted - colour));
    g_output[id.xy] = float4(result, source.a);
    g_history.Store2((id.y * src_w + id.x) * 8u, uint2(pack2(predicted.x, predicted.y), pack2(predicted.z, 0.0)));
}
