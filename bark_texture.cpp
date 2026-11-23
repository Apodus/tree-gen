// Bark Phase B-1 — Single-pass exact integer-periodic 2D torus heightfield +
// species morphology baker (oak, pine, birch, maple).
//
// Replaces 4x redundant non-wrapping simplex FBM with ONE 1024x512 composite
// BarkField synthesized from exact integer-periodic (Pu, Pv) 2D gradient noise
// and periodic 2D Worley/cellular noise (including periodic domain warp).
//
// Invariants:
//   - Bit-exact torus periodicity across both U (col 0 <-> 1023) and V (row 0 <-> 511).
//   - Physically scaled tangent-space normals (18-38 deg ridge/fissure walls).
//   - True multi-radius cavity AO (deep furrow floors dark, ridge crests bright).
//   - Physically grounded roughness (0.48-0.92, glTF MR layout R=255, G=rough, B=0, A=255).
//   - /fp:precise enforced for Debug == Release == Retail byte-identity.
#include "bark_texture.hpp"
#include "det_rng.hpp"
#include "png_encoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace treegen {

    namespace {

        constexpr int   k_w = K_BARK_TEX_W;  // 1024
        constexpr int   k_h = K_BARK_TEX_H;  // 512
        constexpr size_t k_n = static_cast<size_t>(k_w) * static_cast<size_t>(k_h);

        // Physical tile extent in metres (one 1024x512 repeat around trunk girth
        // and axial height). Used to derive physically scaled normal slopes.
        constexpr float k_tile_u_m = 1.80f;
        constexpr float k_tile_v_m = 1.80f;

        inline uint64_t mix64(uint64_t x) {
            x += 0x9E3779B97F4A7C15ULL;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            return x ^ (x >> 31);
        }

        inline int wrap_mod(int a, int p) {
            int r = a % p;
            return (r < 0) ? (r + p) : r;
        }

        inline float clamp01(float x) {
            return (x < 0.0f) ? 0.0f : (x > 1.0f ? 1.0f : x);
        }

        inline float clampf(float x, float lo, float hi) {
            return (x < lo) ? lo : (x > hi ? hi : x);
        }

        inline float lerpf(float a, float b, float t) {
            return a + (b - a) * t;
        }

        inline float smoothstep(float e0, float e1, float x) {
            const float t = clamp01((x - e0) / (e1 - e0));
            return t * t * (3.0f - 2.0f * t);
        }

        inline float quintic(float t) {
            return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
        }

        inline uint64_t hash_cell(int ix, int iy, uint64_t seed) {
            const uint64_t h = (static_cast<uint64_t>(static_cast<uint32_t>(ix)) * 0x9E3779B185EBCA87ULL)
                             ^ (static_cast<uint64_t>(static_cast<uint32_t>(iy)) * 0xC2B2AE3D27D4EB4FULL)
                             ^ seed;
            return mix64(h);
        }

        inline float hash_float01(uint64_t h) {
            // Exact 24-bit mantissa uniform float in [0, 1).
            return static_cast<float>((h >> 40) & 0xFFFFFFu) * (1.0f / 16777216.0f);
        }

        // 16 unit gradient vectors evenly spaced on the unit circle.
        constexpr std::array<std::array<float, 2>, 16> k_grad16 = {{
            { 1.00000000f,  0.00000000f}, { 0.92387953f,  0.38268343f},
            { 0.70710678f,  0.70710678f}, { 0.38268343f,  0.92387953f},
            { 0.00000000f,  1.00000000f}, {-0.38268343f,  0.92387953f},
            {-0.70710678f,  0.70710678f}, {-0.92387953f,  0.38268343f},
            {-1.00000000f,  0.00000000f}, {-0.92387953f, -0.38268343f},
            {-0.70710678f, -0.70710678f}, {-0.38268343f, -0.92387953f},
            { 0.00000000f, -1.00000000f}, { 0.38268343f, -0.92387953f},
            { 0.70710678f, -0.70710678f}, { 0.92387953f, -0.38268343f},
        }};

        // Exact integer-periodic 2D Perlin gradient noise in [-1, 1].
        // Tiles seamlessly on u in [0, 1) with period Pu, v in [0, 1) with period Pv.
        float periodic_grad(float u, float v, int pu, int pv, uint64_t seed) {
            const float x = u * static_cast<float>(pu);
            const float y = v * static_cast<float>(pv);
            const int ix0 = static_cast<int>(std::floor(x));
            const int iy0 = static_cast<int>(std::floor(y));
            const float fx = x - static_cast<float>(ix0);
            const float fy = y - static_cast<float>(iy0);

            const int wx0 = wrap_mod(ix0,     pu);
            const int wx1 = wrap_mod(ix0 + 1, pu);
            const int wy0 = wrap_mod(iy0,     pv);
            const int wy1 = wrap_mod(iy0 + 1, pv);

            auto grad_dot = [&](int gx, int gy, float dx, float dy) -> float {
                const uint64_t h = hash_cell(gx, gy, seed);
                const auto& g = k_grad16[h & 15u];
                return g[0] * dx + g[1] * dy;
            };

            const float n00 = grad_dot(wx0, wy0, fx,        fy);
            const float n10 = grad_dot(wx1, wy0, fx - 1.0f, fy);
            const float n01 = grad_dot(wx0, wy1, fx,        fy - 1.0f);
            const float n11 = grad_dot(wx1, wy1, fx - 1.0f, fy - 1.0f);

            const float sx = quintic(fx);
            const float sy = quintic(fy);
            const float nx0 = lerpf(n00, n10, sx);
            const float nx1 = lerpf(n01, n11, sx);
            // Scale by ~1.35 so output spans close to [-1, 1].
            return clampf(lerpf(nx0, nx1, sy) * 1.35f, -1.0f, 1.0f);
        }

        // Exact integer-periodic FBM in [0, 1].
        float periodic_fbm(float u, float v, int pu0, int pv0, int octaves, float gain, uint64_t seed) {
            int pu = pu0;
            int pv = pv0;
            float amp = 1.0f;
            float sum = 0.0f;
            float norm = 0.0f;
            for (int o = 0; o < octaves; ++o) {
                sum  += amp * periodic_grad(u, v, pu, pv, mix64(seed ^ static_cast<uint64_t>(o * 7919 + 17)));
                norm += amp;
                pu *= 2;
                pv *= 2;
                amp *= gain;
            }
            return clamp01((sum / norm) * 0.5f + 0.5f);
        }

        // Exact integer-periodic ridged multifractal in [0, 1] (0 = sharp furrow floor, 1 = broad ridge crest).
        float periodic_ridged(float u, float v, int pu0, int pv0, int octaves, float gain, uint64_t seed) {
            int pu = pu0;
            int pv = pv0;
            float amp = 1.0f;
            float sum = 0.0f;
            float norm = 0.0f;
            for (int o = 0; o < octaves; ++o) {
                const float g = periodic_grad(u, v, pu, pv, mix64(seed ^ static_cast<uint64_t>(o * 6271 + 31)));
                // Sharp V-valley at g == 0, rounded ridge peak at |g| == 1.
                const float r = std::abs(g);
                sum  += amp * r;
                norm += amp;
                pu *= 2;
                pv *= 2;
                amp *= gain;
            }
            return clamp01(sum / norm);
        }

        struct WorleyHit {
            float f1;         // distance to nearest feature point (in cell units)
            float f2;         // distance to second-nearest feature point
            float edge;       // f2 - f1 (0 at Voronoi cell border, ~0.5 at cell center)
            float cell_val;   // deterministic per-cell value in [0, 1)
            float cell_val2;  // second independent deterministic per-cell value in [0, 1)
            float dx;         // signed vector from nearest cell point to sample (cell units)
            float dy;
        };

        // Exact integer-periodic 2D Worley / cellular noise.
        WorleyHit periodic_worley(float u, float v, int pu, int pv, float jitter, uint64_t seed) {
            const float x = u * static_cast<float>(pu);
            const float y = v * static_cast<float>(pv);
            const int cx = static_cast<int>(std::floor(x));
            const int cy = static_cast<int>(std::floor(y));
            const float fx = x - static_cast<float>(cx);
            const float fy = y - static_cast<float>(cy);

            float d1 = 1e9f;
            float d2 = 1e9f;
            uint64_t best_h = 0;
            float best_dx = 0.0f, best_dy = 0.0f;

            for (int oy = -1; oy <= 1; ++oy) {
                const int wy = wrap_mod(cy + oy, pv);
                for (int ox = -1; ox <= 1; ++ox) {
                    const int wx = wrap_mod(cx + ox, pu);
                    const uint64_t h = hash_cell(wx, wy, seed);
                    const float jx = 0.5f + (hash_float01(h) - 0.5f) * jitter;
                    const float jy = 0.5f + (hash_float01(mix64(h)) - 0.5f) * jitter;
                    const float dx = static_cast<float>(ox) + jx - fx;
                    const float dy = static_cast<float>(oy) + jy - fy;
                    const float d2_sq = dx * dx + dy * dy;
                    if (d2_sq < d1) {
                        d2 = d1;
                        d1 = d2_sq;
                        best_h = h;
                        best_dx = -dx;
                        best_dy = -dy;
                    } else if (d2_sq < d2) {
                        d2 = d2_sq;
                    }
                }
            }

            const float f1 = std::sqrt(d1);
            const float f2 = std::sqrt(d2);
            return WorleyHit{
                f1, f2, f2 - f1,
                hash_float01(mix64(best_h ^ 0x51A7E001ULL)),
                hash_float01(mix64(best_h ^ 0x92B3C002ULL)),
                best_dx, best_dy
            };
        }

        // sRGB encode (IEC 61966-2-1).
        inline uint8_t to_srgb_byte(float linear) {
            linear = clamp01(linear);
            const float s = (linear <= 0.0031308f)
                ? 12.92f * linear
                : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
            const float scaled = s * 255.0f + 0.5f;
            return static_cast<uint8_t>(clampf(scaled, 0.0f, 255.0f));
        }

        inline uint8_t to_linear_byte(float linear) {
            const float scaled = clamp01(linear) * 255.0f + 0.5f;
            return static_cast<uint8_t>(clampf(scaled, 0.0f, 255.0f));
        }

        struct LinearRgb { float r, g, b; };

        inline LinearRgb lerp_rgb(LinearRgb a, LinearRgb b, float t) {
            return { lerpf(a.r, b.r, t), lerpf(a.g, b.g, t), lerpf(a.b, b.b, t) };
        }

        // Composite 1024x512 torus field produced in ONE pass per species.
        struct BarkField {
            float              depth_m = 0.025f;   // physical furrow depth in metres
            float              cavity_gain = 3.2f; // multi-radius horizon cavity gain
            float              depth_ao_w  = 0.34f;// absolute depth AO weight
            std::vector<float> height;             // [0, 1] composite physical height
            std::vector<float> r_lin;              // linear albedo R
            std::vector<float> g_lin;              // linear albedo G
            std::vector<float> b_lin;              // linear albedo B
            std::vector<float> rough;              // linear roughness [0.45, 0.94]
            std::vector<float> extra_cavity;       // extra pit/fissure AO darken [0, 1]

            BarkField()
                : height(k_n, 0.5f), r_lin(k_n, 0.2f), g_lin(k_n, 0.2f),
                  b_lin(k_n, 0.2f), rough(k_n, 0.75f), extra_cavity(k_n, 0.0f) {}
        };

        // Wrap-aware separable box blur on a 1024x512 torus float buffer (O(W*H)).
        void periodic_box_blur(const std::vector<float>& src, std::vector<float>& dst,
                               int rx, int ry) {
            std::vector<float> tmp(k_n, 0.0f);
            const float inv_x = 1.0f / static_cast<float>(2 * rx + 1);
            const float inv_y = 1.0f / static_cast<float>(2 * ry + 1);

            for (int y = 0; y < k_h; ++y) {
                const size_t row = static_cast<size_t>(y) * k_w;
                double sum = 0.0;
                for (int kx = -rx; kx <= rx; ++kx) {
                    sum += static_cast<double>(src[row + static_cast<size_t>(wrap_mod(kx, k_w))]);
                }
                for (int x = 0; x < k_w; ++x) {
                    tmp[row + static_cast<size_t>(x)] = static_cast<float>(sum * inv_x);
                    const int add_x = wrap_mod(x + rx + 1, k_w);
                    const int rem_x = wrap_mod(x - rx,     k_w);
                    sum += static_cast<double>(src[row + static_cast<size_t>(add_x)])
                         - static_cast<double>(src[row + static_cast<size_t>(rem_x)]);
                }
            }

            dst.resize(k_n);
            for (int x = 0; x < k_w; ++x) {
                double sum = 0.0;
                for (int ky = -ry; ky <= ry; ++ky) {
                    sum += static_cast<double>(tmp[static_cast<size_t>(wrap_mod(ky, k_h)) * k_w + static_cast<size_t>(x)]);
                }
                for (int y = 0; y < k_h; ++y) {
                    dst[static_cast<size_t>(y) * k_w + static_cast<size_t>(x)] = static_cast<float>(sum * inv_y);
                    const int add_y = wrap_mod(y + ry + 1, k_h);
                    const int rem_y = wrap_mod(y - ry,     k_h);
                    sum += static_cast<double>(tmp[static_cast<size_t>(add_y) * k_w + static_cast<size_t>(x)])
                         - static_cast<double>(tmp[static_cast<size_t>(rem_y) * k_w + static_cast<size_t>(x)]);
                }
            }
        }

        // ---- Species 0: Oak (deep vertically-anisotropic fissured ridges + cross-cracks + lichen) ----
        void synthesize_oak(BarkField& f, uint64_t seed) {
            f.depth_m     = 0.036f;  // 3.6 cm deep furrows -> crisp 25-42 deg ridge walls
            f.cavity_gain = 3.6f;
            f.depth_ao_w  = 0.34f;

            const uint64_t s_wu = mix64(seed ^ 0x0A1001ULL);
            const uint64_t s_wv = mix64(seed ^ 0x0A1002ULL);
            const uint64_t s_r1 = mix64(seed ^ 0x0A1003ULL);
            const uint64_t s_r2 = mix64(seed ^ 0x0A1004ULL);
            const uint64_t s_cc = mix64(seed ^ 0x0A1005ULL);
            const uint64_t s_gr = mix64(seed ^ 0x0A1006ULL);
            const uint64_t s_lc = mix64(seed ^ 0x0A1007ULL);

            constexpr LinearRgb c_deep  = {0.055f, 0.038f, 0.022f}; // dark umber furrow floor
            constexpr LinearRgb c_wall  = {0.235f, 0.160f, 0.095f}; // rich brown furrow wall
            constexpr LinearRgb c_ridge = {0.370f, 0.295f, 0.215f}; // weathered grey-brown ridge cap
            constexpr LinearRgb c_moss  = {0.240f, 0.280f, 0.185f}; // muted sage-olive lichen fleck

            for (int y = 0; y < k_h; ++y) {
                const float v = static_cast<float>(y) / static_cast<float>(k_h);
                for (int x = 0; x < k_w; ++x) {
                    const float u = static_cast<float>(x) / static_cast<float>(k_w);

                    // Periodic wavy domain warp so vertical furrows braid organically.
                    const float wu = (periodic_fbm(u, v, 4, 2, 2, 0.5f, s_wu) - 0.5f) * 0.050f;
                    const float wv = (periodic_fbm(u, v, 4, 3, 2, 0.5f, s_wv) - 0.5f) * 0.055f;
                    const float uw = u + wu;
                    const float vw = v + wv;

                    // Primary deep vertical furrows (16 around girth x 3 axial = ~5.3:1 in UV).
                    const float r1_raw = periodic_ridged(uw, vw, 16, 3, 3, 0.45f, s_r1);
                    // Broad weathered ridge plateaus separated by deep crisp V-furrows.
                    const float r1 = std::pow(smoothstep(0.05f, 0.64f, r1_raw), 0.72f);

                    // Secondary braided sub-ridges + fine longitudinal wood grain.
                    const float r2 = periodic_ridged(uw + 0.017f, vw, 32, 6, 2, 0.50f, s_r2);
                    const float grain = periodic_fbm(uw, vw, 64, 16, 2, 0.5f, s_gr);

                    // Horizontal cross-cracks cutting across ridge tops.
                    const float cc_raw = periodic_ridged(uw, vw, 12, 10, 2, 0.48f, s_cc);
                    const float cross_crack = smoothstep(0.0f, 0.22f, cc_raw);

                    const float macro = periodic_fbm(u, v, 6, 6, 2, 0.5f, mix64(s_lc ^ 0x33ULL));

                    float h = r1 * 0.64f + r2 * 0.18f + grain * 0.10f + macro * 0.08f;
                    h *= (0.80f + 0.20f * cross_crack);
                    h = clamp01(h);

                    LinearRgb col = (h < 0.42f)
                        ? lerp_rgb(c_deep, c_wall, h / 0.42f)
                        : lerp_rgb(c_wall, c_ridge, (h - 0.42f) / 0.58f);

                    // Subtle weathered lichen/moss flecks strictly on upper ridge crests.
                    const float lic_n = periodic_fbm(u, v, 10, 8, 3, 0.52f, s_lc);
                    const float lic_w = smoothstep(0.65f, 0.84f, h) * smoothstep(0.64f, 0.82f, lic_n) * 0.36f;
                    col = lerp_rgb(col, c_moss, lic_w);

                    const float tone = 0.92f + 0.18f * grain;
                    const size_t idx = static_cast<size_t>(y) * k_w + static_cast<size_t>(x);
                    f.height[idx]       = h;
                    f.r_lin[idx]        = clamp01(col.r * tone);
                    f.g_lin[idx]        = clamp01(col.g * tone);
                    f.b_lin[idx]        = clamp01(col.b * tone);
                    // Weathered ridge caps ~0.68, dusty deep furrows ~0.92 (mean ~0.79 -> ~201/255).
                    f.rough[idx]        = clampf(0.92f - 0.26f * h + 0.04f * lic_w, 0.64f, 0.94f);
                    f.extra_cavity[idx] = (1.0f - smoothstep(0.05f, 0.30f, r1_raw)) * 0.52f
                                        + (1.0f - cross_crack) * 0.18f;
                }
            }
        }

        // ---- Species 1: Pine (elongated vertical ragged reddish-brown scaly plates + deep fissures) ----
        void synthesize_pine(BarkField& f, uint64_t seed) {
            f.depth_m     = 0.025f;  // 2.5 cm fissured scaly plates -> 22-38 deg bevels, p99 <= 58 deg
            f.cavity_gain = 3.4f;
            f.depth_ao_w  = 0.33f;

            const uint64_t s_wu = mix64(seed ^ 0x510001ULL);
            const uint64_t s_wv = mix64(seed ^ 0x510002ULL);
            const uint64_t s_w1 = mix64(seed ^ 0x510003ULL);
            const uint64_t s_w2 = mix64(seed ^ 0x510004ULL);
            const uint64_t s_r1 = mix64(seed ^ 0x510005ULL);
            const uint64_t s_gr = mix64(seed ^ 0x510006ULL);

            constexpr LinearRgb c_fissure = {0.068f, 0.042f, 0.024f}; // dark charcoal-umber fissure
            constexpr LinearRgb c_inner   = {0.440f, 0.215f, 0.095f}; // rich cinnamon-red inner bark
            constexpr LinearRgb c_plate   = {0.455f, 0.275f, 0.145f}; // warm terracotta-red pine plate
            constexpr LinearRgb c_grey    = {0.310f, 0.220f, 0.150f}; // weathered grey-brown outer scale

            for (int y = 0; y < k_h; ++y) {
                const float v = static_cast<float>(y) / static_cast<float>(k_h);
                for (int x = 0; x < k_w; ++x) {
                    const float u = static_cast<float>(x) / static_cast<float>(k_w);

                    // Strong 3-octave periodic domain warp shatters polygonal Voronoi lines into
                    // jagged, irregular, organic pine bark plates.
                    const float wu = (periodic_fbm(u, v, 8, 4, 3, 0.55f, s_wu) - 0.5f) * 0.072f;
                    const float wv = (periodic_fbm(u, v, 8, 4, 3, 0.55f, s_wv) - 0.5f) * 0.090f;
                    const float uw = u + wu;
                    const float vw = v + wv;

                    // Vertically elongated interlocking pine plates (18 around girth x 4 axial = ~4.5:1 tall).
                    const WorleyHit w1 = periodic_worley(uw, vw, 18, 4, 0.85f, s_w1);
                    // Modulate fissure width organically so some cracks are wide canyons and others tight seams.
                    const float crack_mod = 0.14f + 0.16f * periodic_fbm(u, v, 6, 4, 2, 0.5f, mix64(s_r1 ^ 0x99ULL));
                    const float plate_mask = smoothstep(0.02f, crack_mod, w1.edge);
                    const float plate_h = 0.52f + 0.34f * w1.cell_val;

                    // Secondary flaky overlapping sub-scales (36 x 8 elongated scales).
                    const WorleyHit w2 = periodic_worley(uw + 0.015f, vw, 36, 8, 0.82f, s_w2);
                    const float sub_edge = smoothstep(0.01f, 0.18f, w2.edge);
                    const float sub_step = (0.80f + 0.20f * w2.cell_val) * (0.84f + 0.16f * sub_edge);

                    // Vertical braided bark fissures & scaly longitudinal grain.
                    const float r_vert = periodic_ridged(uw, vw, 20, 3, 3, 0.48f, s_r1);
                    const float grain  = periodic_fbm(uw, vw, 64, 16, 2, 0.5f, s_gr);

                    // Blend jagged elongated plates with vertical braided fissures so it reads as
                    // scaly flaky bark, never geometric cobblestone.
                    const float plate_comp = plate_mask * plate_h * sub_step;
                    const float ridge_comp = smoothstep(0.05f, 0.72f, r_vert);
                    float h = plate_comp * 0.62f + ridge_comp * 0.26f + grain * 0.12f;
                    h = clamp01(h);

                    // Color: dark umber deep fissure -> cinnamon-red inner bark bevel ->
                    // warm reddish-brown / weathered grey-brown flaky plate faces.
                    LinearRgb plate_face = lerp_rgb(c_plate, c_grey, w1.cell_val2 * 0.58f);
                    const float p_mul = 0.86f + 0.26f * w1.cell_val;
                    plate_face = { plate_face.r * p_mul, plate_face.g * p_mul, plate_face.b * p_mul };

                    LinearRgb col;
                    if (h < 0.28f) {
                        col = lerp_rgb(c_fissure, c_inner, h / 0.28f);
                    } else if (h < 0.52f) {
                        col = lerp_rgb(c_inner, plate_face, (h - 0.28f) / 0.24f);
                    } else {
                        const float hi_t = (h - 0.52f) / 0.48f;
                        col = {
                            plate_face.r * (0.94f + 0.18f * hi_t),
                            plate_face.g * (0.94f + 0.18f * hi_t),
                            plate_face.b * (0.94f + 0.18f * hi_t)
                        };
                    }

                    const float grain_mod = 0.90f + 0.20f * grain;
                    const size_t idx = static_cast<size_t>(y) * k_w + static_cast<size_t>(x);
                    f.height[idx]       = h;
                    f.r_lin[idx]        = clamp01(col.r * grain_mod);
                    f.g_lin[idx]        = clamp01(col.g * grain_mod);
                    f.b_lin[idx]        = clamp01(col.b * grain_mod);
                    // Smooth flaky resinous plates ~0.58, dusty dark fissures ~0.90.
                    f.rough[idx]        = clampf(0.90f - 0.30f * plate_mask + 0.06f * (1.0f - sub_edge), 0.56f, 0.92f);
                    f.extra_cavity[idx] = (1.0f - plate_mask) * 0.52f
                                        + (1.0f - smoothstep(0.05f, 0.30f, r_vert)) * 0.25f;
                }
            }
        }

        // ---- Species 2: Birch (papery horizontal-banded ivory periderm + full-V lenticels + diamond knots) ----
        void synthesize_birch(BarkField& f, uint64_t seed) {
            f.depth_m     = 0.022f;  // 2.2 cm max (gentle papery bands, crisp 22-35 deg lenticels & knots)
            f.cavity_gain = 3.3f;
            f.depth_ao_w  = 0.38f;

            const uint64_t s_wu = mix64(seed ^ 0xB10001ULL);
            const uint64_t s_wv = mix64(seed ^ 0xB10002ULL);
            const uint64_t s_bd = mix64(seed ^ 0xB10003ULL);
            const uint64_t s_pl = mix64(seed ^ 0xB10004ULL);
            const uint64_t s_ln = mix64(seed ^ 0xB10005ULL);
            const uint64_t s_kn = mix64(seed ^ 0xB10006ULL);
            const uint64_t s_kr = mix64(seed ^ 0xB10007ULL);

            constexpr LinearRgb c_knot  = {0.060f, 0.046f, 0.035f}; // dark crusty diamond knot/fissure
            constexpr LinearRgb c_lent  = {0.070f, 0.052f, 0.038f}; // dark horizontal lenticel pore
            constexpr LinearRgb c_peel  = {0.440f, 0.315f, 0.215f}; // warm tan inner papery peel
            constexpr LinearRgb c_mid   = {0.540f, 0.485f, 0.405f}; // warm parchment mid band
            constexpr LinearRgb c_ivory = {0.695f, 0.650f, 0.565f}; // luminous warm-ivory outer periderm

            for (int y = 0; y < k_h; ++y) {
                const float v = static_cast<float>(y) / static_cast<float>(k_h);
                for (int x = 0; x < k_w; ++x) {
                    const float u = static_cast<float>(x) / static_cast<float>(k_w);

                    const float wu = (periodic_fbm(u, v, 4, 4, 2, 0.5f, s_wu) - 0.5f) * 0.028f;
                    const float wv = (periodic_fbm(u, v, 4, 6, 2, 0.5f, s_wv) - 0.5f) * 0.024f;
                    const float uw = u + wu;
                    const float vw = v + wv;

                    // Horizontal papery outer-bark strata (Pu=4, Pv=20).
                    const float band = periodic_fbm(uw, vw, 4, 20, 3, 0.52f, s_bd);
                    const float peel_raw = periodic_ridged(uw, vw, 6, 12, 2, 0.48f, s_pl);
                    const float peel_ledge = 1.0f - smoothstep(0.0f, 0.17f, peel_raw);

                    float h = 0.64f + 0.24f * band - 0.16f * peel_ledge;

                    // Full-V organically scattered tapered horizontal lenticels (NO [200,320) lock!).
                    const WorleyHit ln = periodic_worley(uw, vw, 16, 36, 0.80f, s_ln);
                    float lenticel = 0.0f;
                    if (ln.cell_val > 0.32f) {
                        const float half_u = 0.18f + 0.27f * ln.cell_val2;
                        const float half_v = 0.14f + 0.13f * ((ln.cell_val - 0.32f) / 0.68f);
                        const float au = std::abs(ln.dx) / half_u;
                        const float av = std::abs(ln.dy) / (half_v * std::max(0.20f, 1.0f - 0.65f * au * au));
                        const float r_ell = std::sqrt(au * au + av * av);
                        lenticel = 1.0f - smoothstep(0.52f, 1.0f, r_ell);
                    }

                    // Organic ragged crusty dark bark knots scattered across full torus (4 x 4).
                    const WorleyHit kn = periodic_worley(uw, vw, 4, 4, 0.76f, s_kn);
                    float knot = 0.0f;
                    if (kn.cell_val > 0.40f) {
                        const float rag1 = (periodic_fbm(uw, vw, 16, 16, 3, 0.55f, s_kr) - 0.5f) * 0.22f;
                        const float d_dia = std::abs(kn.dx) * 1.25f + std::abs(kn.dy) * 2.25f + rag1;
                        const float sz = 0.23f + 0.14f * kn.cell_val2;
                        knot = 1.0f - smoothstep(sz * 0.38f, sz, d_dia);
                    }

                    const float knot_cracks = periodic_ridged(uw, vw, 28, 12, 2, 0.5f, s_kr);
                    h -= lenticel * 0.38f;
                    h = lerpf(h, 0.06f + 0.22f * knot_cracks, knot);
                    h = clamp01(h);

                    LinearRgb col = lerp_rgb(c_mid, c_ivory, smoothstep(0.25f, 0.85f, band));
                    col = lerp_rgb(col, c_peel, peel_ledge * 0.68f);
                    col = lerp_rgb(col, c_lent, lenticel * 0.90f);
                    const LinearRgb knot_col = lerp_rgb(c_knot, {0.17f, 0.13f, 0.09f}, knot_cracks * 0.6f);
                    col = lerp_rgb(col, knot_col, knot);

                    const size_t idx = static_cast<size_t>(y) * k_w + static_cast<size_t>(x);
                    f.height[idx]       = h;
                    f.r_lin[idx]        = clamp01(col.r);
                    f.g_lin[idx]        = clamp01(col.g);
                    f.b_lin[idx]        = clamp01(col.b);
                    float r = 0.49f + 0.11f * (1.0f - band) + 0.16f * peel_ledge;
                    r = lerpf(r, 0.89f, lenticel);
                    r = lerpf(r, 0.93f, knot);
                    f.rough[idx]        = clampf(r, 0.48f, 0.93f);
                    f.extra_cavity[idx] = clamp01(lenticel * 0.68f + knot * (0.60f + 0.30f * (1.0f - knot_cracks)) + peel_ledge * 0.30f);
                }
            }
        }

        // ---- Species 3: Maple (long shaggy/flaky vertical strips + irregular fissures + pale lichen) ----
        void synthesize_maple(BarkField& f, uint64_t seed) {
            f.depth_m     = 0.026f;  // 2.6 cm flaky vertical strips
            f.cavity_gain = 3.4f;
            f.depth_ao_w  = 0.35f;

            const uint64_t s_wu = mix64(seed ^ 0xA40001ULL);
            const uint64_t s_wv = mix64(seed ^ 0xA40002ULL);
            const uint64_t s_w1 = mix64(seed ^ 0xA40003ULL);
            const uint64_t s_r1 = mix64(seed ^ 0xA40004ULL);
            const uint64_t s_r2 = mix64(seed ^ 0xA40005ULL);
            const uint64_t s_gr = mix64(seed ^ 0xA40006ULL);
            const uint64_t s_mt = mix64(seed ^ 0xA40007ULL);

            constexpr LinearRgb c_furrow = {0.075f, 0.062f, 0.048f}; // dark grey-brown fissure
            constexpr LinearRgb c_mid    = {0.245f, 0.205f, 0.165f}; // warm taupe-brown inner plate
            constexpr LinearRgb c_plate  = {0.395f, 0.352f, 0.295f}; // silvery-taupe outer vertical strip
            constexpr LinearRgb c_lichen = {0.275f, 0.305f, 0.240f}; // subtle pale grey-green lichen

            for (int y = 0; y < k_h; ++y) {
                const float v = static_cast<float>(y) / static_cast<float>(k_h);
                for (int x = 0; x < k_w; ++x) {
                    const float u = static_cast<float>(x) / static_cast<float>(k_w);

                    // Strong organic domain warp breaks up any straight vertical or polygonal lines.
                    const float wu = (periodic_fbm(u, v, 6, 3, 3, 0.55f, s_wu) - 0.5f) * 0.068f;
                    const float wv = (periodic_fbm(u, v, 6, 4, 3, 0.55f, s_wv) - 0.5f) * 0.078f;
                    const float uw = u + wu;
                    const float vw = v + wv;

                    // Primary shaggy vertical maple fissures (20 around girth x 3 axial = ~6.7:1 vertical strips).
                    const float r1_raw = periodic_ridged(uw, vw, 20, 3, 3, 0.48f, s_r1);
                    const float r1 = std::pow(smoothstep(0.05f, 0.68f, r1_raw), 0.78f);

                    // Secondary narrow vertical flaky/curling strips (24 x 3 heavily warped Voronoi).
                    const WorleyHit w1 = periodic_worley(uw + 0.013f, vw, 24, 3, 0.84f, s_w1);
                    const float strip_seam = smoothstep(0.02f, 0.22f, w1.edge);
                    // Subtle asymmetric curl across each strip's horizontal profile (real sugar-maple curl).
                    const float curl = 0.72f + 0.28f * clamp01(w1.dx * 0.85f + 0.5f);
                    const float strip_h = strip_seam * (0.62f + 0.28f * w1.cell_val) * curl;

                    // Shallow irregular diagonal flake cracks & fine vertical fiber grain.
                    const float r2    = periodic_ridged(uw, vw, 14, 9, 2, 0.50f, s_r2);
                    const float grain = periodic_fbm(uw, vw, 64, 16, 2, 0.5f, s_gr);

                    float h = r1 * 0.52f + strip_h * 0.34f + grain * 0.14f;
                    h *= (0.86f + 0.14f * smoothstep(0.0f, 0.24f, r2));
                    h = clamp01(h);

                    LinearRgb col = (h < 0.42f)
                        ? lerp_rgb(c_furrow, c_mid, h / 0.42f)
                        : lerp_rgb(c_mid, c_plate, (h - 0.42f) / 0.58f);

                    // Subtle pale grey-green lichen & weathered mottling on outer strip faces.
                    const float mottle = periodic_fbm(u, v, 8, 6, 3, 0.52f, s_mt);
                    const float lic_w = smoothstep(0.56f, 0.80f, h) * smoothstep(0.62f, 0.80f, mottle) * 0.34f;
                    col = lerp_rgb(col, c_lichen, lic_w);

                    const float tone = 0.90f + 0.20f * grain;
                    const size_t idx = static_cast<size_t>(y) * k_w + static_cast<size_t>(x);
                    f.height[idx]       = h;
                    f.r_lin[idx]        = clamp01(col.r * tone);
                    f.g_lin[idx]        = clamp01(col.g * tone);
                    f.b_lin[idx]        = clamp01(col.b * tone);
                    f.rough[idx]        = clampf(0.88f - 0.26f * h + 0.05f * lic_w, 0.60f, 0.90f);
                    f.extra_cavity[idx] = (1.0f - smoothstep(0.05f, 0.30f, r1_raw)) * 0.44f
                                        + (1.0f - strip_seam) * 0.28f;
                }
            }
        }

        BarkField build_bark_field(LeafShape species, uint64_t seed) {
            BarkField f;
            const uint64_t sp_seed = mix64(seed ^ species_fnv_name(species));
            switch (species) {
                case LeafShape::OakLobed:      synthesize_oak  (f, sp_seed); break;
                case LeafShape::PineNeedle:    synthesize_pine (f, sp_seed); break;
                case LeafShape::BirchSerrated: synthesize_birch(f, sp_seed); break;
                case LeafShape::MapleStar:     synthesize_maple(f, sp_seed); break;
            }
            return f;
        }

        std::vector<uint8_t> pack_diffuse(const BarkField& f) {
            std::vector<uint8_t> img(k_n * 4);
            for (size_t i = 0; i < k_n; ++i) {
                img[i * 4 + 0] = to_srgb_byte(f.r_lin[i]);
                img[i * 4 + 1] = to_srgb_byte(f.g_lin[i]);
                img[i * 4 + 2] = to_srgb_byte(f.b_lin[i]);
                img[i * 4 + 3] = 255;
            }
            return img;
        }

        std::vector<uint8_t> pack_normal(const BarkField& f) {
            std::vector<uint8_t> img(k_n * 4);
            const float dx_m = k_tile_u_m / static_cast<float>(k_w);
            const float dy_m = k_tile_v_m / static_cast<float>(k_h);
            const float sx = f.depth_m / (2.0f * dx_m);
            const float sy = f.depth_m / (2.0f * dy_m);

            for (int y = 0; y < k_h; ++y) {
                const int ym = wrap_mod(y - 1, k_h);
                const int yp = wrap_mod(y + 1, k_h);
                const size_t row  = static_cast<size_t>(y)  * k_w;
                const size_t rowm = static_cast<size_t>(ym) * k_w;
                const size_t rowp = static_cast<size_t>(yp) * k_w;
                for (int x = 0; x < k_w; ++x) {
                    const int xm = wrap_mod(x - 1, k_w);
                    const int xp = wrap_mod(x + 1, k_w);
                    const float dh_du = (f.height[row + static_cast<size_t>(xp)]
                                       - f.height[row + static_cast<size_t>(xm)]) * sx;
                    const float dh_dv = (f.height[rowp + static_cast<size_t>(x)]
                                       - f.height[rowm + static_cast<size_t>(x)]) * sy;

                    // Tangent-space normal: (-dh/du_m, -dh/dv_m, 1) normalized.
                    float nx = -dh_du;
                    float ny = -dh_dv;
                    float nz = 1.0f;
                    const float inv_l = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
                    nx *= inv_l; ny *= inv_l; nz *= inv_l;

                    const size_t idx = (row + static_cast<size_t>(x)) * 4;
                    img[idx + 0] = to_linear_byte(nx * 0.5f + 0.5f);
                    img[idx + 1] = to_linear_byte(ny * 0.5f + 0.5f);
                    img[idx + 2] = to_linear_byte(nz * 0.5f + 0.5f);
                    img[idx + 3] = 255;
                }
            }
            return img;
        }

        std::vector<uint8_t> pack_ao(const BarkField& f) {
            // True multi-radius wrap-aware cavity / horizon AO:
            // deep furrow floors dark (0.18-0.35), ridge crests bright (0.85-1.0).
            std::vector<float> blur_small, blur_large;
            periodic_box_blur(f.height, blur_small, 6, 4);
            periodic_box_blur(f.height, blur_large, 18, 10);

            std::vector<uint8_t> img(k_n * 4);
            for (size_t i = 0; i < k_n; ++i) {
                const float local_mean = 0.45f * blur_small[i] + 0.55f * blur_large[i];
                const float cavity = clamp01((local_mean - f.height[i]) * f.cavity_gain);
                const float ao = clampf(0.98f
                                        - 0.50f * cavity
                                        - f.depth_ao_w * (1.0f - f.height[i])
                                        - 0.24f * f.extra_cavity[i],
                                        0.16f, 1.0f);
                const uint8_t b = to_linear_byte(ao);
                img[i * 4 + 0] = b;
                img[i * 4 + 1] = b;
                img[i * 4 + 2] = b;
                img[i * 4 + 3] = 255;
            }
            return img;
        }

        std::vector<uint8_t> pack_roughness(const BarkField& f) {
            std::vector<uint8_t> img(k_n * 4);
            for (size_t i = 0; i < k_n; ++i) {
                img[i * 4 + 0] = 255;                    // R unused (glTF MR)
                img[i * 4 + 1] = to_linear_byte(f.rough[i]); // G roughness
                img[i * 4 + 2] = 0;                      // B metallic = 0
                img[i * 4 + 3] = 255;                    // A
            }
            return img;
        }

        // Lossless channel-compact PNG encoders for opaque bark maps (all GLB
        // consumers decode via stbi_load_from_memory(..., 4) to bit-exact RGBA8).
        std::vector<uint8_t> encode_opaque_rgb_png(const std::vector<uint8_t>& rgba) {
            std::vector<uint8_t> rgb(k_n * 3);
            for (size_t i = 0; i < k_n; ++i) {
                rgb[i * 3 + 0] = rgba[i * 4 + 0];
                rgb[i * 3 + 1] = rgba[i * 4 + 1];
                rgb[i * 3 + 2] = rgba[i * 4 + 2];
            }
            return encode_png_rgb8(k_w, k_h, k_w * 3, rgb.data());
        }

        std::vector<uint8_t> encode_opaque_gray_png(const std::vector<uint8_t>& rgba) {
            std::vector<uint8_t> gray(k_n);
            for (size_t i = 0; i < k_n; ++i) gray[i] = rgba[i * 4 + 0];
            return encode_png_gray8(k_w, k_h, k_w, gray.data());
        }

    } // anon namespace

    uint64_t species_fnv_name(LeafShape species) {
        return fnv1a64(std::string_view{leaf_shape_to_string(species)});
    }

    BarkPngBundle encode_bark_bundle_pngs(LeafShape species, uint64_t seed) {
        const BarkField field = build_bark_field(species, seed);
        BarkPngBundle out;
        out.diffuse_png   = encode_opaque_rgb_png (pack_diffuse(field));
        out.normal_png    = encode_opaque_rgb_png (pack_normal(field));
        out.ao_png        = encode_opaque_gray_png(pack_ao(field));
        out.roughness_png = encode_opaque_rgb_png (pack_roughness(field));
        return out;
    }

    std::vector<uint8_t> bake_bark_diffuse(LeafShape species, uint64_t seed) {
        return pack_diffuse(build_bark_field(species, seed));
    }

    std::vector<uint8_t> encode_bark_png(LeafShape species, uint64_t seed) {
        return encode_opaque_rgb_png(bake_bark_diffuse(species, seed));
    }

    std::vector<uint8_t> bake_bark_normal(LeafShape species, uint64_t seed) {
        return pack_normal(build_bark_field(species, seed));
    }

    std::vector<uint8_t> encode_bark_normal_png(LeafShape species, uint64_t seed) {
        return encode_opaque_rgb_png(bake_bark_normal(species, seed));
    }

    std::vector<uint8_t> bake_bark_ao(LeafShape species, uint64_t seed) {
        return pack_ao(build_bark_field(species, seed));
    }

    std::vector<uint8_t> encode_bark_ao_png(LeafShape species, uint64_t seed) {
        return encode_opaque_gray_png(bake_bark_ao(species, seed));
    }

    std::vector<uint8_t> bake_bark_roughness(LeafShape species, uint64_t seed) {
        return pack_roughness(build_bark_field(species, seed));
    }

    std::vector<uint8_t> encode_bark_roughness_png(LeafShape species, uint64_t seed) {
        return encode_opaque_rgb_png(bake_bark_roughness(species, seed));
    }

}
