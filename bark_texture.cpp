// C2 P1 — Simplex noise FBM + analytic derivatives for bark textures.
//   Replaces 4-octave value noise with 6-octave simplex gradient noise.
//   Normal bake uses analytic dh/dx, dh/dy (eliminates 4 extra samples/px).
//   AO bake uses gradient magnitude (eliminates 49-sample box mean/px).
// /fp:precise enforced by treegen.sharpmake.cs — required for byte-deterministic
// FBM output.
#include "bark_texture.hpp"
#include "det_rng.hpp"
#include "png_encoder.hpp"
#include "simplex_noise.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace treegen {

    namespace {

        // ---- hash helper (mirror det_rng style; self-contained to avoid cross-
        //      coupling bark to rng utilities only used by skeleton growth).
        inline uint64_t mix64(uint64_t x) {
            x += 0x9E3779B97F4A7C15ULL;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            return x ^ (x >> 31);
        }

        constexpr int k_w = 1024;
        constexpr int k_h = 512;

        // ---- Per-species bark profile. Replaces BarkParams with per-species
        //      lacunarity/gain for simplex FBM tuning + 3-key color ramp (P2).
        struct LinearRgb { float r, g, b; };

        struct BarkProfile {
            float     lacunarity;         // frequency multiplier per octave
            float     gain;               // amplitude decay per octave
            float     normal_strength;    // dh/dpixel multiplier in normal bake
            float     ao_scale;           // gradient-magnitude multiplier in AO bake
            float     rough_base;         // baseline roughness in linear [0,1]
            bool      has_lenticels;      // birch-style horizontal stripe pores
            LinearRgb color_crevice;      // dark crevice tone (h=0)
            LinearRgb color_mid;          // mid-range tone (h=0.5)
            LinearRgb color_ridge;        // ridge/surface tone (h=1)
        };

        // ao_scale tuned for analytic-derivative AO (gradient magnitude in
        // texture-space units). Simplex FBM derivatives are in height-per-pixel;
        // typical gradient magnitudes ~0.01. Scales calibrated so mean AO byte
        // lands within the existing test bounds. Relative ratios preserve "more
        // rugged species = higher scale" (pine highest, birch lowest).
        //
        // 3-key color ramp: crevice (h=0) -> mid (h=0.5) -> ridge (h=1).
        // Replaces single baseColor * modulation. Linear RGB.
        constexpr BarkProfile k_bark_profiles[4] = {
            //                  lac   gain  norm  ao    rough lent  crevice              mid                  ridge
            /* OakLobed      */ { 2.0f, 0.50f, 4.0f, 16.0f, 0.75f, false, {0.12f,0.08f,0.05f}, {0.28f,0.18f,0.10f}, {0.38f,0.28f,0.18f} },
            /* PineNeedle    */ { 2.2f, 0.45f, 5.0f, 24.0f, 0.65f, false, {0.25f,0.18f,0.10f}, {0.49f,0.36f,0.20f}, {0.55f,0.42f,0.28f} },
            /* BirchSerrated */ { 2.0f, 0.55f, 1.5f, 12.0f, 0.55f, true,  {0.60f,0.58f,0.52f}, {0.84f,0.83f,0.78f}, {0.92f,0.91f,0.88f} },
            /* MapleStar     */ { 2.1f, 0.48f, 3.0f, 16.0f, 0.70f, false, {0.18f,0.14f,0.10f}, {0.36f,0.29f,0.23f}, {0.48f,0.40f,0.32f} },
        };
        static_assert(int(LeafShape::OakLobed)      == 0, "k_bark_profiles index pinned to LeafShape enum order");
        static_assert(int(LeafShape::PineNeedle)    == 1, "k_bark_profiles index pinned to LeafShape enum order");
        static_assert(int(LeafShape::BirchSerrated) == 2, "k_bark_profiles index pinned to LeafShape enum order");
        static_assert(int(LeafShape::MapleStar)     == 3, "k_bark_profiles index pinned to LeafShape enum order");

        inline const BarkProfile& bark_for(LeafShape s) {
            return k_bark_profiles[static_cast<int>(s)];
        }

        // ---- 6-octave simplex FBM with analytic derivatives.
        // Returns height in [0, 1] and derivatives dh/du, dh/dv in texture-space
        // units (per-pixel). Wraps in U (cylindrical bark mapping) by sampling at
        // two offset points and blending — simplex noise itself is non-periodic,
        // so we use a standard cylindrical wrap trick: evaluate at (cos(2pi*u)*r,
        // sin(2pi*u)*r, v) mapped to 2D via two shifted simplex samples blended
        // by u-phase. Simpler approach: just feed the pixel coords and accept
        // that seams exist only at u=0 where the bark wraps — acceptable for a
        // cylindrical bark texture where the seam is on the back of the tree.
        //
        // For seamless U wrap, we evaluate the noise on a cylinder: the simplex
        // domain point traces a circle in XY as U goes 0..1, with V along a
        // third axis projected to the second simplex coordinate.
        struct fbm_result { float height; float dhdx; float dhdy; };

        fbm_result fbm_simplex(uint64_t seed, int px, int py, const BarkProfile& profile) {
            const float u = static_cast<float>(px) / static_cast<float>(k_w);
            const float v = static_cast<float>(py) / static_cast<float>(k_h);

            constexpr float k_pi2 = 6.283185307179586f;
            constexpr int k_octaves = 6;
            constexpr float k_base_freq = 8.0f;

            float freq = k_base_freq;
            float amp = 1.0f;
            float sum = 0.0f;
            float dsum_dx = 0.0f;
            float dsum_dy = 0.0f;
            float amp_total = 0.0f;

            for (int o = 0; o < k_octaves; ++o) {
                // Cylindrical wrap in U: map U to a circle of radius freq/(2*pi)
                // so that the noise tiles seamlessly. The radius is chosen so the
                // arc length equals freq (matching the V scale).
                const float r = freq / k_pi2;
                const float angle = u * k_pi2;
                const float cx = r * std::cos(angle);
                const float cy = r * std::sin(angle);
                const float fv = v * freq;

                // Use two simplex samples to map the 3D cylinder to 2D:
                // sample A at (cx, fv) and sample B at (cy, fv) with offset seed.
                uint64_t seed_a = mix64(seed ^ static_cast<uint64_t>(o));
                uint64_t seed_b = mix64(seed_a ^ 0xDEADBEEFCAFEULL);

                auto sa = simplex_noise_2d(cx, fv, seed_a);
                auto sb = simplex_noise_2d(cy, fv, seed_b);

                // Blend: weighted sum to produce a cylindrically-tileable signal.
                // Both samples contribute equally; the combined signal tiles
                // because each sample's U-dependent coordinate traces a full
                // circle.
                float noise_val = (sa.value + sb.value) * 0.5f;

                // Chain-rule derivatives for the pixel-space dh/dpx, dh/dpy:
                //   dcx/du = -r*sin(angle)*2*pi/k_w  (u = px/k_w)
                //   dcy/du =  r*cos(angle)*2*pi/k_w
                //   dfv/dv = freq / k_h  (v = py/k_h)
                float du_dpx = 1.0f / static_cast<float>(k_w);
                float dv_dpy = 1.0f / static_cast<float>(k_h);

                float dcx_dpx = -r * std::sin(angle) * k_pi2 * du_dpx;
                float dcy_dpx =  r * std::cos(angle) * k_pi2 * du_dpx;
                float dfv_dpy = freq * dv_dpy;

                // sa depends on (cx, fv); sb depends on (cy, fv).
                // d(noise)/dpx = 0.5*(sa.dx * dcx_dpx + sb.dx * dcy_dpx)
                // d(noise)/dpy = 0.5*(sa.dy * dfv_dpy + sb.dy * dfv_dpy)
                float dn_dpx = 0.5f * (sa.dx * dcx_dpx + sb.dx * dcy_dpx);
                float dn_dpy = 0.5f * (sa.dy * dfv_dpy + sb.dy * dfv_dpy);

                sum      += amp * noise_val;
                dsum_dx  += amp * dn_dpx;
                dsum_dy  += amp * dn_dpy;
                amp_total += amp;

                freq *= profile.lacunarity;
                amp  *= profile.gain;
            }

            // Normalize to [0, 1].
            float inv_amp = 1.0f / amp_total;
            float h = sum * inv_amp * 0.5f + 0.5f;
            float dx = dsum_dx * inv_amp * 0.5f;
            float dy = dsum_dy * inv_amp * 0.5f;

            // Clamp height to [0,1].
            if (h < 0.0f) h = 0.0f;
            if (h > 1.0f) h = 1.0f;

            return { h, dx, dy };
        }

        // sRGB encode (IEC 61966-2-1).
        inline uint8_t to_srgb_byte(float linear) {
            if (linear < 0.0f) linear = 0.0f;
            if (linear > 1.0f) linear = 1.0f;
            float s = (linear <= 0.0031308f)
                ? 12.92f * linear
                : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
            float scaled = s * 255.0f + 0.5f;
            if (scaled < 0.0f)   scaled = 0.0f;
            if (scaled > 255.0f) scaled = 255.0f;
            return static_cast<uint8_t>(scaled);
        }

        inline uint8_t to_linear_byte(float linear) {
            if (linear < 0.0f) linear = 0.0f;
            if (linear > 1.0f) linear = 1.0f;
            float scaled = linear * 255.0f + 0.5f;
            if (scaled > 255.0f) scaled = 255.0f;
            return static_cast<uint8_t>(scaled);
        }

        // ---- Birch lenticel mask (unchanged from C6 P2).
        inline float birch_lenticel_mask(int px, int py, uint64_t seed) {
            if (py < 200 || py >= 320) return 1.0f;

            constexpr int k_row_period_y   = 14;
            constexpr int k_row_thickness  = 4;
            constexpr int k_dash_period_x  = 64;
            constexpr int k_dash_length    = 28;
            constexpr float k_dark_floor   = 0.40f;

            const int row_phase = py % k_row_period_y;
            if (row_phase >= k_row_thickness) return 1.0f;

            uint64_t hr = mix64(seed ^ static_cast<uint64_t>(py / k_row_period_y));
            const int dash_offset = static_cast<int>(hr % static_cast<uint64_t>(k_dash_period_x));
            const int col_phase = ((px + dash_offset) % k_dash_period_x + k_dash_period_x) % k_dash_period_x;
            if (col_phase >= k_dash_length) return 1.0f;

            float t_end = std::min(col_phase, k_dash_length - 1 - col_phase) / 4.0f;
            if (t_end > 1.0f) t_end = 1.0f;
            const float t_row = 1.0f - std::abs(static_cast<float>(row_phase) - 1.5f) / 1.5f;
            const float t = t_end * t_row;
            return 1.0f - (1.0f - k_dark_floor) * t;
        }

    } // anon namespace

    uint64_t species_fnv_name(LeafShape species) {
        return fnv1a64(std::string_view{leaf_shape_to_string(species)});
    }

    std::vector<uint8_t> bake_bark_diffuse(LeafShape species, uint64_t seed) {
        const BarkProfile& bp = bark_for(species);

        std::vector<uint8_t> img(static_cast<size_t>(k_w) * k_h * 4);
        for (int y = 0; y < k_h; ++y) {
            for (int x = 0; x < k_w; ++x) {
                auto fbm = fbm_simplex(seed, x, y, bp);
                const float h = fbm.height; // [0, 1]

                // 3-key color ramp: crevice -> mid -> ridge.
                float lr, lg, lb;
                if (h < 0.5f) {
                    const float t = h * 2.0f; // [0, 1] in lower half
                    lr = bp.color_crevice.r + (bp.color_mid.r - bp.color_crevice.r) * t;
                    lg = bp.color_crevice.g + (bp.color_mid.g - bp.color_crevice.g) * t;
                    lb = bp.color_crevice.b + (bp.color_mid.b - bp.color_crevice.b) * t;
                } else {
                    const float t = (h - 0.5f) * 2.0f; // [0, 1] in upper half
                    lr = bp.color_mid.r + (bp.color_ridge.r - bp.color_mid.r) * t;
                    lg = bp.color_mid.g + (bp.color_ridge.g - bp.color_mid.g) * t;
                    lb = bp.color_mid.b + (bp.color_ridge.b - bp.color_mid.b) * t;
                }

                if (bp.has_lenticels) {
                    const float m = birch_lenticel_mask(x, y, seed);
                    lr *= m;
                    lg *= m;
                    lb *= m;
                }

                const size_t idx = (static_cast<size_t>(y) * k_w + x) * 4;
                img[idx + 0] = to_srgb_byte(lr);
                img[idx + 1] = to_srgb_byte(lg);
                img[idx + 2] = to_srgb_byte(lb);
                img[idx + 3] = 255;
            }
        }
        return img;
    }

    std::vector<uint8_t> encode_bark_png(LeafShape species, uint64_t seed) {
        auto rgba = bake_bark_diffuse(species, seed);
        return encode_png_rgba8(k_w, k_h, k_w * 4, rgba.data());
    }

    // ========================================================================
    // Normal / AO / Roughness bakes — analytic derivatives from simplex FBM.
    // ========================================================================

    std::vector<uint8_t> bake_bark_normal(LeafShape species, uint64_t seed) {
        const BarkProfile& bp = bark_for(species);
        const float strength = bp.normal_strength;

        std::vector<uint8_t> img(static_cast<size_t>(k_w) * k_h * 4);
        for (int y = 0; y < k_h; ++y) {
            for (int x = 0; x < k_w; ++x) {
                auto fbm = fbm_simplex(seed, x, y, bp);

                // Analytic derivatives replace central-difference finite-diff.
                // Sign convention pinned: (nx, ny) = (-dh/dx*s, -dh/dy*s);
                // GL-style +Y-up tangent space normal map.
                float nx = -fbm.dhdx * strength;
                float ny = -fbm.dhdy * strength;
                float nz = 1.0f;
                const float inv_len = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
                nx *= inv_len; ny *= inv_len; nz *= inv_len;

                const size_t idx = (static_cast<size_t>(y) * k_w + x) * 4;
                img[idx + 0] = to_linear_byte(nx * 0.5f + 0.5f);
                img[idx + 1] = to_linear_byte(ny * 0.5f + 0.5f);
                img[idx + 2] = to_linear_byte(nz * 0.5f + 0.5f);
                img[idx + 3] = 255;
            }
        }
        return img;
    }

    std::vector<uint8_t> encode_bark_normal_png(LeafShape species, uint64_t seed) {
        auto rgba = bake_bark_normal(species, seed);
        return encode_png_rgba8(k_w, k_h, k_w * 4, rgba.data());
    }

    std::vector<uint8_t> bake_bark_ao(LeafShape species, uint64_t seed) {
        const BarkProfile& bp = bark_for(species);
        const float ao_scale = bp.ao_scale;

        // Gradient-magnitude AO: AO = 1 - |grad| * ao_scale, clamped [0,1].
        // Crevices have steep gradients → low AO; flat areas → AO near 1.
        // Eliminates the 49-sample box-mean window.
        std::vector<uint8_t> img(static_cast<size_t>(k_w) * k_h * 4);
        for (int y = 0; y < k_h; ++y) {
            for (int x = 0; x < k_w; ++x) {
                auto fbm = fbm_simplex(seed, x, y, bp);
                float grad_mag = std::sqrt(fbm.dhdx * fbm.dhdx + fbm.dhdy * fbm.dhdy);
                float ao = 1.0f - grad_mag * ao_scale;
                if (ao < 0.0f) ao = 0.0f;
                if (ao > 1.0f) ao = 1.0f;

                const uint8_t b = to_linear_byte(ao);
                const size_t idx = (static_cast<size_t>(y) * k_w + x) * 4;
                img[idx + 0] = b;
                img[idx + 1] = b;
                img[idx + 2] = b;
                img[idx + 3] = 255;
            }
        }
        return img;
    }

    std::vector<uint8_t> encode_bark_ao_png(LeafShape species, uint64_t seed) {
        auto rgba = bake_bark_ao(species, seed);
        return encode_png_rgba8(k_w, k_h, k_w * 4, rgba.data());
    }

    std::vector<uint8_t> bake_bark_roughness(LeafShape species, uint64_t seed) {
        const BarkProfile& bp = bark_for(species);
        const float rough_base = bp.rough_base;
        constexpr float k_cavity_boost = 0.10f;

        std::vector<uint8_t> img(static_cast<size_t>(k_w) * k_h * 4);
        for (int y = 0; y < k_h; ++y) {
            for (int x = 0; x < k_w; ++x) {
                auto fbm = fbm_simplex(seed, x, y, bp);
                float r = rough_base + (1.0f - fbm.height) * k_cavity_boost;
                if (r < 0.0f) r = 0.0f;
                if (r > 1.0f) r = 1.0f;

                const size_t idx = (static_cast<size_t>(y) * k_w + x) * 4;
                img[idx + 0] = 255;
                img[idx + 1] = to_linear_byte(r);
                img[idx + 2] = 0;
                img[idx + 3] = 255;
            }
        }
        return img;
    }

    std::vector<uint8_t> encode_bark_roughness_png(LeafShape species, uint64_t seed) {
        auto rgba = bake_bark_roughness(species, seed);
        return encode_png_rgba8(k_w, k_h, k_w * 4, rgba.data());
    }

}
