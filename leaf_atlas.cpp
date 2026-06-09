// C6 P3 — leaf atlas bake.
//   P3a: alpha + diffuse (this file's `encode_leaf_atlas_png`).
//   P3b: normal + translucency (added when those encoders ship).
//   C3 P1: intra-leaf color gradients (tip-to-base, vein darkening, edge tint).
// /fp:precise (treegen.sharpmake.cs) — per-pixel barycentric edge tests, sRGB
// conversion, and vein stamping all contribute to atlas byte hash.
#include "leaf_atlas.hpp"

#include "det_rng.hpp"         // pcg32
#include "leaf_geometry.hpp"   // shape_mesh_for(LeafShape)
#include "leaf_rasterizer.hpp"
#include "leaf_veins.hpp"
#include "png_encoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace treegen {

    namespace {

        // Linear RGB greens. Per the C6 P3 plan; "greens" cover the full crown
        // tone range without per-cell sampling for now.
        struct LinearRgb { float r, g, b; };
        constexpr LinearRgb k_leaf_color[4] = {
            /* OakLobed      */ { 0.20f, 0.38f, 0.12f },
            /* PineNeedle    */ { 0.10f, 0.30f, 0.12f },
            /* BirchSerrated */ { 0.35f, 0.50f, 0.18f },
            /* MapleStar     */ { 0.30f, 0.38f, 0.14f },
        };
        static_assert(int(LeafShape::OakLobed)      == 0, "k_leaf_color index pinned to LeafShape enum order");
        static_assert(int(LeafShape::PineNeedle)    == 1, "k_leaf_color index pinned to LeafShape enum order");
        static_assert(int(LeafShape::BirchSerrated) == 2, "k_leaf_color index pinned to LeafShape enum order");
        static_assert(int(LeafShape::MapleStar)     == 3, "k_leaf_color index pinned to LeafShape enum order");

        // C3 P1 — per-species gradient parameters.
        struct GradientParams {
            float tip_to_base_strength;   // 0 = none, 0.25 = darken base by 25%
            bool  apply_edge_tint;        // false for pine (too thin)
        };
        constexpr GradientParams k_gradient[4] = {
            /* OakLobed      */ { 0.25f, true  },
            /* PineNeedle    */ { 0.15f, false },  // subtle longitudinal only
            /* BirchSerrated */ { 0.20f, true  },
            /* MapleStar     */ { 0.30f, true  },
        };

        inline uint8_t srgb_byte(float linear) {
            if (linear < 0.0f) linear = 0.0f;
            if (linear > 1.0f) linear = 1.0f;
            const float s = (linear <= 0.0031308f)
                ? 12.92f * linear
                : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
            float scaled = s * 255.0f + 0.5f;
            if (scaled < 0.0f)   scaled = 0.0f;
            if (scaled > 255.0f) scaled = 255.0f;
            return static_cast<uint8_t>(scaled);
        }

        inline float smoothstep(float edge0, float edge1, float x) {
            const float t = std::max(0.0f, std::min(1.0f, (x - edge0) / (edge1 - edge0)));
            return t * t * (3.0f - 2.0f * t);
        }

        constexpr std::array<LeafShape, 4> k_species_in_atlas = {
            LeafShape::OakLobed,
            LeafShape::PineNeedle,
            LeafShape::BirchSerrated,
            LeafShape::MapleStar,
        };

        // Bake one cell's alpha mask. Returns a 256×256 byte buffer.
        std::vector<uint8_t> bake_cell_alpha(LeafShape sp) {
            const LeafShapeMesh mesh = shape_mesh_for(sp);
            std::vector<uint8_t> alpha(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            rasterize_leaf_alpha_into_cell(mesh,
                                           alpha.data(),
                                           K_LEAF_CELL_PX,
                                           K_LEAF_CELL_PX,
                                           K_LEAF_CELL_PAD_PX,
                                           K_LEAF_CELL_USABLE_PX);
            return alpha;
        }

        // Compute alpha vertical extent for the cell (y_min/y_max of alpha>0 rows).
        struct AlphaExtent { int y_min; int y_max; };
        AlphaExtent compute_alpha_extent(const uint8_t* cell_alpha) {
            int y_min = K_LEAF_CELL_PX;
            int y_max = 0;
            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    if (cell_alpha[y * K_LEAF_CELL_PX + x] > 0) {
                        if (y < y_min) y_min = y;
                        if (y > y_max) y_max = y;
                    }
                }
            }
            if (y_min > y_max) { y_min = 0; y_max = K_LEAF_CELL_PX - 1; }
            return { y_min, y_max };
        }

        // C3 P1 — composite a cell with intra-leaf color gradients.
        void splat_cell_rgba(uint8_t* atlas_rgba,
                             int cell_row, int cell_col,
                             const uint8_t* cell_alpha,
                             const uint8_t* vein_height,
                             const uint8_t* blurred_alpha,
                             LinearRgb diffuse_linear,
                             const GradientParams& gp) {
            const int x0 = cell_col * K_LEAF_CELL_PX;
            const int y0 = cell_row * K_LEAF_CELL_PX;

            const AlphaExtent ext = compute_alpha_extent(cell_alpha);
            const float y_range = static_cast<float>(ext.y_max - ext.y_min);
            const float inv_y_range = (y_range > 0.0f) ? 1.0f / y_range : 0.0f;

            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    const int ci = y * K_LEAF_CELL_PX + x;
                    const uint8_t a = cell_alpha[ci];
                    const size_t idx = (static_cast<size_t>(y0 + y) * K_LEAF_ATLAS_PX
                                       + static_cast<size_t>(x0 + x)) * 4;
                    if (!a) continue;

                    float lr = diffuse_linear.r;
                    float lg = diffuse_linear.g;
                    float lb = diffuse_linear.b;

                    // 1. Tip-to-base gradient: 0=tip (top), 1=base (bottom).
                    const float v_norm = static_cast<float>(y - ext.y_min) * inv_y_range;
                    const float dark_factor = 1.0f - gp.tip_to_base_strength * v_norm;
                    lr *= dark_factor;
                    lg *= dark_factor;
                    lb *= dark_factor;

                    // 2. Vein-adjacent darkening.
                    const float vein_proximity = static_cast<float>(vein_height[ci]) / 255.0f;
                    const float vein_dark = 1.0f - 0.15f * vein_proximity;
                    lr *= vein_dark;
                    lg *= vein_dark;
                    lb *= vein_dark;

                    // 3. Edge tinting (skip for pine needles — too thin).
                    if (gp.apply_edge_tint) {
                        const float edge_dist = static_cast<float>(blurred_alpha[ci]) / 255.0f;
                        const float edge_factor = smoothstep(0.0f, 0.3f, edge_dist);
                        // Desaturate toward brown at edge.
                        constexpr float brown_r = 0.45f, brown_g = 0.35f, brown_b = 0.20f;
                        lr = lr * edge_factor + brown_r * (1.0f - edge_factor) * lr;
                        lg = lg * edge_factor + brown_g * (1.0f - edge_factor) * lg;
                        lb = lb * edge_factor + brown_b * (1.0f - edge_factor) * lb;
                    }

                    atlas_rgba[idx + 0] = srgb_byte(lr);
                    atlas_rgba[idx + 1] = srgb_byte(lg);
                    atlas_rgba[idx + 2] = srgb_byte(lb);
                    atlas_rgba[idx + 3] = 255;
                }
            }
        }

        // Stamp vein segments into a per-cell height buffer (256x256, uint8_t).
        // Each vein is a 2-px-wide line (two quads along the segment) with value
        // proportional to distance from root. Root end = 255, tip = 128.
        void stamp_veins_into_cell(const std::vector<VeinSegment>& veins,
                                   uint8_t* height,
                                   int cell_px,
                                   int pad_px,
                                   int usable_px) {
            const float fp = static_cast<float>(pad_px);
            const float fu = static_cast<float>(usable_px);
            auto to_px = [fp, fu](vec2 v) -> vec2 {
                const float u = 0.5f * (v.x + 1.0f);
                const float vf = 0.5f * (1.0f - v.y);  // V-flip
                return vec2{fp + u * fu, fp + vf * fu};
            };

            constexpr float k_half_width = 1.5f;  // 1.5 px half-width → ~3px line

            for (const auto& seg : veins) {
                const vec2 a = to_px(seg.from);
                const vec2 b = to_px(seg.to);
                const float dx = b.x - a.x;
                const float dy = b.y - a.y;
                const float len = std::sqrt(dx * dx + dy * dy);
                if (len < 0.01f) continue;

                // Perpendicular direction for line width.
                const float px = -dy / len * k_half_width;
                const float py =  dx / len * k_half_width;

                // Quad corners: from-left, from-right, to-right, to-left.
                const vec2 fl{a.x + px, a.y + py};
                const vec2 fr{a.x - px, a.y - py};
                const vec2 tl{b.x + px, b.y + py};
                const vec2 tr{b.x - px, b.y - py};

                // Root end = 255, tip = 128. Use max-blend (stamp both tris with
                // the root value; the overlap region gets the correct max).
                rasterize_tri_pixel_space(height, cell_px, cell_px, fl, fr, tr, 255);
                rasterize_tri_pixel_space(height, cell_px, cell_px, fl, tr, tl, 200);
            }
        }

        // Box blur radius r on a single-channel buffer. Alpha mask restricts
        // output to alpha>0 pixels only.
        void box_blur_masked(const uint8_t* src, uint8_t* dst,
                             int w, int h, int r,
                             const uint8_t* alpha_mask) {
            // Horizontal pass into temp.
            std::vector<uint8_t> tmp(static_cast<size_t>(w) * h, 0);
            for (int y = 0; y < h; ++y) {
                int sum = 0;
                int count = 0;
                // Init window for x=0.
                for (int kx = 0; kx <= r; ++kx) {
                    if (kx < w && alpha_mask[y * w + kx] > 0) {
                        sum += src[y * w + kx];
                        ++count;
                    }
                }
                for (int x = 0; x < w; ++x) {
                    if (alpha_mask[y * w + x] > 0 && count > 0)
                        tmp[y * w + x] = static_cast<uint8_t>(sum / count);
                    // Slide window.
                    const int add_x = x + r + 1;
                    const int rem_x = x - r;
                    if (add_x < w && alpha_mask[y * w + add_x] > 0) {
                        sum += src[y * w + add_x];
                        ++count;
                    }
                    if (rem_x >= 0 && alpha_mask[y * w + rem_x] > 0) {
                        sum -= src[y * w + rem_x];
                        --count;
                    }
                }
            }
            // Vertical pass from tmp into dst.
            for (int x = 0; x < w; ++x) {
                int sum = 0;
                int count = 0;
                for (int ky = 0; ky <= r; ++ky) {
                    if (ky < h && alpha_mask[ky * w + x] > 0) {
                        sum += tmp[ky * w + x];
                        ++count;
                    }
                }
                for (int y = 0; y < h; ++y) {
                    if (alpha_mask[y * w + x] > 0 && count > 0)
                        dst[y * w + x] = static_cast<uint8_t>(sum / count);
                    const int add_y = y + r + 1;
                    const int rem_y = y - r;
                    if (add_y < h && alpha_mask[add_y * w + x] > 0) {
                        sum += tmp[add_y * w + x];
                        ++count;
                    }
                    if (rem_y >= 0 && alpha_mask[rem_y * w + x] > 0) {
                        sum -= tmp[rem_y * w + x];
                        --count;
                    }
                }
            }
        }

        // ---- Needle-strip cell bake (C3-needle-strips P1) ----
        // 8 parallel tapered needles slotted along V (branch axis).
        // Each needle extends across U (strip width, perpendicular to branch).
        // Needle body alpha=255, gaps alpha=0.

        struct NeedleStripParams {
            int   needle_count       = 8;
            float needle_width_ratio = 0.06f;   // fraction of usable width per needle body
            float gap_ratio          = 0.07f;    // fraction of usable width per gap
            float tip_taper          = 0.3f;     // fraction of height tapered to point at each end
        };

        // Returns a 256x256 alpha mask for the needle-strip cell.
        std::vector<uint8_t> bake_needle_strip_alpha(const NeedleStripParams& np) {
            std::vector<uint8_t> alpha(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            const float usable = static_cast<float>(K_LEAF_CELL_USABLE_PX);
            const float pad    = static_cast<float>(K_LEAF_CELL_PAD_PX);
            const int N        = np.needle_count;
            const float pitch  = 1.0f / static_cast<float>(N); // fraction per needle+gap slot
            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    // Normalize to [0,1] within usable region.
                    const float u = (static_cast<float>(x) - pad) / usable;
                    const float v = (static_cast<float>(y) - pad) / usable;
                    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) continue;

                    // Which needle slot (slotted along V)?
                    const float slot_f = v / pitch;
                    const int slot = static_cast<int>(slot_f);
                    if (slot >= N) continue;
                    const float local_v = slot_f - static_cast<float>(slot); // [0,1) within slot

                    // Needle body occupies [0, needle_width_ratio/pitch] of the slot.
                    const float body_frac = np.needle_width_ratio / pitch;
                    const float center    = 0.5f;
                    const float half_body = body_frac * 0.5f;

                    // Tapered tips: width narrows linearly to 0 at u=0 and u=1.
                    float taper_scale = 1.0f;
                    if (u < np.tip_taper) {
                        taper_scale = u / np.tip_taper;
                    } else if (u > 1.0f - np.tip_taper) {
                        taper_scale = (1.0f - u) / np.tip_taper;
                    }
                    const float tapered_half = half_body * taper_scale;

                    if (local_v >= center - tapered_half && local_v <= center + tapered_half) {
                        alpha[y * K_LEAF_CELL_PX + x] = 255;
                    }
                }
            }
            return alpha;
        }

        // Splat needle-strip diffuse with per-needle hue jitter.
        void bake_needle_strip_diffuse(uint8_t* atlas_rgba,
                                       const uint8_t* strip_alpha,
                                       const NeedleStripParams& np,
                                       uint64_t pcg_seed) {
            constexpr LinearRgb pine_base = { 0.10f, 0.30f, 0.12f };
            const int x0 = K_NEEDLE_STRIP_CELL_COL * K_LEAF_CELL_PX;
            const int y0 = K_NEEDLE_STRIP_CELL_ROW * K_LEAF_CELL_PX;
            const float usable = static_cast<float>(K_LEAF_CELL_USABLE_PX);
            const float pad    = static_cast<float>(K_LEAF_CELL_PAD_PX);
            const float pitch  = 1.0f / static_cast<float>(np.needle_count);

            // Per-needle hue jitter: +/-5% on each channel, seeded from pcg.
            pcg32 rng;
            rng.seed(pcg_seed, 0xC3'01'01ULL);
            // Pre-generate per-needle jitter multipliers.
            std::vector<float> jitter_r(static_cast<size_t>(np.needle_count));
            std::vector<float> jitter_g(static_cast<size_t>(np.needle_count));
            std::vector<float> jitter_b(static_cast<size_t>(np.needle_count));
            for (int n = 0; n < np.needle_count; ++n) {
                jitter_r[n] = 1.0f + (rng.next_float_01() - 0.5f) * 0.10f; // +/-5%
                jitter_g[n] = 1.0f + (rng.next_float_01() - 0.5f) * 0.10f;
                jitter_b[n] = 1.0f + (rng.next_float_01() - 0.5f) * 0.10f;
            }

            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    const int ci = y * K_LEAF_CELL_PX + x;
                    if (!strip_alpha[ci]) continue;

                    const float v = (static_cast<float>(y) - pad) / usable;
                    const int slot = std::min(static_cast<int>(v / pitch), np.needle_count - 1);

                    const float lr = pine_base.r * jitter_r[slot];
                    const float lg = pine_base.g * jitter_g[slot];
                    const float lb = pine_base.b * jitter_b[slot];

                    const size_t idx = (static_cast<size_t>(y0 + y) * K_LEAF_ATLAS_PX
                                       + static_cast<size_t>(x0 + x)) * 4;
                    atlas_rgba[idx + 0] = srgb_byte(lr);
                    atlas_rgba[idx + 1] = srgb_byte(lg);
                    atlas_rgba[idx + 2] = srgb_byte(lb);
                    atlas_rgba[idx + 3] = 255;
                }
            }
        }

        // Splat needle-strip normals: cylindrical per-needle cross-section.
        void bake_needle_strip_normals(uint8_t* atlas_rgba,
                                       const uint8_t* strip_alpha,
                                       const NeedleStripParams& np) {
            const int x0 = K_NEEDLE_STRIP_CELL_COL * K_LEAF_CELL_PX;
            const int y0 = K_NEEDLE_STRIP_CELL_ROW * K_LEAF_CELL_PX;
            const float usable = static_cast<float>(K_LEAF_CELL_USABLE_PX);
            const float pad    = static_cast<float>(K_LEAF_CELL_PAD_PX);
            const float pitch  = 1.0f / static_cast<float>(np.needle_count);
            const float body_frac = np.needle_width_ratio / pitch;

            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    const int ci = y * K_LEAF_CELL_PX + x;
                    if (!strip_alpha[ci]) continue;

                    const float v = (static_cast<float>(y) - pad) / usable;
                    const float slot_f = v / pitch;
                    const int slot = static_cast<int>(slot_f);
                    const float local_v = slot_f - static_cast<float>(std::min(slot, np.needle_count - 1));

                    // Map local_v within needle body to [-1,1] across cross-section.
                    const float center = 0.5f;
                    const float half_body = body_frac * 0.5f;
                    float t = (local_v - center) / half_body; // [-1, 1] if half_body > 0
                    if (t < -1.0f) t = -1.0f;
                    if (t > 1.0f)  t = 1.0f;

                    // Cylindrical normal: N.x = t, N.y = 0, N.z = sqrt(1-t^2).
                    const float nx = t;
                    const float nz = std::sqrt(std::max(0.0f, 1.0f - t * t));

                    const size_t idx = (static_cast<size_t>(y0 + y) * K_LEAF_ATLAS_PX
                                       + static_cast<size_t>(x0 + x)) * 4;
                    atlas_rgba[idx + 0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f,
                                          (nx * 0.5f + 0.5f) * 255.0f + 0.5f)));
                    atlas_rgba[idx + 1] = 128; // N.y = 0
                    atlas_rgba[idx + 2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f,
                                          (nz * 0.5f + 0.5f) * 255.0f + 0.5f)));
                    atlas_rgba[idx + 3] = 255;
                }
            }
        }

        // Splat needle-strip translucency: flat 0.6 baseline.
        void bake_needle_strip_translucency(uint8_t* atlas_rgba,
                                            const uint8_t* strip_alpha) {
            const int x0 = K_NEEDLE_STRIP_CELL_COL * K_LEAF_CELL_PX;
            const int y0 = K_NEEDLE_STRIP_CELL_ROW * K_LEAF_CELL_PX;
            constexpr uint8_t trans_byte = static_cast<uint8_t>(0.6f * 255.0f + 0.5f); // 153

            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    if (!strip_alpha[y * K_LEAF_CELL_PX + x]) continue;
                    const size_t idx = (static_cast<size_t>(y0 + y) * K_LEAF_ATLAS_PX
                                       + static_cast<size_t>(x0 + x)) * 4;
                    atlas_rgba[idx + 0] = trans_byte;
                }
            }
        }

        // Splat needle-strip roughness: flat 0.50 (smooth needle surface).
        void bake_needle_strip_roughness(uint8_t* atlas_rgba,
                                         const uint8_t* strip_alpha) {
            const int x0 = K_NEEDLE_STRIP_CELL_COL * K_LEAF_CELL_PX;
            const int y0 = K_NEEDLE_STRIP_CELL_ROW * K_LEAF_CELL_PX;
            constexpr uint8_t rough_byte = static_cast<uint8_t>(0.50f * 255.0f + 0.5f); // 128

            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    if (!strip_alpha[y * K_LEAF_CELL_PX + x]) continue;
                    const size_t idx = (static_cast<size_t>(y0 + y) * K_LEAF_ATLAS_PX
                                       + static_cast<size_t>(x0 + x)) * 4;
                    atlas_rgba[idx + 0] = 255;       // R unused
                    atlas_rgba[idx + 1] = rough_byte; // G roughness
                    // B=0 (metallic), A=255 already set
                }
            }
        }

    }  // namespace

    std::vector<uint8_t> encode_leaf_atlas_png(uint64_t /*seed*/) {
        std::vector<uint8_t> rgba(
            static_cast<size_t>(K_LEAF_ATLAS_PX) * K_LEAF_ATLAS_PX * 4, 0);

        for (LeafShape sp : k_species_in_atlas) {
            const int si = static_cast<int>(sp);
            const LeafShapeMesh mesh = shape_mesh_for(sp);
            const auto veins = veins_for_species(sp, mesh);

            // Bake per-cell alpha + vein height + blurred alpha.
            auto alpha = bake_cell_alpha(sp);

            std::vector<uint8_t> vein_height(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            stamp_veins_into_cell(veins, vein_height.data(),
                                  K_LEAF_CELL_PX, K_LEAF_CELL_PAD_PX, K_LEAF_CELL_USABLE_PX);

            std::vector<uint8_t> blurred(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            box_blur_masked(alpha.data(), blurred.data(),
                            K_LEAF_CELL_PX, K_LEAF_CELL_PX, 3, alpha.data());

            splat_cell_rgba(rgba.data(),
                            leaf_atlas_cell_row(sp),
                            leaf_atlas_cell_col(sp),
                            alpha.data(),
                            vein_height.data(),
                            blurred.data(),
                            k_leaf_color[si],
                            k_gradient[si]);
        }

        // Needle-strip cell.
        {
            NeedleStripParams nsp;
            auto strip_alpha = bake_needle_strip_alpha(nsp);
            bake_needle_strip_diffuse(rgba.data(), strip_alpha.data(), nsp, 0xC3'01'00ULL);
        }

        return encode_png_rgba8(K_LEAF_ATLAS_PX, K_LEAF_ATLAS_PX,
                                K_LEAF_ATLAS_PX * 4, rgba.data());
    }

    std::vector<uint8_t> encode_leaf_normal_png(uint64_t /*seed*/) {
        // RGBA8: RGB = tangent-space normal * 0.5 + 0.5, A = 255.
        // Default (padding / outside alpha): flat normal (128, 128, 255, 255).
        std::vector<uint8_t> rgba(
            static_cast<size_t>(K_LEAF_ATLAS_PX) * K_LEAF_ATLAS_PX * 4, 0);
        // Fill with flat normal default.
        for (size_t i = 0; i < static_cast<size_t>(K_LEAF_ATLAS_PX) * K_LEAF_ATLAS_PX; ++i) {
            rgba[i * 4 + 0] = 128;
            rgba[i * 4 + 1] = 128;
            rgba[i * 4 + 2] = 255;
            rgba[i * 4 + 3] = 255;
        }

        // Vein normal perturbation. 8.0 tilted vein-edge normals up to ~83° off
        // the leaf surface — physically absurd for a leaf, and the steep facets
        // aliased the (diffuse) sun/omni N·L into a high-frequency speckle that
        // travelled across the leaf under camera/wind motion, reading as a
        // "metallic glimmer". 2.0 keeps the veins legible with gentle relief.
        constexpr float k_normal_strength = 2.0f;

        for (LeafShape sp : k_species_in_atlas) {
            const LeafShapeMesh mesh = shape_mesh_for(sp);
            const auto veins = veins_for_species(sp, mesh);

            // Bake per-cell alpha + vein height.
            std::vector<uint8_t> alpha(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            rasterize_leaf_alpha_into_cell(mesh, alpha.data(),
                                           K_LEAF_CELL_PX, K_LEAF_CELL_PX,
                                           K_LEAF_CELL_PAD_PX, K_LEAF_CELL_USABLE_PX);

            std::vector<uint8_t> vein_height(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            stamp_veins_into_cell(veins, vein_height.data(),
                                  K_LEAF_CELL_PX, K_LEAF_CELL_PAD_PX, K_LEAF_CELL_USABLE_PX);

            // Finite-difference normal from vein heightfield. +Y-up tangent space.
            const int row = leaf_atlas_cell_row(sp);
            const int col = leaf_atlas_cell_col(sp);
            const int x0 = col * K_LEAF_CELL_PX;
            const int y0 = row * K_LEAF_CELL_PX;

            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    if (alpha[y * K_LEAF_CELL_PX + x] == 0) continue;

                    // Central difference on vein heightfield.
                    const int xl = (x > 0) ? x - 1 : x;
                    const int xr = (x < K_LEAF_CELL_PX - 1) ? x + 1 : x;
                    const int yu = (y > 0) ? y - 1 : y;
                    const int yd = (y < K_LEAF_CELL_PX - 1) ? y + 1 : y;

                    const float hL = static_cast<float>(vein_height[y  * K_LEAF_CELL_PX + xl]);
                    const float hR = static_cast<float>(vein_height[y  * K_LEAF_CELL_PX + xr]);
                    const float hU = static_cast<float>(vein_height[yu * K_LEAF_CELL_PX + x]);
                    const float hD = static_cast<float>(vein_height[yd * K_LEAF_CELL_PX + x]);

                    const float dhdx = (hR - hL) * k_normal_strength / 255.0f;
                    const float dhdy = (hD - hU) * k_normal_strength / 255.0f;

                    // Tangent-space: N = normalize(-dhdx, -dhdy, 1).
                    const float nx = -dhdx;
                    const float ny = -dhdy;
                    const float nz = 1.0f;
                    const float inv_len = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);

                    const size_t idx = (static_cast<size_t>(y0 + y) * K_LEAF_ATLAS_PX
                                       + static_cast<size_t>(x0 + x)) * 4;
                    rgba[idx + 0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f,
                                    (nx * inv_len * 0.5f + 0.5f) * 255.0f + 0.5f)));
                    rgba[idx + 1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f,
                                    (ny * inv_len * 0.5f + 0.5f) * 255.0f + 0.5f)));
                    rgba[idx + 2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f,
                                    (nz * inv_len * 0.5f + 0.5f) * 255.0f + 0.5f)));
                    rgba[idx + 3] = 255;
                }
            }
        }

        // Needle-strip normals.
        {
            NeedleStripParams nsp;
            auto strip_alpha = bake_needle_strip_alpha(nsp);
            bake_needle_strip_normals(rgba.data(), strip_alpha.data(), nsp);
        }

        return encode_png_rgba8(K_LEAF_ATLAS_PX, K_LEAF_ATLAS_PX,
                                K_LEAF_ATLAS_PX * 4, rgba.data());
    }

    std::vector<uint8_t> encode_leaf_translucency_png(uint64_t /*seed*/) {
        // C3 P2: physically-motivated translucency. Veins block up to 80% of
        // transmitted light; edge pixels are thinner and transmit more.
        // Formula: trans = (1 - 0.8 * vein_factor) * edge_boost, clamped [0,1].
        std::vector<uint8_t> rgba(
            static_cast<size_t>(K_LEAF_ATLAS_PX) * K_LEAF_ATLAS_PX * 4, 0);
        // Fill A=255.
        for (size_t i = 0; i < static_cast<size_t>(K_LEAF_ATLAS_PX) * K_LEAF_ATLAS_PX; ++i)
            rgba[i * 4 + 3] = 255;

        for (LeafShape sp : k_species_in_atlas) {
            const LeafShapeMesh mesh = shape_mesh_for(sp);
            const auto veins = veins_for_species(sp, mesh);

            std::vector<uint8_t> alpha(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            rasterize_leaf_alpha_into_cell(mesh, alpha.data(),
                                           K_LEAF_CELL_PX, K_LEAF_CELL_PX,
                                           K_LEAF_CELL_PAD_PX, K_LEAF_CELL_USABLE_PX);

            std::vector<uint8_t> vein_height(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            stamp_veins_into_cell(veins, vein_height.data(),
                                  K_LEAF_CELL_PX, K_LEAF_CELL_PAD_PX, K_LEAF_CELL_USABLE_PX);

            // Box-blur alpha (r=3) restricted to alpha>0 pixels — proxy for edge distance.
            std::vector<uint8_t> blurred(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            box_blur_masked(alpha.data(), blurred.data(),
                            K_LEAF_CELL_PX, K_LEAF_CELL_PX, 3, alpha.data());

            const int row = leaf_atlas_cell_row(sp);
            const int col = leaf_atlas_cell_col(sp);
            const int x0 = col * K_LEAF_CELL_PX;
            const int y0 = row * K_LEAF_CELL_PX;

            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    const int ci = y * K_LEAF_CELL_PX + x;
                    if (alpha[ci] == 0) continue;

                    const float vein_factor = static_cast<float>(vein_height[ci]) / 255.0f;
                    float trans = 1.0f - vein_factor * 0.8f;  // veins block up to 80%
                    // Edge boost: thin edges transmit more light.
                    const float edge_dist = static_cast<float>(blurred[ci]) / 255.0f;
                    const float edge_boost = 1.0f + 0.3f * (1.0f - edge_dist);
                    trans *= edge_boost;
                    if (trans > 1.0f) trans = 1.0f;

                    const size_t idx = (static_cast<size_t>(y0 + y) * K_LEAF_ATLAS_PX
                                       + static_cast<size_t>(x0 + x)) * 4;
                    rgba[idx + 0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f,
                                    trans * 255.0f + 0.5f)));
                }
            }
        }

        // Needle-strip translucency.
        {
            NeedleStripParams nsp;
            auto strip_alpha = bake_needle_strip_alpha(nsp);
            bake_needle_strip_translucency(rgba.data(), strip_alpha.data());
        }

        return encode_png_rgba8(K_LEAF_ATLAS_PX, K_LEAF_ATLAS_PX,
                                K_LEAF_ATLAS_PX * 4, rgba.data());
    }

    std::vector<uint8_t> encode_leaf_roughness_png(uint64_t /*seed*/) {
        // C3 P2: per-leaf roughness atlas, glTF metallic-roughness layout.
        // R=255 (unused), G=roughness (LINEAR byte), B=0 (metallic=0), A=255.
        // Leaf center smooth (0.35), vein ridges rough (0.80).
        // Formula: roughness = 0.35 + 0.45 * vein_factor.
        std::vector<uint8_t> rgba(
            static_cast<size_t>(K_LEAF_ATLAS_PX) * K_LEAF_ATLAS_PX * 4, 0);
        // Fill default: R=255, G=128 (0.5 roughness), B=0 (metal=0), A=255.
        for (size_t i = 0; i < static_cast<size_t>(K_LEAF_ATLAS_PX) * K_LEAF_ATLAS_PX; ++i) {
            rgba[i * 4 + 0] = 255;
            rgba[i * 4 + 1] = 128;
            rgba[i * 4 + 3] = 255;
        }

        for (LeafShape sp : k_species_in_atlas) {
            const LeafShapeMesh mesh = shape_mesh_for(sp);
            const auto veins = veins_for_species(sp, mesh);

            std::vector<uint8_t> alpha(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            rasterize_leaf_alpha_into_cell(mesh, alpha.data(),
                                           K_LEAF_CELL_PX, K_LEAF_CELL_PX,
                                           K_LEAF_CELL_PAD_PX, K_LEAF_CELL_USABLE_PX);

            std::vector<uint8_t> vein_height(static_cast<size_t>(K_LEAF_CELL_PX) * K_LEAF_CELL_PX, 0);
            stamp_veins_into_cell(veins, vein_height.data(),
                                  K_LEAF_CELL_PX, K_LEAF_CELL_PAD_PX, K_LEAF_CELL_USABLE_PX);

            const int row = leaf_atlas_cell_row(sp);
            const int col = leaf_atlas_cell_col(sp);
            const int x0 = col * K_LEAF_CELL_PX;
            const int y0 = row * K_LEAF_CELL_PX;

            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    const int ci = y * K_LEAF_CELL_PX + x;
                    if (alpha[ci] == 0) continue;

                    const float vein_factor = static_cast<float>(vein_height[ci]) / 255.0f;
                    const float roughness = 0.35f + 0.45f * vein_factor;
                    const uint8_t rough_byte = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f,
                                               roughness * 255.0f + 0.5f)));

                    const size_t idx = (static_cast<size_t>(y0 + y) * K_LEAF_ATLAS_PX
                                       + static_cast<size_t>(x0 + x)) * 4;
                    rgba[idx + 0] = 255;         // unused
                    rgba[idx + 1] = rough_byte;  // roughness
                    // B=0 (metallic), A=255 already set
                }
            }
        }

        // Needle-strip roughness.
        {
            NeedleStripParams nsp;
            auto strip_alpha = bake_needle_strip_alpha(nsp);
            bake_needle_strip_roughness(rgba.data(), strip_alpha.data());
        }

        return encode_png_rgba8(K_LEAF_ATLAS_PX, K_LEAF_ATLAS_PX,
                                K_LEAF_ATLAS_PX * 4, rgba.data());
    }

}
