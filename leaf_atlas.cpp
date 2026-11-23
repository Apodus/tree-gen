// Botanical Leaf Sprigs & 3D Cluster Uplift — 16-cell 1024x1024 leaf atlas bake.
//   Row 0 (cols 0..3): upgraded single-leaf cells (Oak, Pine, Birch, Maple).
//   Rows 1..3 x Cols 0, 2, 3: 3 distinct botanical sprig variants per broadleaf
//     species (Var A forked terminal fan, Var B curved lateral shoot, Var C open
//     dappled twiglet) with per-leaf 3D tangent normals, per-leaf albedo/roughness,
//     and backlit-safe translucency (>= 0.05 floor) + 6-px edge dilation.
//   Rows 1..3 x Col 1: V-tileable angled pine fascicle shoot (1,1) + two pine
//     whorl tuft variants (2,1) & (3,1).
//
// Strictly seed-independent: all 4 encoders ignore `seed` so all staged GLBs
// emit byte-identical PNGs and deduplicate to a single GPU atlas set.
// /fp:precise (treegen.sharpmake.cs) — every sample contributes to atlas hash.
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
#include <tuple>
#include <vector>

namespace treegen {

    namespace {

        constexpr float k_pi     = 3.14159265358979323846f;
        constexpr float k_two_pi = 6.28318530717958647692f;
        constexpr float k_deg    = k_pi / 180.0f;

        // Linear RGB greens per species (index pinned to LeafShape enum order).
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

        // Row-0 single-leaf gradient parameters.
        struct GradientParams {
            float tip_to_base_strength;
            bool  apply_edge_tint;
        };
        constexpr GradientParams k_gradient[4] = {
            /* OakLobed      */ { 0.25f, true  },
            /* PineNeedle    */ { 0.15f, false },
            /* BirchSerrated */ { 0.20f, true  },
            /* MapleStar     */ { 0.30f, true  },
        };

        constexpr std::array<LeafShape, 4> k_species_in_atlas = {
            LeafShape::OakLobed,
            LeafShape::PineNeedle,
            LeafShape::BirchSerrated,
            LeafShape::MapleStar,
        };

        inline float clampf(float x, float lo, float hi) {
            return x < lo ? lo : (x > hi ? hi : x);
        }

        inline uint8_t u8_from_01(float v) {
            float s = clampf(v, 0.0f, 1.0f) * 255.0f + 0.5f;
            if (s < 0.0f) s = 0.0f;
            if (s > 255.0f) s = 255.0f;
            return static_cast<uint8_t>(s);
        }

        inline uint8_t srgb_byte(float linear) {
            linear = clampf(linear, 0.0f, 1.0f);
            const float s = (linear <= 0.0031308f)
                ? 12.92f * linear
                : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
            return u8_from_01(s);
        }

        inline float smoothstep(float edge0, float edge1, float x) {
            const float denom = edge1 - edge0;
            if (std::abs(denom) < 1e-8f) return x >= edge1 ? 1.0f : 0.0f;
            const float t = clampf((x - edge0) / denom, 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        }

        inline float edge_fn(vec2 a, vec2 b, vec2 p) {
            return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        }

        // Distance from point p to line segment (a, b).
        inline float seg_dist(vec2 p, vec2 a, vec2 b, float* out_t = nullptr) {
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            const float l2 = dx * dx + dy * dy;
            float t = 0.0f;
            if (l2 > 1e-8f) {
                t = clampf(((p.x - a.x) * dx + (p.y - a.y) * dy) / l2, 0.0f, 1.0f);
            }
            if (out_t) *out_t = t;
            const float qx = a.x + t * dx;
            const float qy = a.y + t * dy;
            return std::sqrt((p.x - qx) * (p.x - qx) + (p.y - qy) * (p.y - qy));
        }

        // Unified 256x256 cell buffer holding all 4 PBR maps simultaneously.
        struct CellChannels {
            static constexpr int W = K_LEAF_CELL_PX;
            static constexpr int H = K_LEAF_CELL_PX;
            static constexpr size_t N = static_cast<size_t>(W) * H;

            std::vector<uint8_t> alpha;        // 0 or 255
            std::vector<uint8_t> albedo_r;     // sRGB
            std::vector<uint8_t> albedo_g;     // sRGB
            std::vector<uint8_t> albedo_b;     // sRGB
            std::vector<uint8_t> norm_x;       // tangent normal * 0.5 + 0.5
            std::vector<uint8_t> norm_y;
            std::vector<uint8_t> norm_z;
            std::vector<uint8_t> rough;        // linear roughness (glTF G)
            std::vector<uint8_t> trans;        // linear translucency (R)

            CellChannels()
                : alpha(N, 0)
                , albedo_r(N, 0)
                , albedo_g(N, 0)
                , albedo_b(N, 0)
                , norm_x(N, 128)
                , norm_y(N, 128)
                , norm_z(N, 255)
                , rough(N, 179)   // ~0.70 matte default
                , trans(N, 0) {}
        };

        // 6-px 8-neighbor edge dilation of albedo RGB, normal, roughness, and
        // translucency into alpha==0 pixels strictly inside [pad, W-pad).
        // Leaves alpha unchanged and never touches the 4-px cell padding band.
        void dilate_cell_channels(CellChannels& c, int radius_px, int pad_px) {
            constexpr int W = CellChannels::W;
            constexpr int H = CellChannels::H;
            std::vector<uint8_t> filled(CellChannels::N, 0);
            for (size_t i = 0; i < CellChannels::N; ++i) {
                filled[i] = (c.alpha[i] > 0) ? 1u : 0u;
            }

            std::vector<uint8_t> next_filled = filled;
            std::vector<uint8_t> next_ar = c.albedo_r, next_ag = c.albedo_g, next_ab = c.albedo_b;
            std::vector<uint8_t> next_nx = c.norm_x,   next_ny = c.norm_y,   next_nz = c.norm_z;
            std::vector<uint8_t> next_ro = c.rough,    next_tr = c.trans;

            const int lo = pad_px;
            const int hi = W - pad_px;

            for (int iter = 0; iter < radius_px; ++iter) {
                bool changed = false;
                for (int y = lo; y < hi; ++y) {
                    for (int x = lo; x < hi; ++x) {
                        const size_t ci = static_cast<size_t>(y) * W + x;
                        if (filled[ci]) continue;

                        int cnt = 0;
                        int sr = 0, sg = 0, sb = 0;
                        int snx = 0, sny = 0, snz = 0;
                        int sro = 0, str = 0;
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int ny = y + dy;
                            if (ny < lo || ny >= hi) continue;
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dx == 0 && dy == 0) continue;
                                const int nx = x + dx;
                                if (nx < lo || nx >= hi) continue;
                                const size_t ni = static_cast<size_t>(ny) * W + nx;
                                if (!filled[ni]) continue;
                                sr  += c.albedo_r[ni];
                                sg  += c.albedo_g[ni];
                                sb  += c.albedo_b[ni];
                                snx += c.norm_x[ni];
                                sny += c.norm_y[ni];
                                snz += c.norm_z[ni];
                                sro += c.rough[ni];
                                str += c.trans[ni];
                                ++cnt;
                            }
                        }
                        if (cnt > 0) {
                            next_filled[ci] = 1u;
                            next_ar[ci] = static_cast<uint8_t>((sr  + cnt / 2) / cnt);
                            next_ag[ci] = static_cast<uint8_t>((sg  + cnt / 2) / cnt);
                            next_ab[ci] = static_cast<uint8_t>((sb  + cnt / 2) / cnt);
                            next_nx[ci] = static_cast<uint8_t>((snx + cnt / 2) / cnt);
                            next_ny[ci] = static_cast<uint8_t>((sny + cnt / 2) / cnt);
                            next_nz[ci] = static_cast<uint8_t>((snz + cnt / 2) / cnt);
                            next_ro[ci] = static_cast<uint8_t>((sro + cnt / 2) / cnt);
                            next_tr[ci] = static_cast<uint8_t>((str + cnt / 2) / cnt);
                            changed = true;
                        }
                    }
                }
                filled.swap(next_filled);
                c.albedo_r.swap(next_ar); c.albedo_g.swap(next_ag); c.albedo_b.swap(next_ab);
                c.norm_x.swap(next_nx);   c.norm_y.swap(next_ny);   c.norm_z.swap(next_nz);
                c.rough.swap(next_ro);    c.trans.swap(next_tr);
                next_filled = filled;
                next_ar = c.albedo_r; next_ag = c.albedo_g; next_ab = c.albedo_b;
                next_nx = c.norm_x;   next_ny = c.norm_y;   next_nz = c.norm_z;
                next_ro = c.rough;    next_tr = c.trans;
                if (!changed) break;
            }
        }

        // Write a CellChannels tile into one of the four 1024x1024 RGBA8 atlases.
        enum class AtlasChannelKind { Albedo, Normal, Roughness, Translucency };

        void blit_cell_to_atlas(uint8_t* atlas_rgba, int cell_row, int cell_col,
                                const CellChannels& c, AtlasChannelKind kind) {
            const int x0 = cell_col * K_LEAF_CELL_PX;
            const int y0 = cell_row * K_LEAF_CELL_PX;
            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    const size_t ci  = static_cast<size_t>(y) * K_LEAF_CELL_PX + x;
                    const size_t idx = (static_cast<size_t>(y0 + y) * K_LEAF_ATLAS_PX
                                       + static_cast<size_t>(x0 + x)) * 4;
                    switch (kind) {
                        case AtlasChannelKind::Albedo:
                            atlas_rgba[idx + 0] = c.albedo_r[ci];
                            atlas_rgba[idx + 1] = c.albedo_g[ci];
                            atlas_rgba[idx + 2] = c.albedo_b[ci];
                            atlas_rgba[idx + 3] = c.alpha[ci];
                            break;
                        case AtlasChannelKind::Normal:
                            atlas_rgba[idx + 0] = c.norm_x[ci];
                            atlas_rgba[idx + 1] = c.norm_y[ci];
                            atlas_rgba[idx + 2] = c.norm_z[ci];
                            atlas_rgba[idx + 3] = 255;
                            break;
                        case AtlasChannelKind::Roughness:
                            atlas_rgba[idx + 0] = 255;
                            atlas_rgba[idx + 1] = c.rough[ci];
                            atlas_rgba[idx + 2] = 0;
                            atlas_rgba[idx + 3] = 255;
                            break;
                        case AtlasChannelKind::Translucency:
                            atlas_rgba[idx + 0] = c.trans[ci];
                            atlas_rgba[idx + 1] = 0;
                            atlas_rgba[idx + 2] = 0;
                            atlas_rgba[idx + 3] = 255;
                            break;
                    }
                }
            }
        }

        // ---- Row-0 single-leaf helpers (back-compat for SingleCard/BentCard) ----

        void stamp_veins_into_cell(const std::vector<VeinSegment>& veins,
                                   uint8_t* height,
                                   int cell_px,
                                   int pad_px,
                                   int usable_px) {
            const float fp = static_cast<float>(pad_px);
            const float fu = static_cast<float>(usable_px);
            auto to_px = [fp, fu](vec2 v) -> vec2 {
                const float u  = 0.5f * (v.x + 1.0f);
                const float vf = 0.5f * (1.0f - v.y);  // V-flip
                return vec2{fp + u * fu, fp + vf * fu};
            };

            constexpr float k_half_width = 1.5f;
            for (const auto& seg : veins) {
                const vec2 a = to_px(seg.from);
                const vec2 b = to_px(seg.to);
                const float dx = b.x - a.x;
                const float dy = b.y - a.y;
                const float len = std::sqrt(dx * dx + dy * dy);
                if (len < 0.01f) continue;

                const float px = -dy / len * k_half_width;
                const float py =  dx / len * k_half_width;
                const vec2 fl{a.x + px, a.y + py};
                const vec2 fr{a.x - px, a.y - py};
                const vec2 tl{b.x + px, b.y + py};
                const vec2 tr{b.x - px, b.y - py};

                rasterize_tri_pixel_space(height, cell_px, cell_px, fl, fr, tr, 255);
                rasterize_tri_pixel_space(height, cell_px, cell_px, fl, tr, tl, 200);
            }
        }

        void box_blur_masked(const uint8_t* src, uint8_t* dst,
                             int w, int h, int r,
                             const uint8_t* alpha_mask) {
            std::vector<uint8_t> tmp(static_cast<size_t>(w) * h, 0);
            for (int y = 0; y < h; ++y) {
                int sum = 0, count = 0;
                for (int kx = 0; kx <= r; ++kx) {
                    if (kx < w && alpha_mask[y * w + kx] > 0) { sum += src[y * w + kx]; ++count; }
                }
                for (int x = 0; x < w; ++x) {
                    if (alpha_mask[y * w + x] > 0 && count > 0)
                        tmp[y * w + x] = static_cast<uint8_t>(sum / count);
                    const int add_x = x + r + 1;
                    const int rem_x = x - r;
                    if (add_x < w && alpha_mask[y * w + add_x] > 0) { sum += src[y * w + add_x]; ++count; }
                    if (rem_x >= 0 && alpha_mask[y * w + rem_x] > 0) { sum -= src[y * w + rem_x]; --count; }
                }
            }
            for (int x = 0; x < w; ++x) {
                int sum = 0, count = 0;
                for (int ky = 0; ky <= r; ++ky) {
                    if (ky < h && alpha_mask[ky * w + x] > 0) { sum += tmp[ky * w + x]; ++count; }
                }
                for (int y = 0; y < h; ++y) {
                    if (alpha_mask[y * w + x] > 0 && count > 0)
                        dst[y * w + x] = static_cast<uint8_t>(sum / count);
                    const int add_y = y + r + 1;
                    const int rem_y = y - r;
                    if (add_y < h && alpha_mask[add_y * w + x] > 0) { sum += tmp[add_y * w + x]; ++count; }
                    if (rem_y >= 0 && alpha_mask[rem_y * w + x] > 0) { sum -= tmp[rem_y * w + x]; --count; }
                }
            }
        }

        CellChannels bake_single_leaf_cell(LeafShape sp) {
            CellChannels c;
            const int si = static_cast<int>(sp);
            const LeafShapeMesh mesh = shape_mesh_for(sp);
            const auto veins = veins_for_species(sp, mesh);

            rasterize_leaf_alpha_into_cell(mesh, c.alpha.data(),
                                           K_LEAF_CELL_PX, K_LEAF_CELL_PX,
                                           K_LEAF_CELL_PAD_PX, K_LEAF_CELL_USABLE_PX);

            std::vector<uint8_t> vein_h(CellChannels::N, 0);
            stamp_veins_into_cell(veins, vein_h.data(),
                                  K_LEAF_CELL_PX, K_LEAF_CELL_PAD_PX, K_LEAF_CELL_USABLE_PX);

            std::vector<uint8_t> blurred(CellChannels::N, 0);
            box_blur_masked(c.alpha.data(), blurred.data(),
                            K_LEAF_CELL_PX, K_LEAF_CELL_PX, 3, c.alpha.data());

            int y_min = K_LEAF_CELL_PX, y_max = 0;
            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    if (c.alpha[static_cast<size_t>(y) * K_LEAF_CELL_PX + x] > 0) {
                        y_min = std::min(y_min, y);
                        y_max = std::max(y_max, y);
                    }
                }
            }
            if (y_min > y_max) { y_min = 0; y_max = K_LEAF_CELL_PX - 1; }
            const float inv_y_range = (y_max > y_min) ? (1.0f / static_cast<float>(y_max - y_min)) : 0.0f;

            constexpr float k_normal_strength = 2.0f;
            const LinearRgb base_col = k_leaf_color[si];
            const GradientParams gp  = k_gradient[si];

            for (int y = 0; y < K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < K_LEAF_CELL_PX; ++x) {
                    const size_t ci = static_cast<size_t>(y) * K_LEAF_CELL_PX + x;
                    if (!c.alpha[ci]) continue;

                    // Albedo
                    float lr = base_col.r;
                    float lg = base_col.g;
                    float lb = base_col.b;

                    const float v_norm = static_cast<float>(y - y_min) * inv_y_range;
                    const float dark_factor = 1.0f - gp.tip_to_base_strength * v_norm;
                    lr *= dark_factor; lg *= dark_factor; lb *= dark_factor;

                    const float vein_f = static_cast<float>(vein_h[ci]) / 255.0f;
                    const float vein_dark = 1.0f - 0.15f * vein_f;
                    lr *= vein_dark; lg *= vein_dark; lb *= vein_dark;

                    if (gp.apply_edge_tint) {
                        const float edge_dist = static_cast<float>(blurred[ci]) / 255.0f;
                        const float edge_factor = smoothstep(0.0f, 0.3f, edge_dist);
                        constexpr float brown_r = 0.45f, brown_g = 0.35f, brown_b = 0.20f;
                        lr = lr * edge_factor + brown_r * (1.0f - edge_factor) * lr;
                        lg = lg * edge_factor + brown_g * (1.0f - edge_factor) * lg;
                        lb = lb * edge_factor + brown_b * (1.0f - edge_factor) * lb;
                    }
                    c.albedo_r[ci] = srgb_byte(lr);
                    c.albedo_g[ci] = srgb_byte(lg);
                    c.albedo_b[ci] = srgb_byte(lb);

                    // Normal (central difference on vein heightfield)
                    const int xl = (x > 0) ? x - 1 : x;
                    const int xr = (x < K_LEAF_CELL_PX - 1) ? x + 1 : x;
                    const int yu = (y > 0) ? y - 1 : y;
                    const int yd = (y < K_LEAF_CELL_PX - 1) ? y + 1 : y;
                    const float hL = static_cast<float>(vein_h[static_cast<size_t>(y)  * K_LEAF_CELL_PX + xl]);
                    const float hR = static_cast<float>(vein_h[static_cast<size_t>(y)  * K_LEAF_CELL_PX + xr]);
                    const float hU = static_cast<float>(vein_h[static_cast<size_t>(yu) * K_LEAF_CELL_PX + x]);
                    const float hD = static_cast<float>(vein_h[static_cast<size_t>(yd) * K_LEAF_CELL_PX + x]);
                    const float nx = -(hR - hL) * k_normal_strength / 255.0f;
                    const float ny = -(hD - hU) * k_normal_strength / 255.0f;
                    const float nz = 1.0f;
                    const float inv_l = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
                    c.norm_x[ci] = u8_from_01(nx * inv_l * 0.5f + 0.5f);
                    c.norm_y[ci] = u8_from_01(ny * inv_l * 0.5f + 0.5f);
                    c.norm_z[ci] = u8_from_01(nz * inv_l * 0.5f + 0.5f);

                    // Roughness
                    const float rough = 0.70f + 0.15f * vein_f;
                    c.rough[ci] = u8_from_01(rough);

                    // Translucency (keep strictly >= 0.08 on veins and > 0.90 on blades)
                    const float edge_dist = static_cast<float>(blurred[ci]) / 255.0f;
                    const float edge_boost = 1.0f + 0.25f * (1.0f - edge_dist);
                    float trans = (0.92f - 0.72f * vein_f) * edge_boost;
                    trans = clampf(trans, 0.08f, 0.96f);
                    c.trans[ci] = u8_from_01(trans);
                }
            }

            dilate_cell_channels(c, 6, K_LEAF_CELL_PAD_PX);
            return c;
        }

        // ---- Broadleaf Botanical Sprig Baker (Rows 1..3 x Cols 0, 2, 3) ----

        // Map cluster_cell_grid override to deterministic blade radius scale.
        // Default [4, 3] -> 1.0; [6, 2] -> (1/6) / (1/4) = 0.6667 (lowers coverage).
        float sprig_grid_scale(const std::array<int, 2>& grid) {
            if (grid == k_default_cluster_cell_grid) return 1.0f;
            const float gx = static_cast<float>(std::max(1, grid[0]));
            const float gy = static_cast<float>(std::max(1, grid[1]));
            const float s = std::min(1.0f / gx, 1.0f / gy) / 0.25f;
            return clampf(s, 0.25f, 1.15f);
        }

        // Hard non-squareness envelope test for broadleaf sprig pixels:
        // recessed >= 16 px from all 4 borders AND strictly outside all four 32x32 corner triangles.
        inline bool in_sprig_safe_mask(int x, int y) {
            constexpr int W = K_LEAF_CELL_PX;
            constexpr int R = K_LEAF_SPRIG_RECESS_PX;
            constexpr int C = K_LEAF_SPRIG_CORNER_PX;
            if (x < R || x >= W - R || y < R || y >= W - R) return false;
            if (x + y < C) return false;
            if ((W - 1 - x) + y < C) return false;
            if (x + (W - 1 - y) < C) return false;
            if ((W - 1 - x) + (W - 1 - y) < C) return false;
            return true;
        }

        struct SprigLeafSpec {
            float cx = 128.0f;
            float cy = 128.0f;
            float radius = 32.0f;
            float angle = 0.0f;       // radians: 0 = apex points up (-y)
            float tilt_x = 0.0f;      // tangent-space X tilt (radians)
            float tilt_y = 0.0f;      // tangent-space Y tilt (radians)
            float luma_mul = 1.0f;    // per-leaf linear luma jitter
            float warmth = 0.0f;      // [0, 1] young apical leaf yellow-green shift
            float rough_base = 0.70f; // [0.64, 0.78]
            float trans_base = 0.65f; // [0.54, 0.78]
            vec2  anchor_px{128.0f, 128.0f}; // sub-cluster woody petiole base
            int   subcluster_id = 0;  // 0..3
        };

        // Analytic per-leaf vein height (midrib + species lateral veins) in shape space [-1, 1]^2.
        struct AnalyticVein {
            float height = 0.0f;      // [0, 1]
            float midrib = 0.0f;      // [0, 1] strong woody midrib/petiole weight
            float dnx = 0.0f;         // normal slope X in leaf-local space
            float dny = 0.0f;         // normal slope Y in leaf-local space
        };

        AnalyticVein eval_leaf_veins(LeafShape sp, float lx, float ly) {
            AnalyticVein v{};
            // Central midrib: tapers from base (-0.96) to near apex (+0.85).
            if (ly >= -1.0f && ly <= 0.90f) {
                const float t_len = clampf((ly + 1.0f) * 0.5f, 0.0f, 1.0f);
                const float half_w = 0.105f - 0.065f * t_len;
                const float ax = std::abs(lx);
                if (ax < half_w) {
                    const float u = ax / half_w;
                    const float profile = (1.0f - u * u) * smoothstep(0.90f, 0.72f, ly);
                    v.midrib = profile;
                    v.height = profile;
                    // Ridge normal slopes away from centerline.
                    const float sgn = (lx >= 0.0f) ? 1.0f : -1.0f;
                    v.dnx += sgn * u * (1.0f - u) * 0.55f;
                }
            }

            // Species-specific secondary veins.
            auto add_vein_seg = [&](vec2 a, vec2 b, float hw, float strength) {
                float t = 0.0f;
                const float d = seg_dist(vec2{lx, ly}, a, b, &t);
                const float w_loc = hw * (1.0f - 0.45f * t);
                if (d < w_loc) {
                    const float u = d / w_loc;
                    const float h = (1.0f - u * u) * strength;
                    if (h > v.height) v.height = h;
                    const float dx = b.x - a.x;
                    const float dy = b.y - a.y;
                    const float l = std::sqrt(dx * dx + dy * dy);
                    if (l > 1e-4f) {
                        const float px = -dy / l;
                        const float py =  dx / l;
                        const float side = ((lx - a.x) * px + (ly - a.y) * py) >= 0.0f ? 1.0f : -1.0f;
                        v.dnx += px * side * u * (1.0f - u) * 0.35f * strength;
                        v.dny += py * side * u * (1.0f - u) * 0.35f * strength;
                    }
                }
            };

            if (sp == LeafShape::OakLobed) {
                const float roots[3] = { -0.40f, -0.05f, +0.30f };
                const float tx[3]    = {  0.34f,  0.52f,  0.42f };
                const float ty[3]    = { -0.22f, +0.18f, +0.54f };
                for (int k = 0; k < 3; ++k) {
                    add_vein_seg(vec2{0.0f, roots[k]}, vec2{+tx[k], ty[k]}, 0.055f, 0.65f);
                    add_vein_seg(vec2{0.0f, roots[k]}, vec2{-tx[k], ty[k]}, 0.055f, 0.65f);
                }
            } else if (sp == LeafShape::BirchSerrated) {
                const float roots[4] = { -0.52f, -0.24f, +0.05f, +0.34f };
                const float tx[4]    = {  0.48f,  0.50f,  0.38f,  0.24f };
                const float ty[4]    = { -0.28f, +0.02f, +0.32f, +0.58f };
                for (int k = 0; k < 4; ++k) {
                    add_vein_seg(vec2{0.0f, roots[k]}, vec2{+tx[k], ty[k]}, 0.048f, 0.60f);
                    add_vein_seg(vec2{0.0f, roots[k]}, vec2{-tx[k], ty[k]}, 0.048f, 0.60f);
                }
            } else if (sp == LeafShape::MapleStar) {
                const vec2 base{0.0f, -0.18f};
                add_vein_seg(base, vec2{+0.70f, +0.42f}, 0.068f, 0.80f);
                add_vein_seg(base, vec2{-0.70f, +0.42f}, 0.068f, 0.80f);
                add_vein_seg(base, vec2{+0.46f, -0.22f}, 0.058f, 0.70f);
                add_vein_seg(base, vec2{-0.46f, -0.22f}, 0.058f, 0.70f);
            }
            return v;
        }

        // Check if pixel (x, y) is within `gap` pixels of a DIFFERENT subcluster.
        // Enforces crisp organic breathing gaps between sub-clusters so thick
        // botanical rosettes never accidentally fuse into a single cell-wide blob.
        bool near_other_subcluster(const std::vector<uint8_t>& owner, int x, int y, uint8_t my_tag, int gap) {
            constexpr int W = K_LEAF_CELL_PX;
            const int y0 = std::max(0, y - gap);
            const int y1 = std::min(W - 1, y + gap);
            const int x0 = std::max(0, x - gap);
            const int x1 = std::min(W - 1, x + gap);
            for (int ny = y0; ny <= y1; ++ny) {
                for (int nx = x0; nx <= x1; ++nx) {
                    const uint8_t o = owner[static_cast<size_t>(ny) * W + nx];
                    if (o != 0u && o != my_tag) return true;
                }
            }
            return false;
        }

        // Stamp a thick mip-safe woody petiole stub (>= 4.5 px wide) inside one sub-cluster.
        void stamp_petiole_stub(CellChannels& c, std::vector<uint8_t>& owner,
                                vec2 a, vec2 b, float half_w, int subcluster_id) {
            constexpr int W = K_LEAF_CELL_PX;
            const uint8_t my_tag = static_cast<uint8_t>(subcluster_id + 1);
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            const float len = std::sqrt(dx * dx + dy * dy);
            if (len < 1.0f) return;

            const float ux = dx / len;
            const float uy = dy / len;
            const float px = -uy;
            const float py =  ux;

            const int min_x = std::max(K_LEAF_SPRIG_RECESS_PX, static_cast<int>(std::floor(std::min(a.x, b.x) - half_w - 1.0f)));
            const int max_x = std::min(W - K_LEAF_SPRIG_RECESS_PX - 1, static_cast<int>(std::ceil(std::max(a.x, b.x) + half_w + 1.0f)));
            const int min_y = std::max(K_LEAF_SPRIG_RECESS_PX, static_cast<int>(std::floor(std::min(a.y, b.y) - half_w - 1.0f)));
            const int max_y = std::min(W - K_LEAF_SPRIG_RECESS_PX - 1, static_cast<int>(std::ceil(std::max(a.y, b.y) + half_w + 1.0f)));

            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    if (!in_sprig_safe_mask(x, y)) continue;
                    const vec2 pt{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
                    float t = 0.0f;
                    const float d = seg_dist(pt, a, b, &t);
                    const float hw_t = half_w * (1.0f - 0.25f * t);
                    if (d > hw_t) continue;
                    if (near_other_subcluster(owner, x, y, my_tag, 1)) continue;

                    const size_t ci = static_cast<size_t>(y) * W + x;
                    // Signed cross-section offset [-1, 1] for cylindrical petiole normal.
                    const float cross_s = clampf(((pt.x - a.x) * px + (pt.y - a.y) * py) / std::max(0.5f, hw_t), -1.0f, 1.0f);
                    const float nx = px * cross_s * 0.72f;
                    const float ny = py * cross_s * 0.72f;
                    const float nz = std::sqrt(std::max(0.20f, 1.0f - nx * nx - ny * ny));

                    // Warm woody olive-brown petiole tint.
                    const float shade = 0.72f + 0.28f * (1.0f - std::abs(cross_s));
                    c.alpha[ci]    = 255;
                    c.albedo_r[ci] = srgb_byte(0.18f * shade);
                    c.albedo_g[ci] = srgb_byte(0.17f * shade);
                    c.albedo_b[ci] = srgb_byte(0.08f * shade);
                    c.norm_x[ci]   = u8_from_01(nx * 0.5f + 0.5f);
                    c.norm_y[ci]   = u8_from_01(ny * 0.5f + 0.5f);
                    c.norm_z[ci]   = u8_from_01(nz * 0.5f + 0.5f);
                    c.rough[ci]    = u8_from_01(0.84f);
                    // TRAP 2: woody petiole translucency strictly clamped [0.07, 0.11] (>= 0.05 floor!).
                    c.trans[ci]    = u8_from_01(0.085f + 0.02f * std::abs(cross_s));
                    owner[ci]      = my_tag;
                }
            }
        }

        // Stamp one fully shaded botanical leaf into a sprig CellChannels buffer.
        void stamp_sprig_leaf(CellChannels& c, std::vector<uint8_t>& owner,
                              LeafShape sp, const LeafShapeMesh& mesh,
                              const SprigLeafSpec& lf) {
            constexpr int W = K_LEAF_CELL_PX;
            const uint8_t my_tag = static_cast<uint8_t>(lf.subcluster_id + 1);
            const int si = static_cast<int>(sp);
            const float ca = std::cos(lf.angle);
            const float sa = std::sin(lf.angle);
            const float inv_r = 1.0f / std::max(1.0f, lf.radius);

            auto xf = [&](vec2 v) -> vec2 {
                const float rx = v.x * ca - v.y * sa;
                const float ry = v.x * sa + v.y * ca;
                // V-flip (shape +Y -> smaller pixel y).
                return vec2{ lf.cx + rx * lf.radius, lf.cy - ry * lf.radius };
            };

            // Compute screen-space triangle vertices & leaf bounding box.
            const size_t n_tris = mesh.tris.size() / 3;
            std::vector<vec2> px_verts(mesh.verts.size());
            float bmin_x = 1e9f, bmin_y = 1e9f, bmax_x = -1e9f, bmax_y = -1e9f;
            for (size_t i = 0; i < mesh.verts.size(); ++i) {
                px_verts[i] = xf(mesh.verts[i]);
                bmin_x = std::min(bmin_x, px_verts[i].x);
                bmin_y = std::min(bmin_y, px_verts[i].y);
                bmax_x = std::max(bmax_x, px_verts[i].x);
                bmax_y = std::max(bmax_y, px_verts[i].y);
            }

            // Rasterize local binary leaf coverage mask inside the leaf's pixel bbox
            // so we get clean edge distance for rim lighting & thin-edge translucency.
            const int x0 = std::max(K_LEAF_SPRIG_RECESS_PX, static_cast<int>(std::floor(bmin_x)));
            const int y0 = std::max(K_LEAF_SPRIG_RECESS_PX, static_cast<int>(std::floor(bmin_y)));
            const int x1 = std::min(W - K_LEAF_SPRIG_RECESS_PX, static_cast<int>(std::ceil(bmax_x)));
            const int y1 = std::min(W - K_LEAF_SPRIG_RECESS_PX, static_cast<int>(std::ceil(bmax_y)));
            if (x0 >= x1 || y0 >= y1) return;

            const int bw = x1 - x0;
            const int bh = y1 - y0;
            std::vector<uint8_t> local_mask(static_cast<size_t>(bw) * bh, 0);

            for (size_t t = 0; t < n_tris; ++t) {
                const vec2 a = px_verts[mesh.tris[t * 3 + 0]];
                const vec2 b = px_verts[mesh.tris[t * 3 + 1]];
                const vec2 d = px_verts[mesh.tris[t * 3 + 2]];

                const int tx0 = std::max(x0, static_cast<int>(std::floor(std::min({a.x, b.x, d.x}))));
                const int ty0 = std::max(y0, static_cast<int>(std::floor(std::min({a.y, b.y, d.y}))));
                const int tx1 = std::min(x1, static_cast<int>(std::ceil (std::max({a.x, b.x, d.x}))));
                const int ty1 = std::min(y1, static_cast<int>(std::ceil (std::max({a.y, b.y, d.y}))));

                for (int y = ty0; y < ty1; ++y) {
                    for (int x = tx0; x < tx1; ++x) {
                        const vec2 p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
                        const float w0 = edge_fn(b, d, p);
                        const float w1 = edge_fn(d, a, p);
                        const float w2 = edge_fn(a, b, p);
                        if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) ||
                            (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f)) {
                            local_mask[static_cast<size_t>(y - y0) * bw + (x - x0)] = 255;
                        }
                    }
                }
            }

            // Compute subtle contact AO darken onto already-stamped leaves of the
            // same sub-cluster right outside this upper leaf's silhouette (1-2 px).
            for (int ly_i = 0; ly_i < bh; ++ly_i) {
                const int y = y0 + ly_i;
                for (int lx_i = 0; lx_i < bw; ++lx_i) {
                    if (local_mask[static_cast<size_t>(ly_i) * bw + lx_i]) continue;
                    const int x = x0 + lx_i;
                    const size_t ci = static_cast<size_t>(y) * W + x;
                    if (owner[ci] != my_tag) continue;
                    bool adj = false;
                    for (int dy = -2; dy <= 2 && !adj; ++dy) {
                        const int ny = ly_i + dy;
                        if (ny < 0 || ny >= bh) continue;
                        for (int dx = -2; dx <= 2; ++dx) {
                            const int nx = lx_i + dx;
                            if (nx < 0 || nx >= bw) continue;
                            if (local_mask[static_cast<size_t>(ny) * bw + nx]) { adj = true; break; }
                        }
                    }
                    if (adj) {
                        c.albedo_r[ci] = static_cast<uint8_t>((int(c.albedo_r[ci]) * 86) / 100);
                        c.albedo_g[ci] = static_cast<uint8_t>((int(c.albedo_g[ci]) * 86) / 100);
                        c.albedo_b[ci] = static_cast<uint8_t>((int(c.albedo_b[ci]) * 86) / 100);
                    }
                }
            }

            // Species linear base color (matching C3 P3 AO-tint level).
            constexpr float k_sprig_ao_tint = 0.86f;
            const float tilt_nx = std::sin(lf.tilt_x);
            const float tilt_ny = std::sin(lf.tilt_y);

            for (int ly_i = 0; ly_i < bh; ++ly_i) {
                const int y = y0 + ly_i;
                for (int lx_i = 0; lx_i < bw; ++lx_i) {
                    if (!local_mask[static_cast<size_t>(ly_i) * bw + lx_i]) continue;
                    const int x = x0 + lx_i;
                    if (!in_sprig_safe_mask(x, y)) continue;
                    if (near_other_subcluster(owner, x, y, my_tag, 1)) continue;

                    // Local edge-proximity check (how many of 8 neighbors at r=2 are inside).
                    int inside_nb = 0;
                    for (int dy = -2; dy <= 2; dy += 2) {
                        const int ny = ly_i + dy;
                        if (ny < 0 || ny >= bh) continue;
                        for (int dx = -2; dx <= 2; dx += 2) {
                            const int nx = lx_i + dx;
                            if (nx < 0 || nx >= bw) continue;
                            if (local_mask[static_cast<size_t>(ny) * bw + nx]) ++inside_nb;
                        }
                    }
                    const float edge_interior = static_cast<float>(inside_nb) / 9.0f;

                    // Invert leaf transform to recover exact shape-space (lx, ly) in [-1, 1]^2.
                    const float dx_px = (static_cast<float>(x) + 0.5f - lf.cx) * inv_r;
                    const float dy_px = -(static_cast<float>(y) + 0.5f - lf.cy) * inv_r; // undo V-flip
                    const float lx =  dx_px * ca + dy_px * sa;
                    const float ly = -dx_px * sa + dy_px * ca;

                    const AnalyticVein vn = eval_leaf_veins(sp, lx, ly);

                    // --- 1. Per-leaf Albedo ---
                    float lr = k_leaf_color[si].r * k_sprig_ao_tint * lf.luma_mul;
                    float lg = k_leaf_color[si].g * k_sprig_ao_tint * lf.luma_mul;
                    float lb = k_leaf_color[si].b * k_sprig_ao_tint * lf.luma_mul;

                    // Apical warmth (young outer leaves slightly warmer yellow-green).
                    lr *= (1.0f + 0.14f * lf.warmth);
                    lg *= (1.0f + 0.08f * lf.warmth);
                    lb *= (1.0f - 0.10f * lf.warmth);

                    // Per-leaf base-to-tip gradient (brighter at leaf tip, darker at petiole).
                    const float t_tip = clampf((ly + 1.0f) * 0.5f, 0.0f, 1.0f);
                    const float tip_grad = 0.80f + 0.24f * t_tip;
                    lr *= tip_grad; lg *= tip_grad; lb *= tip_grad;

                    // Subtle central-shoot AO where leaf bases meet the sub-cluster anchor.
                    const float da = std::sqrt((static_cast<float>(x) + 0.5f - lf.anchor_px.x) *
                                               (static_cast<float>(x) + 0.5f - lf.anchor_px.x) +
                                               (static_cast<float>(y) + 0.5f - lf.anchor_px.y) *
                                               (static_cast<float>(y) + 0.5f - lf.anchor_px.y));
                    const float shoot_ao = 0.84f + 0.16f * smoothstep(4.0f, 34.0f, da);
                    lr *= shoot_ao; lg *= shoot_ao; lb *= shoot_ao;

                    // Woody midrib & secondary vein tint.
                    if (vn.midrib > 0.01f) {
                        const float m = 0.35f * vn.midrib;
                        lr = lr * (1.0f - m) + (0.28f * lf.luma_mul) * m;
                        lg = lg * (1.0f - m) + (0.27f * lf.luma_mul) * m;
                        lb = lb * (1.0f - m) + (0.11f * lf.luma_mul) * m;
                    }
                    const float vein_dark = 1.0f - 0.12f * vn.height;
                    lr *= vein_dark; lg *= vein_dark; lb *= vein_dark;

                    // Subtle organic margin warmth at blade rim.
                    const float rim = 1.0f - smoothstep(0.35f, 0.90f, edge_interior);
                    lr = lr * (1.0f - 0.14f * rim) + (lr * 0.72f + 0.02f) * (0.14f * rim);
                    lg = lg * (1.0f - 0.14f * rim) + (lg * 0.68f)         * (0.14f * rim);
                    lb = lb * (1.0f - 0.14f * rim) + (lb * 0.55f)         * (0.14f * rim);

                    // --- 2. Per-leaf 3D Tangent-Space Normal ---
                    // Cross-blade V-cupping (~10 deg) + subtle longitudinal arch + vein ridge.
                    const float cup_x = -std::tanh(lx * 3.8f) * 0.18f;
                    const float cup_y = -ly * 0.10f;
                    const float loc_nx = cup_x + vn.dnx;
                    const float loc_ny = cup_y + vn.dny;

                    // Rotate leaf-local normal into cell tangent space (+x right, +y down).
                    const float rot_nx =  loc_nx * ca - loc_ny * sa;
                    const float rot_ny = -(loc_nx * sa + loc_ny * ca);

                    // Add per-leaf 3D pitch/roll tilt (±18..28 deg) and normalize.
                    float nx = rot_nx + tilt_nx;
                    float ny = rot_ny + tilt_ny;
                    const float xy2 = nx * nx + ny * ny;
                    if (xy2 > 0.72f) {
                        const float sc = std::sqrt(0.72f / xy2);
                        nx *= sc; ny *= sc;
                    }
                    const float nz = std::sqrt(std::max(0.28f, 1.0f - nx * nx - ny * ny));

                    // --- 3. Per-leaf Roughness (glTF G in [0.64, 0.86]) ---
                    const float rough = clampf(lf.rough_base + 0.12f * vn.height + 0.04f * vn.midrib, 0.64f, 0.86f);

                    // --- 4. Per-leaf Translucency (strictly >= 0.065 on ALL alpha>0!) ---
                    float trans = lf.trans_base + 0.10f * (1.0f - edge_interior) + 0.06f * t_tip;
                    // Secondary veins block ~40%, woody midrib drops cleanly into [0.07, 0.12].
                    trans *= (1.0f - 0.42f * vn.height);
                    if (vn.midrib > 0.25f) {
                        const float w_mid = smoothstep(0.25f, 0.75f, vn.midrib);
                        trans = trans * (1.0f - w_mid) + 0.085f * w_mid;
                    }
                    trans = clampf(trans, 0.065f, 0.88f);

                    const size_t ci = static_cast<size_t>(y) * W + x;
                    c.alpha[ci]    = 255;
                    c.albedo_r[ci] = srgb_byte(lr);
                    c.albedo_g[ci] = srgb_byte(lg);
                    c.albedo_b[ci] = srgb_byte(lb);
                    c.norm_x[ci]   = u8_from_01(nx * 0.5f + 0.5f);
                    c.norm_y[ci]   = u8_from_01(ny * 0.5f + 0.5f);
                    c.norm_z[ci]   = u8_from_01(nz * 0.5f + 0.5f);
                    c.rough[ci]    = u8_from_01(rough);
                    c.trans[ci]    = u8_from_01(trans);
                    owner[ci]      = my_tag;
                }
            }
        }

        // Build one broadleaf botanical sprig cell (variant_row in {1, 2, 3}).
        // Strictly seed-independent: RNG keyed solely on (species, variant_row).
        CellChannels bake_broadleaf_sprig_cell(LeafShape sp, int variant_row,
                                               const std::array<int, 2>& cluster_cell_grid) {
            CellChannels c;
            std::vector<uint8_t> owner(CellChannels::N, 0);

            const LeafShapeMesh mesh = shape_mesh_for(sp);
            const float gscale = sprig_grid_scale(cluster_cell_grid);

            // Species-tuned scale & petiole length:
            // - Oak: broad overlapping rosettes along forked/alternating shoots (~0.30-0.36 fill)
            // - Birch: numerous slender smaller leaves along arching shoots (~0.26-0.32 fill)
            // - Maple: shorter petiole offset so cordate bases fill the shoot core (~0.30-0.36 fill)
            float sp_scale = 1.0f;
            float petiole_frac = 0.54f;
            if (sp == LeafShape::BirchSerrated) { sp_scale = 1.09f; petiole_frac = 0.46f; }
            else if (sp == LeafShape::MapleStar) { sp_scale = 0.84f; petiole_frac = 0.48f; }

            pcg32 rng;
            constexpr uint64_t k_stream_sprig = 0xB0'7A'01'5E'EDULL;
            rng.seed(static_cast<uint64_t>(sp) * 17ULL + static_cast<uint64_t>(variant_row) * 131ULL,
                     k_stream_sprig);

            auto rand_sym = [&]() -> float { return rng.next_float_01() * 2.0f - 1.0f; };

            std::vector<SprigLeafSpec> leaves;
            std::vector<std::tuple<vec2, vec2, int>> petioles;

            auto add_subcluster = [&](int sc_id, vec2 anchor, float base_dir_deg,
                                      int count, float spread_deg, float r_min, float r_max,
                                      float warmth_base, float out_tilt_deg) {
                for (int k = 0; k < count; ++k) {
                    const float frac = (count == 1) ? 0.0f
                        : (static_cast<float>(k) / static_cast<float>(count - 1) - 0.5f);
                    const float dir_deg = base_dir_deg + frac * spread_deg + rand_sym() * 9.0f;
                    const float dir_rad = dir_deg * k_deg;
                    const float r = (r_min + (r_max - r_min) * (0.5f + 0.5f * rand_sym())) * sp_scale * gscale;

                    // In pixel coords: angle=0 points apex to -Y, so forward = (-sin(a), -cos(a)).
                    const float fx = -std::sin(dir_rad);
                    const float fy = -std::cos(dir_rad);
                    const float petiole_len = petiole_frac * r;
                    float cx = anchor.x + fx * petiole_len;
                    float cy = anchor.y + fy * petiole_len;

                    // Keep every leaf naturally recessed >= 18 px from borders and away from corners.
                    const float margin = r + static_cast<float>(K_LEAF_SPRIG_RECESS_PX) + 2.0f;
                    cx = clampf(cx, margin, static_cast<float>(K_LEAF_CELL_PX) - margin);
                    cy = clampf(cy, margin, static_cast<float>(K_LEAF_CELL_PX) - margin);

                    // Per-leaf 3D tangent-space tilt (±18..28 deg outward + organic jitter).
                    const float tilt_mag = (out_tilt_deg + rand_sym() * 6.0f) * k_deg;
                    const float tilt_x = clampf(fx * tilt_mag + rand_sym() * 11.0f * k_deg, -0.50f, 0.50f);
                    const float tilt_y = clampf(fy * tilt_mag + rand_sym() * 11.0f * k_deg, -0.50f, 0.50f);

                    SprigLeafSpec lf;
                    lf.cx            = cx;
                    lf.cy            = cy;
                    lf.radius        = r;
                    lf.angle         = dir_rad;
                    lf.tilt_x        = tilt_x;
                    lf.tilt_y        = tilt_y;
                    lf.luma_mul      = 1.0f + rand_sym() * 0.13f;
                    lf.warmth        = clampf(warmth_base + rand_sym() * 0.18f, 0.0f, 1.0f);
                    lf.rough_base    = clampf(0.70f + rand_sym() * 0.06f, 0.64f, 0.78f);
                    lf.trans_base    = clampf(0.64f + rand_sym() * 0.09f, 0.52f, 0.76f);
                    lf.anchor_px     = anchor;
                    lf.subcluster_id = sc_id;

                    // Extend thick petiole stub cleanly into the leaf palm/midrib so it never detaches.
                    const vec2 leaf_join{ cx - fx * (0.08f * r), cy - fy * (0.08f * r) };
                    petioles.emplace_back(anchor, leaf_join, sc_id);
                    leaves.push_back(lf);
                }
            };

            if (sp == LeafShape::BirchSerrated) {
                // Lush asymmetric silver-birch shoots: natural 5-10 px sky pockets, zero
                // sticker trenches, zero central donut hole, distinct silhouettes per row.
                if (variant_row == 1) {
                    // Var A: Forked Y-branch terminal shoot (core crosses center-left, forked apex)
                    add_subcluster(3, vec2{ 82.0f, 184.0f},  136.0f, 3,  86.0f, 39.0f, 46.0f, 0.14f, 23.0f);
                    add_subcluster(2, vec2{168.0f, 174.0f}, -126.0f, 3,  90.0f, 41.0f, 49.0f, 0.26f, 26.0f);
                    add_subcluster(0, vec2{108.0f, 112.0f},   34.0f, 3,  96.0f, 43.0f, 51.0f, 0.44f, 24.0f);
                    add_subcluster(1, vec2{172.0f,  86.0f},  -42.0f, 3,  90.0f, 39.0f, 47.0f, 0.80f, 22.0f);
                } else if (variant_row == 2) {
                    // Var B: Curved S-arc lateral shoot (sweeps lower-right -> core -> upper-left)
                    add_subcluster(3, vec2{148.0f, 197.0f}, -162.0f, 3,  84.0f, 38.0f, 45.0f, 0.16f, 24.0f);
                    add_subcluster(2, vec2{ 68.0f, 146.0f},  104.0f, 3,  88.0f, 41.0f, 49.0f, 0.30f, 26.0f);
                    add_subcluster(0, vec2{146.0f, 126.0f},  -62.0f, 3,  94.0f, 43.0f, 51.0f, 0.48f, 25.0f);
                    add_subcluster(1, vec2{ 98.0f,  72.0f},   16.0f, 3,  88.0f, 38.0f, 46.0f, 0.82f, 22.0f);
                } else {
                    // Var C: Open dappled 3-cluster arching twiglet
                    add_subcluster(2, vec2{112.0f, 172.0f},  162.0f, 3, 108.0f, 42.0f, 50.0f, 0.18f, 24.0f);
                    add_subcluster(1, vec2{168.0f, 124.0f},  -70.0f, 3, 112.0f, 44.0f, 52.0f, 0.40f, 26.0f);
                    add_subcluster(0, vec2{ 94.0f,  84.0f},   32.0f, 3, 108.0f, 41.0f, 49.0f, 0.78f, 22.0f);
                }
            } else if (variant_row == 1) {
                // Var A: Asymmetric forked Y-shoot (core leaf group fills center-left; zero donut ring).
                if (sp == LeafShape::MapleStar) {
                    add_subcluster(3, vec2{ 82.0f, 184.0f},  138.0f, 2,  82.0f, 44.0f, 52.0f, 0.12f, 23.0f);
                    add_subcluster(2, vec2{172.0f, 170.0f}, -124.0f, 3,  88.0f, 45.0f, 53.0f, 0.28f, 26.0f);
                    add_subcluster(0, vec2{112.0f, 116.0f},   32.0f, 3,  96.0f, 47.0f, 55.0f, 0.42f, 24.0f);
                    add_subcluster(1, vec2{172.0f,  84.0f},  -40.0f, 3,  88.0f, 42.0f, 50.0f, 0.78f, 21.0f);
                } else {
                    // Oak Var A: Forked terminal shoot sweeping lower-left -> center -> upper-right/left
                    add_subcluster(3, vec2{ 84.0f, 184.0f},  136.0f, 2,  84.0f, 44.0f, 52.0f, 0.12f, 23.0f);
                    add_subcluster(2, vec2{170.0f, 170.0f}, -122.0f, 3,  90.0f, 45.0f, 53.0f, 0.28f, 26.0f);
                    add_subcluster(0, vec2{108.0f, 114.0f},   36.0f, 3,  98.0f, 47.0f, 55.0f, 0.40f, 24.0f);
                    add_subcluster(1, vec2{172.0f,  84.0f},  -38.0f, 3,  90.0f, 42.0f, 50.0f, 0.78f, 21.0f);
                }
            } else if (variant_row == 2) {
                // Var B (default cluster row): Elongated curved S-arc lateral shoot (distinct from Var A).
                if (sp == LeafShape::MapleStar) {
                    add_subcluster(3, vec2{148.0f, 192.0f}, -164.0f, 2,  82.0f, 44.0f, 52.0f, 0.14f, 24.0f);
                    add_subcluster(2, vec2{ 68.0f, 144.0f},  102.0f, 3,  88.0f, 46.0f, 54.0f, 0.30f, 26.0f);
                    add_subcluster(0, vec2{144.0f, 124.0f},  -64.0f, 3,  94.0f, 47.0f, 55.0f, 0.46f, 25.0f);
                    add_subcluster(1, vec2{ 98.0f,  72.0f},   18.0f, 3,  88.0f, 42.0f, 50.0f, 0.78f, 22.0f);
                } else {
                    add_subcluster(3, vec2{144.0f, 192.0f}, -160.0f, 2,  84.0f, 44.0f, 52.0f, 0.15f, 23.0f);
                    add_subcluster(2, vec2{ 70.0f, 146.0f},  104.0f, 3,  90.0f, 46.0f, 54.0f, 0.28f, 26.0f);
                    add_subcluster(0, vec2{144.0f, 124.0f},  -66.0f, 3,  96.0f, 47.0f, 55.0f, 0.44f, 25.0f);
                    add_subcluster(1, vec2{ 98.0f,  74.0f},   14.0f, 3,  88.0f, 42.0f, 50.0f, 0.80f, 21.0f);
                }
            } else {
                // Var C (Row 3): Open dappled 3-cluster twiglet with natural interior sky pockets.
                add_subcluster(2, vec2{112.0f, 170.0f},  164.0f, 2,  96.0f, 46.0f, 54.0f, 0.18f, 24.0f);
                add_subcluster(1, vec2{166.0f, 124.0f},  -72.0f, 3, 112.0f, 46.0f, 54.0f, 0.38f, 26.0f);
                add_subcluster(0, vec2{ 96.0f,  86.0f},   34.0f, 3, 110.0f, 43.0f, 51.0f, 0.76f, 22.0f);
            }

            // Stamp thick petiole stubs first (underneath leaf bases), then leaves back-to-front.
            for (const auto& [pa, pb, sc_id] : petioles) {
                stamp_petiole_stub(c, owner, pa, pb, 2.4f, sc_id);
            }
            for (const auto& lf : leaves) {
                stamp_sprig_leaf(c, owner, sp, mesh, lf);
            }

            dilate_cell_channels(c, 6, K_LEAF_CELL_PAD_PX);
            return c;
        }

        // ---- Pine Conifer Cells (Rows 1..3 x Col 1) ----

        // Stamp one tapered angled pine needle into a CellChannels buffer.
        // If `wrap_v` is true, y wraps periodically inside [pad, W-pad) for seamless V-tiling.
        void stamp_pine_needle(CellChannels& c, std::vector<uint8_t>* owner,
                               vec2 base, float angle_rad, float len_px, float base_hw,
                               LinearRgb col, float tilt_z_bias, float rough_v, float trans_v,
                               bool wrap_v, bool respect_sprig_mask, int subcluster_id) {
            constexpr int W = K_LEAF_CELL_PX;
            constexpr int pad = K_LEAF_CELL_PAD_PX;
            constexpr int usable = K_LEAF_CELL_USABLE_PX;
            const uint8_t my_tag = static_cast<uint8_t>(subcluster_id + 1);

            const float ux = std::cos(angle_rad);
            const float uy = std::sin(angle_rad);
            const float px = -uy;
            const float py =  ux;

            const vec2 tip{ base.x + ux * len_px, base.y + uy * len_px };
            const float max_hw = base_hw + 1.0f;
            const int x0 = std::max(pad, static_cast<int>(std::floor(std::min(base.x, tip.x) - max_hw)));
            const int x1 = std::min(W - pad - 1, static_cast<int>(std::ceil(std::max(base.x, tip.x) + max_hw)));
            const int y0 = static_cast<int>(std::floor(std::min(base.y, tip.y) - max_hw));
            const int y1 = static_cast<int>(std::ceil(std::max(base.y, tip.y) + max_hw));

            for (int raw_y = y0; raw_y <= y1; ++raw_y) {
                int y = raw_y;
                if (wrap_v) {
                    int rel = (raw_y - pad) % usable;
                    if (rel < 0) rel += usable;
                    y = pad + rel;
                } else {
                    if (y < pad || y >= W - pad) continue;
                }
                for (int x = x0; x <= x1; ++x) {
                    if (respect_sprig_mask && !in_sprig_safe_mask(x, y)) continue;
                    const float dx = static_cast<float>(x) + 0.5f - base.x;
                    const float dy = static_cast<float>(raw_y) + 0.5f - base.y;
                    const float along = dx * ux + dy * uy;
                    if (along < 0.0f || along > len_px) continue;

                    const float t = along / len_px;
                    // Taper cleanly toward pointed tip (keep >= 1.1 px near base so mip-safe).
                    const float hw_t = base_hw * (1.0f - 0.65f * t * t);
                    const float perp = dx * px + dy * py;
                    if (std::abs(perp) > hw_t) continue;
                    if (owner && near_other_subcluster(*owner, x, y, my_tag, 2)) continue;

                    const float s = clampf(perp / std::max(0.5f, hw_t), -1.0f, 1.0f);
                    // Cylindrical cross-section normal rotated along the needle + slight 3D pitch.
                    float nx = px * s * 0.78f + ux * tilt_z_bias;
                    float ny = py * s * 0.78f + uy * tilt_z_bias;
                    const float xy2 = nx * nx + ny * ny;
                    if (xy2 > 0.82f) {
                        const float sc = std::sqrt(0.82f / xy2);
                        nx *= sc; ny *= sc;
                    }
                    const float nz = std::sqrt(std::max(0.18f, 1.0f - nx * nx - ny * ny));

                    // Base-to-tip needle shading: darker sheath base, brighter young needle tip.
                    const float tip_luma = 0.78f + 0.30f * t;
                    const float cyl_ao   = 0.85f + 0.15f * std::sqrt(std::max(0.0f, 1.0f - s * s));
                    const float lr = col.r * tip_luma * cyl_ao;
                    const float lg = col.g * tip_luma * cyl_ao;
                    const float lb = col.b * tip_luma * cyl_ao;

                    // Woody fascicle sheath at very base (t < 0.12): darker & lower trans (>= 0.08).
                    float tr = trans_v * (0.85f + 0.20f * t);
                    float ro = rough_v;
                    if (t < 0.12f) {
                        const float sheath = 1.0f - t / 0.12f;
                        tr = tr * (1.0f - sheath) + 0.09f * sheath;
                        ro = ro + 0.08f * sheath;
                    }
                    tr = clampf(tr, 0.075f, 0.82f);
                    ro = clampf(ro, 0.64f, 0.86f);

                    const size_t ci = static_cast<size_t>(y) * W + x;
                    c.alpha[ci]    = 255;
                    c.albedo_r[ci] = srgb_byte(lr);
                    c.albedo_g[ci] = srgb_byte(lg);
                    c.albedo_b[ci] = srgb_byte(lb);
                    c.norm_x[ci]   = u8_from_01(nx * 0.5f + 0.5f);
                    c.norm_y[ci]   = u8_from_01(ny * 0.5f + 0.5f);
                    c.norm_z[ci]   = u8_from_01(nz * 0.5f + 0.5f);
                    c.rough[ci]    = u8_from_01(ro);
                    c.trans[ci]    = u8_from_01(tr);
                    if (owner) (*owner)[ci] = my_tag;
                }
            }
        }

        // Cell (1, 1): V-tileable angled V-fanned pine needle fascicle shoot for BranchStrip.
        CellChannels bake_pine_fascicle_strip_cell() {
            CellChannels c;
            pcg32 rng;
            rng.seed(0x91'4E'57'1BULL, 0xC3'01'01ULL);
            auto rand_sym = [&]() -> float { return rng.next_float_01() * 2.0f - 1.0f; };

            constexpr LinearRgb pine_base = k_leaf_color[int(LeafShape::PineNeedle)];
            constexpr int k_whorls = 8;
            const float pad = static_cast<float>(K_LEAF_CELL_PAD_PX);
            const float usable = static_cast<float>(K_LEAF_CELL_USABLE_PX);
            const float pitch_y = usable / static_cast<float>(k_whorls);

            for (int w = 0; w < k_whorls; ++w) {
                const float y_w = pad + (static_cast<float>(w) + 0.5f) * pitch_y;
                // Fascicle bundles fanning forward-left and forward-right in V-chevrons.
                for (int side = -1; side <= 1; side += 2) {
                    const float bx = 128.0f + static_cast<float>(side) * (2.0f + rand_sym() * 1.5f);
                    const float by = y_w + rand_sym() * 2.5f;
                    const float fasc_luma = 1.0f + rand_sym() * 0.14f;
                    const float fasc_warm = rng.next_float_01();
                    const LinearRgb col{
                        pine_base.r * fasc_luma * (1.0f + 0.14f * fasc_warm),
                        pine_base.g * fasc_luma * (1.0f + 0.08f * fasc_warm),
                        pine_base.b * fasc_luma * (1.0f - 0.08f * fasc_warm),
                    };
                    // 4 needles per side fascicle at forward V-angles (28..56 deg off +Y/-Y).
                    for (int n = 0; n < 4; ++n) {
                        const float fan = (static_cast<float>(n) / 3.0f) * 28.0f + 26.0f + rand_sym() * 4.0f;
                        // side=+1 fans right & forward (-Y); side=-1 fans left & forward (-Y).
                        const float ang = (side > 0) ? (-fan * k_deg) : (-(180.0f - fan) * k_deg);
                        const float len = 84.0f + rand_sym() * 22.0f;
                        const float hw  = 2.6f + rand_sym() * 0.5f;
                        const float tilt = rand_sym() * 0.25f;
                        const float ro = 0.72f + rand_sym() * 0.07f;
                        const float tr = 0.60f + rand_sym() * 0.08f;
                        stamp_pine_needle(c, nullptr, vec2{bx, by}, ang, len, hw, col, tilt, ro, tr,
                                          /*wrap_v=*/true, /*respect_sprig_mask=*/false, 0);
                    }
                }
            }

            dilate_cell_channels(c, 6, K_LEAF_CELL_PAD_PX);
            return c;
        }

        // Cells (2, 1) and (3, 1): Pine Whorl Tuft A (radial cluster) and Tuft B (asymmetric candle).
        CellChannels bake_pine_whorl_cell(int variant_row, const std::array<int, 2>& cluster_cell_grid) {
            CellChannels c;
            std::vector<uint8_t> owner(CellChannels::N, 0);
            const float gscale = sprig_grid_scale(cluster_cell_grid);

            pcg32 rng;
            rng.seed(static_cast<uint64_t>(variant_row) * 97ULL + 0x91'4E'00ULL, 0xB0'7A'02'5E'EDULL);
            auto rand_sym = [&]() -> float { return rng.next_float_01() * 2.0f - 1.0f; };
            constexpr LinearRgb pine_base = k_leaf_color[int(LeafShape::PineNeedle)];

            auto add_tuft = [&](int sc_id, vec2 anchor, float base_deg, int needles,
                                float spread_deg, float len_min, float len_max, float warmth) {
                for (int k = 0; k < needles; ++k) {
                    const float frac = (needles == 1) ? 0.0f
                        : (static_cast<float>(k) / static_cast<float>(needles - 1) - 0.5f);
                    const float ang = (base_deg + frac * spread_deg + rand_sym() * 9.0f) * k_deg;
                    const float len = (len_min + (len_max - len_min) * (0.5f + 0.5f * rand_sym())) * gscale;
                    const float hw  = (3.4f + rand_sym() * 0.6f) * std::max(0.65f, gscale);
                    const float luma = 1.0f + rand_sym() * 0.14f;
                    const LinearRgb col{
                        pine_base.r * luma * (1.0f + 0.18f * warmth),
                        pine_base.g * luma * (1.0f + 0.12f * warmth),
                        pine_base.b * luma * (1.0f - 0.10f * warmth),
                    };
                    const vec2 jit_base{ anchor.x + rand_sym() * 5.0f, anchor.y + rand_sym() * 5.0f };
                    const float tilt = rand_sym() * 0.30f;
                    const float ro = 0.72f + rand_sym() * 0.07f;
                    const float tr = 0.58f + rand_sym() * 0.10f;
                    stamp_pine_needle(c, &owner, jit_base, ang, len, hw, col, tilt, ro, tr,
                                      /*wrap_v=*/false, /*respect_sprig_mask=*/true, sc_id);
                }
            };

            if (variant_row == 2) {
                // Pine Whorl Tuft A: 4 organic overlapping needle bough tufts around a central whorl.
                add_tuft(0, vec2{128.0f, 114.0f},  -88.0f, 10, 128.0f, 54.0f, 78.0f, 0.58f);
                add_tuft(1, vec2{112.0f, 128.0f},  190.0f,  9, 118.0f, 52.0f, 74.0f, 0.32f);
                add_tuft(2, vec2{146.0f, 128.0f},  -12.0f,  9, 118.0f, 52.0f, 74.0f, 0.32f);
                add_tuft(3, vec2{126.0f, 146.0f},   92.0f,  9, 120.0f, 48.0f, 70.0f, 0.16f);
            } else {
                // Pine Whorl Tuft B: 3 asymmetric terminal pine candle shoots.
                add_tuft(0, vec2{130.0f, 110.0f},  -80.0f, 11, 130.0f, 56.0f, 80.0f, 0.82f);
                add_tuft(1, vec2{108.0f, 136.0f},  202.0f,  9, 114.0f, 50.0f, 72.0f, 0.36f);
                add_tuft(2, vec2{150.0f, 138.0f},  -20.0f,  9, 114.0f, 50.0f, 72.0f, 0.36f);
            }

            dilate_cell_channels(c, 6, K_LEAF_CELL_PAD_PX);
            return c;
        }

        // Bake all 16 cells into the requested 1024x1024 RGBA8 atlas buffer.
        std::vector<uint8_t> encode_atlas_channel(AtlasChannelKind kind,
                                                  const std::array<int, 2>& cluster_cell_grid) {
            std::vector<uint8_t> rgba(
                static_cast<size_t>(K_LEAF_ATLAS_PX) * K_LEAF_ATLAS_PX * 4, 0);

            // Initialize atlas defaults per channel kind.
            const size_t total_px = static_cast<size_t>(K_LEAF_ATLAS_PX) * K_LEAF_ATLAS_PX;
            if (kind == AtlasChannelKind::Normal) {
                for (size_t i = 0; i < total_px; ++i) {
                    rgba[i * 4 + 0] = 128;
                    rgba[i * 4 + 1] = 128;
                    rgba[i * 4 + 2] = 255;
                    rgba[i * 4 + 3] = 255;
                }
            } else if (kind == AtlasChannelKind::Roughness) {
                for (size_t i = 0; i < total_px; ++i) {
                    rgba[i * 4 + 0] = 255;
                    rgba[i * 4 + 1] = 179;
                    rgba[i * 4 + 2] = 0;
                    rgba[i * 4 + 3] = 255;
                }
            } else if (kind == AtlasChannelKind::Translucency) {
                for (size_t i = 0; i < total_px; ++i) {
                    rgba[i * 4 + 3] = 255;
                }
            }

            // Row 0: single-leaf cells for all 4 species.
            for (LeafShape sp : k_species_in_atlas) {
                const CellChannels cell = bake_single_leaf_cell(sp);
                blit_cell_to_atlas(rgba.data(), leaf_atlas_cell_row(sp), leaf_atlas_cell_col(sp), cell, kind);
            }

            // Rows 1..3 x Cols 0, 2, 3: 3 botanical sprig variants per broadleaf species.
            for (LeafShape sp : {LeafShape::OakLobed, LeafShape::BirchSerrated, LeafShape::MapleStar}) {
                const int col = leaf_cluster_cell_col(sp);
                for (int row = K_LEAF_SPRIG_ROW_MIN; row <= K_LEAF_SPRIG_ROW_MAX; ++row) {
                    const CellChannels cell = bake_broadleaf_sprig_cell(sp, row, cluster_cell_grid);
                    blit_cell_to_atlas(rgba.data(), row, col, cell, kind);
                }
            }

            // Rows 1..3 x Col 1: Pine fascicle shoot + two pine whorl tufts.
            {
                const int col = leaf_atlas_cell_col(LeafShape::PineNeedle);
                const CellChannels strip_cell = bake_pine_fascicle_strip_cell();
                blit_cell_to_atlas(rgba.data(), 1, col, strip_cell, kind);
                const CellChannels whorl_a = bake_pine_whorl_cell(2, cluster_cell_grid);
                blit_cell_to_atlas(rgba.data(), 2, col, whorl_a, kind);
                const CellChannels whorl_b = bake_pine_whorl_cell(3, cluster_cell_grid);
                blit_cell_to_atlas(rgba.data(), 3, col, whorl_b, kind);
            }

            return encode_png_rgba8(K_LEAF_ATLAS_PX, K_LEAF_ATLAS_PX,
                                    K_LEAF_ATLAS_PX * 4, rgba.data());
        }

    }  // namespace

    std::vector<uint8_t> encode_leaf_atlas_png(uint64_t /*seed*/, const std::array<int, 2>& cluster_cell_grid) {
        return encode_atlas_channel(AtlasChannelKind::Albedo, cluster_cell_grid);
    }

    std::vector<uint8_t> encode_leaf_normal_png(uint64_t /*seed*/, const std::array<int, 2>& cluster_cell_grid) {
        return encode_atlas_channel(AtlasChannelKind::Normal, cluster_cell_grid);
    }

    std::vector<uint8_t> encode_leaf_translucency_png(uint64_t /*seed*/, const std::array<int, 2>& cluster_cell_grid) {
        return encode_atlas_channel(AtlasChannelKind::Translucency, cluster_cell_grid);
    }

    std::vector<uint8_t> encode_leaf_roughness_png(uint64_t /*seed*/, const std::array<int, 2>& cluster_cell_grid) {
        return encode_atlas_channel(AtlasChannelKind::Roughness, cluster_cell_grid);
    }

}  // namespace treegen
