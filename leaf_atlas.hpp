// C6 P3 — Leaf atlas API. P3a swaps P1's transparent stub for the real polygon
// rasterizer (alpha + diffuse); P3b adds normal + translucency. Per-species
// cell layout reserved in P1 (one row, 4 cols; 12 cells unused).
#pragma once

#include "leaf_shapes.hpp"        // vec2
#include "tree_descriptor.hpp"    // LeafShape

#include <array>
#include <cstdint>
#include <vector>

namespace treegen {

    // 4×4 atlas grid; 256-px cells; 4-px padding inset → 248-px usable region.
    // P3 populates row=0 cols=0..3 only (one cell per species). The padding is
    // alpha=0 so bilinear neighbours of border texels don't leak between cells.
    inline constexpr int K_LEAF_ATLAS_DIM       = 4;
    inline constexpr int K_LEAF_ATLAS_PX        = 1024;
    inline constexpr int K_LEAF_CELL_PX         = 256;
    inline constexpr int K_LEAF_CELL_PAD_PX     = 4;
    inline constexpr int K_LEAF_CELL_USABLE_PX  = K_LEAF_CELL_PX - 2 * K_LEAF_CELL_PAD_PX;  // 248

    static_assert(K_LEAF_CELL_PX * K_LEAF_ATLAS_DIM == K_LEAF_ATLAS_PX,
                  "cell grid covers the whole atlas");

    constexpr int leaf_atlas_cell_row(LeafShape s) {
        switch (s) {
            case LeafShape::OakLobed:      return 0;
            case LeafShape::PineNeedle:    return 0;
            case LeafShape::BirchSerrated: return 0;
            case LeafShape::MapleStar:     return 0;
        }
        return 0;
    }
    constexpr int leaf_atlas_cell_col(LeafShape s) {
        switch (s) {
            case LeafShape::OakLobed:      return 0;
            case LeafShape::PineNeedle:    return 1;
            case LeafShape::BirchSerrated: return 2;
            case LeafShape::MapleStar:     return 3;
        }
        return 0;
    }

    // Map per-shape local 2D coords ([-1, 1] half-extent, +Y apex) to atlas UV.
    // V is flipped so the leaf apex (+Y) maps to the top of the cell (smaller V),
    // matching the emitter's `uv_for_quad_corner` convention. Single source of
    // truth for emitter ↔ atlas alignment — every consumer that needs a tile UV
    // calls this; drift between emitter and atlas is structurally impossible.
    constexpr vec2 tile_uv(LeafShape s, float lx, float ly) {
        const int   row          = leaf_atlas_cell_row(s);
        const int   col          = leaf_atlas_cell_col(s);
        const float cell_size_uv = 1.0f / static_cast<float>(K_LEAF_ATLAS_DIM);                       // 0.25
        const float pad_uv       = static_cast<float>(K_LEAF_CELL_PAD_PX)
                                 / static_cast<float>(K_LEAF_ATLAS_PX);                                // 0.00390625
        const float usable_uv    = cell_size_uv - 2.0f * pad_uv;                                       // ≈ 0.24219
        const float u_local      = 0.5f * (lx + 1.0f);     // [-1, 1] → [0, 1]
        const float v_local      = 0.5f * (1.0f - ly);     // V-flip: ly=+1 → 0 (top); ly=-1 → 1
        return vec2{
            static_cast<float>(col) * cell_size_uv + pad_uv + u_local * usable_uv,
            static_cast<float>(row) * cell_size_uv + pad_uv + v_local * usable_uv,
        };
    }

    // C3 P1 -> Botanical Sprig Uplift: rows 1..3 hold 3 distinct botanical sprig
    // variants per species at the same column as its single-leaf cell (row 0):
    //   Row 1: Var A (forked terminal fan shoot; pine = V-tileable fascicle shoot)
    //   Row 2: Var B (curved alternating/opposite lateral shoot; default cluster row)
    //   Row 3: Var C (open dappled twiglet with interior sky pockets; pine = candle tuft)
    inline constexpr int K_LEAF_SPRIG_ROW_MIN     = 1;
    inline constexpr int K_LEAF_SPRIG_ROW_MAX     = 3;
    inline constexpr int K_LEAF_CLUSTER_CELL_ROW  = 2;
    inline constexpr int K_LEAF_SPRIG_RECESS_PX   = 16;
    inline constexpr int K_LEAF_SPRIG_CORNER_PX   = 32;
    constexpr int leaf_cluster_cell_col(LeafShape s) { return leaf_atlas_cell_col(s); }

    // Map sprig-card local coords ([-1, 1] half-extent, +Y apex) to the chosen
    // variant row (1..3) atlas UV.
    constexpr vec2 sprig_tile_uv(LeafShape s, int variant_row, float lx, float ly) {
        const int   row          = (variant_row < K_LEAF_SPRIG_ROW_MIN)
                                 ? K_LEAF_SPRIG_ROW_MIN
                                 : (variant_row > K_LEAF_SPRIG_ROW_MAX ? K_LEAF_SPRIG_ROW_MAX : variant_row);
        const int   col          = leaf_cluster_cell_col(s);
        const float cell_size_uv = 1.0f / static_cast<float>(K_LEAF_ATLAS_DIM);
        const float pad_uv       = static_cast<float>(K_LEAF_CELL_PAD_PX)
                                 / static_cast<float>(K_LEAF_ATLAS_PX);
        const float usable_uv    = cell_size_uv - 2.0f * pad_uv;
        const float u_local      = 0.5f * (lx + 1.0f);
        const float v_local      = 0.5f * (1.0f - ly);
        return vec2{
            static_cast<float>(col) * cell_size_uv + pad_uv + u_local * usable_uv,
            static_cast<float>(row) * cell_size_uv + pad_uv + v_local * usable_uv,
        };
    }

    // Map cluster-card local coords ([-1, 1] half-extent, +Y apex) to the
    // default cluster cell's atlas UV (row 2).
    constexpr vec2 cluster_tile_uv(LeafShape s, float lx, float ly) {
        return sprig_tile_uv(s, K_LEAF_CLUSTER_CELL_ROW, lx, ly);
    }

    // Needle-strip atlas cell: row=1, col=1 (cell index 5).
    inline constexpr int K_NEEDLE_STRIP_CELL_ROW = 1;
    inline constexpr int K_NEEDLE_STRIP_CELL_COL = 1;

    // Map strip-local (u, v) in [0,1]^2 to atlas UV for the needle-strip cell.
    // u spans across needles (width), v spans along the branch (tiling length).
    // Both u and v are clamped to [0, 1] so strip UVs never bleed into Row 2.
    constexpr vec2 needle_strip_tile_uv(float u, float v) {
        const float cu           = u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u);
        const float cv           = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        const float cell_size_uv = 1.0f / static_cast<float>(K_LEAF_ATLAS_DIM);
        const float pad_uv       = static_cast<float>(K_LEAF_CELL_PAD_PX)
                                 / static_cast<float>(K_LEAF_ATLAS_PX);
        const float usable_uv    = cell_size_uv - 2.0f * pad_uv;
        return vec2{
            static_cast<float>(K_NEEDLE_STRIP_CELL_COL) * cell_size_uv + pad_uv + cu * usable_uv,
            static_cast<float>(K_NEEDLE_STRIP_CELL_ROW) * cell_size_uv + pad_uv + cv * usable_uv,
        };
    }

    // ----- Atlas encoders (strictly seed-independent!) ---------------------
    // All 4 encoders ignore `seed` so all staged GLBs deduplicate to one atlas.
    // `cluster_cell_grid`: default [4, 3] emits full botanical sprigs; non-default
    // [nx, ny] scales leaf blade radius deterministically for sweep overrides.
    // Normal and roughness encode losslessly as opaque RGB8 PNG; translucency
    // encodes losslessly as Gray8 PNG (all consumers decode with req_comp = 4).
    struct LeafAtlasPngBundle {
        std::vector<uint8_t> albedo_png;
        std::vector<uint8_t> normal_png;
        std::vector<uint8_t> translucency_png;
        std::vector<uint8_t> roughness_png;
    };

    // Single-pass bundle encoder (bakes all 16 cells once -> 4x faster than 4 calls).
    LeafAtlasPngBundle encode_leaf_bundle_pngs(
        uint64_t seed, const std::array<int, 2>& cluster_cell_grid = k_default_cluster_cell_grid);

    std::vector<uint8_t> encode_leaf_atlas_png(
        uint64_t seed, const std::array<int, 2>& cluster_cell_grid = k_default_cluster_cell_grid);

    // Tangent-space normal (OpenGL +Y_up = +B = -dP/dV; RGB8 PNG, A=255 on decode).
    std::vector<uint8_t> encode_leaf_normal_png(
        uint64_t seed, const std::array<int, 2>& cluster_cell_grid = k_default_cluster_cell_grid);

    // Translucency (Gray8 PNG: R = trans >= 0.05 on all alpha>0).
    std::vector<uint8_t> encode_leaf_translucency_png(
        uint64_t seed, const std::array<int, 2>& cluster_cell_grid = k_default_cluster_cell_grid);

    // Roughness (glTF MR RGB8 PNG: R = 255, G = roughness, B = 0, A = 255 on decode).
    std::vector<uint8_t> encode_leaf_roughness_png(
        uint64_t seed, const std::array<int, 2>& cluster_cell_grid = k_default_cluster_cell_grid);

}
