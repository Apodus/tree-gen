// C6 P3 — Leaf atlas API. P3a swaps P1's transparent stub for the real polygon
// rasterizer (alpha + diffuse); P3b adds normal + translucency. Per-species
// cell layout reserved in P1 (one row, 4 cols; 12 cells unused).
#pragma once

#include "leaf_shapes.hpp"        // vec2
#include "tree_descriptor.hpp"    // LeafShape

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

    // Needle-strip atlas cell: row=1, col=1 (cell index 5).
    inline constexpr int K_NEEDLE_STRIP_CELL_ROW = 1;
    inline constexpr int K_NEEDLE_STRIP_CELL_COL = 1;

    // Map strip-local (u, v) in [0,1]^2 to atlas UV for the needle-strip cell.
    // u spans across needles (width), v spans along the branch (tiling length).
    constexpr vec2 needle_strip_tile_uv(float u, float v) {
        const float cell_size_uv = 1.0f / static_cast<float>(K_LEAF_ATLAS_DIM);
        const float pad_uv       = static_cast<float>(K_LEAF_CELL_PAD_PX)
                                 / static_cast<float>(K_LEAF_ATLAS_PX);
        const float usable_uv    = cell_size_uv - 2.0f * pad_uv;
        return vec2{
            static_cast<float>(K_NEEDLE_STRIP_CELL_COL) * cell_size_uv + pad_uv + u * usable_uv,
            static_cast<float>(K_NEEDLE_STRIP_CELL_ROW) * cell_size_uv + pad_uv + v * usable_uv,
        };
    }

    // ----- Atlas encoders --------------------------------------------------
    // P3a — alpha + diffuse (RGBA8 1024×1024, RGB = species diffuse, A = mask).
    std::vector<uint8_t> encode_leaf_atlas_png(uint64_t seed);

    // P3b — tangent-space normal (RGBA8: RGB = normal*0.5+0.5, A=255).
    std::vector<uint8_t> encode_leaf_normal_png(uint64_t seed);

    // P3b — translucency (RGBA8: R = trans, GB = 0, A=255).
    std::vector<uint8_t> encode_leaf_translucency_png(uint64_t seed);

    // C3 P2 — roughness (RGBA8: R = roughness, GB = 0, A=255).
    std::vector<uint8_t> encode_leaf_roughness_png(uint64_t seed);

}
