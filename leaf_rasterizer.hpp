// C6 P3 — Triangle rasterizer for the leaf atlas. Edge-function (barycentric)
// scan over each triangle's pixel bounding box. Pure POD; no engine deps.
// /fp:precise (treegen.sharpmake.cs) is required so per-pixel barycentric
// sign tests are reproducible — every per-pixel "is inside" decision
// contributes to the atlas byte hash.
#pragma once

#include "leaf_shapes.hpp"   // vec2, LeafShapeMesh

#include <cstdint>
#include <vector>

namespace treegen {

    // Stamp the polygon `mesh` into `alpha` (size = cell_w * cell_h, row-major,
    // origin = top-left). The shape coordinates are in [-1, 1]² (apex +Y);
    // they are mapped to the cell's USABLE rect, which spans
    //   x_px ∈ [pad_px,                  pad_px + usable_px]
    //   y_px ∈ [pad_px,                  pad_px + usable_px]
    // inside the `cell_w × cell_h` cell. Pixels covered by any triangle of the
    // shape are set to 255; pre-existing pixels are preserved (caller pre-zeros).
    //
    // V-flip convention: shape +Y → smaller pixel y (top of cell), so the leaf
    // apex sits at the top of the atlas tile — same convention as `tile_uv`.
    void rasterize_leaf_alpha_into_cell(
        const LeafShapeMesh& mesh,
        uint8_t*             alpha,         // cell_w * cell_h bytes
        int                  cell_w,        // pixel width  of the cell
        int                  cell_h,        // pixel height of the cell
        int                  pad_px,        // inset on every side
        int                  usable_px);    // = cell_w - 2 * pad_px (must hold)

    // Single-triangle path — exposed so the vein stamper can additively splat
    // 2-px-wide line "triangles" into a separate height buffer. `value` is the
    // byte stamped into every covered pixel; the caller decides additive vs
    // max-blend semantics by reading dst first if needed. Coordinates are in
    // pixel space (origin top-left of the buffer). No clipping outside the
    // buffer; caller responsibility.
    void rasterize_tri_pixel_space(
        uint8_t* dst,
        int      dst_w,
        int      dst_h,
        vec2     a_px,
        vec2     b_px,
        vec2     c_px,
        uint8_t  value);

}
