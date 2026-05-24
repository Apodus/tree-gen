// /fp:precise required — every barycentric edge-function sign decides whether
// a pixel ends up covered, and that decides the atlas byte hash. See
// treegen.sharpmake.cs.
#include "leaf_rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace treegen {

namespace {

inline float edge_fn(vec2 a, vec2 b, vec2 p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Stamp triangle (a, b, c) with `value` into `dst`. Top-left-inclusive (>= 0)
// edge test handles both winding orders by checking that the three edge
// functions share a sign. Pixel center sample is (x + 0.5, y + 0.5).
void rasterize_tri_impl(uint8_t* dst, int dst_w, int dst_h,
                        vec2 a, vec2 b, vec2 c, uint8_t value) {
    const float min_xf = std::min(std::min(a.x, b.x), c.x);
    const float min_yf = std::min(std::min(a.y, b.y), c.y);
    const float max_xf = std::max(std::max(a.x, b.x), c.x);
    const float max_yf = std::max(std::max(a.y, b.y), c.y);

    int min_x = static_cast<int>(std::floor(min_xf));
    int min_y = static_cast<int>(std::floor(min_yf));
    int max_x = static_cast<int>(std::ceil (max_xf));
    int max_y = static_cast<int>(std::ceil (max_yf));
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x > dst_w) max_x = dst_w;
    if (max_y > dst_h) max_y = dst_h;
    if (min_x >= max_x || min_y >= max_y) return;

    for (int y = min_y; y < max_y; ++y) {
        for (int x = min_x; x < max_x; ++x) {
            const vec2 p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
            const float w0 = edge_fn(b, c, p);
            const float w1 = edge_fn(c, a, p);
            const float w2 = edge_fn(a, b, p);
            const bool inside_ccw = (w0 >= 0.0f) && (w1 >= 0.0f) && (w2 >= 0.0f);
            const bool inside_cw  = (w0 <= 0.0f) && (w1 <= 0.0f) && (w2 <= 0.0f);
            if (inside_ccw || inside_cw) {
                dst[y * dst_w + x] = value;
            }
        }
    }
}

// Transform shape-space [-1, 1]² (apex +Y) → pixel space inside the cell's
// usable rect. V-flip: shape +Y → smaller y_px (top of cell).
inline vec2 shape_to_cell_px(vec2 v, int pad_px, int usable_px) {
    const float u_local = 0.5f * (v.x + 1.0f);   // [0, 1]
    const float v_local = 0.5f * (1.0f - v.y);   // V-flip
    const float fu = static_cast<float>(usable_px);
    const float fp = static_cast<float>(pad_px);
    return vec2{fp + u_local * fu, fp + v_local * fu};
}

}  // namespace

void rasterize_leaf_alpha_into_cell(const LeafShapeMesh& mesh,
                                    uint8_t* alpha,
                                    int cell_w, int cell_h,
                                    int pad_px, int usable_px) {
    (void)cell_h;
    const size_t n_tris = mesh.tris.size() / 3;
    for (size_t t = 0; t < n_tris; ++t) {
        const uint16_t ia = mesh.tris[t * 3 + 0];
        const uint16_t ib = mesh.tris[t * 3 + 1];
        const uint16_t ic = mesh.tris[t * 3 + 2];
        const vec2 a = shape_to_cell_px(mesh.verts[ia], pad_px, usable_px);
        const vec2 b = shape_to_cell_px(mesh.verts[ib], pad_px, usable_px);
        const vec2 c = shape_to_cell_px(mesh.verts[ic], pad_px, usable_px);
        rasterize_tri_impl(alpha, cell_w, cell_h, a, b, c, 255);
    }
}

void rasterize_tri_pixel_space(uint8_t* dst, int dst_w, int dst_h,
                               vec2 a_px, vec2 b_px, vec2 c_px, uint8_t value) {
    rasterize_tri_impl(dst, dst_w, dst_h, a_px, b_px, c_px, value);
}

}  // namespace treegen
