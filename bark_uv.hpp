// Cylindrical bark UV mapping — header-only. C4 P1.
//
// Seam-column duplicate convention: a ring of N angular segments emits (N+1)
// verts where the (N+1)-th duplicates vert 0 angularly but holds u=1 instead
// of wrapping to 0 — gives a continuous [seam_offset, seam_offset+1] U range
// the texture sampler can interpolate across without a discontinuity.
//
// V is axial: caller supplies v_axial = axial_distance_m / bark_repeat_m.
// /fp:precise consumer (see treegen.sharpmake.cs) — keep the math here pure.
#pragma once

#include <utility>

namespace treegen::bark_uv {

inline std::pair<float, float> ring_uv(int i, int N, float v_axial, float seam_offset_rad) {
    constexpr float k_inv_2pi = 0.15915494309189535f; // 1 / (2π)
    float u = static_cast<float>(i) / static_cast<float>(N) + seam_offset_rad * k_inv_2pi;
    return {u, v_axial};
}

} // namespace treegen::bark_uv
