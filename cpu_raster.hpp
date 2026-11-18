// CPU triangle rasteriser: depth-tested, nearest-sampled, alpha-cut RGBA8 target.
// Projection is the caller's: vertices arrive in target pixel space. Offline
// measurement only (no SIMD, no tiling).
#pragma once

#include <cstdint>
#include <vector>

namespace treegen::cpu_raster {

// RGBA8, row-major, top row first. UVs repeat; nearest texel.
struct texture_view {
    const uint8_t* rgba = nullptr;
    int            w    = 0;
    int            h    = 0;
};

struct target {
    int                  w = 0;
    int                  h = 0;
    std::vector<uint8_t> rgba;    // w*h*4; alpha 0 = uncovered
    std::vector<float>   depth;   // larger = nearer; k_empty_depth = uncovered

    static constexpr float k_empty_depth = -1e30f;
    target(int width, int height);
    bool covered(int x, int y) const { return rgba[(size_t(y) * w + x) * 4 + 3] != 0; }
};

// x right, y down, in pixels; depth larger = nearer.
struct vertex {
    float x, y, depth;
    float u, v;
};

// Covers pixel centres inside the triangle (either winding; edges inclusive),
// nearer than the stored depth, whose texel alpha >= alpha_cutoff. An untextured
// draw (tex.rgba == nullptr) writes opaque white. Degenerate triangles draw
// nothing. Returns the number of pixels written.
int draw_triangle(target& t, const vertex& a, const vertex& b, const vertex& c,
                  const texture_view& tex, uint8_t alpha_cutoff);

}  // namespace treegen::cpu_raster
