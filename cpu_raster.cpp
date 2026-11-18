#include "cpu_raster.hpp"

#include <algorithm>
#include <cmath>

namespace treegen::cpu_raster {

namespace {

inline float edge_fn(float ax, float ay, float bx, float by, float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

inline void sample(const texture_view& tex, float u, float v, uint8_t out[4]) {
    u -= std::floor(u);
    v -= std::floor(v);
    const int ix = std::clamp(int(u * float(tex.w)), 0, tex.w - 1);
    const int iy = std::clamp(int(v * float(tex.h)), 0, tex.h - 1);
    const uint8_t* src = tex.rgba + (size_t(iy) * tex.w + ix) * 4;
    out[0] = src[0]; out[1] = src[1]; out[2] = src[2]; out[3] = src[3];
}

}  // namespace

target::target(int width, int height)
    : w(width), h(height),
      rgba(size_t(width) * height * 4, 0),
      depth(size_t(width) * height, k_empty_depth) {}

int draw_triangle(target& t, const vertex& a, const vertex& b, const vertex& c,
                  const texture_view& tex, uint8_t alpha_cutoff) {
    const float denom = edge_fn(a.x, a.y, b.x, b.y, c.x, c.y);
    if (std::abs(denom) < 1e-6f) return 0;
    const float inv_denom = 1.0f / denom;
    const bool textured = tex.rgba != nullptr && tex.w > 0 && tex.h > 0;

    const int min_x = std::max(0,   int(std::floor(std::min({ a.x, b.x, c.x }))));
    const int min_y = std::max(0,   int(std::floor(std::min({ a.y, b.y, c.y }))));
    const int max_x = std::min(t.w, int(std::ceil (std::max({ a.x, b.x, c.x }))));
    const int max_y = std::min(t.h, int(std::ceil (std::max({ a.y, b.y, c.y }))));

    int written = 0;
    for (int py = min_y; py < max_y; ++py) {
        for (int px = min_x; px < max_x; ++px) {
            const float cx = float(px) + 0.5f;
            const float cy = float(py) + 0.5f;
            const float w0 = edge_fn(b.x, b.y, c.x, c.y, cx, cy) * inv_denom;
            const float w1 = edge_fn(c.x, c.y, a.x, a.y, cx, cy) * inv_denom;
            const float w2 = 1.0f - w0 - w1;
            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;

            const float d = w0 * a.depth + w1 * b.depth + w2 * c.depth;
            const size_t pi = size_t(py) * t.w + px;
            if (d <= t.depth[pi]) continue;

            uint8_t texel[4] = { 255, 255, 255, 255 };
            if (textured)
                sample(tex, w0 * a.u + w1 * b.u + w2 * c.u, w0 * a.v + w1 * b.v + w2 * c.v, texel);
            if (texel[3] < alpha_cutoff) continue;

            t.depth[pi] = d;
            std::copy(texel, texel + 4, t.rgba.begin() + pi * 4);
            ++written;
        }
    }
    return written;
}

}  // namespace treegen::cpu_raster
