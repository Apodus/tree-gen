// C9 P3 -- see impostor_bake.hpp.
#include "impostor_bake.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace treegen {

namespace {

constexpr int k_azimuths = 8;
constexpr float k_pi = 3.14159265358979323846f;

struct vec3 { float x, y, z; };
struct vec2 { float x, y; };

// Orthographic projection: rotate world point around Z by -azimuth,
// then map the resulting (x, z) into the cell pixel rect.
// Returns pixel-space (px, py) and depth for Z-buffer.
struct projected {
    float px, py, depth;
};

projected ortho_project(vec3 p, float cos_a, float sin_a,
                        float x_min, float x_range,
                        float z_min, float z_range,
                        int cell_w, int cell_h) {
    // Rotate by -azimuth around Z to get view-local coordinates.
    // view_x = screen horizontal, view_y = depth, view_z = screen vertical.
    float vx = p.x * cos_a + p.y * sin_a;
    float vy = -p.x * sin_a + p.y * cos_a;
    float vz = p.z;

    // Map vx -> [0, cell_w), vz -> [0, cell_h). V-flip: larger z = top.
    float u = (vx - x_min) / x_range;
    float v = 1.0f - (vz - z_min) / z_range;
    return { u * float(cell_w), v * float(cell_h), vy };
}

// Edge function for triangle rasterization.
inline float edge_fn(float ax, float ay, float bx, float by, float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// Sample RGBA8 texture with UV wrapping (repeat mode for bark).
inline void sample_tex(const uint8_t* rgba, int w, int h,
                       float u, float v, uint8_t out[4]) {
    // Wrap UVs.
    u = u - std::floor(u);
    v = v - std::floor(v);
    int ix = std::clamp(int(u * float(w)), 0, w - 1);
    int iy = std::clamp(int(v * float(h)), 0, h - 1);
    const uint8_t* src = rgba + (iy * w + ix) * 4;
    out[0] = src[0]; out[1] = src[1]; out[2] = src[2]; out[3] = src[3];
}

// Rasterize one triangle with barycentric UV interpolation + texture sampling.
// Writes into atlas_rgba at cell offset. Z-buffer is per-cell.
void rasterize_textured_tri(
    uint8_t* atlas_rgba, float* zbuf,
    int atlas_w, int cell_x0, int cell_w, int cell_h,
    float ax, float ay, float az,  // projected vertices
    float bx, float by, float bz,
    float cx, float cy, float cz,
    float ua, float va,            // UV at each vertex
    float ub, float vb,
    float uc, float vc,
    const uint8_t* tex_rgba, int tex_w, int tex_h)
{
    int min_x = std::max(0, int(std::floor(std::min({ax, bx, cx}))));
    int min_y = std::max(0, int(std::floor(std::min({ay, by, cy}))));
    int max_x = std::min(cell_w, int(std::ceil(std::max({ax, bx, cx}))));
    int max_y = std::min(cell_h, int(std::ceil(std::max({ay, by, cy}))));

    float denom = edge_fn(ax, ay, bx, by, cx, cy);
    if (std::abs(denom) < 1e-6f) return;
    float inv_denom = 1.0f / denom;

    for (int py = min_y; py < max_y; ++py) {
        for (int px = min_x; px < max_x; ++px) {
            float ppx = float(px) + 0.5f;
            float ppy = float(py) + 0.5f;

            float w0 = edge_fn(bx, by, cx, cy, ppx, ppy) * inv_denom;
            float w1 = edge_fn(cx, cy, ax, ay, ppx, ppy) * inv_denom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;

            float depth = w0 * az + w1 * bz + w2 * cz;
            int zbuf_idx = py * cell_w + px;
            if (depth <= zbuf[zbuf_idx]) continue;  // behind existing
            zbuf[zbuf_idx] = depth;

            // Interpolate UV.
            float su = w0 * ua + w1 * ub + w2 * uc;
            float sv = w0 * va + w1 * vb + w2 * vc;

            uint8_t texel[4];
            if (tex_rgba && tex_w > 0 && tex_h > 0) {
                sample_tex(tex_rgba, tex_w, tex_h, su, sv, texel);
            } else {
                texel[0] = 180; texel[1] = 140; texel[2] = 100; texel[3] = 255;
            }

            // Alpha discard for leaf cards.
            if (texel[3] < 128) continue;

            int atlas_x = cell_x0 + px;
            int atlas_idx = (py * atlas_w + atlas_x) * 4;
            atlas_rgba[atlas_idx + 0] = texel[0];
            atlas_rgba[atlas_idx + 1] = texel[1];
            atlas_rgba[atlas_idx + 2] = texel[2];
            atlas_rgba[atlas_idx + 3] = texel[3];
        }
    }
}

// Compute AABB of positions array.
void compute_aabb(const float* pos, int vcount, float out_min[3], float out_max[3]) {
    out_min[0] = out_min[1] = out_min[2] = 1e30f;
    out_max[0] = out_max[1] = out_max[2] = -1e30f;
    for (int i = 0; i < vcount; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = pos[i * 3 + c];
            if (v < out_min[c]) out_min[c] = v;
            if (v > out_max[c]) out_max[c] = v;
        }
    }
}

}  // namespace

ImpostorAtlas bake_impostor_atlas(const ImpostorInput& input) {
    ImpostorAtlas atlas;
    const int cell_w = atlas.width / k_azimuths;   // 256
    const int cell_h = atlas.height;                // 1024

    atlas.rgba.resize(size_t(atlas.width) * atlas.height * 4, 0);

    // Compute AABB from all input geometry (bark + leaves).
    float aabb_min[3], aabb_max[3];
    if (input.aabb_max[0] > input.aabb_min[0]) {
        std::memcpy(aabb_min, input.aabb_min, sizeof(aabb_min));
        std::memcpy(aabb_max, input.aabb_max, sizeof(aabb_max));
    } else {
        compute_aabb(input.positions, input.vertex_count, aabb_min, aabb_max);
        if (input.leaf_positions && input.leaf_vertex_count > 0) {
            float lmin[3], lmax[3];
            compute_aabb(input.leaf_positions, input.leaf_vertex_count, lmin, lmax);
            for (int c = 0; c < 3; ++c) {
                aabb_min[c] = std::min(aabb_min[c], lmin[c]);
                aabb_max[c] = std::max(aabb_max[c], lmax[c]);
            }
        }
    }

    // Projection extents. For each azimuth, the horizontal extent is the
    // maximum XY diagonal of the AABB (conservative), vertical = Z range.
    float dx = aabb_max[0] - aabb_min[0];
    float dy = aabb_max[1] - aabb_min[1];
    float diag_xy = std::sqrt(dx * dx + dy * dy);
    float z_range = aabb_max[2] - aabb_min[2];
    if (diag_xy < 0.001f) diag_xy = 1.0f;
    if (z_range < 0.001f) z_range = 1.0f;

    // Pad 5% on each side.
    float pad = 0.05f;
    float x_min = -diag_xy * (0.5f + pad);
    float x_max =  diag_xy * (0.5f + pad);
    float z_min = aabb_min[2] - z_range * pad;
    float z_max = aabb_max[2] + z_range * pad;
    float x_range = x_max - x_min;
    float z_range_padded = z_max - z_min;

    for (int ai = 0; ai < k_azimuths; ++ai) {
        float angle = float(ai) * (2.0f * k_pi / float(k_azimuths));
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        int cell_x0 = ai * cell_w;

        // Per-cell Z-buffer.
        std::vector<float> zbuf(size_t(cell_w) * cell_h, -1e30f);

        // Lambda: rasterize a mesh's triangles.
        auto rasterize_mesh = [&](const float* pos, const float* uvs,
                                  const uint32_t* idx, int tri_count,
                                  const uint8_t* tex, int tw, int th) {
            for (int t = 0; t < tri_count; ++t) {
                uint32_t ia = idx[t * 3 + 0];
                uint32_t ib = idx[t * 3 + 1];
                uint32_t ic = idx[t * 3 + 2];

                vec3 pa{ pos[ia*3+0], pos[ia*3+1], pos[ia*3+2] };
                vec3 pb{ pos[ib*3+0], pos[ib*3+1], pos[ib*3+2] };
                vec3 pc{ pos[ic*3+0], pos[ic*3+1], pos[ic*3+2] };

                auto a = ortho_project(pa, cos_a, sin_a, x_min, x_range, z_min, z_range_padded, cell_w, cell_h);
                auto b = ortho_project(pb, cos_a, sin_a, x_min, x_range, z_min, z_range_padded, cell_w, cell_h);
                auto c = ortho_project(pc, cos_a, sin_a, x_min, x_range, z_min, z_range_padded, cell_w, cell_h);

                float ua_ = uvs ? uvs[ia*2+0] : 0.0f;
                float va_ = uvs ? uvs[ia*2+1] : 0.0f;
                float ub_ = uvs ? uvs[ib*2+0] : 0.0f;
                float vb_ = uvs ? uvs[ib*2+1] : 0.0f;
                float uc_ = uvs ? uvs[ic*2+0] : 0.0f;
                float vc_ = uvs ? uvs[ic*2+1] : 0.0f;

                rasterize_textured_tri(
                    atlas.rgba.data(), zbuf.data(),
                    atlas.width, cell_x0, cell_w, cell_h,
                    a.px, a.py, a.depth,
                    b.px, b.py, b.depth,
                    c.px, c.py, c.depth,
                    ua_, va_, ub_, vb_, uc_, vc_,
                    tex, tw, th);
            }
        };

        // Bark mesh.
        if (input.positions && input.index_count > 0) {
            rasterize_mesh(input.positions, input.uvs,
                           input.indices, input.index_count / 3,
                           input.bark_rgba, input.bark_w, input.bark_h);
        }

        // Leaf mesh (separate primitive).
        if (input.leaf_positions && input.leaf_index_count > 0) {
            rasterize_mesh(input.leaf_positions, input.leaf_uvs,
                           input.leaf_indices, input.leaf_index_count / 3,
                           input.leaf_rgba, input.leaf_w, input.leaf_h);
        }
    }

    return atlas;
}

}  // namespace treegen
