// C4 P3 — wind weight bake (see wind_weights.hpp for the formula).
// /fp:precise (treegen.sharpmake.cs).
#include "wind_weights.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace treegen {

namespace {

inline float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

// Classic smoothstep — matches GLSL smoothstep(edge0, edge1, x).
inline float smoothstep(float edge0, float edge1, float x) {
    const float t = clamp01((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

inline uint8_t quantize(float w) {
    // clamp + nearest-int via +0.5 floor — symmetric across runs under /fp:precise.
    const float c = clamp01(w);
    return static_cast<uint8_t>(c * 255.0f + 0.5f);
}

}  // namespace

std::vector<uint8_t> bake_wind_weights(const TreeSkeleton& skel,
                                       const std::vector<int>& per_vertex_node_index,
                                       const std::vector<float>& per_vertex_world_z,
                                       float tree_height_m) {
    assert(per_vertex_node_index.size() == per_vertex_world_z.size());
    assert(tree_height_m > 0.0f);

    // max_depth — clamp ≥4 so degenerate trees (single depth) don't divide-by-0
    // or saturate depth_norm to 1 everywhere.
    int max_depth = 0;
    for (const auto& n : skel.nodes) {
        if (n.depth > max_depth) max_depth = n.depth;
    }
    if (max_depth < 4) max_depth = 4;
    const float inv_max_depth = 1.0f / static_cast<float>(max_depth);
    const float inv_tree_h    = 1.0f / tree_height_m;

    const size_t vcount = per_vertex_node_index.size();
    std::vector<uint8_t> out(vcount * 4);

    for (size_t i = 0; i < vcount; ++i) {
        const int node_idx = per_vertex_node_index[i];
        assert(node_idx >= 0 && node_idx < static_cast<int>(skel.nodes.size()));

        const int depth = skel.nodes[static_cast<size_t>(node_idx)].depth;
        const float depth_norm = clamp01(static_cast<float>(depth) * inv_max_depth);

        const float world_z = per_vertex_world_z[i];
        const float rooted_factor = smoothstep(0.0f, 0.1f, world_z * inv_tree_h);

        const float one_minus_d = 1.0f - depth_norm;
        const float trunk_w  = one_minus_d * one_minus_d * rooted_factor;
        const float branch_w = 4.0f * depth_norm * one_minus_d;
        const float twig_w   = 4.0f * depth_norm * depth_norm * one_minus_d;
        const float leaf_w   = depth_norm * depth_norm * depth_norm;

        out[i * 4 + 0] = quantize(trunk_w);
        out[i * 4 + 1] = quantize(branch_w);
        out[i * 4 + 2] = quantize(twig_w);
        out[i * 4 + 3] = quantize(leaf_w);
    }

    return out;
}

}  // namespace treegen
