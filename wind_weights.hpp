// C4 P3 — per-vertex wind tier weight bake for the `_RYNX_WIND` GLB extension.
//
// Per vertex emits 4 independent normalized bytes (trunk_w, branch_w, twig_w,
// leaf_w) — these do NOT sum to 255; each is an independent wind influence in
// [0,1] quantized via round-half-to-even-style nearest-int (clamp(w)*255+0.5).
//
// Derivation:
//   depth_norm   = skel.nodes[node_index].depth / max(max_depth, 4)
//   rooted_factor = smoothstep(0, 0.1, world_z / tree_height_m)
//   trunk_w  = (1 - depth_norm)^2 * rooted_factor
//   branch_w = 4 * depth_norm * (1 - depth_norm)        // peaks at d=0.5
//   twig_w   = 4 * depth_norm^2 * (1 - depth_norm)      // peaks at d≈0.67
//   leaf_w   = depth_norm^3
//
// max_depth clamped to ≥4 (verifier r2) to prevent degenerate single-depth
// trees blowing up depth_norm.
//
// Rooted invariant: any vertex with world_z < 0.001 * tree_height has
// trunk_byte == 0 (smoothstep returns 0 below the lower edge).
//
// /fp:precise (treegen.sharpmake.cs) → byte-identical across runs.
#pragma once

#include "skeleton.hpp"

#include <cstdint>
#include <vector>

namespace treegen {

// 4 bytes per vertex; output.size() == per_vertex_node_index.size() * 4.
// Packed (trunk, branch, twig, leaf) per vertex.
std::vector<uint8_t> bake_wind_weights(const TreeSkeleton& skel,
                                       const std::vector<int>& per_vertex_node_index,
                                       const std::vector<float>& per_vertex_world_z,
                                       float tree_height_m);

}  // namespace treegen
