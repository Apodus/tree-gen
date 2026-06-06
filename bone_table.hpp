// C8-wind P1 — analytic rotational wind data-model foundation.
//
// Bakes the TreeSkeleton as a flat bone table for a future vertex shader that
// ROTATES each vertex about its branch's pivot joint (preserving segment
// length) instead of translating it. This module is data-only — no engine or
// shader changes (the engine-side decode is a later campaign).
//
// Bone i corresponds 1:1 to skeleton node i. `records` is bone_count*8 floats,
// 8 per bone in node-index order:
//   [ parent_index_as_float, depth_as_float,
//     pivot.x, pivot.y, pivot.z,            // node i's PARENT position (the joint)
//     node_pos.x, node_pos.y, node_pos.z ]  // node i's own position
// For the root (parent_index == -1): parent_index = -1.0f, pivot = node.position
// (the root does not rotate). All positions world-space (matches skeleton.hpp).
//
// Deterministic / /fp:precise-safe: iterates skel.nodes in index order, no
// unordered-container traversal.
#pragma once

#include "skeleton.hpp"

#include <cstdint>
#include <vector>

namespace treegen {

inline constexpr int K_FLOATS_PER_BONE = 8;

struct BoneTable {
    std::vector<float> records;       // bone_count * K_FLOATS_PER_BONE floats
    uint32_t           bone_count = 0;
};

BoneTable build_bone_table(const TreeSkeleton& skel);

}  // namespace treegen
