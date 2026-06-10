// Decimated wind rig — bakes the TreeSkeleton as a SMALL bone table for the
// engine's analytic rotational wind FK (tree.vs.glsl).
//
// The growth skeleton has a node every ~growth_distance (0.3 m) of branch
// path — thousands of nodes per tree. Wind bend is a smooth low-frequency
// curve; it only needs joints where the bend visibly articulates. Bone set:
//   * the root,
//   * fork nodes (>= 2 children) with depth <= order_cap AND at least
//     min_fork_spacing_m of arc since the nearest kept ancestor (the spacing
//     gate is what bounds FK chain LENGTH — without it every fork on a limb
//     becomes a chain link and deep crowns ride the shader's ancestor cap),
//   * subdivision joints every <= max_segment_m of arc along runs
//     (including runs that end at a tip, so long unforked limbs still bend).
// Runs pass THROUGH dropped forks: a dropped fork's side branch anchors at
// the same kept joint as the limb it grew from, with its run-arc offset
// carried over (monotone blend; every vertex has exactly one binding, so no
// vertex-level tears). Everything past the last kept joint rides rigidly,
// pinned to its attachment point's exact blend. Typical result: tens of
// bones instead of thousands, FK chains well under the shader cap (16).
//
// Compliance conservation: in the 1:1 rig every original joint j bent by
// theta_j = sway * (k_base + k_slope * depth_j). A decimated bone B whose run
// collapsed original joints J bends by the SUM of its run's contributions:
//   theta_B = sway * (n_B * k_base + k_slope * sum_depth_B)
// with n_B = |J|, sum_depth_B = sum(depth_j). k_base / k_slope stay runtime
// knobs; total root-to-tip bend angle is preserved exactly (all rotations
// share one bend axis, so angles add). Joints tipward of the last kept bone
// (short twig tails, beyond-order_cap subtrees) drop their compliance — they
// ride rigidly; leaf flutter/tumble supplies the motion there.
//
// Bone i record is 12 floats (3 vec4s GPU-side), bone-index order:
//   [ parent_bone_as_float, n_collapsed, sum_depth, 0,
//     pivot.x,    pivot.y,    pivot.z,    0,     // parent kept joint (rotation pivot)
//     node_pos.x, node_pos.y, node_pos.z, 0 ]    // own joint position
// Root: parent = -1.0f, n = 0, sum_depth = 0, pivot = own position (no
// rotation). Bones are kept-node-index ordered, so parent bone < own bone
// (DAG seal). All positions world-space (matches skeleton.hpp).
//
// Vertex remap (consumed by lod_emitter): for every ORIGINAL node the table
// reports its host bone and its arc-length parameter within the host run:
//   node_to_bone[i] — bone whose run contains node i, or (for rigid riders:
//                     tails past the last kept joint, beyond-cap subtrees)
//                     the parent's binding copied verbatim.
//   node_run_t[i]   — arc-length fraction (0,1] from the run's anchor joint
//                     to node i; kept nodes 1.0; riders = parent's value.
//   node_in_run[i]  — 1 when the blend interpolates along a run; riders copy
//                     their parent's flag with seg-t0 pinned to the parent's
//                     end value, so their blend is CONSTANT (the attachment
//                     point's transform — no kink at the attach).
// A vertex at intra-segment fraction t_seg on segment parent(i)->i blends as
//   blend = 1 - lerp(node_seg_t0[i], node_run_t[i], t_seg)
// which the shader's 2-bone mix turns back into smooth curvature along the
// collapsed run. See vertex_parent_blend().
//
// Deterministic / /fp:precise-safe: iterates skel.nodes in index order only.
#pragma once

#include "skeleton.hpp"

#include <cstdint>
#include <vector>

namespace treegen {

inline constexpr int K_FLOATS_PER_BONE = 12;

struct BoneDecimateParams {
    int   order_cap          = 3;     // forks kept only while node.depth <= cap
    float max_segment_m      = 2.0f;  // subdivision arc spacing along runs
    float min_fork_spacing_m = 1.0f;  // min arc between kept joints on a path
};

struct BoneTable {
    std::vector<float> records;       // bone_count * K_FLOATS_PER_BONE floats
    uint32_t           bone_count = 0;

    // Original-node -> decimated-rig remap (sizes == skel.nodes.size()).
    std::vector<uint16_t> node_to_bone;
    std::vector<float>    node_run_t;
    std::vector<uint8_t>  node_in_run;
    // Run-t at the parent end of node i's segment (0 when the parent is the
    // run anchor). Backing data for vertex_parent_blend(); same size as above.
    std::vector<float>    node_seg_t0;

    // Parent-blend byte for a vertex on segment parent(node)->node at
    // intra-segment fraction t_seg in [0,1] (t_seg = 1 at node). 255 = pure
    // parent-bone transform, 0 = pure host. Continuous across kept joints by
    // construction; rigid riders yield their attachment point's constant.
    uint8_t vertex_parent_blend(int node, float t_seg) const;
};

BoneTable build_bone_table(const TreeSkeleton& skel,
                           const BoneDecimateParams& params = {});

}  // namespace treegen
