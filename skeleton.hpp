// TreeSkeleton — output of C3 P2 space colonization. POD; consumed by C4
// branch-mesh translator and C5 leaf placement. Node-index 0 is root; all
// positions in world-space (tree base at origin, +Z up — matches envelopes.hpp).
//
// Parent-index convention: BranchNode::parent_index == -1 marks the root;
// every other node has parent_index < own_index (postorder traversal is just
// reverse iteration over the dense node array — required by radius_solver).
#pragma once

#include "vec3.hpp"

#include <cstdint>
#include <vector>

namespace treegen {

struct BranchNode {
    int   parent_index    = -1;       // -1 for root; else < own_index (DAG seal)
    vec3  position        = {0,0,0};  // world-space
    vec3  axis            = {0,0,1};  // grow_dir from parent; root = +Z
    float radius          = 0.0f;     // solved by radius_solver (post-grow)
    int   depth           = 0;        // graph depth from root
    float t_along_parent  = 1.0f;     // attachment param on parent segment (1=tip)
    int   wind_tier       = 0;        // 0..3 — C8/animation hint; derived from depth
};

struct LeafSite {
    // C5 P1 — authoritative leaf position. Earlier we derived position from
    // (branch_id, t_along_branch), but the four placement strategies all
    // need to author free-space points (Halton-sampled shell, fanned cluster
    // offsets from a tip, scattered along upper-order branches). Keep the
    // legacy fields for back-compat but `position` is the load-bearing one.
    vec3  position         = {0,0,0};
    vec3  normal           = {0,0,1}; // outward surface normal
    int   branch_id        = -1;      // nearest skeleton node index
    int   type             = 0;       // species-specific leaf class
    float age              = 0.0f;    // 0..1 — for autumn/seasonal variation
    float t_along_branch   = 0.0f;    // legacy; unused by C5+

    // C1 P1 — BranchWalk strategy enrichment. Populated by strat_branch_walk;
    // zeroed for other strategies (backward compat).
    vec3  branch_tangent        = {0,0,1};  // unit tangent along parent segment at site
    vec3  branch_normal         = {0,0,1};  // radial outward from branch axis toward site
    float branch_depth_fraction = 0.0f;     // node.depth / max_depth, in [0,1]
};

struct TreeSkeleton {
    std::vector<BranchNode> nodes;        // root at index 0
    std::vector<LeafSite>   leaf_sites;   // populated by C5 (empty in C3 P2)
    int                     attractors_consumed = 0;  // stats
    int                     iterations_run      = 0;  // stats
};

} // namespace treegen
