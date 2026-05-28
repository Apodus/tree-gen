// C4 P4+P5 — per-LOD face budget allocator.
//
// Given a TreeSkeleton + a target triangle budget for one LOD level, computes
// per-order radial segment counts (trunk / order1 / order2 / order3+) that, in
// combination with the per-branch axial-subdivision count baked into
// `build_bark_mesh`, produce a mesh whose triangle total is ≤ budget.
//
// Algorithm (deterministic):
//   1. Estimate the triangle count predicted by the current `target` radial.
//      Each branch's side wall emits N * axial_segs * 2 tris where N is the
//      per-order radial count.
//   2. If estimate > budget, multiply every order's radial by sqrt(budget/est),
//      clamped to ≥3. Re-estimate; iterate up to 3 passes.
//   3. C2-LOD-quality — if still over budget after radial cut: merge shortest
//      branches into their parents. Merged branches suppress bark emission
//      but retain the skeleton node for leaf placement (silent-cull variant).
//      Sort non-trunk nodes ascending by branch length; merge shortest first
//      until estimate fits. Long structural branches survive by construction.
#pragma once

#include "skeleton.hpp"

#include <cstdint>
#include <vector>

namespace treegen {

struct LodBudget {
    // P5 recalibration — collar mesh at forks (P2-P4 Hermite collar rings +
    // crotch cap) replaces the old bipartite stitch, adding ~8x more fork
    // tris per child. 36k accommodates oak's dense fork topology at full
    // radial quality; still trivial on modern GPU (< 1 draw call worth).
    int l0_tris = 36000;
    int l1_tris = 4000;
    int l2_tris = 1500;
    int l3_tris = 4;
};

struct PerOrderRadial {
    int trunk      = 12;  // depth 0
    int order1     = 8;   // depth 1
    int order2     = 6;   // depth 2
    int order3plus = 4;   // depth ≥ 3
};

// Allocate per-order radial segment counts to hit `budget_tris`. Returns the
// adjusted counts. If radial cut alone cannot fit budget, fills
// `out_culled_node_mask` (length == skel.nodes.size()) with 1 at indices to
// merge (shortest branches first). Merged branches suppress bark emission but
// retain the skeleton node for leaf placement — leaves on merged branches
// still appear (the leaf pipeline is independent of the bark merge mask).
// Trunk (depth 0 root) is never merged. Merge-cascade: when all children of
// a fork are merged, the fork's collar/crotch geometry is also suppressed.
//
// When `out_culled_node_mask` is null and the merge rule fires, falls back to
// the legacy per-order cull (drop entire orders) so existing API consumers
// keep working without the mask plumbing.
PerOrderRadial allocate_radial_for_lod(const TreeSkeleton&    skel,
                                       int                    budget_tris,
                                       const PerOrderRadial&  target,
                                       std::vector<uint8_t>*  out_culled_node_mask = nullptr,
                                       int*                   out_culled_lowest_order = nullptr);

}  // namespace treegen
