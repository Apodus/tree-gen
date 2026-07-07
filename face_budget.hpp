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
#include "tree_descriptor.hpp"   // LeafShape (per-species L0 budget)

#include <cstdint>
#include <vector>

namespace treegen {

struct LodBudget {
    // C3 P2 bark diet — the canopy rebalance moves silhouette work from bark to
    // multi-leaf cluster cards, so the trunk no longer needs full radial detail.
    // L0 10k holds oak's fork topology at reduced radial; meso-detail rides the
    // bark NORMAL map (main.cpp idx1, tree.fs.glsl). Net L0 (bark+leaf) ~40k→~17k.
    // L1 stays at the original 4000 (plan wanted 3000): pine's tall conical bark
    // tracks the budget, and the tri estimator under-counts pine's fork-dense
    // collar at L1 (actual > budget) while OVER-counting it at L0 — opposite-
    // signed, so no single safety margin fits both without starving pine's L0
    // retain floor. L1 was never the fat; the diet is L0 (72% off) + L2 (20% off).
    // L0 is set PER-SPECIES via lod_budget_for_species (below); this default is
    // oak's aggressive diet, shared by pine (also achievable).
    int l0_tris = 10000;
    int l1_tris = 4000;
    int l2_tris = 1200;
    int l3_tris = 4;
};

// Per-species L0 budget. The tri estimator's error is species-dependent (it
// under-counts fork-dense BROADLEAF collar in the light-cut regime, over-counts
// pine), so no single L0 works. The pre-C3 36k cap only ever bound OAK (ref
// ~42k) — birch (~17k) / maple (~19k) natural meshes were always under it, i.e.
// were never the fat; and the estimator can't cut them below natural without
// overshooting the ceiling. So: oak+pine take the achievable 10k diet; birch/
// maple keep an effectively-uncapped L0 (≈ their natural, unchanged from pre-C3).
// L1/L2/L3 stay shared (every species already complies). See face_budget.cpp.
LodBudget lod_budget_for_species(LeafShape species);

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
