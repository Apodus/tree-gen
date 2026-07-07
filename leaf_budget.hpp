// C5 P3 — per-LOD leaf budget + chain-min monotonicity seal.
//
// Sub-samples a flat LeafSite vector into per-LOD kept-index lists with a
// triangle-count downgrade ladder (ProceduralVeined → BentCrossCluster(N) →
// BentCrossCluster(1) → BentCard → SingleCard) when the budget cannot fit the
// requested fraction at the requested geometry type. Chain-min monotonicity
// (kept_L+1 ⊆ kept_L) is enforced *inside* the single public entry point so
// no caller can construct a chain-violating sequence by mis-ordering N
// independent per-LOD calls.
#pragma once

#include "leaf_geometry.hpp"
#include "skeleton.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace treegen {

struct LeafBudget {
    // C3 P3 retune — crossed cluster cards (N cards/site, 8 tris/card) carry the
    // canopy silhouette now (bark is dieted), and this is a TOP-DOWN game where
    // the far L2 pose IS the primary player view, so L2 leaves must stay dense.
    // Budgets raised to give the allocator room for the crossed cards; total L0
    // (bark ~9.5k + leaf ≤8k) still lands ~17k, half the original ~36k.
    int l0_tris = 8000;
    int l1_tris = 5000;
    int l2_tris = 4000;
    int l3_tris = 0;
};

struct LeafBudgetResult {
    std::vector<uint32_t> kept_indices;     // indices into the input sites vector
    LeafGeometryType      emitted_type      = LeafGeometryType::BentCrossCluster;
    // Effective cluster count after the downgrade ladder (only meaningful for
    // BentCrossCluster — the ladder may collapse N->1). Emitters MUST honour
    // this rather than the caller's original cluster_count_per_tip, else they
    // will exceed `estimated_tris`.
    int                   effective_cluster = 1;
    int                   estimated_tris    = 0; // exact (count * tris_per_leaf), <= budget
};

// Single entry point. Internally enforces chain-min monotonicity:
//   keep_L3 ⊆ keep_L2 ⊆ keep_L1 ⊆ keep_L0 ⊆ sites
//   estimated_tris_{L+1} ≤ estimated_tris_L
// Per-LOD allocator stays private in .cpp anon namespace; caller cannot
// construct a chain-violating call sequence via the public API.
std::array<LeafBudgetResult, 4> allocate_leaves_all_lods(
    const std::vector<LeafSite>& sites,
    LeafGeometryType             starting_type,
    LeafShape                    shape,
    int                          cluster_count_per_tip,
    const LeafBudget&            budget,
    uint64_t                     seed_effective);

// C3-needle-strips: segment-based strip budget. Separate from site-based
// leaf allocation. Segments ranked by depth (deeper twigs culled first).
struct StripBudget {
    int l0_strips = 600;
    int l1_strips = 200;
    int l2_strips = 60;
    int l3_strips = 0;
};

struct StripBudgetResult {
    int strips_allocated  = 0;
    int max_strips_per_seg = 3;
    float min_radius_threshold = 0.02f;
};

std::array<StripBudgetResult, 4> allocate_strips_all_lods(
    const TreeSkeleton& skel,
    int                 min_branch_depth,
    const StripBudget&  budget);

}  // namespace treegen
