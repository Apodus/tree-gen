// /fp:precise required for determinism — see treegen.sharpmake.cs. PCG32 +
// integer arithmetic only; no FP used in the allocator math itself, but the
// shared code is built under the same flags.
#include "leaf_budget.hpp"

#include "det_rng.hpp"
#include "leaf_geometry.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace treegen {

namespace {

// Per-stream PCG32 stream id. Matches the per-stream-id convention
// (cf. k_stream_leaf_geom = 0xC5'02'01ULL in leaf_geometry.cpp).
constexpr uint64_t k_stream_leaf_subsample = 0xC5'03'01ULL;

// Per-LOD subsample fractions (matches plan / vision targets).
constexpr float k_lod_subsample_fraction[4] = { 1.00f, 0.30f, 0.10f, 0.00f };

int budget_for_lod(const LeafBudget& b, int lod_index) {
    switch (lod_index) {
        case 0: return b.l0_tris;
        case 1: return b.l1_tris;
        case 2: return b.l2_tris;
        default: return b.l3_tris;
    }
}

// Stable hash-based sort: produce a deterministic permutation of indices
// [0..N) ordered by PCG32 hash (rank_key) per site index. Tiebreak by index
// (PCG32 collisions are 2^-32 rare but the seal is free).
std::vector<uint32_t> sort_sites_by_rank(size_t n, uint64_t seed_effective) {
    std::vector<std::pair<uint32_t, uint32_t>> keyed;  // (rank_key, idx)
    keyed.reserve(n);

    pcg32 rng;
    const uint64_t stream = seed_effective ^ k_stream_leaf_subsample;
    for (size_t i = 0; i < n; ++i) {
        // Re-seed per index so the rank_key is a pure function of (seed, idx)
        // and independent of iteration order or n.
        rng.seed(static_cast<uint64_t>(i), stream);
        const uint32_t rank_key = rng.next_u32();
        keyed.emplace_back(rank_key, static_cast<uint32_t>(i));
    }

    std::sort(keyed.begin(), keyed.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;  // tiebreak by index
              });

    std::vector<uint32_t> sorted_indices;
    sorted_indices.reserve(n);
    for (const auto& k : keyed) sorted_indices.push_back(k.second);
    return sorted_indices;
}

}  // anonymous namespace

std::array<LeafBudgetResult, 4> allocate_leaves_all_lods(
    const std::vector<LeafSite>& sites,
    LeafGeometryType             starting_type,
    LeafShape                    shape,
    int                          cluster_count_per_tip,
    const LeafBudget&            budget,
    uint64_t                     seed_effective)
{
    std::array<LeafBudgetResult, 4> out;
    const int N = static_cast<int>(sites.size());

    // Pre-compute the global rank permutation; per-LOD prefix-take honours
    // both subsample fractions and chain-min monotonicity.
    const std::vector<uint32_t> sorted_indices =
        sort_sites_by_rank(static_cast<size_t>(N), seed_effective);

    int prev_K = N;  // chain-min cap from previous LOD; L0 starts unbounded.

    // C3-LOD-quality: geometry type is frozen across all LODs (no downgrade
    // ladder). Budget compliance comes purely from count reduction via cap_L.
    const int frozen_cluster = cluster_count_per_tip > 0 ? cluster_count_per_tip : 1;
    const int frozen_tpl     = tris_per_leaf(starting_type, shape, frozen_cluster);
    const int tpl_safe       = frozen_tpl > 0 ? frozen_tpl : K_TRIS_PER_LEAF_SINGLE_CARD;

    for (int L = 0; L < 4; ++L) {
        const int budget_L = budget_for_lod(budget, L);
        const int raw_K_L  = static_cast<int>(k_lod_subsample_fraction[L] * static_cast<float>(N));

        const int cap_L = budget_L / tpl_safe;  // integer division — pessimistic, never over-budget
        int K_L = std::min({ prev_K, raw_K_L, cap_L });
        if (K_L < 0) K_L = 0;

        LeafBudgetResult& r = out[static_cast<size_t>(L)];
        r.kept_indices.assign(sorted_indices.begin(),
                              sorted_indices.begin() + K_L);
        r.emitted_type      = starting_type;
        r.effective_cluster = frozen_cluster;
        r.estimated_tris    = K_L * tpl_safe;

        // Fail early (CLAUDE.md) — chain-min + budget invariants are structural,
        // not runtime-best-effort.
        assert(r.estimated_tris <= budget_L);
        assert(K_L <= prev_K);

        prev_K = K_L;
    }

    return out;
}

std::array<StripBudgetResult, 4> allocate_strips_all_lods(
    const TreeSkeleton& skel,
    int                 min_branch_depth,
    const StripBudget&  budget)
{
    // Count qualifying segments.
    int total_segs = 0;
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const auto& n = skel.nodes[i];
        if (n.depth < min_branch_depth) continue;
        if (n.parent_index < 0) continue;
        ++total_segs;
    }

    constexpr int k_lod_strip_budget[4] = { 600, 200, 60, 0 };
    std::array<StripBudgetResult, 4> out;

    for (int L = 0; L < 4; ++L) {
        int budget_L = k_lod_strip_budget[L];
        switch (L) {
            case 0: budget_L = budget.l0_strips; break;
            case 1: budget_L = budget.l1_strips; break;
            case 2: budget_L = budget.l2_strips; break;
            default: budget_L = budget.l3_strips; break;
        }

        StripBudgetResult& r = out[static_cast<size_t>(L)];
        r.strips_allocated = std::min(total_segs, budget_L);

        if (L == 0)      { r.max_strips_per_seg = 3; r.min_radius_threshold = 0.02f; }
        else if (L == 1) { r.max_strips_per_seg = 1; r.min_radius_threshold = 0.03f; }
        else if (L == 2) { r.max_strips_per_seg = 1; r.min_radius_threshold = 0.05f; }
        else             { r.max_strips_per_seg = 0; r.min_radius_threshold = 1.0f;  }
    }

    return out;
}

}  // namespace treegen
