// C4 P4+P5 — see face_budget.hpp for algorithm contract.
#include "face_budget.hpp"

#include "vec3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace treegen {

LodBudget lod_budget_for_species(LeafShape species) {
    LodBudget b;  // default = oak/pine diet (l0=10000) + shared l1/l2/l3
    switch (species) {
        // Broadleaf whose natural L0 sits below the estimator's cuttable floor:
        // keep effectively uncapped (≈ natural). Ceilings comfortably above the
        // observed natural (~17k / ~19k) so the allocator leaves them uncut.
        case LeafShape::BirchSerrated: b.l0_tris = 20000; break;
        case LeafShape::MapleStar:     b.l0_tris = 24000; break;
        // OakLobed (ref ~42k → dieted to ~9.5k) + PineNeedle (ref ~11k → ~8.8k)
        // both hit the 10k diet cleanly.
        default: break;
    }
    return b;
}

namespace {

// Default axial segment length pin — matches BarkMeshOptions::axial_segment_length_m.
// Kept as a constant here (not parameterized) because radial varies per LOD;
// axial stays fixed for the budget estimate.
constexpr float k_axial_segment_length_m = 0.5f;

int radial_for_depth(int depth, const PerOrderRadial& r) {
    if (depth == 0) return r.trunk;
    if (depth == 1) return r.order1;
    if (depth == 2) return r.order2;
    return r.order3plus;
}

// Compute branch length (parent → node) for `node_index`. Returns 0 for root.
float branch_length(const TreeSkeleton& skel, size_t node_index) {
    const auto& node = skel.nodes[node_index];
    if (node.parent_index < 0) return 0.0f;
    const auto& parent = skel.nodes[size_t(node.parent_index)];
    return length(node.position - parent.position);
}

// Estimate triangle count for `skel` rendered with the given radial counts +
// optional per-node cull mask. Mirrors:
//   1) branch_mesh.cpp's per-segment cylinder:
//        tube_tris = max(1, floor(seg_len / axial_seg_len)) * N * 2
//   2) fork_blend.cpp's per-fork collar mesh (P2-P4 replacement for the
//      old bipartite sector stitch). Per surviving child at a fork:
//        - N_collar intermediate rings connected by quad strips: each of
//          N_collar strips emits N_p * 2 tris (inter-collar + collar-to-child).
//        - Parent sector → first collar ring bipartite stitch: ≈ 2*N_p - 2 tris
//          (sector ≈ full ring approximation for budget convergence).
//      Per fork: crotch cap ≈ N_kids tris (1 triangle per adjacent pair).
//      N_collar is angle-adaptive (2-5); k_n_collar_est=3 covers the mean.
// `culled` (when non-empty) is length skel.nodes.size(); 1 means "merged —
// skip bark emission, retain for leaf placement". Mirroring both
// contributions matters: underestimate → merge_shortest_branches stops too
// early → actual mesh overshoots the budget by the un-counted stitches.
int estimate_tris(const TreeSkeleton&         skel,
                  const PerOrderRadial&       r,
                  const std::vector<uint8_t>& culled)
{
    long long tris = 0;

    // (1) Tube tris.
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        if (!culled.empty() && culled[i]) continue;
        const auto& node = skel.nodes[i];
        if (node.parent_index < 0) continue;
        const float seg_len = branch_length(skel, i);
        if (seg_len < 1e-5f) continue;

        int N = radial_for_depth(node.depth, r);
        if (N == 0) continue;  // legacy per-order cull
        if (N < 3) N = 3;       // build_bark_mesh floor

        const int axial_segs = std::max(1, int(std::floor(seg_len / k_axial_segment_length_m)));
        tris += static_cast<long long>(axial_segs) * static_cast<long long>(N) * 2LL;
    }

    // (2) Collar mesh tris (replaces old fork-stitch tris).
    // Per surviving child at a fork with sufficient sector allocation:
    //   - (n_collar-1) inter-collar quad strips, each N_p*2 tris
    //   - 1 last-collar → child s=0 connection: strip N_p*2 or bipartite N_p+N_c-2
    //   - sector stitch: parent sector → first collar ring (bipartite)
    // Per fork: crotch cap ≈ N_kids tris.
    // Children whose angular sector would be < 2 parent verts receive no collar
    // geometry (extract_sector returns empty → apply_skin_rim_blend skips them).
    constexpr int k_n_collar_est = 2; // median collar ring count (actual: 2-5)
    std::vector<std::vector<int>> children(skel.nodes.size());
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const int p = skel.nodes[i].parent_index;
        if (p >= 0) children[size_t(p)].push_back(int(i));
    }
    for (size_t F = 0; F < skel.nodes.size(); ++F) {
        if (children[F].size() < 2u) continue;                  // not a fork zone
        if (skel.nodes[F].parent_index < 0) continue;           // root fork has no parent rim — skipped in apply_skin_rim_blend
        if (!culled.empty() && culled[F]) continue;             // fork node itself culled → no parent rim

        int N_p = radial_for_depth(skel.nodes[F].depth, r);
        if (N_p == 0) continue;
        if (N_p < 3) N_p = 3;

        // Count surviving children to estimate sector sizes.
        int num_survivors = 0;
        for (int ci : children[F]) {
            if (!culled.empty() && culled[size_t(ci)]) continue;
            int N_c = radial_for_depth(skel.nodes[size_t(ci)].depth, r);
            if (N_c == 0) continue;
            ++num_survivors;
        }
        if (num_survivors == 0) continue;

        // Average sector size: N_p / num_survivors. Children whose sector
        // would be < 2 verts get no collar geometry in the actual mesh.
        const int avg_sector = N_p / num_survivors;
        const bool sectors_viable = (avg_sector >= 2);

        int collar_children = 0;
        for (int ci : children[F]) {
            if (!culled.empty() && culled[size_t(ci)]) continue;
            int N_c = radial_for_depth(skel.nodes[size_t(ci)].depth, r);
            if (N_c == 0) continue;
            if (N_c < 3) N_c = 3;

            if (sectors_viable) {
                ++collar_children;
                // Inter-collar quad strips: (n_collar-1) × N_p × 2
                tris += static_cast<long long>(k_n_collar_est - 1) * static_cast<long long>(N_p) * 2LL;
                // Last-collar → child: strip when N matches, bipartite otherwise.
                if (N_c == N_p)
                    tris += static_cast<long long>(N_p) * 2LL;
                else
                    tris += static_cast<long long>(N_p) + static_cast<long long>(N_c) - 2LL;
            }
        }
        if (collar_children > 0) {
            // Sector stitch (parent rim → first collar ring): bipartite sum
            // Σ(sector_k + N_p - 2) = N_p + collar_children*(N_p - 2)
            tris += static_cast<long long>(N_p) + static_cast<long long>(collar_children) * (static_cast<long long>(N_p) - 2LL);
            // Crotch cap: ~collar_children tris
            tris += static_cast<long long>(collar_children);
        }
    }

    // (3) C12 P3 — tip end-cap tris: N triangles per leaf node (fan to centroid).
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        if (!culled.empty() && culled[i]) continue;
        if (skel.nodes[i].parent_index < 0) continue;
        // Count children that survive culling.
        int nc = 0;
        for (int ci : children[i]) {
            if (!culled.empty() && culled[size_t(ci)]) continue;
            ++nc;
        }
        if (nc > 0) continue; // not a leaf
        int N = radial_for_depth(skel.nodes[i].depth, r);
        if (N == 0) continue;
        if (N < 3) N = 3;
        tris += N;
    }

    // Conservative 10% safety margin — the collar mesh estimator systematically
    // underestimates actual tri counts (angle-adaptive collar ring counts,
    // bipartite stitch geometry, etc.). Overestimating is safe (allocator picks
    // slightly lower radials → actual mesh guaranteed under budget).
    tris = tris + tris / 10;
    return tris > std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : int(tris);
}

void scale_radial(PerOrderRadial& r, float k) {
    auto apply = [&](int& v) {
        if (v == 0) return;  // already culled (legacy)
        const int scaled = int(std::floor(float(v) * k + 0.5f));
        v = std::max(3, scaled);  // floor matches build_bark_mesh clamp
    };
    apply(r.trunk);
    apply(r.order1);
    apply(r.order2);
    apply(r.order3plus);
}

// C2-LOD-quality — merge shortest branches into their parents until tri
// estimate fits budget. "Merge" = suppress bark emission for the branch
// (out_mask[i] = 1) but retain the skeleton node for leaf placement. Parent
// radius is unchanged (silent-cull variant: at L1/L2 distances the silhouette
// difference from area-conserving merge is imperceptible).
//
// Merge-cascade: when all children of a fork are merged, the fork is no
// longer a fork — no collar mesh or crotch cap is emitted for it. Both
// build_bark_mesh (zone pruning) and estimate_tris (num_survivors==0 skip)
// already handle this structurally.
//
// Trunk (depth 0 root) is never merged. `out_mask` length ==
// skel.nodes.size(); 1 marks "merged — skip bark, keep for leaves".
//
// Algorithm: collect (length, node_index) for all non-trunk, non-degenerate
// nodes; sort ascending by length; merge from the front until estimate ≤
// budget. Re-estimate every K=32 merges (32 short twigs ≈ 192 tris).
void merge_shortest_branches(const TreeSkeleton&    skel,
                             const PerOrderRadial&  r,
                             int                    budget_tris,
                             std::vector<uint8_t>&  out_mask)
{
    out_mask.assign(skel.nodes.size(), 0u);

    struct Entry { float length; size_t index; };
    std::vector<Entry> entries;
    entries.reserve(skel.nodes.size());
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const auto& node = skel.nodes[i];
        if (node.parent_index < 0) continue;  // root always survives
        // C11: depth-inflation fix extended depth-0 into the crown (central
        // leader). Below crown base the trunk chain is structural and long
        // (won't be culled first anyway). Above, leader segments are cullable.
        const float len = branch_length(skel, i);
        if (len < 1e-5f) continue;
        entries.push_back({len, i});
    }
    // Stable ascending sort by length, tiebreak by index — deterministic.
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.length != b.length) return a.length < b.length;
        return a.index < b.index;
    });

    constexpr int k_recheck_interval = 32;
    int merged_since_check = 0;
    int last_est = estimate_tris(skel, r, out_mask);
    if (last_est <= budget_tris) return;

    for (size_t k = 0; k < entries.size(); ++k) {
        out_mask[entries[k].index] = 1u;  // merged: no bark, keep for leaves
        ++merged_since_check;
        if (merged_since_check >= k_recheck_interval) {
            merged_since_check = 0;
            last_est = estimate_tris(skel, r, out_mask);
            if (last_est <= budget_tris) return;
        }
    }
    // Final check covers the tail.
    last_est = estimate_tris(skel, r, out_mask);
    (void)last_est;  // best-effort; if still over budget the trunk dominates
                    // and there's nothing more we can merge.
}

} // anonymous namespace

PerOrderRadial allocate_radial_for_lod(const TreeSkeleton&    skel,
                                       int                    budget_tris,
                                       const PerOrderRadial&  target,
                                       std::vector<uint8_t>*  out_culled_node_mask,
                                       int*                   out_culled_lowest_order)
{
    PerOrderRadial r = target;
    if (out_culled_lowest_order) *out_culled_lowest_order = -1;
    if (out_culled_node_mask) out_culled_node_mask->clear();

    std::vector<uint8_t> empty_mask;

    // Pass 1+2+3: proportional cut — scale every order by sqrt(budget/estimate).
    // Iterating 3 passes lets the N≥3 floor's nonlinearity converge.
    for (int pass = 0; pass < 3; ++pass) {
        const int est = estimate_tris(skel, r, empty_mask);
        if (est <= budget_tris) return r;
        const float ratio = float(budget_tris) / float(est);
        if (ratio >= 1.0f) return r;
        const float k = std::sqrt(ratio);
        scale_radial(r, k);
    }

    // Still over budget. Preferred path: merge-by-length — caller passed
    // a mask buffer; fill it with shortest-first merges. Merged branches
    // suppress bark but retain skeleton nodes for leaf placement.
    if (out_culled_node_mask) {
        merge_shortest_branches(skel, r, budget_tris, *out_culled_node_mask);
        return r;
    }

    // Fallback path (legacy P4): no mask buffer → cull-by-order. Drop
    // order3+ first; if still over, order2; then order1. Trunk never culled.
    int culled = -1;
    int* depth_targets[3] = { &r.order3plus, &r.order2, &r.order1 };
    int  depth_markers[3] = { 3, 2, 1 };
    for (int i = 0; i < 3; ++i) {
        if (estimate_tris(skel, r, empty_mask) <= budget_tris) break;
        if (*depth_targets[i] != 0) {
            *depth_targets[i] = 0;
            culled = depth_markers[i];
        }
    }
    if (out_culled_lowest_order) *out_culled_lowest_order = culled;
    return r;
}

}  // namespace treegen
