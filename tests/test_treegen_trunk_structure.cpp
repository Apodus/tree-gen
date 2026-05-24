// [treegen_depth_order] + [treegen_central_leader] + [treegen_branching_onset]
// C11 — trunk structure: depth-inflation fix, central leader, gradual onset.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"
#include "../tree_descriptor.hpp"

#include <cmath>
#include <vector>

namespace ts = treegen;

TEST_CASE("[treegen_depth_order] continuation inherits depth; split increments",
          "[treegen][treegen_depth_order]") {
    namespace tsp = rynx::test_support;

    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* rel : scenarios) {
        const auto fixture = tsp::find_repo_file(rel);
        REQUIRE_FALSE(fixture.empty());

        ts::Scenario s = ts::load_scenario(fixture);
        REQUIRE(s.kind == "tree");

        const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
        ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
        REQUIRE(skel.nodes.size() > 50u);

        // Build child lists to identify single-child (continuation) vs multi-child (split) parents.
        const size_t n = skel.nodes.size();
        std::vector<std::vector<int>> children(n);
        for (size_t i = 1; i < n; ++i) {
            children[static_cast<size_t>(skel.nodes[i].parent_index)].push_back(static_cast<int>(i));
        }

        // Invariant: child.depth in {parent.depth, parent.depth+1}.
        // Continuation (primary) keeps parent depth; split increments by 1.
        // No child should ever jump by more than 1.
        int depth_violations = 0;
        int has_continuation = 0; // at least one child inherits depth
        int has_split = 0;        // at least one child increments depth
        for (size_t p = 0; p < n; ++p) {
            const auto& kids = children[p];
            if (kids.empty()) continue;
            const int pd = skel.nodes[p].depth;
            for (int ci : kids) {
                const int cd = skel.nodes[static_cast<size_t>(ci)].depth;
                if (cd == pd) ++has_continuation;
                else if (cd == pd + 1) ++has_split;
                else {
                    INFO("species=" << s.tree.species << " parent=" << p
                         << " child=" << ci << " pd=" << pd << " cd=" << cd);
                    ++depth_violations;
                }
            }
        }
        INFO("species=" << s.tree.species
             << " continuations=" << has_continuation << " splits=" << has_split);
        REQUIRE(depth_violations == 0);
        // Sanity: tree must have both continuations and splits.
        REQUIRE(has_continuation > 0);
        REQUIRE(has_split > 0);
    }
}

TEST_CASE("[treegen_central_leader] depth-0 nodes exist above crown_base_z",
          "[treegen][treegen_central_leader]") {
    namespace tsp = rynx::test_support;

    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* rel : scenarios) {
        const auto fixture = tsp::find_repo_file(rel);
        REQUIRE_FALSE(fixture.empty());

        ts::Scenario s = ts::load_scenario(fixture);
        const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
        ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

        const float crown_base_z = s.tree.branching.crown_base_fraction * s.tree.height_m;

        int depth0_above_crown = 0;
        for (size_t i = 0; i < skel.nodes.size(); ++i) {
            if (skel.nodes[i].depth == 0 && skel.nodes[i].position.z > crown_base_z + 0.01f) {
                ++depth0_above_crown;
            }
        }
        INFO("species=" << s.tree.species << " depth0_above_crown=" << depth0_above_crown
             << " crown_base_z=" << crown_base_z);
        REQUIRE(depth0_above_crown >= 1);
    }
}

TEST_CASE("[treegen_branching_onset] ramp delays first lateral branch",
          "[treegen][treegen_branching_onset]") {
    namespace tsp = rynx::test_support;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;

    // With ramp: the first depth>=1 branch should appear at a higher z.
    ts::TreeDescriptor td_ramp = s.tree;
    td_ramp.crown_onset_ramp_fraction = 0.30f; // strong ramp for clear signal
    ts::TreeSkeleton skel_ramp = ts::grow_skeleton(td_ramp, seed_effective);

    ts::TreeDescriptor td_flat = s.tree;
    td_flat.crown_onset_ramp_fraction = 0.0f;
    ts::TreeSkeleton skel_flat = ts::grow_skeleton(td_flat, seed_effective);

    // Find lowest z of any depth>=1 node (first lateral branch off trunk).
    float first_branch_z_ramp = td_ramp.height_m;
    for (const auto& n : skel_ramp.nodes) {
        if (n.depth >= 1 && n.position.z < first_branch_z_ramp)
            first_branch_z_ramp = n.position.z;
    }
    float first_branch_z_flat = td_flat.height_m;
    for (const auto& n : skel_flat.nodes) {
        if (n.depth >= 1 && n.position.z < first_branch_z_flat)
            first_branch_z_flat = n.position.z;
    }

    INFO("first_branch_z: ramp=" << first_branch_z_ramp << " flat=" << first_branch_z_flat);
    // Ramp should push the first branch higher (or at least not lower).
    REQUIRE(first_branch_z_ramp >= first_branch_z_flat);
}
