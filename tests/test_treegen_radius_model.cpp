// [treegen_radius_floor] + [treegen_trunk_taper] + [treegen_lod_count] — C10.
// Pins: min_branch_radius_m floor, trunk_taper_rate linear taper, 3-LOD GLB.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../lod_emitter.hpp"
#include "../radius_solver.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"
#include "../tree_descriptor.hpp"

#include <algorithm>
#include <cmath>

namespace ts = treegen;

TEST_CASE("[treegen_radius_floor] every non-root node radius >= min_branch_radius_m",
          "[treegen][treegen_radius_floor]") {
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

        const float floor = s.tree.branching.min_branch_radius_m;
        INFO("species=" << s.tree.species << " min_branch_radius_m=" << floor);
        REQUIRE(floor > 0.0f);

        int violations = 0;
        for (size_t i = 1; i < skel.nodes.size(); ++i) {
            if (skel.nodes[i].radius < floor - 1e-7f) {
                ++violations;
            }
        }
        REQUIRE(violations == 0);

        // Root is trunk_base_radius_m (NOT floored — it's already large).
        REQUIRE(std::abs(skel.nodes[0].radius - s.tree.trunk_base_radius_m) < 1e-6f);
    }
}

TEST_CASE("[treegen_trunk_taper] trunk chain tapers linearly from base to crown_base_z",
          "[treegen][treegen_trunk_taper]") {
    namespace tsp = rynx::test_support;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

    const float trunk_base   = s.tree.trunk_base_radius_m;
    const float taper_rate   = s.tree.trunk_taper_rate;
    const float crown_base_z = s.tree.branching.crown_base_fraction * s.tree.height_m;
    REQUIRE(crown_base_z > 0.0f);

    // Collect depth==0 trunk nodes at or below crown_base_z.
    int trunk_count = 0;
    float max_error = 0.0f;
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        const auto& node = skel.nodes[i];
        if (node.depth != 0) continue;
        if (node.position.z > crown_base_z) continue;

        const float t = node.position.z / crown_base_z;
        const float expected = trunk_base * (1.0f + (taper_rate - 1.0f) * t);
        // Floor may override if expected < min_branch_radius_m, but trunk radii
        // are always well above the floor, so this checks the taper directly.
        const float actual = node.radius;
        const float err = std::abs(actual - expected);
        if (err > max_error) max_error = err;
        INFO("i=" << i << " z=" << node.position.z << " t=" << t
             << " expected=" << expected << " actual=" << actual);
        REQUIRE(err < 1e-5f);
        ++trunk_count;
    }
    // Must have found at least a few trunk nodes in the taper zone.
    REQUIRE(trunk_count >= 3);
    INFO("trunk_count=" << trunk_count << " max_error=" << max_error);
}

TEST_CASE("[treegen_lod_count] GLB has 3 LOD meshes (L3 billboard stub suppressed)",
          "[treegen][treegen_lod_count]") {
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

        ts::BarkMeshOptions base_opts;
        base_opts.tree_height_m   = s.tree.height_m;
        base_opts.seam_offset_rad = 0.0f;

        ts::LodBudget budget;
        auto lods = ts::emit_all_lods(skel, base_opts, budget, s.tree.height_m, seed_effective);

        INFO("species=" << s.tree.species << " lods=" << lods.size());
        REQUIRE(lods.size() == 3u);
        REQUIRE(lods[0].lod_index == 0);
        REQUIRE(lods[1].lod_index == 1);
        REQUIRE(lods[2].lod_index == 2);

        // All LODs have non-degenerate bark geometry.
        for (int i = 0; i < 3; ++i) {
            REQUIRE(lods[i].indices_u32.size() / 3 > 0u);
        }
    }
}
