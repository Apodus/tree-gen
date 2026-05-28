// [treegen_collision_fit] — P1 trunk capsule fit. Pins compute_trunk_capsule
// across all 4 species: radius>0, half_length>radius, half_length<height_m.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../collision_fit.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"
#include "../tree_descriptor.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace {

struct SpeciesFixture {
    const char* scenario_path;
    const char* label;
};

constexpr std::array<SpeciesFixture, 4> k_species = {{
    {"tools/rynx-treegen/scenarios/c3_oak.json",   "oak"},
    {"tools/rynx-treegen/scenarios/c3_pine.json",  "pine"},
    {"tools/rynx-treegen/scenarios/c3_birch.json", "birch"},
    {"tools/rynx-treegen/scenarios/c3_maple.json", "maple"},
}};

} // anonymous namespace

TEST_CASE("treegen_collision_fit: all species produce valid trunk capsule",
          "[treegen][treegen_collision_fit]") {
    namespace ts = rynx::test_support;

    for (const auto& sp : k_species) {
        const auto fixture = ts::find_repo_file(sp.scenario_path);
        INFO("species=" << sp.label << " fixture=" << fixture);
        REQUIRE_FALSE(fixture.empty());

        treegen::Scenario s = treegen::load_scenario(fixture);
        REQUIRE(s.kind == "tree");

        const uint64_t seed_effective = 0ull ^ s.scenario_fnv;
        treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed_effective);
        REQUIRE(skel.nodes.size() > 1u);

        const treegen::TrunkCapsule cap = treegen::compute_trunk_capsule(skel, s.tree);

        INFO("half_length=" << cap.half_length << " radius=" << cap.radius
             << " height_m=" << s.tree.height_m);

        REQUIRE(cap.radius > 0.0f);
        REQUIRE(cap.half_length > cap.radius);
        REQUIRE(cap.half_length < s.tree.height_m);
    }
}
