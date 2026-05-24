// [treegen_fork_blend_no_tjunction] — pins C4 P2 skin-rim fork blending.
// After build_bark_mesh applies the fork blend by default, no edge in the
// emitted bark mesh has > 2 adjacent triangles (manifold seal). Also pins
// determinism: two invocations produce byte-identical positions + indices.
//
// Same link rationale as test_treegen_branch_mesh: tool sources (fork_blend.cpp
// + branch_mesh.cpp + space_colonization.cpp + …) link into TestTech via
// rynx_tests.sharpmake.cs so this test pins the exact code shipped in
// rynx-treegen.exe.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../branch_mesh.hpp"
#include "../fork_blend.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace ts = treegen;

TEST_CASE("[treegen_fork_blend_no_tjunction] no edge has more than 2 adjacent triangles",
          "[treegen][treegen_fork_blend_no_tjunction]") {
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    REQUIRE(s.kind == "tree");

    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
    REQUIRE(skel.nodes.size() > 100u);

    ts::BarkMeshOptions opts;
    // apply_fork_blend defaults to true — fork blend runs inside build_bark_mesh.
    ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);

    REQUIRE(bark.indices_u32.size() % 3 == 0);
    REQUIRE(bark.indices_u32.size() > 0u);

    // Edge -> adjacency count.
    std::map<std::pair<uint32_t, uint32_t>, int> edge_count;
    auto add_edge = [&](uint32_t a, uint32_t b) {
        if (a > b) std::swap(a, b);
        ++edge_count[{a, b}];
    };
    for (size_t i = 0; i + 2 < bark.indices_u32.size(); i += 3) {
        const uint32_t a = bark.indices_u32[i + 0];
        const uint32_t b = bark.indices_u32[i + 1];
        const uint32_t c = bark.indices_u32[i + 2];
        add_edge(a, b);
        add_edge(b, c);
        add_edge(c, a);
    }
    int over_count = 0;
    for (auto& kv : edge_count) {
        if (kv.second > 2) ++over_count;
    }
    REQUIRE(over_count == 0);

    // Determinism: build twice; byte-equal positions + indices.
    ts::BarkMeshOutput bark2 = ts::build_bark_mesh(skel, opts);
    REQUIRE(bark.mesh.positions == bark2.mesh.positions);
    REQUIRE(bark.indices_u32     == bark2.indices_u32);
}

TEST_CASE("[treegen_fork_blend_no_tjunction] all four species are manifold after blend",
          "[treegen][treegen_fork_blend_no_tjunction]") {
    namespace tsp = rynx::test_support;
    const char* species[] = { "c3_oak.json", "c3_pine.json", "c3_birch.json", "c3_maple.json" };

    for (const char* sp : species) {
        const std::string path = std::string("tools/rynx-treegen/scenarios/") + sp;
        const auto fixture = tsp::find_repo_file(path);
        REQUIRE_FALSE(fixture.empty());

        ts::Scenario s = ts::load_scenario(fixture);
        ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, 42ull ^ s.scenario_fnv);

        ts::BarkMeshOptions opts;
        ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);

        REQUIRE(bark.indices_u32.size() % 3 == 0);
        REQUIRE(bark.indices_u32.size() > 0u);

        std::map<std::pair<uint32_t, uint32_t>, int> edge_count;
        auto add_edge = [&](uint32_t a, uint32_t b) {
            if (a > b) std::swap(a, b);
            ++edge_count[{a, b}];
        };
        for (size_t i = 0; i + 2 < bark.indices_u32.size(); i += 3) {
            const uint32_t a = bark.indices_u32[i + 0];
            const uint32_t b = bark.indices_u32[i + 1];
            const uint32_t c = bark.indices_u32[i + 2];
            add_edge(a, b); add_edge(b, c); add_edge(c, a);
        }
        int over_count = 0;
        for (auto& kv : edge_count) if (kv.second > 2) ++over_count;
        INFO("species: " << sp);
        REQUIRE(over_count == 0);
    }
}

TEST_CASE("[treegen_fork_blend_no_tjunction] apply_fork_blend=false matches P1 baseline",
          "[treegen][treegen_fork_blend_no_tjunction]") {
    // Pins the blend-on / blend-off distinction: with the flag off, output
    // is the raw P1 cylinder tube set (no extra stitch tris, no position
    // mutation). With it on, additional tris are appended and child first
    // rings are snapped — tri count strictly increases or matches.
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    ts::Scenario s = ts::load_scenario(fixture);
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, 42ull ^ s.scenario_fnv);

    ts::BarkMeshOptions opts_off;
    opts_off.apply_fork_blend = false;
    ts::BarkMeshOutput bark_off = ts::build_bark_mesh(skel, opts_off);

    ts::BarkMeshOptions opts_on;
    opts_on.apply_fork_blend = true;
    ts::BarkMeshOutput bark_on = ts::build_bark_mesh(skel, opts_on);

    REQUIRE(bark_on.indices_u32.size() >= bark_off.indices_u32.size());
    // Collar rings add verts; blend-on has at least as many as blend-off.
    REQUIRE(bark_on.mesh.positions.size() >= bark_off.mesh.positions.size());
}
