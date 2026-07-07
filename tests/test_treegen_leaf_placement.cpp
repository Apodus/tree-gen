// [treegen_leaf_placement] -- pins BranchWalk leaf-site generation (C1 P4).

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../json_reader.hpp"
#include "../leaf_placement.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../skeleton_json.hpp"
#include "../space_colonization.hpp"
#include "../tree_descriptor.hpp"
#include "../vec3.hpp"

#include <cmath>
#include <cstdint>
#include <string>

namespace {

float vlen(treegen::vec3 v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

} // anonymous namespace

TEST_CASE("treegen_leaf_placement: branch_walk oak produces sane site set",
          "[treegen][treegen_leaf_placement]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    INFO("fixture=" << fixture);
    REQUIRE_FALSE(fixture.empty());

    treegen::Scenario s = treegen::load_scenario(fixture);
    REQUIRE(s.kind == "tree");
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed_effective);

    auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
    auto sites = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed_effective);
    INFO("sites=" << sites.size());
    REQUIRE(sites.size() > 100u);

    // All normals are unit-length.
    for (const auto& site : sites) {
        const float len = vlen(site.normal);
        INFO("normal len=" << len);
        REQUIRE(std::abs(len - 1.0f) < 1e-4f);
    }

    // branch_id valid index into nodes.
    for (const auto& site : sites) {
        REQUIRE(site.branch_id >= 0);
        REQUIRE(site.branch_id < static_cast<int>(skel.nodes.size()));
    }
}

TEST_CASE("treegen_leaf_placement: deterministic across runs (same seed)",
          "[treegen][treegen_leaf_placement][treegen_leaf_determinism]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed = 42ull ^ s.scenario_fnv;

    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);
    auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);

    auto sites1 = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);
    auto sites2 = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);

    REQUIRE(sites1.size() == sites2.size());
    for (size_t i = 0; i < sites1.size(); ++i) {
        INFO("i=" << i);
        REQUIRE(sites1[i].position.x == sites2[i].position.x);
        REQUIRE(sites1[i].position.y == sites2[i].position.y);
        REQUIRE(sites1[i].position.z == sites2[i].position.z);
        REQUIRE(sites1[i].normal.x   == sites2[i].normal.x);
        REQUIRE(sites1[i].normal.y   == sites2[i].normal.y);
        REQUIRE(sites1[i].normal.z   == sites2[i].normal.z);
        REQUIRE(sites1[i].branch_id  == sites2[i].branch_id);
    }

    // Skeleton-JSON dump is byte-identical when sites are part of the dump.
    treegen::TreeSkeleton dump_skel = skel;
    dump_skel.leaf_sites = sites1;
    const std::string d1 = treegen::dump_skeleton_json(dump_skel);
    dump_skel.leaf_sites = sites2;
    const std::string d2 = treegen::dump_skeleton_json(dump_skel);
    REQUIRE(d1 == d2);
}

TEST_CASE("treegen_leaf_placement: no sites at depth < min_depth",
          "[treegen][treegen_leaf_placement]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);

    auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
    opts.leaf_min_branch_depth = 4;

    auto sites = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);
    REQUIRE(sites.size() > 10u);

    for (const auto& site : sites) {
        const int depth = skel.nodes[static_cast<size_t>(site.branch_id)].depth;
        INFO("branch_id=" << site.branch_id << " depth=" << depth);
        REQUIRE(depth >= opts.leaf_min_branch_depth);
    }
}

TEST_CASE("treegen_leaf_placement: all species generate valid leaf meshes",
          "[treegen][treegen_leaf_placement]") {
    namespace ts = rynx::test_support;

    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* scenario_rel : scenarios) {
        const auto fixture = ts::find_repo_file(scenario_rel);
        REQUIRE_FALSE(fixture.empty());
        treegen::Scenario s = treegen::load_scenario(fixture);
        const uint64_t seed = 42ull ^ s.scenario_fnv;
        treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);

        auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
        auto sites = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);
        INFO("species=" << s.tree.species << " sites=" << sites.size());
        REQUIRE(sites.size() > 50u);

        for (const auto& site : sites) {
            const float len = vlen(site.normal);
            REQUIRE(std::abs(len - 1.0f) < 1e-4f);
            REQUIRE(site.branch_id >= 0);
            REQUIRE(site.branch_id < static_cast<int>(skel.nodes.size()));
        }
    }
}

TEST_CASE("treegen_leaf_placement: TreeDescriptor.Leaves round-trips through JSON",
          "[treegen][treegen_leaf_placement][treegen_leaf_descriptor_roundtrip]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);

    REQUIRE(s.tree.leaves.shape == treegen::LeafShape::OakLobed);
    REQUIRE(std::abs(s.tree.leaves.leaf_density_per_meter - 4.5f) < 1e-4f);  // C3 P2 rebalance

    const std::string serialized = treegen::serialize_tree_descriptor(s.tree);
    auto parsed = treegen::json::parse(serialized);
    REQUIRE(parsed.ok);
    treegen::TreeDescriptor td2 = treegen::parse_tree_descriptor(parsed.root);

    REQUIRE(td2.leaves.geometry_type        == s.tree.leaves.geometry_type);
    REQUIRE(td2.leaves.shape                == s.tree.leaves.shape);
    REQUIRE(std::abs(td2.leaves.leaf_size_m          - s.tree.leaves.leaf_size_m)          < 1e-5f);
    REQUIRE(td2.leaves.cluster_count_per_tip == s.tree.leaves.cluster_count_per_tip);
    REQUIRE(td2.leaves.leaf_min_branch_depth == s.tree.leaves.leaf_min_branch_depth);
    REQUIRE(std::abs(td2.leaves.leaf_density_per_meter   - s.tree.leaves.leaf_density_per_meter)   < 1e-5f);
    REQUIRE(std::abs(td2.leaves.leaf_depth_density_curve - s.tree.leaves.leaf_depth_density_curve) < 1e-5f);
    REQUIRE(std::abs(td2.leaves.leaf_phototropic_bias    - s.tree.leaves.leaf_phototropic_bias)    < 1e-5f);

    // Serialize twice -> byte-identical (determinism).
    const std::string serialized2 = treegen::serialize_tree_descriptor(td2);
    REQUIRE(serialized == serialized2);
}

TEST_CASE("treegen_leaf_placement: branch_walk sites lie near qualifying skeleton segments",
          "[treegen][treegen_leaf_placement][treegen_leaf_branch_walk]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);

    auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
    opts.leaf_min_branch_depth = 2;

    auto sites = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);
    INFO("branch_walk sites=" << sites.size());
    REQUIRE(sites.size() > 10u);

    // Each site should be within radius + leaf_size + epsilon of its parent segment.
    for (const auto& site : sites) {
        REQUIRE(site.branch_id >= 1);
        REQUIRE(site.branch_id < static_cast<int>(skel.nodes.size()));
        const auto& child  = skel.nodes[static_cast<size_t>(site.branch_id)];
        const auto& parent = skel.nodes[static_cast<size_t>(child.parent_index)];

        const treegen::vec3 seg = child.position - parent.position;
        const float seg_len2 = treegen::dot(seg, seg);
        float t = 0.0f;
        if (seg_len2 > 1e-10f)
            t = std::clamp(treegen::dot(site.position - parent.position, seg) / seg_len2, 0.0f, 1.0f);
        const treegen::vec3 closest = parent.position + seg * t;
        const float dist = vlen(site.position - closest);
        const float max_radius = std::max(parent.radius, child.radius);
        const float tolerance = std::max(0.005f, max_radius) + opts.leaf_size_m * 0.5f + 1e-3f;
        INFO("dist=" << dist << " tolerance=" << tolerance);
        REQUIRE(dist <= tolerance);
    }
}

TEST_CASE("treegen_leaf_placement: branch_walk tangent/normal are unit and orthogonal",
          "[treegen][treegen_leaf_placement][treegen_leaf_branch_walk]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);

    auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
    opts.leaf_min_branch_depth = 2;

    auto sites = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);
    REQUIRE(sites.size() > 10u);

    for (const auto& site : sites) {
        const float tlen = vlen(site.branch_tangent);
        const float nlen = vlen(site.branch_normal);
        INFO("tangent_len=" << tlen << " normal_len=" << nlen);
        REQUIRE(std::abs(tlen - 1.0f) < 1e-4f);
        REQUIRE(std::abs(nlen - 1.0f) < 1e-4f);

        const float d = treegen::dot(site.branch_tangent, site.branch_normal);
        INFO("dot(tangent,normal)=" << d);
        REQUIRE(std::abs(d) < 1e-4f);
    }
}

TEST_CASE("treegen_leaf_placement: branch_walk depth-density monotonicity with curve > 1",
          "[treegen][treegen_leaf_placement][treegen_leaf_branch_walk]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);

    auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
    opts.leaf_min_branch_depth = 1;
    opts.leaf_density_per_meter = 12.0f;
    opts.leaf_depth_density_curve = 2.0f;

    auto sites = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);
    REQUIRE(sites.size() > 20u);

    int max_depth = 0;
    for (const auto& n : skel.nodes) max_depth = std::max(max_depth, n.depth);
    REQUIRE(max_depth > 2);

    std::vector<int> count_by_depth(static_cast<size_t>(max_depth + 1), 0);
    for (const auto& site : sites) {
        const int d = skel.nodes[static_cast<size_t>(site.branch_id)].depth;
        count_by_depth[static_cast<size_t>(d)]++;
    }

    int last_nz = -1, second_nz = -1;
    for (int d = max_depth; d >= opts.leaf_min_branch_depth; --d) {
        if (count_by_depth[static_cast<size_t>(d)] > 0) {
            if (last_nz < 0) last_nz = d;
            else if (second_nz < 0) { second_nz = d; break; }
        }
    }
    INFO("last_nz_depth=" << last_nz << " count=" << count_by_depth[static_cast<size_t>(last_nz)]
         << " second_nz_depth=" << second_nz << " count=" << count_by_depth[static_cast<size_t>(second_nz)]);
    REQUIRE(last_nz > second_nz);
    REQUIRE(count_by_depth[static_cast<size_t>(last_nz)] > 0);
    REQUIRE(count_by_depth[static_cast<size_t>(second_nz)] > 0);
}

TEST_CASE("treegen_leaf_placement: branch_walk phototropic bias shifts normals toward world_up",
          "[treegen][treegen_leaf_placement][treegen_leaf_branch_walk_phototropic]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);

    auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);

    // Generate with bias=0 (pure radial) and bias=0.3 (phototropic blend).
    opts.leaf_phototropic_bias = 0.0f;
    auto sites_b0 = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);
    opts.leaf_phototropic_bias = 0.3f;
    auto sites_b03 = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);

    // Same count (bias only changes normal direction, not placement).
    REQUIRE(sites_b0.size() == sites_b03.size());
    REQUIRE(sites_b03.size() > 10u);

    // Mean Z component of normals must be higher with bias=0.3 than with bias=0.
    double sum_z_b0 = 0.0, sum_z_b03 = 0.0;
    for (size_t i = 0; i < sites_b0.size(); ++i) {
        const float len = vlen(sites_b03[i].normal);
        REQUIRE(std::abs(len - 1.0f) < 1e-4f);
        sum_z_b0  += static_cast<double>(sites_b0[i].normal.z);
        sum_z_b03 += static_cast<double>(sites_b03[i].normal.z);
    }
    const double mean_z_b0  = sum_z_b0  / static_cast<double>(sites_b0.size());
    const double mean_z_b03 = sum_z_b03 / static_cast<double>(sites_b03.size());
    INFO("mean_z bias=0: " << mean_z_b0 << " bias=0.3: " << mean_z_b03);
    REQUIRE(mean_z_b03 > mean_z_b0 + 0.05);
}

TEST_CASE("treegen_leaf_placement: branch_walk bias=1.0 gives world_up",
          "[treegen][treegen_leaf_placement][treegen_leaf_branch_walk_phototropic]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);

    auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
    opts.leaf_phototropic_bias = 1.0f;

    auto sites = treegen::leaf_placement::generate_leaf_sites(skel, opts, seed);
    REQUIRE(sites.size() > 10u);

    for (const auto& site : sites) {
        INFO("normal=(" << site.normal.x << "," << site.normal.y << "," << site.normal.z << ")");
        REQUIRE(std::abs(site.normal.x) < 1e-5f);
        REQUIRE(std::abs(site.normal.y) < 1e-5f);
        REQUIRE(std::abs(site.normal.z - 1.0f) < 1e-5f);
    }
}
