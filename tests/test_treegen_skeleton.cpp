// [treegen_skeleton] — pins C3 P2's space-colonization output. Same split
// rationale as test_treegen_descriptor: tool sources (space_colonization.cpp +
// radius_solver.cpp + skeleton_json.cpp) link into TestTech via
// rynx_tests.sharpmake.cs so tests pin the exact code shipped in rynx-treegen.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../envelopes.hpp"
#include "../radius_solver.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../skeleton_json.hpp"
#include "../space_colonization.hpp"
#include "../tree_descriptor.hpp"
#include "../vec3.hpp"

#include <cmath>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

namespace {

float vlen(treegen::vec3 v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

} // anonymous namespace

TEST_CASE("treegen_skeleton: deterministic across runs on c3_oak fixture",
          "[treegen][treegen_skeleton][treegen_skeleton_determinism]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    INFO("fixture=" << fixture);
    REQUIRE_FALSE(fixture.empty());

    treegen::Scenario s = treegen::load_scenario(fixture);
    REQUIRE(s.kind == "tree");

    // CLI defaults: cli_seed=0; seed_effective = cli_seed ^ scenario_fnv.
    const uint64_t seed_effective = 0ull ^ s.scenario_fnv;

    treegen::TreeSkeleton skel1 = treegen::grow_skeleton(s.tree, seed_effective);
    treegen::TreeSkeleton skel2 = treegen::grow_skeleton(s.tree, seed_effective);

    REQUIRE(skel1.nodes.size() == skel2.nodes.size());
    REQUIRE(skel1.nodes.size() > 1u);  // grew at least something
    REQUIRE(skel1.attractors_consumed == skel2.attractors_consumed);
    REQUIRE(skel1.iterations_run == skel2.iterations_run);

    const std::string dump1 = treegen::dump_skeleton_json(skel1);
    const std::string dump2 = treegen::dump_skeleton_json(skel2);
    REQUIRE(dump1 == dump2);   // BYTE-IDENTICAL — the determinism gate.
    REQUIRE(dump1.size() > 256u); // sanity: non-empty
}

TEST_CASE("treegen_skeleton: topology invariants on c3_oak fixture",
          "[treegen][treegen_skeleton][treegen_skeleton_topology]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed_effective);

    REQUIRE(skel.nodes.size() >= 2u);

    // Exactly one root (parent_index == -1) at index 0.
    int root_count = 0;
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        if (skel.nodes[i].parent_index == -1) {
            ++root_count;
            REQUIRE(i == 0u);
        }
    }
    REQUIRE(root_count == 1);

    // Every non-root parent_index < own_index (DAG seal — postorder ⇔ reverse).
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        REQUIRE(skel.nodes[i].parent_index >= 0);
        REQUIRE(skel.nodes[i].parent_index < static_cast<int>(i));
    }

    // No cycles via BFS from root: every node reachable, visited exactly once.
    std::vector<int> visit(skel.nodes.size(), 0);
    std::queue<int> q;
    q.push(0);
    visit[0] = 1;
    int reached = 1;
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        for (size_t i = 1; i < skel.nodes.size(); ++i) {
            if (skel.nodes[i].parent_index == cur && !visit[i]) {
                visit[i] = 1;
                ++reached;
                q.push(static_cast<int>(i));
            }
        }
    }
    REQUIRE(reached == static_cast<int>(skel.nodes.size()));

    // No zero-length segments (parent → child) except for root (no parent).
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const auto& child  = skel.nodes[i];
        const auto& parent = skel.nodes[static_cast<size_t>(child.parent_index)];
        const float seg_len = vlen(child.position - parent.position);
        INFO("segment i=" << i << " parent=" << child.parent_index << " len=" << seg_len);
        REQUIRE(seg_len > 1e-4f);
    }

    // Da Vinci radius monotonicity: child.radius <= parent.radius
    // (within FP slack — the n-th-root operation may introduce ~1 ULP drift).
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const auto& child  = skel.nodes[i];
        const auto& parent = skel.nodes[static_cast<size_t>(child.parent_index)];
        INFO("child.radius=" << child.radius << " parent.radius=" << parent.radius);
        REQUIRE(child.radius <= parent.radius + 1e-5f);
    }

    // Root radius is force-set to trunk_base_radius_m.
    REQUIRE(std::abs(skel.nodes[0].radius - s.tree.trunk_base_radius_m) < 1e-6f);
}

TEST_CASE("treegen_skeleton: every node inside envelope (modulo tip overshoot)",
          "[treegen][treegen_skeleton][treegen_skeleton_envelope]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);

    const uint64_t seed_effective = 0ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed_effective);

    // Tip overshoot: a tip aimed at an attractor near the envelope boundary
    // can step OUT by up to growth_distance. Compare against the AABB,
    // padded by growth_distance, which is the strict upper bound on overshoot.
    const treegen::Aabb box = treegen::envelope_aabb(s.tree.envelope.shape, s.tree.envelope,
                                                     s.tree.height_m);
    const float pad = s.tree.branching.growth_distance;

    int outside = 0;
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        const auto& n = skel.nodes[i];
        // Root is at origin, which may be outside the spheroid (z=0 below
        // the OblateSpheroid bottom pole sits at z=center_z - b). It's the
        // intended root; skip it.
        if (i == 0) continue;
        const bool inside_padded =
            n.position.x >= box.min.x - pad && n.position.x <= box.max.x + pad &&
            n.position.y >= box.min.y - pad && n.position.y <= box.max.y + pad &&
            n.position.z >= box.min.z - pad && n.position.z <= box.max.z + pad;
        if (!inside_padded) {
            INFO("node i=" << i << " pos=("
                 << n.position.x << "," << n.position.y << "," << n.position.z << ")");
            ++outside;
        }
    }
    REQUIRE(outside == 0);
}

TEST_CASE("treegen_skeleton: radius solver synthetic 3-leaf-1-internal-1-root tree",
          "[treegen][treegen_skeleton][treegen_skeleton_radius_solver]") {
    // Build: 0 root → 1 internal → {2,3,4} leaves. Solver with n=2:
    //   pipe model: leaf_r_raw=0.005*trunk_base, internal=sqrt(3)*leaf_r_raw,
    //   root=internal (single child).
    // Proportional scale (Step 4): scale = trunk_base / pipe_root, all nodes
    // multiplied. Root single child → pipe_root==internal → root==trunk_base,
    //   internal==trunk_base, leaves==trunk_base/sqrt(3).
    treegen::TreeSkeleton skel;
    skel.nodes.resize(5);
    skel.nodes[0].parent_index = -1; skel.nodes[0].position = {0,0,0}; skel.nodes[0].depth = 0;
    skel.nodes[1].parent_index =  0; skel.nodes[1].position = {0,0,1}; skel.nodes[1].depth = 1;
    skel.nodes[2].parent_index =  1; skel.nodes[2].position = {1,0,2}; skel.nodes[2].depth = 2;
    skel.nodes[3].parent_index =  1; skel.nodes[3].position = {-1,0,2}; skel.nodes[3].depth = 2;
    skel.nodes[4].parent_index =  1; skel.nodes[4].position = {0,1,2}; skel.nodes[4].depth = 2;

    const float trunk_base = 0.30f;
    treegen::TreeDescriptor td;
    td.trunk_base_radius_m = trunk_base;
    td.taper_exponent      = 2.0f;
    td.height_m            = 10.0f;
    td.branching.crown_base_fraction = 0.0f; // no trunk taper zone
    td.branching.min_branch_radius_m = 0.0f; // disable floor for this test
    treegen::solve_radii(skel, td);

    // After proportional scaling: leaves = trunk_base / sqrt(3).
    const float expected_leaf = trunk_base / std::sqrt(3.0f);
    INFO("expected_leaf=" << expected_leaf);
    REQUIRE(std::abs(skel.nodes[2].radius - expected_leaf) < 1e-5f);
    REQUIRE(std::abs(skel.nodes[3].radius - expected_leaf) < 1e-5f);
    REQUIRE(std::abs(skel.nodes[4].radius - expected_leaf) < 1e-5f);

    // Internal = trunk_base (single child of root → same pipe-model sum → same scale).
    INFO("internal=" << skel.nodes[1].radius);
    REQUIRE(std::abs(skel.nodes[1].radius - trunk_base) < 1e-5f);

    // Root = trunk_base (proportional scale target).
    REQUIRE(std::abs(skel.nodes[0].radius - trunk_base) < 1e-6f);
}

TEST_CASE("treegen_skeleton: strong +X phototropism biases mean tip X-coord positive",
          "[treegen][treegen_skeleton][treegen_skeleton_tropism_bias]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);

    // Configure strong +X phototropism on top of the oak descriptor.
    s.tree.tropisms.phototropism = 2.0f;
    s.tree.tropisms.light_dir    = treegen::vec3{1.0f, 0.0f, 0.0f};
    s.tree.tropisms.gravitropism = 0.0f;
    s.tree.branching.tropism_strength = 1.0f;

    const uint64_t seed_effective = 0ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed_effective);

    REQUIRE(skel.nodes.size() > 2u);

    // Mean X of all non-root nodes; should be positive under +X bias.
    double sum_x = 0.0;
    int count = 0;
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        sum_x += static_cast<double>(skel.nodes[i].position.x);
        ++count;
    }
    const double mean_x = sum_x / static_cast<double>(count);
    INFO("mean_x=" << mean_x << " over " << count << " nodes");
    REQUIRE(mean_x > 0.0);

    // Sanity: with neutral tropisms baseline, mean X should be ~0 (envelope
    // is centred). This case asserts the tropism actually moves the
    // distribution, not just that growth happened.
    treegen::Scenario baseline = treegen::load_scenario(fixture);
    treegen::TreeSkeleton skel_base = treegen::grow_skeleton(baseline.tree,
                                       0ull ^ baseline.scenario_fnv);
    double sum_x_base = 0.0; int n_base = 0;
    for (size_t i = 1; i < skel_base.nodes.size(); ++i) {
        sum_x_base += static_cast<double>(skel_base.nodes[i].position.x);
        ++n_base;
    }
    const double mean_x_base = n_base > 0 ? sum_x_base / static_cast<double>(n_base) : 0.0;
    INFO("baseline mean_x=" << mean_x_base);
    REQUIRE(mean_x > mean_x_base);
}
