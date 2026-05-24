// C12 — [treegen_fork_normals] + [treegen_mesh_efficiency] + [treegen_tip_caps].
// Pins: fork-blend normal recompute, continuation merge vertex savings, tip end-caps.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../branch_mesh.hpp"
#include "../fork_blend.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"
#include "../vec3.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace ts = treegen;

// ---- helpers ---------------------------------------------------------------

namespace {

// Build child-count array from skeleton.
std::vector<int> child_counts(const ts::TreeSkeleton& skel) {
    std::vector<int> cc(skel.nodes.size(), 0);
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        const int p = skel.nodes[i].parent_index;
        if (p >= 0) ++cc[static_cast<size_t>(p)];
    }
    return cc;
}

} // anonymous namespace

// ---- [treegen_fork_normals] ------------------------------------------------

TEST_CASE("[treegen_fork_normals] blended vertex normals within 15 deg of radial direction",
          "[treegen][treegen_fork_normals]") {
    namespace tsp = rynx::test_support;

    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    constexpr float k_cos_15deg = 0.9659258f; // cos(15 deg)

    for (const char* rel : scenarios) {
        const auto fixture = tsp::find_repo_file(rel);
        REQUIRE_FALSE(fixture.empty());

        ts::Scenario s = ts::load_scenario(fixture);
        REQUIRE(s.kind == "tree");

        const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
        ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

        ts::BarkMeshOptions opts;
        opts.tree_height_m = s.tree.height_m;
        ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);

        // Collect fork zones to identify which rings were blended.
        auto zones = ts::collect_fork_zones(skel, opts.fork_zone_factor);

        // For each blended ring, verify that per-vertex normal is close to
        // the radial direction (pos - ring_center).
        int checked = 0;
        int violations = 0;

        for (const auto& rm : bark.ring_metadata) {
            // Only check rings that are within a fork zone.
            bool in_zone = false;
            for (const auto& z : zones) {
                for (int ci : z.child_indices) {
                    if (ci == rm.branch_node_index && rm.axial_distance_m < z.zone_length_m) {
                        in_zone = true;
                        break;
                    }
                }
                if (in_zone) break;
            }
            if (!in_zone) continue;

            const int N = rm.N;
            // Compute ring center.
            ts::vec3 center = {0.0f, 0.0f, 0.0f};
            for (int i = 0; i < N; ++i) {
                const size_t vi = static_cast<size_t>(rm.vert_start + i);
                center.x += bark.mesh.positions[vi * 3 + 0];
                center.y += bark.mesh.positions[vi * 3 + 1];
                center.z += bark.mesh.positions[vi * 3 + 2];
            }
            center = center * (1.0f / static_cast<float>(N));

            for (int i = 0; i < N; ++i) {
                const size_t vi = static_cast<size_t>(rm.vert_start + i);
                const ts::vec3 pos = {
                    bark.mesh.positions[vi * 3 + 0],
                    bark.mesh.positions[vi * 3 + 1],
                    bark.mesh.positions[vi * 3 + 2],
                };
                const ts::vec3 nrm = {
                    bark.mesh.normals[vi * 3 + 0],
                    bark.mesh.normals[vi * 3 + 1],
                    bark.mesh.normals[vi * 3 + 2],
                };
                const ts::vec3 radial = ts::normalized(pos - center);
                const float d = ts::dot(nrm, radial);
                ++checked;
                if (d < k_cos_15deg) ++violations;
            }
        }

        INFO("species=" << s.tree.species
             << " checked=" << checked << " violations=" << violations);
        REQUIRE(checked > 0);
        REQUIRE(violations == 0);
    }
}

// ---- [treegen_mesh_efficiency] ---------------------------------------------

TEST_CASE("[treegen_mesh_efficiency] continuation merge reduces vertex count",
          "[treegen][treegen_mesh_efficiency]") {
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

        // Count how many single-child continuations exist where N matches.
        const auto cc = child_counts(skel);
        int continuation_nodes = 0;
        for (size_t i = 1; i < skel.nodes.size(); ++i) {
            const int pi = skel.nodes[i].parent_index;
            if (pi >= 0 && cc[static_cast<size_t>(pi)] == 1) {
                ++continuation_nodes;
            }
        }
        // Must have continuations to test.
        REQUIRE(continuation_nodes > 0);

        // Build with default opts (continuation merge active).
        ts::BarkMeshOptions opts;
        opts.tree_height_m     = s.tree.height_m;
        opts.root_flare_factor = s.tree.root_flare_factor;
        opts.apply_fork_blend  = false; // isolate merge effect
        ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);
        const size_t vcount_merged = bark.mesh.positions.size() / 3;

        // The merged count should be strictly less than old count. We can't
        // easily build the old code, so we estimate: each continuation saves
        // (N+1) verts where N is the radial seg count at that depth.
        // Instead, verify at least 5% savings vs a naive estimate.
        // Naive vert count = sum over all non-root segments of (axial_segs+1)*(N+1).
        size_t naive_vcount = 0;
        for (size_t i = 1; i < skel.nodes.size(); ++i) {
            const auto& node = skel.nodes[i];
            if (node.parent_index < 0) continue;
            const auto& parent = skel.nodes[static_cast<size_t>(node.parent_index)];
            const float seg_len = ts::length(node.position - parent.position);
            if (seg_len < 1e-5f) continue;
            const int axial_segs = std::max(1, static_cast<int>(std::floor(seg_len / opts.axial_segment_length_m)));
            int N = node.depth == 0 ? opts.radial_seg_trunk
                  : node.depth == 1 ? opts.radial_seg_order1
                  : node.depth == 2 ? opts.radial_seg_order2
                  : opts.radial_seg_order3_plus;
            if (N < 3) N = 3;
            naive_vcount += static_cast<size_t>((axial_segs + 1) * (N + 1));
        }
        // Add tip cap centroid verts.
        int leaf_tips = 0;
        for (size_t i = 1; i < skel.nodes.size(); ++i) {
            if (cc[i] == 0 && skel.nodes[i].parent_index >= 0) ++leaf_tips;
        }
        // Merged vcount should be less than naive + tips.
        INFO("species=" << s.tree.species
             << " naive=" << naive_vcount
             << " merged=" << vcount_merged
             << " tips=" << leaf_tips
             << " continuation_nodes=" << continuation_nodes);
        // Each continuation node saves (N+1) verts (one ring skipped).
        // Verify savings > 5% of naive.
        const size_t savings = naive_vcount + static_cast<size_t>(leaf_tips) - vcount_merged;
        const float pct = static_cast<float>(savings) / static_cast<float>(naive_vcount) * 100.0f;
        INFO("savings=" << savings << " pct=" << pct);
        REQUIRE(pct > 5.0f);
    }
}

// ---- [treegen_tip_caps] ----------------------------------------------------

TEST_CASE("[treegen_tip_caps] every leaf node has end-cap geometry",
          "[treegen][treegen_tip_caps]") {
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

        const auto cc = child_counts(skel);

        // Count leaf nodes (non-root, 0 children).
        int leaf_count = 0;
        for (size_t i = 1; i < skel.nodes.size(); ++i) {
            if (cc[i] == 0 && skel.nodes[i].parent_index >= 0) {
                ++leaf_count;
            }
        }
        REQUIRE(leaf_count > 0);

        ts::BarkMeshOptions opts;
        opts.tree_height_m     = s.tree.height_m;
        opts.root_flare_factor = s.tree.root_flare_factor;
        ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);

        // Each tip cap emits 1 centroid vertex. Count vertices whose
        // position matches a leaf-node position (within tolerance).
        int cap_count = 0;
        const size_t vcount = bark.mesh.positions.size() / 3;
        for (size_t i = 1; i < skel.nodes.size(); ++i) {
            if (cc[i] != 0 || skel.nodes[i].parent_index < 0) continue;
            // This is a leaf node. Check if there's a vertex at its position.
            const ts::vec3 np = skel.nodes[i].position;
            bool found = false;
            for (size_t v = 0; v < vcount; ++v) {
                // Centroid vertices should be at exact node position.
                const float dx = bark.mesh.positions[v * 3 + 0] - np.x;
                const float dy = bark.mesh.positions[v * 3 + 1] - np.y;
                const float dz = bark.mesh.positions[v * 3 + 2] - np.z;
                if (dx * dx + dy * dy + dz * dz < 1e-10f) {
                    found = true;
                    break;
                }
            }
            if (found) ++cap_count;
        }

        INFO("species=" << s.tree.species
             << " leaf_count=" << leaf_count << " cap_count=" << cap_count);
        REQUIRE(cap_count == leaf_count);
    }
}
