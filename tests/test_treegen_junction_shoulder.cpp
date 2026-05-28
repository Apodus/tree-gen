// [treegen_junction_shoulder] — pins junction shoulder swell at fork junctions.
// [treegen_junction_approach_monotonic] — pins monotonic radius increase in
// the swell zone approaching forks (last ~30% of branch).
// C1 P1 structural seal: these tests kill the fork-junction-necking visual
// artifact by pinning that approaching-fork rings are widened and that the
// swell is monotonically non-decreasing.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../branch_mesh.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"
#include "../vec3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ts = treegen;

// ---- helpers ---------------------------------------------------------------

namespace {

ts::vec3 ring_center(const ts::BarkMeshOutput& bark, const ts::RingMetadata& rm) {
    ts::vec3 c = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < rm.N; ++i) {
        const size_t vi = static_cast<size_t>(rm.vert_start + i);
        c.x += bark.mesh.positions[vi * 3 + 0];
        c.y += bark.mesh.positions[vi * 3 + 1];
        c.z += bark.mesh.positions[vi * 3 + 2];
    }
    return c * (1.0f / static_cast<float>(rm.N));
}

float ring_avg_radius(const ts::BarkMeshOutput& bark, const ts::RingMetadata& rm) {
    ts::vec3 c = ring_center(bark, rm);
    float sum = 0.0f;
    for (int i = 0; i < rm.N; ++i) {
        const size_t vi = static_cast<size_t>(rm.vert_start + i);
        const float dx = bark.mesh.positions[vi * 3 + 0] - c.x;
        const float dy = bark.mesh.positions[vi * 3 + 1] - c.y;
        const float dz = bark.mesh.positions[vi * 3 + 2] - c.z;
        sum += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    return sum / static_cast<float>(rm.N);
}

struct SpeciesFixture {
    ts::Scenario      scenario;
    ts::TreeSkeleton  skel;
    std::string       path;
};

SpeciesFixture load_species(const char* scenario_rel) {
    namespace tsp = rynx::test_support;
    SpeciesFixture f;
    f.path = tsp::find_repo_file(scenario_rel);
    REQUIRE_FALSE(f.path.empty());
    f.scenario = ts::load_scenario(f.path);
    REQUIRE(f.scenario.kind == "tree");
    f.skel = ts::grow_skeleton(f.scenario.tree, 42ull ^ f.scenario.scenario_fnv);
    return f;
}

std::vector<int> child_counts(const ts::TreeSkeleton& skel) {
    std::vector<int> cc(skel.nodes.size(), 0);
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        const int p = skel.nodes[i].parent_index;
        if (p >= 0) ++cc[static_cast<size_t>(p)];
    }
    return cc;
}

} // anonymous namespace

// ---- [treegen_junction_shoulder] -------------------------------------------

TEST_CASE("[treegen_junction_shoulder] fork parent-rim ring is swelled",
          "[treegen][treegen_junction_shoulder]") {
    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* rel : scenarios) {
        auto fix = load_species(rel);
        const auto cc = child_counts(fix.skel);

        ts::BarkMeshOptions opts;
        opts.tree_height_m             = fix.scenario.tree.height_m;
        opts.root_flare_factor         = fix.scenario.tree.root_flare_factor;
        opts.junction_shoulder_factor  = 0.15f;
        ts::BarkMeshOutput bark = ts::build_bark_mesh(fix.skel, opts);

        int checked = 0;

        for (size_t f = 0; f < fix.skel.nodes.size(); ++f) {
            if (cc[f] < 2) continue;
            const auto& fork_node = fix.skel.nodes[f];
            if (fork_node.parent_index < 0) continue;

            // Find the parent-rim ring: branch_node_index == f,
            // collar_fork_node < 0, max axial_index.
            const ts::RingMetadata* parent_rim = nullptr;
            for (const auto& rm : bark.ring_metadata) {
                if (rm.collar_fork_node >= 0) continue;
                if (rm.branch_node_index != static_cast<int>(f)) continue;
                if (!parent_rim || rm.axial_index > parent_rim->axial_index)
                    parent_rim = &rm;
            }
            if (!parent_rim) continue;

            const float measured = ring_avg_radius(bark, *parent_rim);
            const float parent_r = fix.skel.nodes[static_cast<size_t>(fork_node.parent_index)].radius;
            // Conservative 5% threshold to account for species variation.
            const float expected_min = std::min(
                fork_node.radius * 1.05f, parent_r);

            INFO("species=" << fix.scenario.tree.species
                 << " fork=" << f
                 << " measured=" << measured
                 << " expected_min=" << expected_min);
            REQUIRE(measured >= expected_min - 1e-5f);
            ++checked;
        }

        INFO("species=" << fix.scenario.tree.species << " checked=" << checked);
        REQUIRE(checked > 0);
    }
}

// ---- [treegen_junction_approach_monotonic] ---------------------------------

TEST_CASE("[treegen_junction_approach_monotonic] swell zone radii are non-decreasing",
          "[treegen][treegen_junction_approach_monotonic]") {
    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* rel : scenarios) {
        auto fix = load_species(rel);
        const auto cc = child_counts(fix.skel);

        ts::BarkMeshOptions opts;
        opts.tree_height_m             = fix.scenario.tree.height_m;
        opts.root_flare_factor         = fix.scenario.tree.root_flare_factor;
        opts.junction_shoulder_factor  = 0.15f;
        ts::BarkMeshOutput bark = ts::build_bark_mesh(fix.skel, opts);

        int violations = 0;
        int checked_forks = 0;

        for (size_t f = 0; f < fix.skel.nodes.size(); ++f) {
            if (cc[f] < 2) continue;

            // Collect non-collar rings for this branch, sorted by axial_index.
            std::vector<const ts::RingMetadata*> branch_rings;
            for (const auto& rm : bark.ring_metadata) {
                if (rm.collar_fork_node >= 0) continue;
                if (rm.branch_node_index != static_cast<int>(f)) continue;
                branch_rings.push_back(&rm);
            }
            if (branch_rings.empty()) continue;

            std::sort(branch_rings.begin(), branch_rings.end(),
                [](const ts::RingMetadata* a, const ts::RingMetadata* b) {
                    return a->axial_index < b->axial_index;
                });

            const int axial_segs = branch_rings.back()->axial_segs;
            if (axial_segs < 2) continue;

            // Swell zone: rings where axial_index / axial_segs >= 0.7.
            const int swell_start = static_cast<int>(std::floor(0.7f * static_cast<float>(axial_segs)));

            float prev_radius = -1.0f;
            for (const auto* rm : branch_rings) {
                if (rm->axial_index < swell_start) continue;
                const float r = ring_avg_radius(bark, *rm);
                if (prev_radius >= 0.0f && r < prev_radius - 1e-5f) {
                    INFO("species=" << fix.scenario.tree.species
                         << " fork=" << f
                         << " axial_index=" << rm->axial_index
                         << " radius=" << r
                         << " prev_radius=" << prev_radius);
                    ++violations;
                }
                prev_radius = r;
            }
            ++checked_forks;
        }

        INFO("species=" << fix.scenario.tree.species
             << " checked_forks=" << checked_forks
             << " violations=" << violations);
        REQUIRE(violations == 0);
    }
}
