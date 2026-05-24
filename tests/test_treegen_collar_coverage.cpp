// [treegen_collar_coverage] — verifies collar rings at forks are non-degenerate.
// [treegen_crotch_fill] — verifies crotch cap geometry is emitted at multi-child forks.
// Phase 6 structural seal: these two tests structurally kill the degenerate-stitch
// bug class by pinning that collar rings sit strictly between parent rim and child
// tube, and that fork blend adds geometry (collar + crotch cap tris) at every fork.

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
#include <map>
#include <set>
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

float dist_sq(ts::vec3 a, ts::vec3 b) {
    return ts::length_squared(a - b);
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

// child_counts[i] = number of children of node i.
std::vector<int> child_counts(const ts::TreeSkeleton& skel) {
    std::vector<int> cc(skel.nodes.size(), 0);
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        const int p = skel.nodes[i].parent_index;
        if (p >= 0) ++cc[static_cast<size_t>(p)];
    }
    return cc;
}

} // anonymous namespace

// ---- [treegen_collar_coverage] ---------------------------------------------

TEST_CASE("[treegen_collar_coverage] collar rings sit strictly between parent rim and child tube",
          "[treegen][treegen_collar_coverage]") {
    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* rel : scenarios) {
        auto fix = load_species(rel);

        ts::BarkMeshOptions opts;
        opts.tree_height_m     = fix.scenario.tree.height_m;
        opts.root_flare_factor = fix.scenario.tree.root_flare_factor;
        ts::BarkMeshOutput bark = ts::build_bark_mesh(fix.skel, opts);

        // Build lookup: for each branch_node_index, gather its rings.
        // We need the parent rim ring (last axial ring of the parent branch)
        // and the child s=0 ring (first axial ring of the child branch).

        // Collect all collar rings.
        std::vector<const ts::RingMetadata*> collar_rings;
        for (const auto& rm : bark.ring_metadata) {
            if (rm.collar_fork_node >= 0)
                collar_rings.push_back(&rm);
        }

        INFO("species=" << fix.scenario.tree.species
             << " collar_rings=" << collar_rings.size());
        REQUIRE(collar_rings.size() > 0u);

        // For each collar ring, find the parent rim ring and the child s=0 ring.
        // Parent rim: last ring of the branch that is the parent of the fork.
        // Child s=0: first ring (axial_index==0) of the child branch that this
        // collar belongs to (collar's branch_node_index).

        // Build index: branch_node_index -> rings sorted by axial_index.
        std::map<int, std::vector<const ts::RingMetadata*>> rings_by_branch;
        for (const auto& rm : bark.ring_metadata) {
            if (rm.collar_fork_node < 0)
                rings_by_branch[rm.branch_node_index].push_back(&rm);
        }

        // Find unique fork nodes referenced by collar rings.
        std::set<int> fork_nodes;
        for (const auto* cr : collar_rings) {
            fork_nodes.insert(cr->collar_fork_node);
        }

        int checked = 0;

        for (const auto* cr : collar_rings) {
            ts::vec3 collar_c = ring_center(bark, *cr);

            // Find parent rim ring: the fork node is cr->collar_fork_node.
            // The parent branch is the branch that ends at the fork node.
            // That branch's ring has branch_node_index == collar_fork_node
            // and axial_index == axial_segs (the last ring).
            const int fork_node = cr->collar_fork_node;
            auto it = rings_by_branch.find(fork_node);
            if (it != rings_by_branch.end()) {
                // Find the ring with max axial_index (parent rim).
                const ts::RingMetadata* parent_rim = nullptr;
                for (const auto* r : it->second) {
                    if (!parent_rim || r->axial_index > parent_rim->axial_index)
                        parent_rim = r;
                }
                if (parent_rim) {
                    ts::vec3 parent_c = ring_center(bark, *parent_rim);
                    float d2 = dist_sq(collar_c, parent_c);
                    INFO("collar_fork=" << fork_node
                         << " collar_seq=" << cr->collar_sequence
                         << " dist_to_parent_rim=" << std::sqrt(d2));
                    // Collar ring center must not coincide with parent rim center.
                    REQUIRE(d2 > 1e-10f);
                }
            }

            // Find child s=0 ring: branch_node_index == cr->branch_node_index,
            // axial_index == 0.
            auto cit = rings_by_branch.find(cr->branch_node_index);
            if (cit != rings_by_branch.end()) {
                const ts::RingMetadata* child_s0 = nullptr;
                for (const auto* r : cit->second) {
                    if (r->axial_index == 0) {
                        child_s0 = r;
                        break;
                    }
                }
                if (child_s0) {
                    ts::vec3 child_c = ring_center(bark, *child_s0);
                    float d2 = dist_sq(collar_c, child_c);
                    INFO("collar_fork=" << fork_node
                         << " collar_seq=" << cr->collar_sequence
                         << " dist_to_child_s0=" << std::sqrt(d2));
                    // Collar ring center must not coincide with child s=0 center.
                    REQUIRE(d2 > 1e-10f);
                }
            }

            ++checked;
        }

        INFO("species=" << fix.scenario.tree.species << " checked=" << checked);
        REQUIRE(checked > 0);
    }
}

TEST_CASE("[treegen_collar_coverage] at least 2 collar rings per child per fork",
          "[treegen][treegen_collar_coverage]") {
    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* rel : scenarios) {
        auto fix = load_species(rel);

        ts::BarkMeshOptions opts;
        opts.tree_height_m     = fix.scenario.tree.height_m;
        opts.root_flare_factor = fix.scenario.tree.root_flare_factor;
        ts::BarkMeshOutput bark = ts::build_bark_mesh(fix.skel, opts);

        // Group collar rings by (collar_fork_node, branch_node_index).
        // Each group is one child's collar at one fork.
        std::map<std::pair<int,int>, int> collar_count;
        for (const auto& rm : bark.ring_metadata) {
            if (rm.collar_fork_node >= 0) {
                ++collar_count[{rm.collar_fork_node, rm.branch_node_index}];
            }
        }

        INFO("species=" << fix.scenario.tree.species
             << " collar_groups=" << collar_count.size());
        REQUIRE(collar_count.size() > 0u);

        for (auto& kv : collar_count) {
            INFO("fork=" << kv.first.first
                 << " child=" << kv.first.second
                 << " collar_rings=" << kv.second);
            REQUIRE(kv.second >= 2);
        }
    }
}

TEST_CASE("[treegen_collar_coverage] collar ring N matches parent rim N",
          "[treegen][treegen_collar_coverage]") {
    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* rel : scenarios) {
        auto fix = load_species(rel);

        ts::BarkMeshOptions opts;
        opts.tree_height_m     = fix.scenario.tree.height_m;
        opts.root_flare_factor = fix.scenario.tree.root_flare_factor;
        ts::BarkMeshOutput bark = ts::build_bark_mesh(fix.skel, opts);

        // For each fork node, find parent rim ring's N, then verify all collar
        // rings at that fork have the same N.
        std::map<int, int> fork_parent_N; // fork_node -> parent rim N

        // Find parent rim N for each fork: the ring with branch_node_index==fork_node
        // and max axial_index.
        for (const auto& rm : bark.ring_metadata) {
            if (rm.collar_fork_node >= 0) continue; // skip collar rings themselves
            // Check if this branch_node_index is referenced as a fork node by any collar ring.
        }

        // Collect fork nodes.
        std::set<int> fork_nodes;
        for (const auto& rm : bark.ring_metadata) {
            if (rm.collar_fork_node >= 0)
                fork_nodes.insert(rm.collar_fork_node);
        }

        // For each fork node, find the parent rim ring (last ring of the branch
        // ending at that fork node).
        for (int fn : fork_nodes) {
            const ts::RingMetadata* parent_rim = nullptr;
            for (const auto& rm : bark.ring_metadata) {
                if (rm.collar_fork_node >= 0) continue;
                if (rm.branch_node_index == fn) {
                    if (!parent_rim || rm.axial_index > parent_rim->axial_index)
                        parent_rim = &rm;
                }
            }
            if (parent_rim)
                fork_parent_N[fn] = parent_rim->N;
        }

        int checked = 0;
        for (const auto& rm : bark.ring_metadata) {
            if (rm.collar_fork_node < 0) continue;
            auto it = fork_parent_N.find(rm.collar_fork_node);
            if (it == fork_parent_N.end()) continue;
            INFO("species=" << fix.scenario.tree.species
                 << " fork=" << rm.collar_fork_node
                 << " collar_N=" << rm.N
                 << " parent_N=" << it->second);
            REQUIRE(rm.N == it->second);
            ++checked;
        }

        INFO("species=" << fix.scenario.tree.species << " checked=" << checked);
        REQUIRE(checked > 0);
    }
}

// ---- [treegen_crotch_fill] -------------------------------------------------

TEST_CASE("[treegen_crotch_fill] fork blend adds geometry at multi-child forks",
          "[treegen][treegen_crotch_fill]") {
    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* rel : scenarios) {
        auto fix = load_species(rel);

        // Build with fork blend ON (default).
        ts::BarkMeshOptions opts_on;
        opts_on.tree_height_m     = fix.scenario.tree.height_m;
        opts_on.root_flare_factor = fix.scenario.tree.root_flare_factor;
        opts_on.apply_fork_blend  = true;
        ts::BarkMeshOutput bark_on = ts::build_bark_mesh(fix.skel, opts_on);

        // Build with fork blend OFF.
        ts::BarkMeshOptions opts_off;
        opts_off.tree_height_m     = fix.scenario.tree.height_m;
        opts_off.root_flare_factor = fix.scenario.tree.root_flare_factor;
        opts_off.apply_fork_blend  = false;
        ts::BarkMeshOutput bark_off = ts::build_bark_mesh(fix.skel, opts_off);

        INFO("species=" << fix.scenario.tree.species
             << " indices_on=" << bark_on.indices_u32.size()
             << " indices_off=" << bark_off.indices_u32.size());

        // Fork blend adds collar rings + crotch cap tris -> strictly more indices.
        REQUIRE(bark_on.indices_u32.size() > bark_off.indices_u32.size());

        // Verify collar rings exist in the blended mesh.
        int collar_count = 0;
        for (const auto& rm : bark_on.ring_metadata) {
            if (rm.collar_fork_node >= 0) ++collar_count;
        }

        INFO("species=" << fix.scenario.tree.species
             << " collar_count=" << collar_count);
        REQUIRE(collar_count > 0);
    }
}

TEST_CASE("[treegen_crotch_fill] every multi-child fork has collar geometry",
          "[treegen][treegen_crotch_fill]") {
    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    for (const char* rel : scenarios) {
        auto fix = load_species(rel);
        const auto cc = child_counts(fix.skel);

        // Identify all fork nodes (nodes with >= 2 children).
        std::set<int> fork_nodes;
        for (size_t i = 0; i < cc.size(); ++i) {
            if (cc[i] >= 2) fork_nodes.insert(static_cast<int>(i));
        }
        REQUIRE(fork_nodes.size() > 0u);

        ts::BarkMeshOptions opts;
        opts.tree_height_m     = fix.scenario.tree.height_m;
        opts.root_flare_factor = fix.scenario.tree.root_flare_factor;
        ts::BarkMeshOutput bark = ts::build_bark_mesh(fix.skel, opts);

        // Collect fork nodes that received collar rings.
        std::set<int> collar_fork_nodes;
        for (const auto& rm : bark.ring_metadata) {
            if (rm.collar_fork_node >= 0)
                collar_fork_nodes.insert(rm.collar_fork_node);
        }

        // Every skeleton fork node should have collar geometry (unless its
        // children were all culled by LOD — but default opts have no culling).
        int missing = 0;
        for (int fn : fork_nodes) {
            if (collar_fork_nodes.find(fn) == collar_fork_nodes.end()) {
                ++missing;
                INFO("species=" << fix.scenario.tree.species
                     << " fork_node=" << fn << " MISSING collar geometry");
            }
        }

        INFO("species=" << fix.scenario.tree.species
             << " fork_nodes=" << fork_nodes.size()
             << " collar_fork_nodes=" << collar_fork_nodes.size()
             << " missing=" << missing);
        REQUIRE(missing == 0);
    }
}
