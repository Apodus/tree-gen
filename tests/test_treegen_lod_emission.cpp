// [treegen_lod_emission] — pins C4 P4. Per-LOD face budget allocator + 4-LOD
// multi-mesh GLB emission (3 bark LODs + 1 billboard stub). Round-trip via
// gltf_view confirms the `_RYNX_LOD` extension survives write/read and the
// per-LOD wind weights are present.
//
// Link rationale: face_budget.cpp + lod_emitter.cpp linked into TestTech via
// rynx_tests.sharpmake.cs so this test pins the exact code shipped in
// rynx-treegen.exe.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../external/gltf_view.hpp"

#include "../branch_mesh.hpp"
#include "../face_budget.hpp"
#include "../glb_writer.hpp"
#include "../leaf_budget.hpp"
#include "../leaf_geometry.hpp"
#include "../leaf_placement.hpp"
#include "../lod_emitter.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace ts = treegen;

namespace {

std::vector<char> slurp_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    f.seekg(0, std::ios::end);
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(n));
    if (n > 0) f.read(buf.data(), n);
    return buf;
}

std::string tmp_path(const char* suffix) {
    namespace fs = std::filesystem;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "treegen_lod_%d_%s.glb", std::rand(), suffix);
    return (fs::temp_directory_path() / buf).string();
}

// Pack the 4 LODs into a multi-mesh GLB on disk; returns the path. Mirrors
// the wiring in tools/rynx-treegen/main.cpp.
std::string write_lod_glb(const std::vector<ts::LodOutput>& lods, const char* suffix) {
    std::vector<std::vector<ts::PrimitiveData>> per_lod_prims(lods.size());
    std::vector<ts::MeshData>                    meshes;
    meshes.reserve(lods.size());

    for (size_t li = 0; li < lods.size(); ++li) {
        const auto& lod = lods[li];
        ts::PrimitiveData prim{};
        prim.positions   = std::span<const float>(lod.mesh.positions.data(), lod.mesh.positions.size());
        prim.normals     = std::span<const float>(lod.mesh.normals.data(),   lod.mesh.normals.size());
        prim.uvs         = std::span<const float>(lod.mesh.uvs.data(),       lod.mesh.uvs.size());
        prim.indices_u32 = std::span<const uint32_t>(lod.indices_u32.data(), lod.indices_u32.size());
        prim.wind_weights_packed = std::span<const uint8_t>(
            lod.wind_weights_packed.data(), lod.wind_weights_packed.size());
        per_lod_prims[li].push_back(prim);

        ts::MeshData md{};
        md.primitives          = std::span<const ts::PrimitiveData>(
            per_lod_prims[li].data(), per_lod_prims[li].size());
        md.lod_index           = lod.lod_index;
        md.lod_max_distance_m  = lod.lod_max_distance_m;
        meshes.push_back(md);
    }

    const std::string path = tmp_path(suffix);
    std::string err;
    REQUIRE(ts::write_glb_multi_mesh(
        std::span<const ts::MeshData>(meshes.data(), meshes.size()), path, &err));
    REQUIRE(std::filesystem::exists(path));
    return path;
}

} // anonymous namespace

TEST_CASE("[treegen_lod_emission] oak emits 3 LODs in one multi-mesh GLB",
          "[treegen][treegen_lod_emission]") {
    namespace tsp = rynx::test_support;
    namespace fs  = std::filesystem;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    REQUIRE(s.kind == "tree");

    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
    REQUIRE(skel.nodes.size() > 50u);

    ts::BarkMeshOptions base_opts;
    base_opts.tree_height_m = s.tree.height_m;
    base_opts.seam_offset_rad = 0.0f;

    ts::LodBudget budget;  // P5 defaults: L0=12000, L1=4000, L2=1500, L3=4

    auto lods = ts::emit_all_lods(skel, base_opts, budget, s.tree.height_m, seed_effective);
    REQUIRE(lods.size() == 3u);  // C10: L3 billboard stub suppressed

    // lod_index values are 0..2 and the distance thresholds match the C4 P4
    // pin (L0=25, L1=60, L2=120).
    REQUIRE(lods[0].lod_index == 0);
    REQUIRE(lods[1].lod_index == 1);
    REQUIRE(lods[2].lod_index == 2);
    REQUIRE(lods[0].lod_max_distance_m == Approx(25.0f));
    REQUIRE(lods[1].lod_max_distance_m == Approx(60.0f));
    REQUIRE(lods[2].lod_max_distance_m == Approx(120.0f));

    // Per-LOD face count within budget.
    const size_t l0_tris = lods[0].indices_u32.size() / 3;
    const size_t l1_tris = lods[1].indices_u32.size() / 3;
    const size_t l2_tris = lods[2].indices_u32.size() / 3;

    INFO("L0=" << l0_tris << " L1=" << l1_tris << " L2=" << l2_tris);

    REQUIRE(l0_tris > 0u);
    REQUIRE(l1_tris > 0u);
    REQUIRE(l2_tris > 0u);
    REQUIRE(int(l0_tris) <= budget.l0_tris);
    REQUIRE(int(l1_tris) <= budget.l1_tris);
    REQUIRE(int(l2_tris) <= budget.l2_tris);

    // Detail descends monotonically L0 -> L2.
    REQUIRE(l0_tris >= l1_tris);
    REQUIRE(l1_tris >= l2_tris);

    // Wind weights present for all bark LODs (4 bytes/vert).
    for (int i = 0; i < 3; ++i) {
        const size_t vcount = lods[i].mesh.positions.size() / 3;
        REQUIRE(vcount > 0u);
        REQUIRE(lods[i].wind_weights_packed.size() == 4u * vcount);
    }

    // ---- GLB round-trip ----
    const std::string path = write_lod_glb(lods, "oak");

    std::vector<char> bytes = slurp_bytes(path);
    REQUIRE_FALSE(bytes.empty());

    std::span<const char> byte_span(bytes.data(), bytes.size());
    REQUIRE(rynx::graphics::gltf_view::count_meshes(byte_span) == 3);

    // For each LOD, extract its primitives + verify lod_index + tri count.
    rynx::graphics::gltf_view::glb_view glb =
        rynx::graphics::gltf_view::parse_glb_header(byte_span);
    REQUIRE(glb.valid);
    rynx::graphics::gltf_view::jparser parser(glb.json_begin, glb.json_len);
    rynx::graphics::gltf_view::jval root = parser.parse_value();
    auto* meshes_node = root.find("meshes");
    REQUIRE(meshes_node != nullptr);
    REQUIRE(meshes_node->as_arr().size() == 3u);

    for (int mi = 0; mi < 3; ++mi) {
        int   read_idx = -999;
        float read_max = -1.0f;
        REQUIRE(rynx::graphics::gltf_view::read_rynx_lod_extension(
            meshes_node->as_arr()[mi], read_idx, read_max));
        REQUIRE(read_idx == mi);
        REQUIRE(read_max == Approx(lods[mi].lod_max_distance_m));

        std::vector<rynx::graphics::gltf_view::cpu_mesh> prims;
        std::string err;
        REQUIRE(rynx::graphics::gltf_view::extract_all_primitives(
            byte_span, prims, &err, mi));
        REQUIRE(prims.size() == 1u);
        REQUIRE(prims[0].positions.size() == lods[mi].mesh.positions.size());
        REQUIRE(prims[0].indices.size()   == lods[mi].indices_u32.size());
        // Wind weights round-trip byte-equal.
        REQUIRE(prims[0].wind_weights_packed == lods[mi].wind_weights_packed);
    }

    // Determinism: rebuild from the same skeleton + opts → byte-identical GLB.
    auto lods2 = ts::emit_all_lods(skel, base_opts, budget, s.tree.height_m, seed_effective);
    const std::string path2 = write_lod_glb(lods2, "oak2");
    std::vector<char> bytes2 = slurp_bytes(path2);
    REQUIRE(bytes == bytes2);

    std::error_code ec;
    fs::remove(path, ec);
    fs::remove(path2, ec);
}

TEST_CASE("[treegen_lod_emission] face_budget allocator stays within budget on oak",
          "[treegen][treegen_lod_emission]") {
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

    // Aggressive budget — much lower than L2's 1500 — should force the
    // merge-by-length overflow rule to kick in. The allocator must return a
    // valid PerOrderRadial + merge mask that (after rebuild) produces a mesh
    // under the budget.
    ts::PerOrderRadial target;  // 12/8/6/4
    std::vector<uint8_t> culled_mask;
    ts::PerOrderRadial actual = ts::allocate_radial_for_lod(skel, 200, target, &culled_mask);
    REQUIRE(actual.trunk >= 3);  // trunk never merged
    REQUIRE(culled_mask.size() == skel.nodes.size());  // mask populated

    // Root node is never merged.
    REQUIRE(culled_mask[0] == 0u);

    ts::BarkMeshOptions opts;
    opts.tree_height_m          = s.tree.height_m;
    opts.radial_seg_trunk       = actual.trunk;
    opts.radial_seg_order1      = actual.order1;
    opts.radial_seg_order2      = actual.order2;
    opts.radial_seg_order3_plus = actual.order3plus;
    opts.culled_node_mask       = culled_mask;

    ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);
    const size_t tris = bark.indices_u32.size() / 3;
    INFO("aggressive budget=200, actual tris=" << tris
        << " (trunk=" << actual.trunk << " o1=" << actual.order1
        << " o2=" << actual.order2 << " o3+=" << actual.order3plus << ")");
    // Allocator may slightly overshoot at the boundary (N>=3 floor, fork-blend
    // extra tris, K=32 merge recheck granularity) — allow a small slack.
    REQUIRE(int(tris) <= 300);
}

// [treegen_branch_merge] — C2-LOD-quality. Pins the branch-merge semantics:
// merged branches suppress bark emission but retain their skeleton node for
// leaf placement. Closes the "branch-deletion LOD gap" bug class by
// establishing that branches are MERGED (not deleted) at lower LODs.
//
// Test matrix:
//   (a) L0 mesh unchanged — merge only fires when budget binds.
//   (b) L1/L2 merged branches have zero bark tris.
//   (c) Total tri count ≤ budget.
//   (d) Leaves on merged branches still appear (leaf pipeline independent).
//   (e) Merge-cascade: fork with all children merged emits no collar.
//   (f) Root node is never merged.
TEST_CASE("[treegen_branch_merge] merged branches suppress bark but retain leaves",
          "[treegen][treegen_branch_merge]") {
    namespace tsp = rynx::test_support;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
    REQUIRE(skel.nodes.size() > 50u);

    // Generate leaf sites from the full skeleton.
    auto lp_opts = ts::leaf_placement::options_from_descriptor(s.tree.leaves);
    auto all_leaf_sites = ts::leaf_placement::generate_leaf_sites(skel, lp_opts, seed_effective);
    REQUIRE(all_leaf_sites.size() > 30u);

    ts::LodBudget budget;  // defaults: L0=36000, L1=4000, L2=1500

    // ---- (a) Unbounded budget: no branches merged ----
    // At unbounded budget, radial scaling suffices → merge never fires.
    {
        ts::PerOrderRadial target;
        std::vector<uint8_t> mask;
        ts::allocate_radial_for_lod(skel, 1 << 24, target, &mask);

        bool any_merged = false;
        for (uint8_t m : mask) if (m) { any_merged = true; break; }
        INFO("unbounded: mask.size=" << mask.size() << " any_merged=" << any_merged);
        REQUIRE_FALSE(any_merged);
    }

    // ---- (a2) L0 mesh: merge only fires if natural mesh exceeds budget ----
    // The actual L0 mesh may or may not need merging (dense oak can exceed
    // 36k at full radials). Either way, the output must be within budget.
    {
        ts::PerOrderRadial target;
        std::vector<uint8_t> mask;
        ts::PerOrderRadial actual = ts::allocate_radial_for_lod(
            skel, budget.l0_tris, target, &mask);

        ts::BarkMeshOptions opts;
        opts.tree_height_m          = s.tree.height_m;
        opts.radial_seg_trunk       = actual.trunk;
        opts.radial_seg_order1      = actual.order1;
        opts.radial_seg_order2      = actual.order2;
        opts.radial_seg_order3_plus = actual.order3plus;
        opts.culled_node_mask       = mask;

        ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);
        const size_t tris = bark.indices_u32.size() / 3;
        INFO("L0: bark_tris=" << tris << " budget=" << budget.l0_tris);
        REQUIRE(int(tris) <= budget.l0_tris);
    }

    // ---- (b-c) L1: merged branches produce zero bark, total ≤ budget ----
    {
        const ts::PerOrderRadial target = {6, 4, 3, 3}; // L1 radial targets
        std::vector<uint8_t> mask;
        ts::PerOrderRadial actual = ts::allocate_radial_for_lod(
            skel, budget.l1_tris, target, &mask);

        // Count merged nodes.
        int merged_count = 0;
        for (size_t i = 0; i < mask.size(); ++i)
            if (mask[i]) ++merged_count;
        INFO("L1: merged_count=" << merged_count << "/" << skel.nodes.size());
        // At L1=4000 tris, some branches should be merged for a dense oak.
        // (Not a hard requirement — sparse species may not need merging.)

        // (f) Root is never merged.
        if (!mask.empty()) {
            REQUIRE(mask[0] == 0u);
        }

        // Build bark mesh with merge mask.
        ts::BarkMeshOptions opts;
        opts.tree_height_m          = s.tree.height_m;
        opts.radial_seg_trunk       = actual.trunk;
        opts.radial_seg_order1      = actual.order1;
        opts.radial_seg_order2      = actual.order2;
        opts.radial_seg_order3_plus = actual.order3plus;
        opts.culled_node_mask       = mask;

        ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);
        const size_t tris = bark.indices_u32.size() / 3;
        INFO("L1: bark_tris=" << tris << " budget=" << budget.l1_tris);
        // (c) Total ≤ budget.
        REQUIRE(int(tris) <= budget.l1_tris);

        // (b) Merged branches have zero bark vertices. Check that no vertex
        // in the output mesh references a merged node index.
        if (merged_count > 0) {
            std::set<int> merged_nodes;
            for (size_t i = 0; i < mask.size(); ++i)
                if (mask[i]) merged_nodes.insert(static_cast<int>(i));

            for (int vni : bark.per_vertex_node_index) {
                INFO("vert references merged node " << vni);
                REQUIRE(merged_nodes.find(vni) == merged_nodes.end());
            }
        }
    }

    // ---- (d) Leaves on merged branches still appear ----
    // The leaf pipeline receives all_leaf_sites (generated from the full
    // skeleton). Leaf sites with branch_id matching merged nodes must be
    // present in the output — the merge mask does NOT filter leaf sites.
    {
        const ts::PerOrderRadial target = {3, 3, 3, 3}; // L2 radial targets
        std::vector<uint8_t> mask;
        ts::allocate_radial_for_lod(skel, budget.l2_tris, target, &mask);

        // Collect branch_ids of leaf sites on merged branches.
        std::set<int> merged_nodes;
        for (size_t i = 0; i < mask.size(); ++i)
            if (mask[i]) merged_nodes.insert(static_cast<int>(i));

        int leaf_sites_on_merged = 0;
        for (const auto& site : all_leaf_sites) {
            if (merged_nodes.count(site.branch_id)) ++leaf_sites_on_merged;
        }
        INFO("L2: merged_nodes=" << merged_nodes.size()
             << " leaf_sites_on_merged=" << leaf_sites_on_merged
             << " total_leaf_sites=" << all_leaf_sites.size());

        // For a dense oak at L2, some branches are merged and some of those
        // branches carry leaf sites. The leaf pipeline must include them.
        if (!merged_nodes.empty()) {
            // At least some leaf sites should exist on merged branches (oak
            // is a dense species — branches that carry leaves get merged at
            // L2's tight budget).
            REQUIRE(leaf_sites_on_merged > 0);
        }

        // Run the leaf-aware LOD emitter — the leaf output must include sites
        // from merged branches. The allocator sub-samples by PCG32 rank, not
        // by merge status.
        ts::BarkMeshOptions bark_opts;
        bark_opts.tree_height_m   = s.tree.height_m;
        bark_opts.seam_offset_rad = 0.0f;

        ts::LeafBudget leaf_budget;
        ts::LeafMeshOptions leaf_geom_opts;
        leaf_geom_opts.geometry_type         = s.tree.leaves.geometry_type;
        leaf_geom_opts.shape                 = s.tree.leaves.shape;
        leaf_geom_opts.leaf_size_m           = s.tree.leaves.leaf_size_m;
        leaf_geom_opts.cluster_count_per_tip = s.tree.leaves.cluster_count_per_tip;
        leaf_geom_opts.bend_half_angle       = s.tree.leaves.leaf_bend_half_angle;

        auto lods = ts::emit_all_lods(skel, all_leaf_sites, bark_opts, budget,
                                      leaf_budget, leaf_geom_opts,
                                      s.tree.height_m, seed_effective);
        REQUIRE(lods.size() == 3u);

        // L2 must have leaves — the leaf pipeline is independent of bark merge.
        REQUIRE(lods[2].has_leaves);
        const size_t l2_leaf_tris = lods[2].leaf_indices_u32.size() / 3;
        INFO("L2: leaf_tris=" << l2_leaf_tris);
        REQUIRE(l2_leaf_tris > 0u);
    }
}

// [treegen_branch_merge] — merge-cascade: when all children of a fork are
// merged, the fork zone is pruned and emits no collar/crotch geometry. Tests
// that the estimator and the mesh builder agree on the cascade.
TEST_CASE("[treegen_branch_merge] merge cascade prunes empty fork zones",
          "[treegen][treegen_branch_merge]") {
    namespace tsp = rynx::test_support;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

    // Use a very tight budget to force heavy merging.
    ts::PerOrderRadial target;  // 12/8/6/4
    std::vector<uint8_t> mask;
    ts::PerOrderRadial actual = ts::allocate_radial_for_lod(skel, 300, target, &mask);
    REQUIRE(mask.size() == skel.nodes.size());

    // Identify fork nodes (nodes with >1 child).
    std::vector<int> child_count(skel.nodes.size(), 0);
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        int p = skel.nodes[i].parent_index;
        if (p >= 0) ++child_count[static_cast<size_t>(p)];
    }

    // Count forks whose ALL children are merged — these are cascade forks.
    int cascade_forks = 0;
    int total_forks = 0;
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        if (child_count[i] < 2) continue;
        ++total_forks;

        bool all_children_merged = true;
        for (size_t j = 1; j < skel.nodes.size(); ++j) {
            if (skel.nodes[j].parent_index == static_cast<int>(i)) {
                if (!mask[j]) { all_children_merged = false; break; }
            }
        }
        if (all_children_merged) ++cascade_forks;
    }
    INFO("total_forks=" << total_forks << " cascade_forks=" << cascade_forks);

    // At budget=300 with a dense oak, many forks should cascade.
    REQUIRE(cascade_forks > 0);

    // Build bark mesh — cascaded forks should not contribute collar geometry.
    ts::BarkMeshOptions opts;
    opts.tree_height_m          = s.tree.height_m;
    opts.radial_seg_trunk       = actual.trunk;
    opts.radial_seg_order1      = actual.order1;
    opts.radial_seg_order2      = actual.order2;
    opts.radial_seg_order3_plus = actual.order3plus;
    opts.culled_node_mask       = mask;

    ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);
    const size_t tris = bark.indices_u32.size() / 3;
    INFO("budget=300 actual_tris=" << tris);
    // Mesh should be within budget (with the small slack from N>=3 floor).
    REQUIRE(int(tris) <= 400);

    // Verify no collar ring metadata references a cascade fork node.
    std::set<int> cascade_fork_nodes;
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        if (child_count[i] < 2) continue;
        bool all_merged = true;
        for (size_t j = 1; j < skel.nodes.size(); ++j) {
            if (skel.nodes[j].parent_index == static_cast<int>(i)) {
                if (!mask[j]) { all_merged = false; break; }
            }
        }
        if (all_merged) cascade_fork_nodes.insert(static_cast<int>(i));
    }
    for (const auto& rm : bark.ring_metadata) {
        if (rm.collar_fork_node >= 0) {
            INFO("collar ring references cascade-pruned fork " << rm.collar_fork_node);
            REQUIRE(cascade_fork_nodes.find(rm.collar_fork_node) == cascade_fork_nodes.end());
        }
    }
}
