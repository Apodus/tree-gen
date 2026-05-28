// [treegen_lod_leaves] — pins C5 P3. Per-LOD leaf budget + leaf-aware LOD
// chain integration. Tool sources (leaf_budget.cpp + leaf_geometry.cpp +
// leaf_shapes.cpp + lod_emitter.cpp + ...) link into TestTech via
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

#include <algorithm>
#include <array>
#include <cmath>
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
    char buf[80];
    std::snprintf(buf, sizeof(buf), "treegen_lod_leaves_%d_%s.glb", std::rand(), suffix);
    return (fs::temp_directory_path() / buf).string();
}

// Pack the 4 LODs into a multi-mesh GLB on disk; bark = prim[0],
// leaves (when present) = prim[1] with MASK material.
std::string write_lod_glb(const std::vector<ts::LodOutput>& lods, const char* suffix) {
    std::vector<std::vector<ts::PrimitiveData>> per_lod_prims(lods.size());
    std::vector<ts::MeshData>                    meshes;
    meshes.reserve(lods.size());

    for (size_t li = 0; li < lods.size(); ++li) {
        const auto& lod = lods[li];
        ts::PrimitiveData bark_prim{};
        bark_prim.positions   = std::span<const float>(lod.mesh.positions.data(), lod.mesh.positions.size());
        bark_prim.normals     = std::span<const float>(lod.mesh.normals.data(),   lod.mesh.normals.size());
        bark_prim.uvs         = std::span<const float>(lod.mesh.uvs.data(),       lod.mesh.uvs.size());
        bark_prim.indices_u32 = std::span<const uint32_t>(lod.indices_u32.data(), lod.indices_u32.size());
        bark_prim.wind_weights_packed = std::span<const uint8_t>(
            lod.wind_weights_packed.data(), lod.wind_weights_packed.size());
        per_lod_prims[li].push_back(bark_prim);

        if (lod.has_leaves) {
            ts::PrimitiveData leaf_prim{};
            leaf_prim.positions   = std::span<const float>(
                lod.leaf_positions.data(), lod.leaf_positions.size());
            leaf_prim.normals     = std::span<const float>(
                lod.leaf_normals.data(), lod.leaf_normals.size());
            leaf_prim.uvs         = std::span<const float>(
                lod.leaf_uvs.data(), lod.leaf_uvs.size());
            leaf_prim.indices_u32 = std::span<const uint32_t>(
                lod.leaf_indices_u32.data(), lod.leaf_indices_u32.size());
            leaf_prim.wind_weights_packed = std::span<const uint8_t>(
                lod.leaf_wind_weights_packed.data(),
                lod.leaf_wind_weights_packed.size());
            leaf_prim.material.alpha_mode   = "MASK";
            leaf_prim.material.alpha_cutoff = 0.5f;
            per_lod_prims[li].push_back(leaf_prim);
        }

        ts::MeshData md{};
        md.primitives            = std::span<const ts::PrimitiveData>(
            per_lod_prims[li].data(), per_lod_prims[li].size());
        md.lod_index             = lod.lod_index;
        md.lod_max_distance_m    = lod.lod_max_distance_m;
        md.lod_screen_height_px  = lod.lod_screen_height_px;
        meshes.push_back(md);
    }

    const std::string path = tmp_path(suffix);
    std::string err;
    REQUIRE(ts::write_glb_multi_mesh(
        std::span<const ts::MeshData>(meshes.data(), meshes.size()), path, &err));
    REQUIRE(std::filesystem::exists(path));
    return path;
}

bool is_prefix_of(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() > b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

}  // anonymous namespace

TEST_CASE("[treegen_lod_leaves] oak default-budget per-LOD invariants + multi-primitive GLB round-trip",
          "[treegen][treegen_lod_leaves]") {
    namespace tsp = rynx::test_support;
    namespace fs  = std::filesystem;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    INFO("fixture=" << fixture);
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    REQUIRE(s.kind == "tree");

    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

    // Populate skel.leaf_sites via the C5 P1 pipeline so the leaf-aware
    // overload has real sites to subsample.
    auto lp_opts = ts::leaf_placement::options_from_descriptor(s.tree.leaves);
    skel.leaf_sites = ts::leaf_placement::generate_leaf_sites(skel, lp_opts, seed_effective);
    REQUIRE(skel.leaf_sites.size() > 100u);

    ts::BarkMeshOptions base_opts;
    base_opts.tree_height_m   = s.tree.height_m;
    base_opts.seam_offset_rad = 0.0f;

    ts::LodBudget   bark_budget;  // P5 defaults: 12000/4000/1500/4
    ts::LeafBudget  leaf_budget;  // defaults: 4000/1200/400/0

    ts::LeafMeshOptions leaf_geom_opts;
    leaf_geom_opts.geometry_type         = s.tree.leaves.geometry_type;
    leaf_geom_opts.shape                 = s.tree.leaves.shape;
    leaf_geom_opts.leaf_size_m           = s.tree.leaves.leaf_size_m;
    leaf_geom_opts.cluster_count_per_tip = s.tree.leaves.cluster_count_per_tip;
    leaf_geom_opts.bend_half_angle       = s.tree.leaves.leaf_bend_half_angle;

    auto lods = ts::emit_all_lods(skel, skel.leaf_sites, base_opts, bark_budget,
                                  leaf_budget, leaf_geom_opts,
                                  s.tree.height_m, seed_effective);
    REQUIRE(lods.size() == 3u);  // C10: L3 billboard stub suppressed

    // ---- Bark + leaf per-LOD budget compliance ----
    for (int i = 0; i < 3; ++i) {
        const size_t bark_tris = lods[i].indices_u32.size() / 3;
        const size_t leaf_tris = lods[i].has_leaves
            ? lods[i].leaf_indices_u32.size() / 3 : 0u;
        const int bark_budget_L = (i == 0) ? bark_budget.l0_tris
                                : (i == 1) ? bark_budget.l1_tris
                                            : bark_budget.l2_tris;
        const int leaf_budget_L = (i == 0) ? leaf_budget.l0_tris
                                : (i == 1) ? leaf_budget.l1_tris
                                            : leaf_budget.l2_tris;
        INFO("LOD " << i << " bark_tris=" << bark_tris << "/" << bark_budget_L
             << " leaf_tris=" << leaf_tris << "/" << leaf_budget_L);
        REQUIRE(int(bark_tris) <= bark_budget_L);
        REQUIRE(int(leaf_tris) <= leaf_budget_L);
    }

    // ---- Subsample monotonicity (un-capped — leaf count strictly satisfies
    // the chain). vcount = wind_weights_packed.size() / 4. ----
    const size_t leaves_L0 = lods[0].has_leaves
        ? lods[0].leaf_wind_weights_packed.size() / 4 : 0u;
    const size_t leaves_L1 = lods[1].has_leaves
        ? lods[1].leaf_wind_weights_packed.size() / 4 : 0u;
    const size_t leaves_L2 = lods[2].has_leaves
        ? lods[2].leaf_wind_weights_packed.size() / 4 : 0u;
    INFO("leaves L0=" << leaves_L0 << " L1=" << leaves_L1 << " L2=" << leaves_L2);
    // L0 must have leaves on a default oak fixture (density > 0).
    REQUIRE(leaves_L0 > 0u);
    REQUIRE(leaves_L0 >= leaves_L1);
    REQUIRE(leaves_L1 >= leaves_L2);

    // ---- Per-leaf-vertex wind weights are depth-interpolated, material slot == 1 ----
    for (int i = 0; i < 3; ++i) {
        if (!lods[i].has_leaves) continue;
        const size_t wn = lods[i].leaf_wind_weights_packed.size();
        REQUIRE(wn >= 4u);
        REQUIRE(wn % 4u == 0u);
        // C1 P3: weights vary with branch depth; just verify nonzero + material slot.
        bool any_nonzero = false;
        for (size_t w = 0; w < wn; ++w)
            if (lods[i].leaf_wind_weights_packed[w] > 0u) { any_nonzero = true; break; }
        REQUIRE(any_nonzero);
        REQUIRE(lods[i].leaf_material_slots.size() >= 1u);
        REQUIRE(lods[i].leaf_material_slots[0] == 1);
    }

    // ---- Multi-primitive GLB round-trip ----
    const std::string path = write_lod_glb(lods, "oak");
    std::vector<char> bytes = slurp_bytes(path);
    REQUIRE_FALSE(bytes.empty());

    std::span<const char> byte_span(bytes.data(), bytes.size());
    REQUIRE(rynx::graphics::gltf_view::count_meshes(byte_span) == 3);  // C10: 3 LODs

    for (int mi = 0; mi < 3; ++mi) {
        std::vector<rynx::graphics::gltf_view::cpu_mesh> prims;
        std::string err;
        REQUIRE(rynx::graphics::gltf_view::extract_all_primitives(
            byte_span, prims, &err, mi));
        if (lods[mi].has_leaves) {
            REQUIRE(prims.size() == 2u);
        } else {
            REQUIRE(prims.size() == 1u);
        }
    }

    // ---- Determinism: re-emit, byte-identical GLB ----
    auto lods2 = ts::emit_all_lods(skel, skel.leaf_sites, base_opts, bark_budget,
                                   leaf_budget, leaf_geom_opts,
                                   s.tree.height_m, seed_effective);
    const std::string path2 = write_lod_glb(lods2, "oak2");
    std::vector<char> bytes2 = slurp_bytes(path2);
    REQUIRE(bytes == bytes2);

    std::error_code ec;
    fs::remove(path, ec);
    fs::remove(path2, ec);
}

TEST_CASE("[treegen_lod_leaves] downgrade-cap monotonicity (F1 regression seal)",
          "[treegen][treegen_lod_leaves]") {
    // Synthetic N=300 default-constructed LeafSites. The allocator is purely
    // index-driven (PCG32 hash over site index for rank_key); POD defaults
    // suffice — we never read position/normal.
    std::vector<ts::LeafSite> sites(300);

    // Budget chosen so the chain-min invariant is the only thing that keeps
    // L2 from inverting L1: raw_K_L1 = floor(0.30*300) = 90, cap_L1 = 20/2
    // = 10 → K_L1 = 10. raw_K_L2 = 30, cap_L2 = 30/2 = 15. Without chain-min,
    // K_L2 = 15 > K_L1 = 10 (the inversion).
    ts::LeafBudget budget;
    budget.l0_tris = 4000;
    budget.l1_tris = 20;
    budget.l2_tris = 30;
    budget.l3_tris = 0;

    const auto alloc = ts::allocate_leaves_all_lods(
        sites,
        ts::LeafGeometryType::SingleCard,
        ts::LeafShape::OakLobed,  // irrelevant for SingleCard
        /*cluster_count_per_tip*/ 1,
        budget,
        /*seed_effective*/ 0xC5'03'00'00'00'00'00'01ULL);

    INFO("kept_L0=" << alloc[0].kept_indices.size()
        << " kept_L1=" << alloc[1].kept_indices.size()
        << " kept_L2=" << alloc[2].kept_indices.size()
        << " kept_L3=" << alloc[3].kept_indices.size());

    REQUIRE(alloc[0].kept_indices.size() == 300u);
    REQUIRE(alloc[1].kept_indices.size() == 10u);
    REQUIRE(alloc[2].kept_indices.size() == 10u);  // chain-min vs L1, not 15
    REQUIRE(alloc[3].kept_indices.size() == 0u);

    // Per-LOD estimated_tris ≤ corresponding budget.
    REQUIRE(alloc[0].estimated_tris <= budget.l0_tris);
    REQUIRE(alloc[1].estimated_tris <= budget.l1_tris);
    REQUIRE(alloc[2].estimated_tris <= budget.l2_tris);
    REQUIRE(alloc[3].estimated_tris <= budget.l3_tris);

    // Set containment: kept_indices_L2 ⊆ kept_indices_L1 ⊆ kept_indices_L0.
    // Implemented as prefix-take so containment is the strong form (prefix).
    REQUIRE(is_prefix_of(alloc[1].kept_indices, alloc[0].kept_indices));
    REQUIRE(is_prefix_of(alloc[2].kept_indices, alloc[1].kept_indices));
    REQUIRE(is_prefix_of(alloc[3].kept_indices, alloc[2].kept_indices));
}

TEST_CASE("[treegen_lod_leaves] frozen geometry type across all LODs (C3-LOD-quality)",
          "[treegen][treegen_lod_leaves]") {
    // C3-LOD-quality: geometry type is frozen — no downgrade ladder. Budget
    // compliance comes purely from count reduction. Verify all LODs emit the
    // same type regardless of budget pressure.
    std::vector<ts::LeafSite> sites(300);

    // ProceduralVeined (MapleStar = 20 tris/leaf). Tight L2 budget forces
    // count reduction, NOT type change.
    {
        ts::LeafBudget budget;
        budget.l0_tris = 100000;
        budget.l1_tris = 100000;
        budget.l2_tris = 100;  // cap_L2 = 100/20 = 5 leaves
        budget.l3_tris = 0;

        const auto alloc = ts::allocate_leaves_all_lods(
            sites,
            ts::LeafGeometryType::ProceduralVeined,
            ts::LeafShape::MapleStar,
            /*cluster_count_per_tip*/ 1,
            budget,
            /*seed_effective*/ 0xC5'03'00'00'00'00'00'02ULL);

        INFO("ProceduralVeined: L0=" << int(alloc[0].emitted_type)
            << " L1=" << int(alloc[1].emitted_type)
            << " L2=" << int(alloc[2].emitted_type)
            << " kept L0/L1/L2=" << alloc[0].kept_indices.size()
            << "/" << alloc[1].kept_indices.size()
            << "/" << alloc[2].kept_indices.size());

        // Frozen type: all LODs stay ProceduralVeined.
        REQUIRE(alloc[0].emitted_type == ts::LeafGeometryType::ProceduralVeined);
        REQUIRE(alloc[1].emitted_type == ts::LeafGeometryType::ProceduralVeined);
        REQUIRE(alloc[2].emitted_type == ts::LeafGeometryType::ProceduralVeined);

        // Budget compliance via count reduction.
        REQUIRE(alloc[2].estimated_tris <= budget.l2_tris);
        REQUIRE(alloc[2].kept_indices.size() <= 5u);  // cap_L2 = 100/20 = 5

        // Monotonicity preserved.
        REQUIRE(alloc[0].kept_indices.size() >= alloc[1].kept_indices.size());
        REQUIRE(alloc[1].kept_indices.size() >= alloc[2].kept_indices.size());
    }

    // BentCrossCluster(N=2) = 8 tris/leaf. Tight L2 budget.
    {
        ts::LeafBudget budget;
        budget.l0_tris = 100000;
        budget.l1_tris = 100000;
        budget.l2_tris = 100;
        budget.l3_tris = 0;

        const auto alloc = ts::allocate_leaves_all_lods(
            sites,
            ts::LeafGeometryType::BentCrossCluster,
            ts::LeafShape::OakLobed,
            /*cluster_count_per_tip*/ 2,
            budget,
            /*seed_effective*/ 0xC2'03'00'00'00'00'00'01ULL);

        // All LODs stay BentCrossCluster.
        REQUIRE(alloc[0].emitted_type == ts::LeafGeometryType::BentCrossCluster);
        REQUIRE(alloc[1].emitted_type == ts::LeafGeometryType::BentCrossCluster);
        REQUIRE(alloc[2].emitted_type == ts::LeafGeometryType::BentCrossCluster);
        // Cluster count preserved.
        REQUIRE(alloc[0].effective_cluster == 2);
        REQUIRE(alloc[1].effective_cluster == 2);
        REQUIRE(alloc[2].effective_cluster == 2);
        // Budget compliance: cap_L2 = 100/8 = 12 leaves.
        REQUIRE(alloc[2].estimated_tris <= budget.l2_tris);
        REQUIRE(alloc[2].kept_indices.size() <= 12u);

        REQUIRE(alloc[0].kept_indices.size() >= alloc[1].kept_indices.size());
        REQUIRE(alloc[1].kept_indices.size() >= alloc[2].kept_indices.size());
    }

    // BentCard = 4 tris/leaf. Tight L2 budget.
    {
        ts::LeafBudget budget;
        budget.l0_tris = 100000;
        budget.l1_tris = 100000;
        budget.l2_tris = 50;
        budget.l3_tris = 0;

        const auto alloc = ts::allocate_leaves_all_lods(
            sites,
            ts::LeafGeometryType::BentCard,
            ts::LeafShape::OakLobed,
            /*cluster_count_per_tip*/ 1,
            budget,
            /*seed_effective*/ 0xC2'03'00'00'00'00'00'02ULL);

        // All LODs stay BentCard.
        REQUIRE(alloc[0].emitted_type == ts::LeafGeometryType::BentCard);
        REQUIRE(alloc[1].emitted_type == ts::LeafGeometryType::BentCard);
        REQUIRE(alloc[2].emitted_type == ts::LeafGeometryType::BentCard);
        // Budget compliance: cap_L2 = 50/4 = 12 leaves.
        REQUIRE(alloc[2].estimated_tris <= budget.l2_tris);
    }
}

TEST_CASE("[treegen_lod_leaves] LOD leaf scale increases with LOD index",
          "[treegen][treegen_lod_leaves]") {
    // C3-LOD-quality: leaves are scaled up at lower LODs to maintain canopy
    // coverage. Verify by checking that per-leaf vertex extent grows across
    // LODs in the full emit pipeline.
    namespace tsp = rynx::test_support;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

    auto lp_opts = ts::leaf_placement::options_from_descriptor(s.tree.leaves);
    skel.leaf_sites = ts::leaf_placement::generate_leaf_sites(skel, lp_opts, seed_effective);
    REQUIRE(skel.leaf_sites.size() > 100u);

    ts::BarkMeshOptions base_opts;
    base_opts.tree_height_m   = s.tree.height_m;
    base_opts.seam_offset_rad = 0.0f;

    ts::LodBudget   bark_budget;
    ts::LeafBudget  leaf_budget;

    ts::LeafMeshOptions leaf_geom_opts;
    leaf_geom_opts.geometry_type         = s.tree.leaves.geometry_type;
    leaf_geom_opts.shape                 = s.tree.leaves.shape;
    leaf_geom_opts.leaf_size_m           = s.tree.leaves.leaf_size_m;
    leaf_geom_opts.cluster_count_per_tip = s.tree.leaves.cluster_count_per_tip;
    leaf_geom_opts.bend_half_angle       = s.tree.leaves.leaf_bend_half_angle;

    auto lods = ts::emit_all_lods(skel, skel.leaf_sites, base_opts, bark_budget,
                                  leaf_budget, leaf_geom_opts,
                                  s.tree.height_m, seed_effective);
    REQUIRE(lods.size() == 3u);

    // Compute per-leaf AABB extent for each LOD (using first leaf's verts).
    // With frozen type and scale factor L0=1.0 L1=1.4 L2=2.0, the per-leaf
    // extent should grow monotonically.
    auto leaf_extent = [](const ts::LodOutput& lod, int verts_per_leaf_n) -> float {
        if (!lod.has_leaves || lod.leaf_positions.size() < static_cast<size_t>(verts_per_leaf_n * 3))
            return 0.0f;
        float max_dist = 0.0f;
        const float cx = lod.leaf_positions[0];
        const float cy = lod.leaf_positions[1];
        const float cz = lod.leaf_positions[2];
        for (int i = 1; i < verts_per_leaf_n; ++i) {
            const float dx = lod.leaf_positions[static_cast<size_t>(i * 3 + 0)] - cx;
            const float dy = lod.leaf_positions[static_cast<size_t>(i * 3 + 1)] - cy;
            const float dz = lod.leaf_positions[static_cast<size_t>(i * 3 + 2)] - cz;
            max_dist = std::max(max_dist, dx * dx + dy * dy + dz * dz);
        }
        return std::sqrt(max_dist);
    };

    const int vpl = ts::verts_per_leaf(leaf_geom_opts.geometry_type,
                                       leaf_geom_opts.shape,
                                       leaf_geom_opts.cluster_count_per_tip);
    const float e0 = leaf_extent(lods[0], vpl);
    const float e1 = leaf_extent(lods[1], vpl);
    const float e2 = leaf_extent(lods[2], vpl);

    INFO("leaf_extent L0=" << e0 << " L1=" << e1 << " L2=" << e2);
    REQUIRE(e0 > 0.0f);
    REQUIRE(e1 > e0 * 1.1f);   // L1 scale 1.4x — expect at least 1.1x
    REQUIRE(e2 > e1 * 1.1f);   // L2 scale 2.0x — expect at least 1.1x above L1
}

TEST_CASE("[treegen_lod_leaves] determinism — second emit byte-equal to first",
          "[treegen][treegen_lod_leaves]") {
    // Separately tagged so a determinism failure isolates from the broader
    // oak case (which mixes budget + round-trip + monotonicity).
    namespace tsp = rynx::test_support;
    namespace fs  = std::filesystem;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
    auto lp_opts = ts::leaf_placement::options_from_descriptor(s.tree.leaves);
    skel.leaf_sites = ts::leaf_placement::generate_leaf_sites(skel, lp_opts, seed_effective);

    ts::BarkMeshOptions base_opts;
    base_opts.tree_height_m   = s.tree.height_m;
    base_opts.seam_offset_rad = 0.0f;
    ts::LodBudget   bark_budget;
    ts::LeafBudget  leaf_budget;
    ts::LeafMeshOptions leaf_geom_opts;
    leaf_geom_opts.geometry_type         = s.tree.leaves.geometry_type;
    leaf_geom_opts.shape                 = s.tree.leaves.shape;
    leaf_geom_opts.leaf_size_m           = s.tree.leaves.leaf_size_m;
    leaf_geom_opts.cluster_count_per_tip = s.tree.leaves.cluster_count_per_tip;
    leaf_geom_opts.bend_half_angle       = s.tree.leaves.leaf_bend_half_angle;

    auto lods_a = ts::emit_all_lods(skel, skel.leaf_sites, base_opts, bark_budget,
                                    leaf_budget, leaf_geom_opts,
                                    s.tree.height_m, seed_effective);
    auto lods_b = ts::emit_all_lods(skel, skel.leaf_sites, base_opts, bark_budget,
                                    leaf_budget, leaf_geom_opts,
                                    s.tree.height_m, seed_effective);
    const std::string path_a = write_lod_glb(lods_a, "det_a");
    const std::string path_b = write_lod_glb(lods_b, "det_b");
    const std::vector<char> bytes_a = slurp_bytes(path_a);
    const std::vector<char> bytes_b = slurp_bytes(path_b);
    REQUIRE(bytes_a == bytes_b);

    std::error_code ec;
    fs::remove(path_a, ec);
    fs::remove(path_b, ec);
}
