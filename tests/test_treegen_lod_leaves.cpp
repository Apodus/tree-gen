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

TEST_CASE("[treegen_lod_leaves] ladder downgrades when ProceduralVeined overshoots",
          "[treegen][treegen_lod_leaves]") {
    // MapleStar = 20 tris/leaf (ProceduralVeined). At L2, raw_K_L2 = 30.
    // Procedural cost = 30*20 = 600 tris. Set l2_tris = 100 so procedural
    // overshoots but SingleCard (2 tris/leaf, 30*2 = 60) fits.
    std::vector<ts::LeafSite> sites(300);

    ts::LeafBudget budget;
    budget.l0_tris = 100000;  // L0 fits ProceduralVeined easily
    budget.l1_tris = 100000;
    budget.l2_tris = 100;     // forces L2 to downgrade
    budget.l3_tris = 0;

    const auto alloc = ts::allocate_leaves_all_lods(
        sites,
        ts::LeafGeometryType::ProceduralVeined,
        ts::LeafShape::MapleStar,
        /*cluster_count_per_tip*/ 1,
        budget,
        /*seed_effective*/ 0xC5'03'00'00'00'00'00'02ULL);

    INFO("L0 emitted_type=" << int(alloc[0].emitted_type)
        << " L1 emitted_type=" << int(alloc[1].emitted_type)
        << " L2 emitted_type=" << int(alloc[2].emitted_type)
        << " L0 est=" << alloc[0].estimated_tris
        << " L1 est=" << alloc[1].estimated_tris
        << " L2 est=" << alloc[2].estimated_tris);

    // L0 + L1 should still be Procedural (budget fits).
    REQUIRE(alloc[0].emitted_type == ts::LeafGeometryType::ProceduralVeined);
    REQUIRE(alloc[1].emitted_type == ts::LeafGeometryType::ProceduralVeined);
    // L2 downgraded — not Procedural.
    REQUIRE(alloc[2].emitted_type != ts::LeafGeometryType::ProceduralVeined);

    // Monotonicity preserved (kept counts + tri counts both descend).
    REQUIRE(alloc[0].kept_indices.size() >= alloc[1].kept_indices.size());
    REQUIRE(alloc[1].kept_indices.size() >= alloc[2].kept_indices.size());
    REQUIRE(alloc[2].kept_indices.size() >= alloc[3].kept_indices.size());

    // Budget compliance.
    REQUIRE(alloc[0].estimated_tris <= budget.l0_tris);
    REQUIRE(alloc[1].estimated_tris <= budget.l1_tris);
    REQUIRE(alloc[2].estimated_tris <= budget.l2_tris);
}

TEST_CASE("[treegen_lod_leaves] BentCrossCluster downgrade ladder",
          "[treegen][treegen_lod_leaves]") {
    // BentCrossCluster(N=2) = 8 tris/leaf. 300 sites.
    // L0: budget=100000 fits; L1: budget=100000 fits.
    // L2: raw_K=30, cost=30*8=240. budget=100 → must downgrade.
    // Ladder: BentCrossCluster(2)→BentCrossCluster(1)→BentCard→SingleCard.
    // BentCrossCluster(1)=4 tris, 30*4=120 > 100, still overshoots.
    // BentCard=4 tris, 30*4=120 > 100, still overshoots.
    // SingleCard=2 tris, 30*2=60 <= 100, fits.
    std::vector<ts::LeafSite> sites(300);

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

    // L0 + L1 stay BentCrossCluster (budget fits).
    REQUIRE(alloc[0].emitted_type == ts::LeafGeometryType::BentCrossCluster);
    REQUIRE(alloc[1].emitted_type == ts::LeafGeometryType::BentCrossCluster);
    // L2 must downgrade past BentCrossCluster (too expensive).
    REQUIRE(alloc[2].emitted_type != ts::LeafGeometryType::BentCrossCluster);
    REQUIRE(alloc[2].emitted_type != ts::LeafGeometryType::BentCard);

    // Budget compliance.
    REQUIRE(alloc[0].estimated_tris <= budget.l0_tris);
    REQUIRE(alloc[1].estimated_tris <= budget.l1_tris);
    REQUIRE(alloc[2].estimated_tris <= budget.l2_tris);

    // Monotonicity.
    REQUIRE(alloc[0].kept_indices.size() >= alloc[1].kept_indices.size());
    REQUIRE(alloc[1].kept_indices.size() >= alloc[2].kept_indices.size());
}

TEST_CASE("[treegen_lod_leaves] BentCard downgrades through full ladder to SingleCard",
          "[treegen][treegen_lod_leaves]") {
    // Starting from BentCard (4 tris/leaf). 300 sites, L2 raw_K=30.
    // Budget L2=50 → 30*4=120 > 50, must downgrade to SingleCard (2 tris),
    // cap=25, 25*2=50<=50.
    std::vector<ts::LeafSite> sites(300);

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

    REQUIRE(alloc[0].emitted_type == ts::LeafGeometryType::BentCard);
    REQUIRE(alloc[1].emitted_type == ts::LeafGeometryType::BentCard);
    REQUIRE(alloc[2].emitted_type == ts::LeafGeometryType::SingleCard);

    REQUIRE(alloc[2].estimated_tris <= budget.l2_tris);
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
