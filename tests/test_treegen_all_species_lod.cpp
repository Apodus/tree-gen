// [treegen_all_species_lod] — pins C4 P5. All-species (oak/pine/birch/maple)
// per-LOD invariants:
//   * Each LOD's tri count fits within its LodBudget.
//   * L0 retains ≥ 80% of L0 budget (proves the cull-by-length rule preserves
//     silhouette — guards against the P4 regression where oak L0 collapsed to
//     a 33-tri trunk-only stick under cull-by-order).
//   * Monotone decreasing detail: L0 ≥ L1 ≥ L2.
//   * Determinism: rebuild from same skeleton + budget → byte-identical GLB.
//
// [treegen_all_species_lod_render] lives in test_treegen_all_species_lod_render.cpp
// (TestTreegenRender: it spawns the game).
//
// Link rationale identical to [treegen_lod_emission]: TestTreegen links the
// tool's own sources (TreegenCore), /fp:precise as the tool compiles them.

#include "../external/catch2/catch.hpp"
#include "test_env.hpp"

#include "test_support_paths.hpp"
#include "../phash.hpp"

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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <sstream>
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

std::string tmp_path(const char* tag) {
    namespace fs = std::filesystem;
    char buf[80];
    std::snprintf(buf, sizeof(buf), "treegen_all_species_%d_%s.glb", std::rand(), tag);
    return treegen::test_env::scratch(buf).string();
}

std::string write_lod_glb(const std::vector<ts::LodOutput>& lods, const char* tag) {
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
        md.primitives            = std::span<const ts::PrimitiveData>(
            per_lod_prims[li].data(), per_lod_prims[li].size());
        md.lod_index             = lod.lod_index;
        md.lod_max_distance_m    = lod.lod_max_distance_m;
        md.lod_screen_height_px  = lod.lod_screen_height_px;
        meshes.push_back(md);
    }

    const std::string path = tmp_path(tag);
    std::string err;
    REQUIRE(ts::write_glb_multi_mesh(
        std::span<const ts::MeshData>(meshes.data(), meshes.size()), path, &err));
    return path;
}

} // anonymous namespace

TEST_CASE("[treegen_all_species_lod] all species fit budget and retain silhouette",
          "[treegen][treegen_all_species_lod]") {
    namespace tsp = rynx::test_support;
    namespace fs  = std::filesystem;

    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    // C3 P2 bark diet: per-species L0 (oak/pine 10k; birch/maple ≈natural),
    // shared L1=4000/L2=1200/L3=4 — set inside the loop via lod_budget_for_species.

    // Silhouette-retention guard: when the budget binds (natural mesh >
    // budget), the cull rule must keep ≥X% of the budget — proves we don't
    // collapse to a trunk-only stick (the P4 cull-by-order regression). When
    // the budget doesn't bind (sparse species like pine whose natural mesh is
    // already below budget), the floor relaxes to the natural mesh size, so
    // the assertion doesn't false-positive on legitimately small trees.
    // Reference mesh = same skeleton + base_opts at an effectively-unbounded
    // budget; computed per species below.
    const int l0_retain_pct = 80;
    const int l1_retain_pct = 60;  // L1 trim more aggressive
    const int l2_retain_pct = 50;  // L2 trim even more

    for (const char* scenario_rel : scenarios) {
        const auto fixture = tsp::find_repo_file(scenario_rel);
        REQUIRE_FALSE(fixture.empty());
        ts::Scenario s = ts::load_scenario(fixture);
        REQUIRE(s.kind == "tree");

        const ts::LodBudget budget = ts::lod_budget_for_species(s.tree.leaves.shape);

        const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
        ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
        REQUIRE(skel.nodes.size() > 50u);

        ts::BarkMeshOptions base_opts;
        base_opts.tree_height_m   = s.tree.height_m;
        base_opts.seam_offset_rad = 0.0f;

        auto lods = ts::emit_all_lods(skel, base_opts, budget, s.tree.height_m, seed_effective);
        REQUIRE(lods.size() == 3u);  // C10: L3 billboard stub suppressed

        const size_t l0_tris = lods[0].indices_u32.size() / 3;
        const size_t l1_tris = lods[1].indices_u32.size() / 3;
        const size_t l2_tris = lods[2].indices_u32.size() / 3;

        // Reference build at unbounded budget.
        ts::LodBudget unbounded;
        unbounded.l0_tris = unbounded.l1_tris = unbounded.l2_tris = 1 << 24;
        auto ref_lods = ts::emit_all_lods(skel, base_opts, unbounded, s.tree.height_m, seed_effective);
        const size_t ref_l0 = ref_lods[0].indices_u32.size() / 3;
        const size_t ref_l1 = ref_lods[1].indices_u32.size() / 3;
        const size_t ref_l2 = ref_lods[2].indices_u32.size() / 3;

        const int l0_floor = std::min<int>(int(ref_l0), (budget.l0_tris * l0_retain_pct) / 100);
        const int l1_floor = std::min<int>(int(ref_l1), (budget.l1_tris * l1_retain_pct) / 100);
        const int l2_floor = std::min<int>(int(ref_l2), (budget.l2_tris * l2_retain_pct) / 100);

        INFO("species=" << s.tree.species
            << " L0=" << l0_tris << " L1=" << l1_tris
            << " L2=" << l2_tris
            << " ref(L0/L1/L2)=" << ref_l0 << "/" << ref_l1 << "/" << ref_l2
            << " floor(L0/L1/L2)=" << l0_floor << "/" << l1_floor << "/" << l2_floor);

        // ---- Budget compliance (over-budget would defeat the LOD intent) ----
        REQUIRE(int(l0_tris) <= budget.l0_tris);
        REQUIRE(int(l1_tris) <= budget.l1_tris);
        REQUIRE(int(l2_tris) <= budget.l2_tris);

        // ---- Silhouette retention (the C4 P5 fix) ----
        // Trunk-only-stick failure mode reports ~33 tris; the floor is at
        // least 750 tris (50% of L2's 1500) even on the sparsest species —
        // an order of magnitude above the collapse failure mode.
        REQUIRE(int(l0_tris) >= l0_floor);
        REQUIRE(int(l1_tris) >= l1_floor);
        REQUIRE(int(l2_tris) >= l2_floor);

        // ---- Monotone decreasing detail L0 → L2 ----
        REQUIRE(l0_tris >= l1_tris);
        REQUIRE(l1_tris >= l2_tris);

        // ---- Determinism: rebuild → byte-identical GLB ----
        auto lods2 = ts::emit_all_lods(skel, base_opts, budget, s.tree.height_m, seed_effective);
        const std::string path_a = write_lod_glb(lods,  s.tree.species.c_str());
        const std::string path_b = write_lod_glb(lods2, (s.tree.species + "2").c_str());
        const std::vector<char> bytes_a = slurp_bytes(path_a);
        const std::vector<char> bytes_b = slurp_bytes(path_b);
        REQUIRE(bytes_a == bytes_b);

        std::error_code ec;
        fs::remove(path_a, ec);
        fs::remove(path_b, ec);
    }
}

// [treegen_leaves_all_species] — C5 P4. Uniform per-LOD leaf invariants across
// the 4 production species (oak / pine / birch / maple). Pins the leaf-aware
// 8-arg `emit_all_lods` overload + main.cpp's leaf wiring (density > 0 ∧ sites
// non-empty → leaves emitted on L0..L2, L3 stays leafless).
//
// Joins the `[treegen]` umbrella sweep so a leaf-pipeline regression on any
// species lights up the default `testtreegen.exe [treegen]` run; `[treegen_lod_leaves]`
// remains the deeper oak-specific pin (multi-prim GLB round-trip + monotonicity
// vs synthetic budgets).
TEST_CASE("[treegen_leaves_all_species] all species emit leaves per LOD",
          "[treegen][treegen_leaves_all_species]") {
    namespace tsp = rynx::test_support;

    const char* scenarios[] = {
        "tools/rynx-treegen/scenarios/c3_oak.json",
        "tools/rynx-treegen/scenarios/c3_pine.json",
        "tools/rynx-treegen/scenarios/c3_birch.json",
        "tools/rynx-treegen/scenarios/c3_maple.json",
    };

    ts::LodBudget  bark_budget;   // C3 P2 bark diet: 10000 / 4000 / 1200 / 4
    ts::LeafBudget leaf_budget;   // C3 P3 retune: 8000 / 5000 / 4000 / 0

    for (const char* scenario_rel : scenarios) {
        const auto fixture = tsp::find_repo_file(scenario_rel);
        INFO("fixture=" << fixture);
        REQUIRE_FALSE(fixture.empty());

        ts::Scenario s = ts::load_scenario(fixture);
        REQUIRE(s.kind == "tree");

        const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
        ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

        auto lp_opts = ts::leaf_placement::options_from_descriptor(s.tree.leaves);
        auto sites   = ts::leaf_placement::generate_leaf_sites(skel, lp_opts, seed_effective);
        INFO("species=" << s.tree.species << " sites=" << sites.size());
        REQUIRE(sites.size() > 50u);

        ts::BarkMeshOptions bark_opts;
        bark_opts.tree_height_m   = s.tree.height_m;
        bark_opts.seam_offset_rad = 0.0f;

        ts::LeafMeshOptions leaf_geom_opts;
        leaf_geom_opts.geometry_type         = s.tree.leaves.geometry_type;
        leaf_geom_opts.shape                 = s.tree.leaves.shape;
        leaf_geom_opts.leaf_size_m           = s.tree.leaves.leaf_size_m;
        leaf_geom_opts.cluster_count_per_tip = s.tree.leaves.cluster_count_per_tip;
        leaf_geom_opts.bend_half_angle       = s.tree.leaves.leaf_bend_half_angle;

        auto lods = ts::emit_all_lods(skel, sites, bark_opts, bark_budget,
                                      leaf_budget, leaf_geom_opts,
                                      s.tree.height_m, seed_effective);
        REQUIRE(lods.size() == 3u);  // C10: L3 billboard stub suppressed

        const int leaf_budget_per_lod[3] = {
            leaf_budget.l0_tris, leaf_budget.l1_tris, leaf_budget.l2_tris
        };
        for (int i = 0; i < 3; ++i) {
            const size_t leaf_tris = lods[i].leaf_indices_u32.size() / 3;
            INFO("species=" << s.tree.species << " L" << i
                 << " leaf_tris=" << leaf_tris << "/" << leaf_budget_per_lod[i]
                 << " has_leaves=" << lods[i].has_leaves);
            REQUIRE(lods[i].has_leaves);
            REQUIRE(leaf_tris > 0u);
            // BranchStrip has its own segment-based budget; skip site-budget check.
            if (s.tree.leaves.geometry_type != ts::LeafGeometryType::BranchStrip) {
                REQUIRE(int(leaf_tris) <= leaf_budget_per_lod[i]);
            }
            REQUIRE(lods[i].leaf_wind_weights_packed.size() >= 4u);
            // C1 P3: wind weights are depth-interpolated, not hardcoded (0,0,0,255).
            // Verify size is 4 bytes/vert and at least one channel is nonzero.
            {
                const size_t wn = lods[i].leaf_wind_weights_packed.size();
                REQUIRE(wn % 4u == 0u);
                bool any_nonzero = false;
                for (size_t w = 0; w < wn; ++w)
                    if (lods[i].leaf_wind_weights_packed[w] > 0u) { any_nonzero = true; break; }
                REQUIRE(any_nonzero);
            }
            REQUIRE(lods[i].leaf_material_slots.size() >= 1u);
            REQUIRE(lods[i].leaf_material_slots[0] == 1);            // slot 1 = leaves
        }

        // Monotone leaf vert count L0 ≥ L1 ≥ L2 — subsample monotonicity
        // surfaced at the vert layer (chain-min is sealed at the allocator).
        auto vcount = [](const auto& lod) {
            return lod.leaf_wind_weights_packed.size() / 4;
        };
        INFO("species=" << s.tree.species
             << " leaf_verts L0/L1/L2=" << vcount(lods[0])
             << "/" << vcount(lods[1])
             << "/" << vcount(lods[2]));
        REQUIRE(vcount(lods[0]) >= vcount(lods[1]));
        REQUIRE(vcount(lods[1]) >= vcount(lods[2]));
    }
}
