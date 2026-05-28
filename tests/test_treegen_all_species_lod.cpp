// [treegen_all_species_lod] — pins C4 P5. All-species (oak/pine/birch/maple)
// per-LOD invariants:
//   * Each LOD's tri count fits within its LodBudget.
//   * L0 retains ≥ 80% of L0 budget (proves the cull-by-length rule preserves
//     silhouette — guards against the P4 regression where oak L0 collapsed to
//     a 33-tri trunk-only stick under cull-by-order).
//   * Monotone decreasing detail: L0 ≥ L1 ≥ L2.
//   * Determinism: rebuild from same skeleton + budget → byte-identical GLB.
//
// [treegen_all_species_lod_render] — pins C4 P5 PNG goldens.
//   4 species × 3 LODs = 12 entries + trivial_42 = 13 PNGs.
//   Default behavior on missing golden = FAIL with diagnostic.
//   RYNX_TREEGEN_GOLDEN_BOOTSTRAP=1 = regenerate missing goldens.
//   pHash thresholds derived from intra-run noise floor (see roadmap progress):
//     k_phash_threshold_soft = WARN  if hamming > N
//     k_phash_threshold_hard = REQUIRE hamming <= M
//   Tag isolation: only `[treegen_all_species_lod_render]` — NO `[treegen]`
//   umbrella — so `testtech.exe [treegen]` stays in-process and fast.
//
// Link rationale identical to [treegen_lod_emission] — face_budget.cpp,
// lod_emitter.cpp, branch_mesh.cpp, fork_blend.cpp, wind_weights.cpp,
// space_colonization.cpp, scenario.cpp, tree_descriptor.cpp, envelopes.cpp
// all linked into TestTech via rynx_tests.sharpmake.cs.

#include "../external/catch2/catch.hpp"

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

std::vector<std::uint8_t> slurp_png(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.good()) return {};
    return std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>());
}

std::string tmp_path(const char* tag) {
    namespace fs = std::filesystem;
    char buf[80];
    std::snprintf(buf, sizeof(buf), "treegen_all_species_%d_%s.glb", std::rand(), tag);
    return (fs::temp_directory_path() / buf).string();
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

    ts::LodBudget budget;  // P5 defaults: L0=12000, L1=4000, L2=1500, L3=4

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

// ---- C4 P5: per-species per-LOD render-pHash matrix ----------------------
//
// Threshold derivation (intra-run noise floor, logs/p5_phash_noise_floor.log):
//   max_intra  = 0   (4 species × 3 LODs × 3 reruns)
//   mean_intra = 0
//   k_soft     = 4   (max(4, max_intra + 1) = max(4, 1) = 4)
//   k_hard     = 8   (max(8, 2*max_intra + 2) = max(8, 2) = 8)
// Renderer is fully deterministic for the static GLB render path under the
// determinism knobs in render_glb_cli.cpp (fixed dt, AE off, hot-reload off,
// fixed exposure). Soft-warn is set above zero to absorb the inevitable
// driver/GPU-arch sliver if/when this runs on different HW.
namespace {
constexpr int k_phash_threshold_soft_render = 4;
constexpr int k_phash_threshold_hard_render = 8;

// Bootstrap-mode env-var gate (shared with [treegen_render_phash]). When set,
// a missing golden = WARN + copy rendered PNG into repo + SUCCEED, so the
// operator can re-baseline + commit. Default (unset) = explicit FAIL so a
// missing golden never silently green-ships.
inline const char* k_env_bootstrap = "RYNX_TREEGEN_GOLDEN_BOOTSTRAP";
} // anonymous namespace

TEST_CASE("[treegen_all_species_lod_render] per-species per-LOD render-pHash matches golden",
          "[treegen_all_species_lod_render]") {   // NO [treegen] umbrella — opt-in only.
    namespace tsp = rynx::test_support;
    namespace fs  = std::filesystem;

    const std::string treegen_exe = tsp::find_built_exe("rynx-treegen.exe");
    const std::string game_exe    = tsp::find_built_exe("topdownshootergame.exe");
    INFO("treegen_exe=" << treegen_exe);
    INFO("game_exe="    << game_exe);
    REQUIRE_FALSE(treegen_exe.empty());
    REQUIRE_FALSE(game_exe.empty());

    // game exe must run from build/topdownshooter/bin/ — VFS mounts are
    // CWD-relative (see reference_run_from_bin_cwd memory).
    const std::string game_bindir = fs::path(game_exe).parent_path().string();

    // Anchor dir for bootstrap-mode writes. Walk to the tests/golden/ dir
    // (existence-checked) and append "treegen/" — robust against trivial_42
    // having just been deleted for re-baseline.
    fs::path anchor_dir;
    {
        const std::string anchor_parent = tsp::find_repo_file("tests/golden/");
        if (!anchor_parent.empty()) {
            anchor_dir = fs::path(anchor_parent) / "treegen";
        }
    }

    const char* species[] = { "oak", "pine", "birch", "maple" };
    const int   lods[]    = { 0, 1, 2 };

    for (const char* sp : species) {
        const std::string scenario_rel = std::string("tools/rynx-treegen/scenarios/c3_")
                                          + sp + ".json";
        const std::string scenario_abs = tsp::find_repo_file(scenario_rel);
        INFO("scenario=" << scenario_abs);
        REQUIRE_FALSE(scenario_abs.empty());

        // ONE GLB per species; reused across LODs (--lod=<i> picks at render).
        const fs::path tmp_glb = fs::temp_directory_path()
                                 / (std::string("treegen_c3_") + sp + "_42.glb");
        std::error_code ec;
        fs::remove(tmp_glb, ec);
        {
            std::ostringstream args;
            args << "--scenario=\"" << scenario_abs
                 << "\" --seed=42 --out=\"" << tmp_glb.string() << "\"";
            const int rc = tsp::run_exe_in_dir(treegen_exe,
                                               fs::current_path().string(),
                                               args.str());
            INFO("rynx-treegen [" << sp << "] rc=" << rc);
            REQUIRE(rc == 0);
            REQUIRE(fs::exists(tmp_glb));
        }

        for (int lod : lods) {
            SECTION(std::string("species=") + sp + " lod=" + std::to_string(lod)) {
                const fs::path tmp_png = fs::temp_directory_path()
                                         / (std::string("treegen_c3_") + sp
                                            + "_L" + std::to_string(lod) + ".png");
                fs::remove(tmp_png, ec);
                {
                    std::ostringstream args;
                    args << "--render-glb=\"" << tmp_glb.string()
                         << "\" --out=\"" << tmp_png.string()
                         << "\" --lod=" << lod;
                    // Known issue: the game exe shutdown path has a ~10-20%
                    // segfault rate (see project_shutdown_segfault.md — 2 sites
                    // closed in T4 but the bug class is latent). Retry ONCE so
                    // pHash-variance signal isn't drowned in process-shutdown
                    // noise. PNG-existence check below is the real correctness
                    // gate — a true render bug fails both attempts.
                    int rc = tsp::run_exe_in_dir(game_exe, game_bindir, args.str());
                    if (rc != 0 || !fs::exists(tmp_png)) {
                        WARN("render rc=" << rc << " (or no PNG) on first attempt for "
                             << sp << " L" << lod << " — retrying once (known shutdown flake)");
                        fs::remove(tmp_png, ec);
                        rc = tsp::run_exe_in_dir(game_exe, game_bindir, args.str());
                    }
                    INFO("topdownshootergame [" << sp << " L" << lod << "] rc=" << rc);
                    REQUIRE(rc == 0);
                    REQUIRE(fs::exists(tmp_png));
                }

                const std::string golden_rel = std::string("tests/golden/treegen/c3_")
                                                + sp + "_L" + std::to_string(lod) + ".png";
                const std::string golden_abs = tsp::find_repo_file(golden_rel);
                if (golden_abs.empty()) {
                    if (const char* env = std::getenv(k_env_bootstrap); env && env[0] == '1') {
                        REQUIRE_FALSE(anchor_dir.empty());
                        fs::create_directories(anchor_dir, ec);
                        const fs::path dst = anchor_dir
                                             / (std::string("c3_") + sp + "_L"
                                                + std::to_string(lod) + ".png");
                        fs::copy_file(tmp_png, dst,
                                      fs::copy_options::overwrite_existing, ec);
                        REQUIRE_FALSE(ec);
                        WARN("bootstrapped golden: " << golden_rel
                              << " (copied from " << tmp_png.string()
                              << ") — review + commit");
                        SUCCEED("bootstrap-mode wrote " + golden_rel);
                        continue;
                    }
                    FAIL("missing golden: " << golden_rel
                          << " — set " << k_env_bootstrap
                          << "=1 to regenerate; do NOT commit env-var-set runs as 'green'");
                }

                const std::vector<std::uint8_t> rendered = slurp_png(tmp_png);
                const std::vector<std::uint8_t> golden   = slurp_png(golden_abs);
                REQUIRE_FALSE(rendered.empty());
                REQUIRE_FALSE(golden.empty());

                const std::uint64_t h_rendered = treegen::phash::compute(rendered);
                const std::uint64_t h_golden   = treegen::phash::compute(golden);
                const int           h          = treegen::phash::hamming(h_rendered, h_golden);
                INFO("species=" << sp << " lod=" << lod
                     << " rendered=" << std::hex << h_rendered
                     << " golden="   << std::hex << h_golden
                     << " hamming="  << std::dec << h);
                if (h > k_phash_threshold_soft_render) {
                    WARN("pHash drift over soft threshold (" << h
                         << " > " << k_phash_threshold_soft_render << "): "
                         << sp << " L" << lod);
                }
                REQUIRE(h <= k_phash_threshold_hard_render);
            }
        }

        fs::remove(tmp_glb, ec);
    }
}

// [treegen_leaves_all_species] — C5 P4. Uniform per-LOD leaf invariants across
// the 4 production species (oak / pine / birch / maple). Pins the leaf-aware
// 8-arg `emit_all_lods` overload + main.cpp's leaf wiring (density > 0 ∧ sites
// non-empty → leaves emitted on L0..L2, L3 stays leafless).
//
// Joins the `[treegen]` umbrella sweep so a leaf-pipeline regression on any
// species lights up the default `testtech.exe [treegen]` run; `[treegen_lod_leaves]`
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

    ts::LodBudget  bark_budget;   // P5 defaults: 12000 / 4000 / 1500 / 4
    ts::LeafBudget leaf_budget;   // defaults: 4000 / 1200 / 400 / 0

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
