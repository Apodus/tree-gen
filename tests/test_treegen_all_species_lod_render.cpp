// [treegen_all_species_lod_render] — pins the C4 P5 PNG goldens:
// 4 species × 3 LODs = 12 entries, one GLB per species (--lod=<i> picks at
// render). Every entry is graded (CHECK), so a red names ALL the drifted
// entries rather than the first one reached.
//
// Thresholds: the render runs in lockstep, so a repeat run on one host is
// stable (every entry's pHash identical across 3 runs and under 3x TestTech
// load; the trivial cylinder 12 of 12 byte-identical). The slack is for a
// different GPU or driver, not for run-to-run noise -- there is none left to
// absorb, and widening it would hide a race rather than remove it.
//   k_soft = 4   WARN above
//   k_hard = 8   CHECK at or below

#include "../external/catch2/catch.hpp"

#include "render_golden.hpp"
#include "test_support_paths.hpp"

#include <filesystem>
#include <string>

namespace {
constexpr int k_phash_threshold_soft_render = 4;
constexpr int k_phash_threshold_hard_render = 8;
} // anonymous namespace

TEST_CASE("[treegen_all_species_lod_render] per-species per-LOD render-pHash matches golden",
          "[treegen_all_species_lod_render]") {
    namespace fs = std::filesystem;
    namespace rg = treegen::render_golden;

    const char* species[] = { "oak", "pine", "birch", "maple" };
    const int   lods[]    = { 0, 1, 2 };

    for (const char* sp : species) {
        const std::string scenario = rynx::test_support::find_repo_file(
            std::string("tools/rynx-treegen/scenarios/c3_") + sp + ".json");
        INFO("scenario=" << scenario);
        REQUIRE_FALSE(scenario.empty());

        const auto glb = rg::generate_glb("--scenario=\"" + scenario + "\" --seed=42",
                                          std::string("treegen_c3_") + sp + "_42.glb");
        for (const int lod : lods) {
            const std::string entry = std::string("c3_") + sp + "_L" + std::to_string(lod);
            const auto png = rg::render(glb, lod, "treegen_" + entry + ".png");
            const auto g = rg::grade_against_golden(png, entry + ".png");
            std::error_code ec;
            fs::remove(png, ec);
            if (!g) continue;

            INFO("species=" << sp << " lod=" << lod
                 << " rendered=" << std::hex << g->rendered
                 << " golden="   << std::hex << g->golden
                 << " hamming="  << std::dec << g->hamming);
            if (g->hamming > k_phash_threshold_soft_render) {
                WARN("pHash drift over soft threshold (" << g->hamming
                     << " > " << k_phash_threshold_soft_render << "): " << entry);
            }
            CHECK(g->hamming <= k_phash_threshold_hard_render);
        }
        std::error_code ec;
        fs::remove(glb, ec);
    }
}
