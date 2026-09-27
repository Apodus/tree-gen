// [treegen_species_library] — C7 P3. Process-spawning golden test: 4 species
// x 4 seeds = 16 entries.
//
// Each entry: rynx-treegen.exe --species <name> --seed <seed> produces GLB,
// the game's --render-glb mode renders PNG, pHash compares to golden. Every
// entry is graded (CHECK), so a red names ALL the drifted entries rather than
// the first one reached.
//
// Verifier amendment D1: disk size gate + timing WARN.

#include "../external/catch2/catch.hpp"

#include "render_golden.hpp"
#include "../species_registry.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>

namespace {

constexpr int k_phash_threshold_soft = 4;
constexpr int k_phash_threshold_hard = 8;
constexpr int k_seeds[] = { 42, 123, 7, 999 };

} // anonymous namespace

TEST_CASE("[treegen_species_library] 16-entry species x seed golden test + vision gates",
          "[treegen_species_library]") {
    namespace fs = std::filesystem;
    namespace rg = treegen::render_golden;

    const auto species = treegen::get_species_names();
    REQUIRE(species.size() == 4u);

    const auto t0 = std::chrono::steady_clock::now();
    std::uintmax_t total_glb_bytes = 0;

    for (const auto& sp : species) {
        for (const int seed : k_seeds) {
            const std::string entry = sp + "_s" + std::to_string(seed);
            const auto glb = rg::generate_glb("--species " + sp + " --seed=" + std::to_string(seed),
                                              "treegen_lib_" + entry + ".glb");
            std::error_code ec;
            total_glb_bytes += fs::file_size(glb, ec);

            const auto png = rg::render(glb, 0, "treegen_lib_" + entry + ".png");
            const auto g = rg::grade_against_golden(png, entry + ".png");
            fs::remove(glb, ec);
            fs::remove(png, ec);
            if (!g) continue;

            INFO("species=" << sp << " seed=" << seed
                 << " rendered=" << std::hex << g->rendered
                 << " golden="   << std::hex << g->golden
                 << " hamming="  << std::dec << g->hamming);
            if (g->hamming > k_phash_threshold_soft) {
                WARN("pHash drift over soft threshold (" << g->hamming
                     << " > " << k_phash_threshold_soft << "): " << entry);
            }
            CHECK(g->hamming <= k_phash_threshold_hard);
        }
    }

    // ---- Vision gate D1: disk size < 80 MB ----
    // Raised from 50→80 MB: C1 P1 crotch cap at all LODs + _RYNX_LOD extension fields.
    INFO("total_glb_bytes=" << total_glb_bytes);
    REQUIRE(total_glb_bytes < 80u * 1024u * 1024u);

    // ---- Vision gate D1: timing WARN ----
    const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    INFO("elapsed_seconds=" << elapsed_s);
    if (elapsed_s > 180.0) {
        WARN("species_library loop took " << elapsed_s << "s (> 180s timing WARN threshold)");
    }
}
