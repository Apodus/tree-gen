// [treegen_species_library] — C7 P3. Process-spawning golden test: 4 species
// x 4 seeds = 16 entries. NOT under [treegen] umbrella (process-spawning).
//
// Each entry: rynx-treegen.exe --species <name> --seed <seed> produces GLB,
// topdownshootergame.exe --render-glb renders PNG, pHash compares to golden.
//
// Verifier amendment D1: disk size gate (total < 50MB) + timing WARN.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"
#include "../phash.hpp"
#include "../species_registry.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int k_phash_threshold_soft = 4;
constexpr int k_phash_threshold_hard = 8;
constexpr int k_seeds[] = { 42, 123, 7, 999 };
constexpr int k_seed_count = 4;
inline const char* k_env_bootstrap = "RYNX_TREEGEN_GOLDEN_BOOTSTRAP";

std::vector<std::uint8_t> slurp_png(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.good()) return {};
    return std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>());
}

} // anonymous namespace

TEST_CASE("[treegen_species_library] 16-entry species x seed golden test + vision gates",
          "[treegen_species_library]") {
    namespace fs  = std::filesystem;
    namespace tsp = rynx::test_support;

    const std::string treegen_exe = tsp::find_built_exe("rynx-treegen.exe");
    const std::string game_exe    = tsp::find_built_exe("topdownshootergame.exe");
    INFO("treegen_exe=" << treegen_exe);
    INFO("game_exe="    << game_exe);
    REQUIRE_FALSE(treegen_exe.empty());
    REQUIRE_FALSE(game_exe.empty());

    const std::string game_bindir = fs::path(game_exe).parent_path().string();

    // Golden anchor.
    fs::path anchor_dir;
    {
        const std::string anchor_parent = tsp::find_repo_file("tests/golden/");
        if (!anchor_parent.empty()) {
            anchor_dir = fs::path(anchor_parent) / "treegen";
        }
    }

    const auto species = treegen::get_species_names();
    REQUIRE(species.size() == 4u);

    const auto t0 = std::chrono::steady_clock::now();
    std::uintmax_t total_glb_bytes = 0;
    int entries_passed = 0;

    // Straight loop — no SECTION — so disk/timing gates accumulate correctly.
    for (const auto& sp : species) {
        for (int si = 0; si < k_seed_count; ++si) {
            const int seed = k_seeds[si];
            const fs::path tmp_glb = fs::temp_directory_path()
                / ("treegen_lib_" + sp + "_" + std::to_string(seed) + ".glb");
            const fs::path tmp_png = fs::temp_directory_path()
                / ("treegen_lib_" + sp + "_s" + std::to_string(seed) + ".png");
            std::error_code ec;
            fs::remove(tmp_glb, ec);
            fs::remove(tmp_png, ec);

            // Step 1: generate GLB.
            {
                std::ostringstream args;
                args << "--species " << sp
                     << " --seed=" << seed
                     << " --out=\"" << tmp_glb.string() << "\"";
                const int rc = tsp::run_exe_in_dir(treegen_exe,
                    fs::current_path().string(), args.str());
                INFO("rynx-treegen [" << sp << " seed=" << seed << "] rc=" << rc);
                REQUIRE(rc == 0);
                REQUIRE(fs::exists(tmp_glb));
            }

            total_glb_bytes += fs::file_size(tmp_glb, ec);

            // Step 2: render PNG.
            {
                std::ostringstream args;
                args << "--render-glb=\"" << tmp_glb.string()
                     << "\" --out=\"" << tmp_png.string()
                     << "\" --lod=0";
                int rc = tsp::run_exe_in_dir(game_exe, game_bindir, args.str());
                if (rc != 0 || !fs::exists(tmp_png)) {
                    WARN("render rc=" << rc << " on first attempt for "
                         << sp << " s" << seed << " — retrying once");
                    fs::remove(tmp_png, ec);
                    rc = tsp::run_exe_in_dir(game_exe, game_bindir, args.str());
                }
                INFO("topdownshootergame [" << sp << " s" << seed << "] rc=" << rc);
                REQUIRE(rc == 0);
                REQUIRE(fs::exists(tmp_png));
            }

            // Step 3: golden lookup.
            const std::string golden_rel = "tests/golden/treegen/"
                + sp + "_s" + std::to_string(seed) + ".png";
            const std::string golden_abs = tsp::find_repo_file(golden_rel);
            if (golden_abs.empty()) {
                if (const char* env = std::getenv(k_env_bootstrap);
                    env && env[0] == '1') {
                    REQUIRE_FALSE(anchor_dir.empty());
                    fs::create_directories(anchor_dir, ec);
                    const fs::path dst = anchor_dir
                        / (sp + "_s" + std::to_string(seed) + ".png");
                    fs::copy_file(tmp_png, dst,
                        fs::copy_options::overwrite_existing, ec);
                    REQUIRE_FALSE(ec);
                    WARN("bootstrapped golden: " << golden_rel);
                    fs::remove(tmp_glb, ec);
                    fs::remove(tmp_png, ec);
                    ++entries_passed;
                    continue;
                }
                fs::remove(tmp_glb, ec);
                fs::remove(tmp_png, ec);
                FAIL("missing golden: " << golden_rel
                      << " — set " << k_env_bootstrap
                      << "=1 to regenerate");
            }

            // Step 4: pHash compare.
            const auto rendered = slurp_png(tmp_png);
            const auto golden   = slurp_png(golden_abs);
            REQUIRE_FALSE(rendered.empty());
            REQUIRE_FALSE(golden.empty());

            const std::uint64_t h_r = treegen::phash::compute(rendered);
            const std::uint64_t h_g = treegen::phash::compute(golden);
            const int h = treegen::phash::hamming(h_r, h_g);
            INFO("species=" << sp << " seed=" << seed
                 << " rendered=" << std::hex << h_r
                 << " golden="   << std::hex << h_g
                 << " hamming="  << std::dec << h);
            if (h > k_phash_threshold_soft) {
                WARN("pHash drift over soft threshold (" << h
                     << " > " << k_phash_threshold_soft << "): "
                     << sp << " s" << seed);
            }
            REQUIRE(h <= k_phash_threshold_hard);

            ++entries_passed;

            // Cleanup.
            fs::remove(tmp_glb, ec);
            fs::remove(tmp_png, ec);
        }
    }

    // All 16 entries must have passed.
    REQUIRE(entries_passed == 16);

    // ---- Vision gate D1: disk size < 50 MB ----
    INFO("total_glb_bytes=" << total_glb_bytes);
    REQUIRE(total_glb_bytes < 50u * 1024u * 1024u);

    // ---- Vision gate D1: timing WARN ----
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    INFO("elapsed_seconds=" << elapsed_s);
    if (elapsed_s > 180.0) {
        WARN("species_library loop took " << elapsed_s
             << "s (> 180s timing WARN threshold)");
    }
}
