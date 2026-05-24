// C1 P4 — end-to-end golden harness for the treegen pipeline.
//
// Spawns rynx-treegen.exe → produces a GLB. Spawns topdownshootergame.exe
// --render-glb=<glb> --out=<png> from build/topdownshooter/bin/ — produces a
// PNG. Computes 8x8 DCT pHash, compares to the checked-in golden, requires
// Hamming <= 6 (C1 vision tolerance).
//
// C4 P5: retrofit env-var-gated bootstrap (matches [treegen_all_species_lod_render]).
//   Default behavior on missing golden = FAIL with diagnostic (closes the
//   silent-green hole — pre-P5 this test SUCCEED'd on missing golden).
//   RYNX_TREEGEN_GOLDEN_BOOTSTRAP=1 = copy rendered PNG into repo + WARN + SUCCEED.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"
#include "../phash.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> slurp_png(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.good()) return {};
    return std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>());
}

} // anonymous namespace

TEST_CASE("treegen: render trivial cylinder phash within 6 of golden",
          "[treegen][treegen_render_phash]") {
    namespace fs = std::filesystem;
    namespace ts = rynx::test_support;

    auto treegen_exe = ts::find_built_exe("rynx-treegen.exe");
    auto game_exe    = ts::find_built_exe("topdownshootergame.exe");
    INFO("treegen_exe=" << treegen_exe);
    INFO("game_exe=" << game_exe);
    REQUIRE_FALSE(treegen_exe.empty());
    REQUIRE_FALSE(game_exe.empty());

    const auto game_bindir = fs::path(game_exe).parent_path().string();

    // Scenario JSON lives alongside the tool sources (tools/rynx-treegen/
    // scenarios/). Search for it via the tool exe's path; fall back to
    // find_repo_file. Both produce absolute paths.
    auto scenario_path = ts::find_repo_file(
        "tools/rynx-treegen/scenarios/trivial_cylinder.json");
    INFO("scenario_path=" << scenario_path);
    REQUIRE_FALSE(scenario_path.empty());

    auto tmp_dir = fs::temp_directory_path();
    auto tmp_glb = tmp_dir / "treegen_c1_42.glb";
    auto tmp_png = tmp_dir / "treegen_c1_42.png";

    // Wipe any stale outputs from a prior run so REQUIRE(fs::exists(...))
    // diagnoses *this* run's failure rather than a stale file from yesterday.
    std::error_code ec;
    fs::remove(tmp_glb, ec);
    fs::remove(tmp_png, ec);

    // Step 1: rynx-treegen → GLB.
    {
        std::ostringstream args;
        args << "--scenario=\"" << scenario_path
             << "\" --seed=42 --out=\"" << tmp_glb.string() << "\"";
        int rc = ts::run_exe_in_dir(treegen_exe, fs::current_path().string(),
                                    args.str());
        INFO("rynx-treegen rc=" << rc);
        REQUIRE(rc == 0);
        REQUIRE(fs::exists(tmp_glb));
    }

    // Step 2: topdownshootergame --render-glb → PNG.
    // CWD MUST be build/topdownshooter/bin/ (VFS mounts are CWD-relative,
    // per reference_run_from_bin_cwd).
    {
        std::ostringstream args;
        args << "--render-glb=\"" << tmp_glb.string()
             << "\" --out=\"" << tmp_png.string() << "\"";
        int rc = ts::run_exe_in_dir(game_exe, game_bindir, args.str());
        INFO("topdownshootergame rc=" << rc);
        REQUIRE(rc == 0);
        REQUIRE(fs::exists(tmp_png));
    }

    // Step 3: locate the golden PNG. C4 P5: missing golden = explicit FAIL
    // by default. Env-var RYNX_TREEGEN_GOLDEN_BOOTSTRAP=1 promotes the path
    // to "copy rendered PNG into repo + WARN + SUCCEED" so the operator can
    // re-baseline + commit. Pre-P5 this branch was an unconditional WARN+SUCCEED
    // which silently green-shipped if a golden was ever deleted.
    const std::string golden_rel = "tests/golden/treegen/trivial_42.png";
    auto golden_path = ts::find_repo_file(golden_rel);
    if (golden_path.empty()) {
        if (const char* env = std::getenv("RYNX_TREEGEN_GOLDEN_BOOTSTRAP");
            env && env[0] == '1') {
            // Resolve a sibling golden dir via any existing repo file; here we
            // walk up from this TU's known siblings — falling back to writing
            // alongside CWD if necessary so the bootstrap path always lands.
            auto anchor_dir_parent = ts::find_repo_file("tests/golden/");
            fs::path dst;
            if (!anchor_dir_parent.empty()) {
                dst = fs::path(anchor_dir_parent) / "treegen" / "trivial_42.png";
            } else {
                dst = fs::current_path() / "tests" / "golden" / "treegen" / "trivial_42.png";
            }
            std::error_code ec2;
            fs::create_directories(dst.parent_path(), ec2);
            fs::copy_file(tmp_png, dst, fs::copy_options::overwrite_existing, ec2);
            REQUIRE_FALSE(ec2);
            WARN("bootstrapped golden: " << golden_rel
                  << " (copied from " << tmp_png.string()
                  << ") — review + commit");
            SUCCEED("bootstrap-mode wrote " + golden_rel);
            return;
        }
        FAIL("missing golden: " << golden_rel
              << " — set RYNX_TREEGEN_GOLDEN_BOOTSTRAP=1 to regenerate; "
                 "do NOT commit env-var-set runs as 'green'");
    }

    // Step 4: pHash compare.
    auto rendered_bytes = slurp_png(tmp_png);
    auto golden_bytes   = slurp_png(golden_path);
    REQUIRE_FALSE(rendered_bytes.empty());
    REQUIRE_FALSE(golden_bytes.empty());

    const std::uint64_t h_rendered = treegen::phash::compute(rendered_bytes);
    const std::uint64_t h_golden   = treegen::phash::compute(golden_bytes);
    const int dist = treegen::phash::hamming(h_rendered, h_golden);
    INFO("rendered_phash=" << std::hex << h_rendered
         << " golden_phash="  << std::hex << h_golden
         << " hamming=" << std::dec << dist);
    REQUIRE(dist <= 6);
}
