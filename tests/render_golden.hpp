#pragma once

// The GPU render gates' one shared step: rynx-treegen -> GLB -> the game's
// --render-glb mode -> PNG -> 8x8 DCT pHash against a checked-in golden. One
// copy, so the three gates cannot disagree about scratch files, the retry, the
// bootstrap mode or where goldens live.
//
// Built only into TestTreegenRender, whose build supplies RYNX_TREEGEN_EXE,
// RYNX_TREEGEN_RENDER_EXE and RYNX_TREEGEN_GOLDEN_DIR and depends on building
// both executables.

#include "../external/catch2/catch.hpp"
#include "test_env.hpp"
#include "test_support_paths.hpp"
#include "../phash.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef RYNX_TREEGEN_GOLDEN_DIR
#error "render_golden.hpp needs RYNX_TREEGEN_GOLDEN_DIR (TestTreegenRender's build defines it)"
#endif

namespace treegen::render_golden {

// RYNX_TREEGEN_GOLDEN_BOOTSTRAP=1: a MISSING golden is written from this run's
// render (WARN) for review and commit. Unset: a missing golden FAILS, so a
// deleted golden can never silently green-ship.
inline bool bootstrap_mode() {
    const char* env = std::getenv("RYNX_TREEGEN_GOLDEN_BOOTSTRAP");
    return env && env[0] == '1';
}

// rynx-treegen <tool_args> --out=<scratch/leaf>. Run from the scratch
// directory: the tool finds scenarios from its own location, not the cwd.
inline std::filesystem::path generate_glb(const std::string& tool_args, std::string_view leaf) {
    namespace fs = std::filesystem;
    const fs::path glb = test_env::scratch(leaf);
    std::ostringstream args;
    args << tool_args << " --out=\"" << glb.string() << "\"";
    const int rc = rynx::test_support::run_exe_in_dir(
        test_env::treegen_exe().string(), glb.parent_path().string(), args.str());
    INFO("rynx-treegen " << args.str() << " rc=" << rc);
    REQUIRE(rc == 0);
    REQUIRE(fs::exists(glb));
    return glb;
}

// The game's --render-glb mode, from its own bin directory (its VFS mounts are
// cwd-relative).
inline std::filesystem::path render(const std::filesystem::path& glb, int lod, std::string_view leaf) {
    namespace fs = std::filesystem;
    const fs::path exe = test_env::render_exe();
    const fs::path png = test_env::scratch(leaf);
    std::ostringstream args;
    args << "--render-glb=\"" << glb.string() << "\" --out=\"" << png.string() << "\" --lod=" << lod;
    auto attempt = [&] {
        std::error_code ec;
        fs::remove(png, ec);
        return rynx::test_support::run_exe_in_dir(exe.string(), exe.parent_path().string(), args.str());
    };
    // The game's shutdown path has crashed after writing the PNG (see
    // project_shutdown_segfault). One retry, loudly, so a teardown crash does
    // not read as a render regression; a render that fails twice is red.
    int rc = attempt();
    if (rc != 0 || !fs::exists(png)) {
        WARN("render " << args.str() << " rc=" << rc << " on the first attempt; retrying once");
        rc = attempt();
    }
    INFO("render " << args.str() << " rc=" << rc);
    REQUIRE(rc == 0);
    REQUIRE(fs::exists(png));
    return png;
}

struct grade {
    std::uint64_t rendered = 0;
    std::uint64_t golden = 0;
    int hamming = 0;
};

inline std::vector<std::uint8_t> slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// pHash distance of `png` from RYNX_TREEGEN_GOLDEN_DIR/<golden_leaf>. nullopt
// when bootstrap mode wrote a missing golden instead.
inline std::optional<grade> grade_against_golden(const std::filesystem::path& png, const std::string& golden_leaf) {
    namespace fs = std::filesystem;
    const fs::path golden = fs::path(RYNX_TREEGEN_GOLDEN_DIR) / golden_leaf;
    if (!fs::exists(golden)) {
        if (!bootstrap_mode())
            FAIL("missing golden " << golden.string()
                 << " -- set RYNX_TREEGEN_GOLDEN_BOOTSTRAP=1 to write it from this render, then review and commit");
        fs::create_directories(golden.parent_path());
        fs::copy_file(png, golden, fs::copy_options::overwrite_existing);
        WARN("bootstrapped golden " << golden.string() << " -- review and commit");
        return std::nullopt;
    }
    const auto rendered_bytes = slurp(png);
    const auto golden_bytes = slurp(golden);
    REQUIRE_FALSE(rendered_bytes.empty());
    REQUIRE_FALSE(golden_bytes.empty());
    grade g;
    g.rendered = treegen::phash::compute(rendered_bytes);
    g.golden = treegen::phash::compute(golden_bytes);
    g.hamming = treegen::phash::hamming(g.rendered, g.golden);
    return g;
}

} // namespace treegen::render_golden
