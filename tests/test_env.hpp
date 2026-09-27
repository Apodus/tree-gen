#pragma once

// What the treegen tests spawn, and where they write.
//
// Executables are compile-time paths set by the test executable's build, which
// also depends on building them: never a search. The search this replaces
// walked a ladder of cwd-relative directories and took the first hit, which is
// how an untracked month-old rynx-treegen.exe in a verifier slot kept passing
// for current source, and how a tree that had not built it read as a
// regression.
//
// Scratch files go to a directory owned by this process. Fixed names under the
// shared temp directory let two concurrent test processes (two worktrees, a
// verify beside a local run) overwrite each other's GLBs mid-case, and
// std::rand() names collide just the same: every process seeds it identically.

#include "../external/catch2/catch.hpp"

#include <filesystem>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <process.h>
#define TREEGEN_TEST_GETPID _getpid
#else
#include <unistd.h>
#define TREEGEN_TEST_GETPID getpid
#endif

namespace treegen::test_env {

// FAILS, naming the path, when a build-provided executable is absent. With the
// build dependency in place that means the build is broken, which is not a
// treegen regression and must not read as one -- nor pass as a skip.
inline std::filesystem::path require_built(const char* path) {
    const std::filesystem::path exe = path;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(exe, ec))
        FAIL(exe.string() << " does not exist: this test executable's build dependency did not produce it "
                             "(a broken build, not a treegen failure)");
    return exe;
}

#ifdef RYNX_TREEGEN_EXE
inline std::filesystem::path treegen_exe() { return require_built(RYNX_TREEGEN_EXE); }
#endif

#ifdef RYNX_TREEGEN_RENDER_EXE
// The game executable whose --render-glb mode renders a GLB to PNG.
inline std::filesystem::path render_exe() { return require_built(RYNX_TREEGEN_RENDER_EXE); }
#endif

// <temp>/rynx-treegen-tests-<pid>/<leaf>. Created on first use and removed
// when the process exits.
inline std::filesystem::path scratch(std::string_view leaf) {
    struct owned_dir {
        std::filesystem::path path;
        owned_dir()
            : path(std::filesystem::temp_directory_path()
                   / ("rynx-treegen-tests-" + std::to_string(TREEGEN_TEST_GETPID()))) {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);  // a dead process with a recycled pid
            std::filesystem::create_directories(path);
        }
        ~owned_dir() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    };
    static const owned_dir dir;
    return dir.path / leaf;
}

} // namespace treegen::test_env
