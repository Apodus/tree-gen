#pragma once

// Test support: CWD-robust shader-path discovery.
//
// Pre-T2.6.6, tech tests like test_spd_pyramid_shaders + test_wave_intrinsic_shaders
// hardcoded paths like "../../../build/shaders/foo.refl.spv" — those resolve
// correctly ONLY when TestTech.exe is invoked from rynx/build/tests/. Run
// from the project root (or anywhere else) and the relative path produces a
// non-existent file → REQUIRE(fh.good()) fires.
//
// This helper searches a series of candidate roots and returns the first one
// where the requested shader exists, OR an empty string if none match. Tests
// should pass the returned path to their existing load_spirv / file-read code;
// FAIL/REQUIRE-if-empty at the callsite makes the diagnostic clear.

#include <string>
#include <string_view>

namespace rynx::test_support {

// Find a built shader artifact by filename. Searches:
//   ./build/shaders/<filename>
//   ./<filename>
//   ../build/shaders/<filename>
//   ../../build/shaders/<filename>
//   ../../../build/shaders/<filename>
//   ../../../../build/shaders/<filename>
//
// Returns the FIRST path where std::filesystem::exists() returns true.
// Returns an empty string if none match. Callers should REQUIRE(!result.empty())
// (or FAIL("shader not found: " + filename)) so the diagnostic surfaces the
// missing-asset case clearly without an opaque "file open failed" deep in
// load_spirv.
//
// The candidate list covers TestTech.exe invocation from:
//   - rynx/build/tests/ (the documented convention; ../../../build/shaders/)
//   - rynx/build/ (../build/shaders/)
//   - rynx/ (build/shaders/ — when build directory was symlinked or moved)
//   - project root (build/shaders/ — most common dev/CI invocation)
//   - one extra level up for safety
std::string find_built_shader(std::string_view filename);

// Find a built exe by leaf filename. Same candidate-root strategy as
// find_built_shader, but the roots target standard tool / game bin
// directories (build/topdownshooter/bin, tools/rynx-treegen/, etc.).
// Returns absolute path of the FIRST existing match (so child process
// invocations are CWD-independent), or empty string if none match.
//
// Callers should REQUIRE_FALSE(result.empty()) and forward the result to
// run_exe_in_dir below.
std::string find_built_exe(std::string_view filename);

// Find a checked-in file by repo-relative path (e.g. "tests/golden/treegen/trivial_42.png").
// Walks the same candidate-root ladder as find_built_shader but appends the
// full path argument (which may contain subdirs). Returns absolute path of
// the first existing match, or empty string if none match.
std::string find_repo_file(std::string_view repo_relative_path);

// Run exe_abs from working_dir_abs with args. Returns the child process's
// exit code (0 on success). Uses `cmd /c "cd /d <wd> && <exe> <args>"` on
// Windows so the child's CWD is exactly working_dir_abs — required for any
// exe whose VFS mounts are CWD-relative (e.g. topdownshootergame.exe per
// reference_run_from_bin_cwd).
int run_exe_in_dir(std::string_view exe_abs,
                   std::string_view working_dir_abs,
                   std::string_view args);

} // namespace rynx::test_support
