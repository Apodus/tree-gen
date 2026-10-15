#pragma once

// Test support: where the repository's data lives, and running a child process.

#include <filesystem>
#include <string>
#include <string_view>

namespace rynx::test_support {

// The repository root for this test executable: its own directory plus the
// build-defined relative path (TREEGEN_ROOT_FROM_EXE), verified by the committed
// .treegen-root marker (../treegen_root.hpp). Not the cwd, not a compiled
// absolute path. A miss throws, naming what was looked for: nothing downstream
// can mean anything without it.
const std::filesystem::path& treegen_root();

// <treegen_root()>/<relative> as an absolute path, or an empty string when that
// file does not exist (callers REQUIRE_FALSE(empty())).
std::string treegen_file(std::string_view relative);

// Run exe_abs from working_dir_abs with args. Returns the child process's
// exit code (0 on success). Uses `cmd /c "cd /d <wd> && <exe> <args>"` on
// Windows so the child's CWD is exactly working_dir_abs, required for any
// exe whose VFS mounts are CWD-relative (e.g. topdownshootergame.exe).
int run_exe_in_dir(std::string_view exe_abs,
                   std::string_view working_dir_abs,
                   std::string_view args);

} // namespace rynx::test_support
