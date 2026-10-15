#pragma once

// Where this repository's data (scenarios/, goldens) lives, for an executable
// built from it: found at runtime from the running exe's own directory, never
// from the cwd and never from a compiled absolute path, so any copy of the tree
// that keeps the exe-to-root layout resolves to itself.
//
// The build gives each exe the RELATIVE path from its directory to the root
// (CMake and rynx's Sharpmake define TREEGEN_ROOT_FROM_EXE); the root must hold
// the committed marker .treegen-root, so a stale or misplaced layout is a miss.

#include <filesystem>
#include <optional>
#include <string_view>

namespace treegen {

inline constexpr std::string_view k_root_marker = ".treegen-root";

std::filesystem::path executable_path();
inline std::filesystem::path executable_directory() { return executable_path().parent_path(); }

// `exe_dir` / `relative_to_exe`, normalised, when that directory holds the marker.
std::optional<std::filesystem::path> locate_root(const std::filesystem::path& exe_dir,
                                                 std::string_view relative_to_exe);

} // namespace treegen
