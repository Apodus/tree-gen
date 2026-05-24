#include "test_support_paths.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>

namespace rynx::test_support {

std::string find_built_shader(std::string_view filename) {
    // Candidate roots, in priority order. Each is tried as `<root>/<filename>`.
    // Covers TestTech.exe invocation from rynx/build/tests/, rynx/build/,
    // rynx/, project root, and one extra level up.
    constexpr std::array<const char*, 6> k_candidates = {
        "build/shaders/",                  // project root invocation
        "./",                              // executable dir if shader is alongside
        "../build/shaders/",               // one up from rynx/build/tests/
        "../../build/shaders/",            // two up
        "../../../build/shaders/",         // three up (documented convention)
        "../../../../build/shaders/",      // four up (safety)
    };

    for (const char* root : k_candidates) {
        std::string candidate;
        candidate.reserve(64);
        candidate.append(root);
        candidate.append(filename);
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && !ec) {
            return candidate;
        }
    }
    return {};
}

std::string find_built_exe(std::string_view filename) {
    // Candidate roots, in priority order. Each is tried as `<root>/<filename>`.
    // Covers TestTech.exe invocation from rynx/build/tests/, rynx/build/, rynx/,
    // and the project root. Tool exes live under tools/ or build/<game>/bin/.
    constexpr std::array<const char*, 14> k_candidates = {
        // Project root invocation (most common dev/CI).
        "tools/rynx-treegen/",
        "build/topdownshooter/bin/",
        "build/cavecrew/bin/",
        "build/bike/bin/",
        "./",
        // One level up (e.g. invoked from rynx/).
        "../tools/rynx-treegen/",
        "../build/topdownshooter/bin/",
        // Two levels up.
        "../../tools/rynx-treegen/",
        "../../build/topdownshooter/bin/",
        // Three levels up (rynx/build/tests/ → repo root).
        "../../../tools/rynx-treegen/",
        "../../../build/topdownshooter/bin/",
        // Four levels up (safety).
        "../../../../tools/rynx-treegen/",
        "../../../../build/topdownshooter/bin/",
        "../../../../",
    };

    for (const char* root : k_candidates) {
        std::string candidate;
        candidate.reserve(96);
        candidate.append(root);
        candidate.append(filename);
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && !ec) {
            // Return ABSOLUTE path so callers can chdir freely without losing the exe.
            auto abs = std::filesystem::absolute(candidate, ec);
            if (!ec) return abs.string();
            return candidate;
        }
    }
    return {};
}

std::string find_repo_file(std::string_view repo_relative_path) {
    // Mirrors find_built_shader/find_built_exe but takes an arbitrary
    // repo-relative path (may include subdirs). Resolves to absolute on hit.
    constexpr std::array<const char*, 6> k_candidates = {
        "./",
        "../",
        "../../",
        "../../../",
        "../../../../",
        "../../../../../",
    };

    // Standalone treegen repo: callers pass game_one-relative paths like
    // "tools/rynx-treegen/scenarios/c3_oak.json". When running from the
    // standalone treegen repo root, strip the prefix so we find
    // "scenarios/c3_oak.json" directly.
    constexpr std::string_view k_treegen_prefix = "tools/rynx-treegen/";
    std::string_view stripped = repo_relative_path;
    if (stripped.substr(0, k_treegen_prefix.size()) == k_treegen_prefix)
        stripped = stripped.substr(k_treegen_prefix.size());

    for (std::string_view path : { repo_relative_path, stripped }) {
        for (const char* root : k_candidates) {
            std::string candidate;
            candidate.reserve(128);
            candidate.append(root);
            candidate.append(path);
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec) && !ec) {
                auto abs = std::filesystem::absolute(candidate, ec);
                if (!ec) return abs.string();
                return candidate;
            }
        }
    }
    return {};
}

int run_exe_in_dir(std::string_view exe_abs,
                   std::string_view working_dir_abs,
                   std::string_view args) {
    // Windows: `cmd /c "cd /d <wd> && <exe> <args>"`.
    // The outer-quoted form (`cmd /c "..."`) is the documented contract for
    // forwarding the entire post-/c string to cmd.exe verbatim.
    std::ostringstream cmd;
    cmd << "cmd /c \"cd /d \"" << working_dir_abs << "\" && \""
        << exe_abs << "\"";
    if (!args.empty()) {
        cmd << ' ' << args;
    }
    cmd << "\"";
    return std::system(cmd.str().c_str());
}

} // namespace rynx::test_support
