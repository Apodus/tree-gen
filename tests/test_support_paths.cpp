#include "test_support_paths.hpp"

#include "../treegen_root.hpp"

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef TREEGEN_ROOT_FROM_EXE
#error "the treegen tests find the repository from the test exe's own location: the build must define TREEGEN_ROOT_FROM_EXE (the relative path from the exe's directory to the repository root)"
#endif

namespace rynx::test_support {

const std::filesystem::path& treegen_root() {
    static const std::filesystem::path root = [] {
        const auto exe_dir = treegen::executable_directory();
        const auto found = treegen::locate_root(exe_dir, TREEGEN_ROOT_FROM_EXE);
        if (!found)
            throw std::runtime_error("no " + std::string(treegen::k_root_marker) + " at " +
                                     (exe_dir / TREEGEN_ROOT_FROM_EXE).lexically_normal().string() +
                                     " (" + TREEGEN_ROOT_FROM_EXE + " from the exe's directory " + exe_dir.string() + ")");
        return *found;
    }();
    return root;
}

std::string treegen_file(std::string_view relative) {
    const std::filesystem::path path = treegen_root() / std::filesystem::path(relative);
    std::error_code ec;
    return std::filesystem::exists(path, ec) && !ec ? path.string() : std::string{};
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
