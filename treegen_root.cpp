#include "treegen_root.hpp"

#include <cstdlib>
#include <string>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace treegen {

std::filesystem::path executable_path() {
#ifdef _WIN32
    std::wstring path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (n == 0) std::abort();  // the running exe has a path
        if (n < path.size()) { path.resize(n); break; }
        path.resize(path.size() * 2);  // truncated: n == size
    }
#else
    std::string path(4096, '\0');
    const ssize_t n = readlink("/proc/self/exe", path.data(), path.size());
    if (n <= 0) std::abort();
    path.resize(static_cast<size_t>(n));
#endif
    return std::filesystem::path(path);
}

std::optional<std::filesystem::path> locate_root(const std::filesystem::path& exe_dir,
                                                 std::string_view relative_to_exe) {
    std::filesystem::path root = (exe_dir / std::filesystem::path(relative_to_exe)).lexically_normal();
    if (!root.has_filename()) root = root.parent_path();  // "a/b/" -> "a/b"
    std::error_code ec;
    if (!std::filesystem::is_regular_file(root / k_root_marker, ec)) return std::nullopt;
    return root;
}

} // namespace treegen
