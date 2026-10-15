// [treegen_root] -- the repository root is found at runtime from the exe's own
// location plus the committed .treegen-root marker; never from the cwd, never from
// a compiled absolute path, never by searching. The relocation arms run a copy of
// this exe inside a fresh tree, with and without the marker.

#include "../external/catch2/catch.hpp"
#include "test_env.hpp"

#include "test_support_paths.hpp"
#include "../treegen_root.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

namespace fs  = std::filesystem;
namespace tsp = rynx::test_support;

constexpr const char* k_probe_out_env = "TREEGEN_ROOT_PROBE_OUT";

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

struct restore_cwd {
    fs::path cwd = fs::current_path();
    ~restore_cwd() { std::error_code ec; fs::current_path(cwd, ec); }
};

// A copy of this exe where TREEGEN_ROOT_FROM_EXE lands inside `base`, and the
// root it resolves to (marker optional). Deep enough that the build's relative
// path cannot climb out of `base`.
struct planted {
    fs::path exe;
    fs::path root;
};

planted plant(const fs::path& base, bool with_marker) {
    int climbs = 0;
    for (const auto& part : fs::path(TREEGEN_ROOT_FROM_EXE))
        if (part == "..") ++climbs;
    fs::path dir = base;
    for (int i = 0; i <= climbs; ++i) dir /= "d" + std::to_string(i);
    fs::create_directories(dir);

    planted p;
    p.root = (dir / TREEGEN_ROOT_FROM_EXE).lexically_normal();
    if (!p.root.has_filename()) p.root = p.root.parent_path();
    fs::create_directories(p.root);
    if (with_marker) std::ofstream(p.root / treegen::k_root_marker).close();

    p.exe = dir / treegen::executable_path().filename();
    std::error_code ec;
    fs::create_hard_link(treegen::executable_path(), p.exe, ec);
    if (ec) fs::copy_file(treegen::executable_path(), p.exe);
    return p;
}

// Runs the probe case of the exe at `exe` from `cwd`, stdout+stderr to `log`.
int run_probe(const fs::path& exe, const fs::path& cwd, const fs::path& log, const fs::path& out) {
    set_env(k_probe_out_env, out.string());
    return tsp::run_exe_in_dir(exe.string(), cwd.string(),
                               "\"[treegen_root_probe]\" > \"" + log.string() + "\" 2>&1");
}

} // namespace

TEST_CASE("treegen_root: resolves from the exe's location, the same from any cwd",
          "[treegen][treegen_root]") {
    const fs::path& root = tsp::treegen_root();
    CHECK(fs::is_regular_file(root / treegen::k_root_marker));
    CHECK_FALSE(tsp::treegen_file("scenarios/c3_oak.json").empty());

    restore_cwd restore;
    fs::current_path(fs::temp_directory_path());
    const auto fresh = treegen::locate_root(treegen::executable_directory(), TREEGEN_ROOT_FROM_EXE);  // uncached
    REQUIRE(fresh);
    CHECK(fs::equivalent(*fresh, root));
}

TEST_CASE("treegen_root: the marker is required and nothing is searched for",
          "[treegen][treegen_root]") {
    const fs::path base = treegen::test_env::scratch("root_marker_required");
    fs::create_directories(base / "a" / "b");

    CHECK_FALSE(treegen::locate_root(base / "a", "b"));            // directory without the marker
    CHECK_FALSE(treegen::locate_root(base / "a", "missing"));      // no such directory
    std::ofstream(base / treegen::k_root_marker).close();          // a marker in an ancestor is no substitute
    CHECK_FALSE(treegen::locate_root(base / "a", "b"));
    std::ofstream(base / "a" / "b" / treegen::k_root_marker).close();
    const auto found = treegen::locate_root(base / "a", "b");
    REQUIRE(found);
    CHECK(fs::equivalent(*found, base / "a" / "b"));
    CHECK(treegen::locate_root(base / "a" / "b", "../.."));        // the relative path may climb
}

TEST_CASE("treegen_root: a relocated copy resolves to the copy (relocation control)",
          "[treegen_root]") {   // spawns this exe: NO [treegen] umbrella
    const fs::path base = treegen::test_env::scratch("root_relocation");
    fs::create_directories(base);

    // With the marker: the copy's root is the copy's, not the tree this exe was built in.
    const planted with = plant(base / "with", true);
    const fs::path out = base / "root.txt";
    const int rc = run_probe(with.exe, base, base / "found.log", out);
    INFO("child output:\n" << slurp(base / "found.log"));
    REQUIRE(rc == 0);
    const fs::path reported = slurp(out);
    INFO("child reported " << reported.string());
    CHECK(fs::equivalent(reported, with.root));
    CHECK_FALSE(fs::equivalent(reported, tsp::treegen_root()));

    // Without the marker: the same exe in a tree with nothing to find FAILS, naming where it looked.
    const planted bare = plant(base / "bare", false);
    const fs::path bare_out = base / "bare_root.txt";
    const int bare_rc = run_probe(bare.exe, base, base / "missed.log", bare_out);
    const std::string missed = slurp(base / "missed.log");
    INFO("child output:\n" << missed);
    CHECK(bare_rc != 0);
    CHECK(missed.find("no .treegen-root at") != std::string::npos);
    CHECK_FALSE(fs::exists(bare_out));
}

// Child side of the relocation arms: writes treegen_root() to $TREEGEN_ROOT_PROBE_OUT.
TEST_CASE("treegen_root probe (child of the relocation control)", "[.][treegen_root_probe]") {
    const char* out = std::getenv(k_probe_out_env);
    if (!out) return;
    const std::string root = tsp::treegen_root().string();
    std::ofstream(out, std::ios::binary) << root;
}

TEST_CASE("treegen_root: the tool finds the same root, from a foreign cwd",
          "[treegen_root]") {   // spawns the tool: NO [treegen] umbrella
    const fs::path exe  = treegen::test_env::treegen_exe();
    const fs::path out  = treegen::test_env::scratch("tool_root.txt");
    const fs::path cwd  = out.parent_path();
    const int rc = tsp::run_exe_in_dir(exe.string(), cwd.string(), "--print-root > \"" + out.string() + "\"");
    REQUIRE(rc == 0);
    std::string reported = slurp(out);
    while (!reported.empty() && (reported.back() == '\n' || reported.back() == '\r')) reported.pop_back();
    CHECK(fs::equivalent(reported, tsp::treegen_root()));
}
