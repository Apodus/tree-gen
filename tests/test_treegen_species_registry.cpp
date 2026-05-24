// [treegen_species_registry] — C7 P1. Pins species_registry (enumerate,
// resolve) and --species CLI flag (process-level exit code + GLB output).

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"
#include "../species_registry.hpp"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

TEST_CASE("treegen_species_registry: enumerate returns sorted 4 names",
          "[treegen][treegen_species_registry]") {
    const auto names = treegen::get_species_names();
    REQUIRE(names.size() == 4u);
    REQUIRE(names[0] == "birch");
    REQUIRE(names[1] == "maple");
    REQUIRE(names[2] == "oak");
    REQUIRE(names[3] == "pine");
    REQUIRE(std::is_sorted(names.begin(), names.end()));
}

TEST_CASE("treegen_species_registry: unknown species returns empty path",
          "[treegen][treegen_species_registry]") {
    REQUIRE(treegen::get_scenario_path("willow").empty());
    REQUIRE(treegen::get_scenario_path("").empty());
}

TEST_CASE("treegen_species_registry: each species resolves to an existing path",
          "[treegen][treegen_species_registry]") {
    namespace fs = std::filesystem;
    for (const auto& name : treegen::get_species_names()) {
        INFO("species=" << name);
        const std::string path = treegen::get_scenario_path(name);
        REQUIRE_FALSE(path.empty());
        std::error_code ec;
        REQUIRE(fs::exists(path, ec));
    }
}

TEST_CASE("treegen_species_registry: --species oak --seed 42 produces GLB",
          "[treegen_species_registry]") {    // NO [treegen] umbrella — process-spawning.
    namespace fs  = std::filesystem;
    namespace tsp = rynx::test_support;

    const std::string exe = tsp::find_built_exe("rynx-treegen.exe");
    REQUIRE_FALSE(exe.empty());

    const fs::path tmp_glb = fs::temp_directory_path() / "treegen_species_registry_oak.glb";
    std::error_code ec;
    fs::remove(tmp_glb, ec);

    std::ostringstream args;
    args << "--species oak --seed 42 --out=\"" << tmp_glb.string() << "\"";
    const int rc = tsp::run_exe_in_dir(exe, fs::current_path().string(), args.str());
    INFO("rc=" << rc);
    REQUIRE(rc == 0);
    REQUIRE(fs::exists(tmp_glb));
    REQUIRE(fs::file_size(tmp_glb, ec) > 1000u);
    fs::remove(tmp_glb, ec);
}
