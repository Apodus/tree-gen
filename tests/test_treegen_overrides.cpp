// [treegen_overrides] -- the crown-probe sweep overrides (crown-shadows S-P0a,
// K17 "Overrides"): leaves.lod_leaf_triangle_budget, leaves.shell_fraction,
// leaves.cluster_cell_grid are inert when absent (or set to their defaults)
// and live when present; the two site-based ones are refused on a
// branch_strip scenario.
#include "../external/catch2/catch.hpp"
#include "generated_trees.hpp"
#include "test_support_paths.hpp"

#include "../crown_probe.hpp"
#include "../generate_tree.hpp"
#include "../json_reader.hpp"
#include "../scenario.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

std::string scenario_text(const std::string& species) {
    std::ifstream f(rynx::test_support::treegen_file("scenarios/c3_" + species + ".json"), std::ios::binary);
    REQUIRE(f.good());
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// c3_<species>.json with `fields` (JSON members, comma-terminated) inserted
// first in its "leaves" object; the scenario name (which seeds the skeleton)
// is unchanged.
treegen::Scenario with_leaves_fields(const std::string& species, const std::string& fields) {
    std::string text = scenario_text(species);
    const size_t at = text.find("\"leaves\": {");
    REQUIRE(at != std::string::npos);
    text.insert(at + std::string("\"leaves\": {").size(), "\n    " + fields);
    return treegen::parse_scenario(text, "c3_" + species + ".json + " + fields);
}

struct Readout {
    size_t sites = 0;
    size_t l0_leaf_triangles = 0;
    double alpha_coverage = 0.0;
};
Readout read(const treegen::GeneratedTree& g) {
    const treegen::crown::Probe probe(treegen::crown::load_crown_from_glb(g.glb, 0).mesh);
    double card = 0.0, opaque = 0.0;
    for (uint32_t t = 0; t < probe.mesh().triangles.size(); ++t) {
        const auto& tr = probe.mesh().triangles[t];
        if (tr.site < 0) continue;
        const double a = 0.5 * treegen::length(treegen::cross(tr.p[1] - tr.p[0], tr.p[2] - tr.p[0]));
        card += a;
        opaque += a * probe.triangle_coverage(t);
    }
    return {probe.mesh().sites.size(), probe.mesh().leaf_triangles, opaque / card};
}

}  // namespace

TEST_CASE("[treegen_overrides] absent in every tracked scenario; explicit defaults change no byte",
          "[treegen][treegen_overrides]") {
    namespace tsp = rynx::test_support;
    int scenarios = 0;
    for (const auto& e : std::filesystem::directory_iterator(tsp::treegen_root() / "scenarios")) {
        if (e.path().extension() != ".json") continue;
        std::ifstream f(e.path(), std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        INFO(e.path().filename().string());
        for (const char* key : {"lod_leaf_triangle_budget", "shell_fraction", "cluster_cell_grid"})
            CHECK(text.find(key) == std::string::npos);
        ++scenarios;
    }
    CHECK(scenarios >= 4);
    // Absent = the extraction table ([treegen_generate_tree]). Present at the
    // defaults = the same bytes.
    const treegen::Scenario s = with_leaves_fields(
        "oak", "\"lod_leaf_triangle_budget\": [8000, 5000, 4000, 0], \"shell_fraction\": 0.4, \"cluster_cell_grid\": [4, 3],");
    CHECK(treegen::generate_tree(s, 42).glb == tsp::generated_tree("oak", 42).glb);
}

TEST_CASE("[treegen_overrides] shell_fraction, leaf budget and cluster grid are live", "[treegen][treegen_overrides]") {
    const Readout base = read(rynx::test_support::generated_tree("oak", 42));
    REQUIRE(base.l0_leaf_triangles < 8000u);   // the L0 budget does not bind today, so a lower prune shows

    SECTION("shell_fraction 0.30 keeps more of the interior") {
        const Readout r = read(treegen::generate_tree(with_leaves_fields("oak", "\"shell_fraction\": 0.30,"), 42));
        INFO("sites " << base.sites << " -> " << r.sites);
        CHECK(r.sites > base.sites);
    }
    SECTION("lod_leaf_triangle_budget[0] halved") {
        const Readout r = read(treegen::generate_tree(
            with_leaves_fields("oak", "\"lod_leaf_triangle_budget\": [4000, 4000, 4000, 0],"), 42));
        INFO("L0 leaf triangles " << base.l0_leaf_triangles << " -> " << r.l0_leaf_triangles);
        CHECK(r.l0_leaf_triangles <= 4000u);
        CHECK(r.l0_leaf_triangles < base.l0_leaf_triangles);
    }
    SECTION("cluster_cell_grid [6, 2] lowers alpha coverage") {
        // Cell coverage scales as nx * ny * min(1/nx, 1/ny)^2: 12/16 at [4, 3],
        // 12/36 at [6, 2] (leaf_atlas.cpp's slot radius).
        const Readout r = read(treegen::generate_tree(with_leaves_fields("oak", "\"cluster_cell_grid\": [6, 2],"), 42));
        INFO("alpha coverage " << base.alpha_coverage << " -> " << r.alpha_coverage);
        CHECK(r.alpha_coverage < 0.75 * base.alpha_coverage);
    }
}

TEST_CASE("[treegen_overrides] site-based overrides are refused on branch_strip", "[treegen][treegen_overrides]") {
    for (const char* field : {"\"shell_fraction\": 0.30,", "\"lod_leaf_triangle_budget\": [100, 100, 100, 0],"}) {
        INFO(field);
        try {
            (void)with_leaves_fields("pine", field);
            FAIL("accepted on a branch_strip scenario");
        } catch (const std::runtime_error& e) {
            CHECK(std::string(e.what()).find("branch_strip") != std::string::npos);
        }
    }
    // The grid is not site-based: accepted on pine.
    CHECK_NOTHROW(with_leaves_fields("pine", "\"cluster_cell_grid\": [6, 2],"));
}
