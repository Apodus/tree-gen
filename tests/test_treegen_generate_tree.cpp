// treegen::generate_tree is the tool's pipeline, byte for byte (crown-shadows
// S-P0a, K17 "Pipeline"). The table below was recorded BEFORE the extraction,
// with HEAD's Release tool (treegen 8c2006e: `rynx-treegen --scenario=<c3_*.json>
// --seed=42 --out=<glb>`, the staging command), one row per tracked tree
// scenario (= per registry species). Debug, Release and Retail wrote identical
// bytes (K10 step 0), and the extraction was checked once by hand on all 16
// staged trees (seeds 7, 42, 123, 999); the train row holds the four. The phase
// that next changes treegen output regenerates this table (K9).
#include "../external/catch2/catch.hpp"
#include "generated_trees.hpp"
#include "test_support_paths.hpp"

#include "../det_rng.hpp"
#include "../generate_tree.hpp"
#include "../scenario.hpp"
#include "../species_registry.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <string_view>

namespace {

struct GlbRow {
    const char* species;
    uint64_t    seed;
    size_t      bytes;
    uint64_t    fnv1a64;  // det_rng.hpp fnv1a64 over the GLB bytes
};

// Botanical Leaf Sprigs & 3D Cluster Uplift regenerated table (16-cell 1024^2
// asymmetric botanical sprig atlas, 3x4-vert sprig bouquets, twig-straddling
// placement, alpha-localized F2_tensor Option 2A crown volume, clumping = 0.93 / 1.05).
constexpr std::array<GlbRow, 4> k_glb_table = {{
    {"oak",   42, 6948788, 0x9ae4a06e0a34349full},
    {"maple", 42, 7829632, 0x0534dfbfe21f4fa2ull},
    {"birch", 42, 6921260, 0x3ce6827204302903ull},
    {"pine",  42, 5800888, 0x98fee4c14488cfdfull},
}};

uint64_t glb_fnv(const std::vector<uint8_t>& b) {
    return treegen::fnv1a64(std::string_view(reinterpret_cast<const char*>(b.data()), b.size()));
}

}  // namespace

TEST_CASE("[treegen_generate_tree] generate_tree reproduces the tool's GLB bytes",
          "[treegen][treegen_generate_tree]") {
    namespace tsp = rynx::test_support;

    // Degenerate guard: one row per registry species at seed 42, and a row for
    // every tracked tree scenario (the table cannot silently cover less).
    std::set<std::string> species_at_42;
    for (const auto& r : k_glb_table) if (r.seed == 42) species_at_42.insert(r.species);
    for (const auto& name : treegen::get_species_names()) {
        INFO("registry species " << name);
        CHECK(species_at_42.count(name) == 1);
    }
    int tree_scenarios = 0;
    for (const auto& e : std::filesystem::directory_iterator(tsp::treegen_root() / "scenarios")) {
        if (e.path().extension() != ".json") continue;
        if (treegen::load_scenario(e.path().string()).kind != "tree") continue;
        ++tree_scenarios;
        const std::string stem = e.path().stem().string();  // c3_<species>
        INFO("tracked scenario " << stem);
        REQUIRE(stem.rfind("c3_", 0) == 0);
        CHECK(species_at_42.count(stem.substr(3)) == 1);
    }
    CHECK(tree_scenarios == static_cast<int>(species_at_42.size()));
    REQUIRE(species_at_42.size() == 4u);

    for (const auto& r : k_glb_table) {
        const treegen::GeneratedTree& g = tsp::generated_tree(r.species, r.seed);
        INFO(r.species << " seed " << r.seed << ": " << g.glb.size() << " bytes, fnv 0x" << std::hex << glb_fnv(g.glb));
        CHECK(g.glb.size() == r.bytes);
        CHECK(glb_fnv(g.glb) == r.fnv1a64);
    }
}

// --species <name> resolves to c3_<name>.json: the registry path and the
// scenario path are one generation, not two rows that could drift.
TEST_CASE("[treegen_generate_tree] species registry path equals the scenario path",
          "[treegen][treegen_generate_tree]") {
    namespace tsp = rynx::test_support;
    for (const auto& name : treegen::get_species_names()) {
        const std::string via_registry = treegen::get_scenario_path(tsp::treegen_root(), name);
        INFO("species " << name);
        REQUIRE_FALSE(via_registry.empty());
        CHECK(std::filesystem::equivalent(via_registry, tsp::treegen_file("scenarios/c3_" + name + ".json")));
    }
}
