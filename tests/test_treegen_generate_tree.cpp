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
#include <cstring>
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

// Crown Lighting L-P2 + L-P3 regenerated table (J1 march_lowpass_0_6m crown-normal
// blend: 0.70 pine, 0.45 broadleaf, 0.35 shrubs + J7 1.42x CLM5 needle translucency
// in the shared 4x4 atlas, +1,192 B uniform PNG delta across all 6 species):
constexpr std::array<GlbRow, 6> k_glb_table = {{
    {"oak",           42, 6220032, 0x4cc7a8b6892d39baull},
    {"maple",         42, 7167884, 0x4a750f6ed45a7dfdull},
    {"birch",         42, 5427408, 0x198050d5730919b6ull},
    {"pine",          42, 5491952, 0x6e857e82e88234d2ull},
    {"juniper_shrub", 42, 4809556, 0xeb90eb0017af7edfull},
    {"alder_shrub",   42, 4542352, 0x1c169dd65c5a5e1full},
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
    REQUIRE(species_at_42.size() == 6u);

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

TEST_CASE("[treegen_generate_tree] cross-seed bark/leaf image dedup and bark sampler REPEAT U+V",
          "[treegen][treegen_generate_tree][treegen_bark_b1]") {
    namespace tsp = rynx::test_support;
    const treegen::GeneratedTree& g42  = tsp::generated_tree("oak", 42);
    const treegen::GeneratedTree& g123 = tsp::generated_tree("oak", 123);

    // Same species across seeds produces byte-identical bark (0..3) and leaf (4..7) PNGs.
    REQUIRE(g42.images.size()  == 8u);
    REQUIRE(g123.images.size() == 8u);
    for (size_t i = 0; i < 8u; ++i) {
        INFO("image index " << i);
        REQUIRE(g42.images[i].png_bytes == g123.images[i].png_bytes);
    }

    // Per-tree geometry & seam_offset_rad UVs still differ across seeds.
    REQUIRE(!g42.lods.empty());
    REQUIRE(!g123.lods.empty());
    REQUIRE(g42.lods[0].mesh.uvs != g123.lods[0].mesh.uvs);

    // GLB JSON chunk pins Bark Sampler 0 to REPEAT U+V (wrapS=10497, wrapT=10497)
    // for the periodic 2D torus bark maps + cumulative root-to-tip v_axial.
    REQUIRE(g42.glb.size() > 20u);
    uint32_t json_len = 0;
    std::memcpy(&json_len, g42.glb.data() + 12, 4);
    REQUIRE(20u + json_len <= g42.glb.size());
    const std::string_view json_sv(reinterpret_cast<const char*>(g42.glb.data() + 20), json_len);
    REQUIRE(json_sv.find(R"("samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497})")
            != std::string_view::npos);
}
