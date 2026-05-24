// [treegen_wind_weights] — pins C4 P3 wind-weight bake. Asserts the formula's
// load-bearing invariants (rooted_w == 0 at z=0, byte-equal determinism) and a
// full GLB round-trip via gltf_view to prove the `_RYNX_WIND` payload
// survives write/read.
//
// Same link rationale as test_treegen_branch_mesh: tool-side sources
// (wind_weights.cpp / branch_mesh.cpp / fork_blend.cpp / …) link into TestTech
// via rynx_tests.sharpmake.cs so this test pins exactly the code shipped in
// rynx-treegen.exe.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../external/gltf_view.hpp"

#include "../branch_mesh.hpp"
#include "../glb_writer.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"
#include "../wind_weights.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace ts = treegen;

namespace {

std::vector<char> slurp_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    f.seekg(0, std::ios::end);
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(n));
    if (n > 0) f.read(buf.data(), n);
    return buf;
}

std::string tmp_path(const char* suffix) {
    namespace fs = std::filesystem;
    char tmp[64];
    std::snprintf(tmp, sizeof(tmp), "treegen_wind_%d_%s.glb", std::rand(), suffix);
    return (fs::temp_directory_path() / tmp).string();
}

} // anonymous namespace

TEST_CASE("[treegen_wind_weights] formula assertions on oak fixture",
          "[treegen][treegen_wind_weights]") {
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    REQUIRE(s.kind == "tree");

    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

    ts::BarkMeshOptions opts;
    opts.tree_height_m = s.tree.height_m;
    ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);

    const size_t vcount = bark.mesh.positions.size() / 3;
    REQUIRE(vcount > 0u);
    REQUIRE(bark.wind_weights_packed.size() == 4u * vcount);

    // Rooted invariant: any vertex with world_z < 0.001 * tree_height has
    // smoothstep(0, 0.1, z/h) == 0 → trunk_byte == 0.
    const float root_z_threshold = 0.0001f * s.tree.height_m; // well below 0.1*h
    int rooted_count = 0;
    for (size_t i = 0; i < vcount; ++i) {
        const float z = bark.mesh.positions[3 * i + 2];
        if (z < root_z_threshold) {
            const uint8_t trunk_byte = bark.wind_weights_packed[4 * i + 0];
            REQUIRE(trunk_byte == 0u);
            ++rooted_count;
        }
    }
    REQUIRE(rooted_count > 0); // sanity: some verts ARE at the root

    // Per-vertex bytes finite (all uint8 obviously, but the count loop above
    // shouldn't have missed any vertex).
    REQUIRE(bark.wind_weights_packed.size() % 4u == 0u);

    // Determinism: bake the same skeleton twice → byte-equal output.
    ts::BarkMeshOutput bark2 = ts::build_bark_mesh(skel, opts);
    REQUIRE(bark.wind_weights_packed == bark2.wind_weights_packed);
}

TEST_CASE("[treegen_wind_weights] direct bake_wind_weights formula edges",
          "[treegen][treegen_wind_weights]") {
    // Synthetic skeleton: root (depth=0) at z=0 + tip (depth=4) at z=10.
    ts::TreeSkeleton skel;
    {
        ts::BranchNode root;
        root.parent_index = -1; root.position = {0, 0, 0};
        root.radius = 0.3f; root.depth = 0;
        skel.nodes.push_back(root);
    }
    for (int d = 1; d <= 4; ++d) {
        ts::BranchNode n;
        n.parent_index = d - 1;
        n.position = {0, 0, static_cast<float>(d) * 2.5f};
        n.radius = 0.1f; n.depth = d;
        skel.nodes.push_back(n);
    }
    const float tree_h = 10.0f;

    // Verts mapped 1:1 to nodes.
    std::vector<int>   per_vertex_node_index = {0, 1, 2, 3, 4};
    std::vector<float> per_vertex_world_z    = {0.0f, 2.5f, 5.0f, 7.5f, 10.0f};

    auto bytes = ts::bake_wind_weights(skel, per_vertex_node_index, per_vertex_world_z, tree_h);
    REQUIRE(bytes.size() == 20u); // 5 verts * 4

    // Root (z=0, depth=0): trunk_w = 1 * smoothstep(0, 0.1, 0) = 0.
    REQUIRE(bytes[0 * 4 + 0] == 0u);   // trunk
    REQUIRE(bytes[0 * 4 + 1] == 0u);   // branch (4*0*1=0)
    REQUIRE(bytes[0 * 4 + 2] == 0u);   // twig
    REQUIRE(bytes[0 * 4 + 3] == 0u);   // leaf

    // Tip (depth=4, max_depth=4 clamp → depth_norm=1.0): leaf_w == 1.0 →
    // 255. trunk = 0 (one_minus_d=0). branch = 4*1*0 = 0. twig = 4*1*0 = 0.
    REQUIRE(bytes[4 * 4 + 0] == 0u);    // trunk
    REQUIRE(bytes[4 * 4 + 1] == 0u);    // branch
    REQUIRE(bytes[4 * 4 + 2] == 0u);    // twig
    REQUIRE(bytes[4 * 4 + 3] == 255u);  // leaf

    // Middle (depth=2, d=0.5): branch_w = 4*0.5*0.5 = 1.0 → 255. twig = 4*0.25*0.5 = 0.5
    // → 128. trunk = 0.25 * smoothstep(0, 0.1, 5.0/10.0=0.5)=0.25*1.0=0.25 → 64.
    // leaf = 0.125 → 32.
    REQUIRE(bytes[2 * 4 + 0] == 64u);
    REQUIRE(bytes[2 * 4 + 1] == 255u);
    REQUIRE(bytes[2 * 4 + 2] == 128u);
    REQUIRE(bytes[2 * 4 + 3] == 32u);
}

TEST_CASE("[treegen_wind_weights] max_depth clamp handles shallow skeleton",
          "[treegen][treegen_wind_weights]") {
    // Degenerate single-depth tree: root + 1 tip at depth=1.
    // Without the max_depth >= 4 clamp depth_norm would saturate to 1.0
    // and trunk_w would collapse.
    ts::TreeSkeleton skel;
    {
        ts::BranchNode root;
        root.parent_index = -1; root.position = {0, 0, 0};
        root.radius = 0.3f; root.depth = 0;
        skel.nodes.push_back(root);
    }
    {
        ts::BranchNode tip;
        tip.parent_index = 0; tip.position = {0, 0, 5.0f};
        tip.radius = 0.1f; tip.depth = 1;
        skel.nodes.push_back(tip);
    }
    std::vector<int>   pvni = {1};
    std::vector<float> pvz  = {5.0f};
    auto bytes = ts::bake_wind_weights(skel, pvni, pvz, 10.0f);

    // depth_norm = 1/4 = 0.25 (NOT 1.0). trunk = 0.75^2 * 1.0 = 0.5625 → 143.
    REQUIRE(bytes.size() == 4u);
    REQUIRE(bytes[0] == 143u);
}

TEST_CASE("[treegen_wind_weights] GLB round-trip preserves _RYNX_WIND bytes",
          "[treegen][treegen_wind_weights]") {
    namespace fs = std::filesystem;
    namespace tsp = rynx::test_support;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

    ts::BarkMeshOptions opts;
    opts.tree_height_m = s.tree.height_m;
    ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);

    REQUIRE(bark.wind_weights_packed.size() == 4u * (bark.mesh.positions.size() / 3));

    ts::PrimitiveData prim{};
    prim.positions   = std::span<const float>(bark.mesh.positions.data(),    bark.mesh.positions.size());
    prim.normals     = std::span<const float>(bark.mesh.normals.data(),      bark.mesh.normals.size());
    prim.uvs         = std::span<const float>(bark.mesh.uvs.data(),          bark.mesh.uvs.size());
    prim.indices_u32 = std::span<const uint32_t>(bark.indices_u32.data(),    bark.indices_u32.size());
    prim.wind_weights_packed = std::span<const uint8_t>(
        bark.wind_weights_packed.data(), bark.wind_weights_packed.size());

    const std::string path = tmp_path("oak_wind_roundtrip");
    std::string err;
    REQUIRE(ts::write_glb(std::span<const ts::PrimitiveData>(&prim, 1), path, &err));
    REQUIRE(fs::exists(path));

    std::vector<char> bytes = slurp_bytes(path);
    REQUIRE_FALSE(bytes.empty());

    std::vector<rynx::graphics::gltf_view::cpu_mesh> out;
    std::string parse_err;
    REQUIRE(rynx::graphics::gltf_view::extract_all_primitives(
        std::span<const char>(bytes.data(), bytes.size()), out, &parse_err));

    REQUIRE(out.size() == 1u);
    REQUIRE(out[0].positions.size() == bark.mesh.positions.size());
    REQUIRE(out[0].wind_weights_packed == bark.wind_weights_packed);

    std::error_code ec;
    fs::remove(path, ec);
}
