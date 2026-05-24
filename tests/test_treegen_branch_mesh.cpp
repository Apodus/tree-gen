// [treegen_branch_mesh_p1] — pins C4 P1 bark mesh extrusion invariants. Same
// link rationale as test_treegen_skeleton: tool sources (branch_mesh.cpp +
// space_colonization.cpp + …) link into TestTech via rynx_tests.sharpmake.cs
// so this test pins the exact code shipped in rynx-treegen.exe.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../bark_uv.hpp"
#include "../branch_mesh.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace ts = treegen;

TEST_CASE("[treegen_branch_mesh_p1] oak skeleton produces valid bark mesh",
          "[treegen][treegen_branch_mesh_p1]") {
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    REQUIRE(s.kind == "tree");

    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
    REQUIRE(skel.nodes.size() > 100u);

    ts::BarkMeshOptions opts;
    opts.seam_offset_rad = 0.0f;
    ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);

    // Layout invariants.
    REQUIRE(bark.mesh.positions.size() % 3 == 0);
    REQUIRE(bark.indices_u32.size()    % 3 == 0);
    REQUIRE(bark.mesh.positions.size() / 3 == bark.per_vertex_node_index.size());
    REQUIRE(bark.mesh.normals.size()       == bark.mesh.positions.size());
    REQUIRE(bark.mesh.uvs.size() * 3       == bark.mesh.positions.size() * 2);

    // Non-empty.
    REQUIRE(bark.mesh.positions.size() > 0u);
    REQUIRE(bark.indices_u32.size()    > 0u);

    // Index range valid.
    const uint32_t vcount = static_cast<uint32_t>(bark.mesh.positions.size() / 3);
    for (uint32_t idx : bark.indices_u32) {
        REQUIRE(idx < vcount);
    }

    // No NaN / Inf positions, normals, or UVs.
    for (float v : bark.mesh.positions) REQUIRE(std::isfinite(v));
    for (float v : bark.mesh.normals)   REQUIRE(std::isfinite(v));
    for (float v : bark.mesh.uvs)       REQUIRE(std::isfinite(v));

    // Every per-vertex node index references a real skeleton node.
    const int N = static_cast<int>(skel.nodes.size());
    for (int n : bark.per_vertex_node_index) {
        REQUIRE(n >= 0);
        REQUIRE(n < N);
    }

    // Normals approximately unit (radial direction is unit by construction).
    for (size_t i = 0; i < bark.mesh.normals.size(); i += 3) {
        const float nx = bark.mesh.normals[i + 0];
        const float ny = bark.mesh.normals[i + 1];
        const float nz = bark.mesh.normals[i + 2];
        const float L  = std::sqrt(nx * nx + ny * ny + nz * nz);
        REQUIRE(std::abs(L - 1.0f) < 1e-4f);
    }

    // Determinism: same inputs → byte-identical position + index + UV vectors.
    ts::BarkMeshOutput bark2 = ts::build_bark_mesh(skel, opts);
    REQUIRE(bark.mesh.positions       == bark2.mesh.positions);
    REQUIRE(bark.mesh.normals         == bark2.mesh.normals);
    REQUIRE(bark.mesh.uvs             == bark2.mesh.uvs);
    REQUIRE(bark.indices_u32          == bark2.indices_u32);
    REQUIRE(bark.per_vertex_node_index == bark2.per_vertex_node_index);
}

TEST_CASE("[treegen_branch_mesh_p1] seam-column UV duplicate continuity",
          "[treegen][treegen_branch_mesh_p1]") {
    // Direct ring_uv check: i=0 and i=N share angular position; U should
    // differ by exactly 1.0 (continuous wrap, no [0,1] modulo).
    constexpr int N = 12;
    auto [u0, v0] = ts::bark_uv::ring_uv(0, N, 0.5f, 0.25f);
    auto [uN, vN] = ts::bark_uv::ring_uv(N, N, 0.5f, 0.25f);
    REQUIRE(std::abs(uN - u0 - 1.0f) < 1e-6f);
    REQUIRE(v0 == 0.5f);
    REQUIRE(vN == 0.5f);

    // Halfway around the ring → U = 0.5 + seam_offset/(2π).
    auto [u_half, _v] = ts::bark_uv::ring_uv(N / 2, N, 0.0f, 0.0f);
    (void)_v;
    REQUIRE(std::abs(u_half - 0.5f) < 1e-6f);
}

TEST_CASE("[treegen_branch_mesh_p1] every triangle has three distinct vertices",
          "[treegen][treegen_branch_mesh_p1]") {
    // Degenerate-triangle gate. A zero-area triangle indicates an emit_strip
    // bug or a degenerate ring (radius=0 + zero-len segment that slipped past
    // the 1e-5f guard).
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    ts::Scenario s = ts::load_scenario(fixture);
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, 0ull ^ s.scenario_fnv);

    ts::BarkMeshOptions opts;
    ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);

    REQUIRE(bark.indices_u32.size() % 3 == 0);
    for (size_t t = 0; t < bark.indices_u32.size(); t += 3) {
        const uint32_t a = bark.indices_u32[t + 0];
        const uint32_t b = bark.indices_u32[t + 1];
        const uint32_t c = bark.indices_u32[t + 2];
        REQUIRE(a != b);
        REQUIRE(b != c);
        REQUIRE(a != c);
    }
}
