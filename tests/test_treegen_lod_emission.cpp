// [treegen_lod_emission] — pins C4 P4. Per-LOD face budget allocator + 4-LOD
// multi-mesh GLB emission (3 bark LODs + 1 billboard stub). Round-trip via
// gltf_view confirms the `_RYNX_LOD` extension survives write/read and the
// per-LOD wind weights are present.
//
// Link rationale: face_budget.cpp + lod_emitter.cpp linked into TestTech via
// rynx_tests.sharpmake.cs so this test pins the exact code shipped in
// rynx-treegen.exe.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../external/gltf_view.hpp"

#include "../branch_mesh.hpp"
#include "../face_budget.hpp"
#include "../glb_writer.hpp"
#include "../lod_emitter.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"

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
    char buf[64];
    std::snprintf(buf, sizeof(buf), "treegen_lod_%d_%s.glb", std::rand(), suffix);
    return (fs::temp_directory_path() / buf).string();
}

// Pack the 4 LODs into a multi-mesh GLB on disk; returns the path. Mirrors
// the wiring in tools/rynx-treegen/main.cpp.
std::string write_lod_glb(const std::vector<ts::LodOutput>& lods, const char* suffix) {
    std::vector<std::vector<ts::PrimitiveData>> per_lod_prims(lods.size());
    std::vector<ts::MeshData>                    meshes;
    meshes.reserve(lods.size());

    for (size_t li = 0; li < lods.size(); ++li) {
        const auto& lod = lods[li];
        ts::PrimitiveData prim{};
        prim.positions   = std::span<const float>(lod.mesh.positions.data(), lod.mesh.positions.size());
        prim.normals     = std::span<const float>(lod.mesh.normals.data(),   lod.mesh.normals.size());
        prim.uvs         = std::span<const float>(lod.mesh.uvs.data(),       lod.mesh.uvs.size());
        prim.indices_u32 = std::span<const uint32_t>(lod.indices_u32.data(), lod.indices_u32.size());
        prim.wind_weights_packed = std::span<const uint8_t>(
            lod.wind_weights_packed.data(), lod.wind_weights_packed.size());
        per_lod_prims[li].push_back(prim);

        ts::MeshData md{};
        md.primitives          = std::span<const ts::PrimitiveData>(
            per_lod_prims[li].data(), per_lod_prims[li].size());
        md.lod_index           = lod.lod_index;
        md.lod_max_distance_m  = lod.lod_max_distance_m;
        meshes.push_back(md);
    }

    const std::string path = tmp_path(suffix);
    std::string err;
    REQUIRE(ts::write_glb_multi_mesh(
        std::span<const ts::MeshData>(meshes.data(), meshes.size()), path, &err));
    REQUIRE(std::filesystem::exists(path));
    return path;
}

} // anonymous namespace

TEST_CASE("[treegen_lod_emission] oak emits 3 LODs in one multi-mesh GLB",
          "[treegen][treegen_lod_emission]") {
    namespace tsp = rynx::test_support;
    namespace fs  = std::filesystem;

    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    REQUIRE(s.kind == "tree");

    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
    REQUIRE(skel.nodes.size() > 50u);

    ts::BarkMeshOptions base_opts;
    base_opts.tree_height_m = s.tree.height_m;
    base_opts.seam_offset_rad = 0.0f;

    ts::LodBudget budget;  // P5 defaults: L0=12000, L1=4000, L2=1500, L3=4

    auto lods = ts::emit_all_lods(skel, base_opts, budget, s.tree.height_m, seed_effective);
    REQUIRE(lods.size() == 3u);  // C10: L3 billboard stub suppressed

    // lod_index values are 0..2 and the distance thresholds match the C4 P4
    // pin (L0=25, L1=60, L2=120).
    REQUIRE(lods[0].lod_index == 0);
    REQUIRE(lods[1].lod_index == 1);
    REQUIRE(lods[2].lod_index == 2);
    REQUIRE(lods[0].lod_max_distance_m == Approx(25.0f));
    REQUIRE(lods[1].lod_max_distance_m == Approx(60.0f));
    REQUIRE(lods[2].lod_max_distance_m == Approx(120.0f));

    // Per-LOD face count within budget.
    const size_t l0_tris = lods[0].indices_u32.size() / 3;
    const size_t l1_tris = lods[1].indices_u32.size() / 3;
    const size_t l2_tris = lods[2].indices_u32.size() / 3;

    INFO("L0=" << l0_tris << " L1=" << l1_tris << " L2=" << l2_tris);

    REQUIRE(l0_tris > 0u);
    REQUIRE(l1_tris > 0u);
    REQUIRE(l2_tris > 0u);
    REQUIRE(int(l0_tris) <= budget.l0_tris);
    REQUIRE(int(l1_tris) <= budget.l1_tris);
    REQUIRE(int(l2_tris) <= budget.l2_tris);

    // Detail descends monotonically L0 -> L2.
    REQUIRE(l0_tris >= l1_tris);
    REQUIRE(l1_tris >= l2_tris);

    // Wind weights present for all bark LODs (4 bytes/vert).
    for (int i = 0; i < 3; ++i) {
        const size_t vcount = lods[i].mesh.positions.size() / 3;
        REQUIRE(vcount > 0u);
        REQUIRE(lods[i].wind_weights_packed.size() == 4u * vcount);
    }

    // ---- GLB round-trip ----
    const std::string path = write_lod_glb(lods, "oak");

    std::vector<char> bytes = slurp_bytes(path);
    REQUIRE_FALSE(bytes.empty());

    std::span<const char> byte_span(bytes.data(), bytes.size());
    REQUIRE(rynx::graphics::gltf_view::count_meshes(byte_span) == 3);

    // For each LOD, extract its primitives + verify lod_index + tri count.
    rynx::graphics::gltf_view::glb_view glb =
        rynx::graphics::gltf_view::parse_glb_header(byte_span);
    REQUIRE(glb.valid);
    rynx::graphics::gltf_view::jparser parser(glb.json_begin, glb.json_len);
    rynx::graphics::gltf_view::jval root = parser.parse_value();
    auto* meshes_node = root.find("meshes");
    REQUIRE(meshes_node != nullptr);
    REQUIRE(meshes_node->as_arr().size() == 3u);

    for (int mi = 0; mi < 3; ++mi) {
        int   read_idx = -999;
        float read_max = -1.0f;
        REQUIRE(rynx::graphics::gltf_view::read_rynx_lod_extension(
            meshes_node->as_arr()[mi], read_idx, read_max));
        REQUIRE(read_idx == mi);
        REQUIRE(read_max == Approx(lods[mi].lod_max_distance_m));

        std::vector<rynx::graphics::gltf_view::cpu_mesh> prims;
        std::string err;
        REQUIRE(rynx::graphics::gltf_view::extract_all_primitives(
            byte_span, prims, &err, mi));
        REQUIRE(prims.size() == 1u);
        REQUIRE(prims[0].positions.size() == lods[mi].mesh.positions.size());
        REQUIRE(prims[0].indices.size()   == lods[mi].indices_u32.size());
        // Wind weights round-trip byte-equal.
        REQUIRE(prims[0].wind_weights_packed == lods[mi].wind_weights_packed);
    }

    // Determinism: rebuild from the same skeleton + opts → byte-identical GLB.
    auto lods2 = ts::emit_all_lods(skel, base_opts, budget, s.tree.height_m, seed_effective);
    const std::string path2 = write_lod_glb(lods2, "oak2");
    std::vector<char> bytes2 = slurp_bytes(path2);
    REQUIRE(bytes == bytes2);

    std::error_code ec;
    fs::remove(path, ec);
    fs::remove(path2, ec);
}

TEST_CASE("[treegen_lod_emission] face_budget allocator stays within budget on oak",
          "[treegen][treegen_lod_emission]") {
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    ts::Scenario s = ts::load_scenario(fixture);
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);

    // Aggressive budget — much lower than L2's 1500 — should force the
    // P5 cull-by-length overflow rule to kick in. The allocator must return a
    // valid PerOrderRadial + cull mask that (after rebuild) produces a mesh
    // under the budget.
    ts::PerOrderRadial target;  // 12/8/6/4
    std::vector<uint8_t> culled_mask;
    ts::PerOrderRadial actual = ts::allocate_radial_for_lod(skel, 200, target, &culled_mask);
    REQUIRE(actual.trunk >= 3);  // trunk never culled
    REQUIRE(culled_mask.size() == skel.nodes.size());  // mask populated

    // Root node is never culled. C11: depth-0 leader nodes above crown base
    // are now cullable (depth-inflation fix extended depth-0 into the crown).
    REQUIRE(culled_mask[0] == 0u);

    ts::BarkMeshOptions opts;
    opts.tree_height_m          = s.tree.height_m;
    opts.radial_seg_trunk       = actual.trunk;
    opts.radial_seg_order1      = actual.order1;
    opts.radial_seg_order2      = actual.order2;
    opts.radial_seg_order3_plus = actual.order3plus;
    opts.culled_node_mask       = culled_mask;

    ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);
    const size_t tris = bark.indices_u32.size() / 3;
    INFO("aggressive budget=200, actual tris=" << tris
        << " (trunk=" << actual.trunk << " o1=" << actual.order1
        << " o2=" << actual.order2 << " o3+=" << actual.order3plus << ")");
    // Allocator may slightly overshoot at the boundary (N>=3 floor, fork-blend
    // extra tris, K=32 cull recheck granularity) — allow a small slack.
    REQUIRE(int(tris) <= 300);
}
