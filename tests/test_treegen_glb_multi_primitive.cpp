// [treegen_glb_multi_primitive] — C2 P2 round-trip pin.
//
// Builds a 2-primitive cylinder (trunk + leaves disk), writes a GLB via the
// (now multi-primitive) treegen::write_glb, parses it back through
// gltf_view::extract_all_primitives, and asserts:
//   - 2 cpu_mesh instances returned.
//   - Per-primitive vert + tri counts match the source.
//   - material_index 0 / 1 land on the right primitive.

#include "../external/catch2/catch.hpp"

#include "../external/gltf_view.hpp"

#include "../glb_writer.hpp"
#include "../trivial_cylinder.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

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
    std::snprintf(tmp, sizeof(tmp), "treegen_multi_%d_%s.glb", std::rand(), suffix);
    return (fs::temp_directory_path() / tmp).string();
}

treegen::PrimitiveData make_prim(const treegen::cpu_mesh_out& cm,
                                 float r, float g, float b) {
    treegen::PrimitiveData p{};
    p.positions = std::span<const float>(cm.positions.data(), cm.positions.size());
    p.normals   = std::span<const float>(cm.normals.data(), cm.normals.size());
    p.uvs       = std::span<const float>(cm.uvs.data(), cm.uvs.size());
    p.indices   = std::span<const uint16_t>(cm.indices.data(), cm.indices.size());
    p.material.base_color[0] = r;
    p.material.base_color[1] = g;
    p.material.base_color[2] = b;
    p.material.base_color[3] = 1.0f;
    return p;
}

} // anonymous namespace

TEST_CASE("treegen: build_trivial_two_primitive_cylinder + write_glb + roundtrip",
          "[treegen][gltf][treegen_glb_multi_primitive]") {
    namespace fs = std::filesystem;

    auto prims = treegen::build_trivial_two_primitive_cylinder(0.5f, 2.0f);
    REQUIRE(prims.size() == 2);

    const auto& trunk = prims[0];
    const auto& leaves = prims[1];

    // Trunk matches the pinned single-prim counts.
    REQUIRE(trunk.positions.size() / 3 == treegen::K_TRIVIAL_VERT_COUNT);
    REQUIRE(trunk.indices.size()   / 3 == treegen::K_TRIVIAL_TRI_COUNT);
    // Leaves: 1 center + 12 rim = 13 verts; 12 fan tris.
    REQUIRE(leaves.positions.size() / 3 == 13);
    REQUIRE(leaves.indices.size()   / 3 == 12);

    treegen::PrimitiveData pdata[2] = {
        make_prim(trunk,  0.45f, 0.30f, 0.20f), // bark
        make_prim(leaves, 0.30f, 0.65f, 0.20f), // foliage
    };

    std::string path = tmp_path("two_prim");
    std::string err;
    REQUIRE(treegen::write_glb(std::span<const treegen::PrimitiveData>(pdata, 2), path, &err));
    REQUIRE(fs::exists(path));

    std::vector<char> bytes = slurp_bytes(path);
    REQUIRE_FALSE(bytes.empty());

    std::vector<rynx::graphics::gltf_view::cpu_mesh> out;
    std::string parse_err;
    REQUIRE(rynx::graphics::gltf_view::extract_all_primitives(
        std::span<const char>(bytes.data(), bytes.size()), out, &parse_err));

    REQUIRE(out.size() == 2);

    // Counts match per-primitive.
    REQUIRE(out[0].positions.size() / 3 == treegen::K_TRIVIAL_VERT_COUNT);
    REQUIRE(out[0].indices.size()   / 3 == treegen::K_TRIVIAL_TRI_COUNT);
    REQUIRE(out[1].positions.size() / 3 == 13);
    REQUIRE(out[1].indices.size()   / 3 == 12);

    // Distinct per-primitive material indices.
    REQUIRE(out[0].material_index == 0);
    REQUIRE(out[1].material_index == 1);

    // UVs preserved.
    REQUIRE(out[0].texcoords.size() == (out[0].positions.size() / 3) * 2);
    REQUIRE(out[1].texcoords.size() == (out[1].positions.size() / 3) * 2);

    // No wind extension in this fixture.
    REQUIRE(out[0].wind_weights_packed.empty());
    REQUIRE(out[1].wind_weights_packed.empty());

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("treegen: extract_first_primitive still returns primitive 0 on multi-prim GLB",
          "[treegen][gltf][treegen_glb_multi_primitive]") {
    namespace fs = std::filesystem;

    auto prims = treegen::build_trivial_two_primitive_cylinder(0.5f, 2.0f);
    treegen::PrimitiveData pdata[2] = {
        make_prim(prims[0], 0.45f, 0.30f, 0.20f),
        make_prim(prims[1], 0.30f, 0.65f, 0.20f),
    };

    std::string path = tmp_path("first_of_multi");
    std::string err;
    REQUIRE(treegen::write_glb(std::span<const treegen::PrimitiveData>(pdata, 2), path, &err));

    std::vector<char> bytes = slurp_bytes(path);
    rynx::graphics::gltf_view::cpu_mesh first;
    std::string parse_err;
    REQUIRE(rynx::graphics::gltf_view::extract_first_primitive(
        std::span<const char>(bytes.data(), bytes.size()), first, &parse_err));

    // First primitive is the trunk.
    REQUIRE(first.positions.size() / 3 == treegen::K_TRIVIAL_VERT_COUNT);
    REQUIRE(first.material_index == 0);

    std::error_code ec;
    fs::remove(path, ec);
}
