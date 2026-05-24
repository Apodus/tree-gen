// [treegen] — pins the tool-side primitive-build + GLB-writer chain.
//
// All three tests run headless (no GPU) by going through the new
// gltf_view::extract_first_primitive (header-only) for the round-trip.
// The trivial_cylinder + glb_writer tool sources are linked directly into
// TestTech via rynx_tests.sharpmake.cs — they are tool-self-contained and
// drag no engine state in.
//
// [treegen_cylinder]    — pinned vert + tri counts (first-write calibration).
// [treegen_glb_roundtrip] — write_glb then parse back via gltf_view; counts match.
// [treegen_determinism] — two writes byte-identical (gate for golden harness).

#include "../external/catch2/catch.hpp"

#include "../external/gltf_view.hpp"

#include "../glb_writer.hpp"
#include "../trivial_cylinder.hpp"

#include <cstdint>
#include <cstdio>
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
    // %p in MSVC isn't always unique enough; mix in random + pid for safety.
    char tmp[64];
    std::snprintf(tmp, sizeof(tmp), "treegen_test_%d_%s.glb",
                  std::rand(), suffix);
    return (fs::temp_directory_path() / tmp).string();
}

treegen::PrimitiveData make_default_prim(const treegen::cpu_mesh_out& cm) {
    treegen::PrimitiveData p{};
    p.positions = std::span<const float>(cm.positions.data(), cm.positions.size());
    p.normals   = std::span<const float>(cm.normals.data(), cm.normals.size());
    p.uvs       = std::span<const float>(cm.uvs.data(), cm.uvs.size());
    p.indices   = std::span<const uint16_t>(cm.indices.data(), cm.indices.size());
    return p;
}

} // anonymous namespace

TEST_CASE("treegen: build_trivial_cylinder produces pinned vert + tri counts",
          "[treegen][treegen_cylinder]") {
    treegen::cpu_mesh_out cm = treegen::build_trivial_cylinder(0.5f, 2.0f, 12, 4);
    REQUIRE(cm.positions.size() % 3 == 0);
    REQUIRE(cm.indices.size()   % 3 == 0);
    REQUIRE(cm.positions.size() / 3 == treegen::K_TRIVIAL_VERT_COUNT);
    REQUIRE(cm.indices.size()   / 3 == treegen::K_TRIVIAL_TRI_COUNT);
    REQUIRE(treegen::K_TRIVIAL_TRI_COUNT == 120);
    // Normals and UVs match positions count.
    REQUIRE(cm.normals.size() == cm.positions.size());
    REQUIRE(cm.uvs.size()     == (cm.positions.size() / 3) * 2);
}

TEST_CASE("treegen: write_glb + roundtrip via gltf_view",
          "[treegen][treegen_glb_roundtrip]") {
    namespace fs = std::filesystem;

    treegen::cpu_mesh_out cm = treegen::build_trivial_cylinder(0.5f, 2.0f, 12, 4);
    treegen::PrimitiveData prim = make_default_prim(cm);

    std::string path = tmp_path("roundtrip");
    std::string err;
    treegen::PrimitiveData prims[1] = { prim };
    REQUIRE(treegen::write_glb(std::span<const treegen::PrimitiveData>(prims, 1), path, &err));
    REQUIRE(fs::exists(path));

    std::vector<char> bytes = slurp_bytes(path);
    REQUIRE_FALSE(bytes.empty());

    rynx::graphics::gltf_view::cpu_mesh out;
    std::string parse_err;
    REQUIRE(rynx::graphics::gltf_view::extract_first_primitive(
        std::span<const char>(bytes.data(), bytes.size()), out, &parse_err));

    REQUIRE(out.positions.size() / 3 == treegen::K_TRIVIAL_VERT_COUNT);
    REQUIRE(out.indices.size()   / 3 == treegen::K_TRIVIAL_TRI_COUNT);
    REQUIRE(out.normals.size()   == out.positions.size());
    // UVs are emitted by the writer; the round-trip should see them.
    REQUIRE(out.texcoords.size() == (out.positions.size() / 3) * 2);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("treegen: write_glb determinism byte-identical",
          "[treegen][treegen_determinism]") {
    namespace fs = std::filesystem;

    treegen::cpu_mesh_out cm = treegen::build_trivial_cylinder(0.5f, 2.0f, 12, 4);
    treegen::PrimitiveData prim = make_default_prim(cm);
    treegen::PrimitiveData prims[1] = { prim };

    std::string p1 = tmp_path("det_a");
    std::string p2 = tmp_path("det_b");
    std::string err;
    REQUIRE(treegen::write_glb(std::span<const treegen::PrimitiveData>(prims, 1), p1, &err));
    REQUIRE(treegen::write_glb(std::span<const treegen::PrimitiveData>(prims, 1), p2, &err));

    std::vector<char> a = slurp_bytes(p1);
    std::vector<char> b = slurp_bytes(p2);
    REQUIRE(a.size() == b.size());
    REQUIRE_FALSE(a.empty());
    REQUIRE(std::memcmp(a.data(), b.data(), a.size()) == 0);

    std::error_code ec;
    fs::remove(p1, ec);
    fs::remove(p2, ec);
}
