// [treegen_glb_textures] — C6 P4 pins for embedded PNG textures in GLB.
//
// 5 cases:
//   1. bark-only GLB embeds 4 images (sampler=0, no translucency extension)
//   2. leaf+bark GLB embeds 7 images (bark sampler=0, leaf sampler=1,
//      _RYNX_LEAF_TRANSLUCENCY in extensionsUsed)
//   3. image PNG bytes round-trip (stbi_load_from_memory decode check)
//   4. determinism (two textured writes byte-identical)
//   5. no images = backwards compatible (byte-identical to old 3-arg overload)

#include "../external/catch2/catch.hpp"

#include "../bark_texture.hpp"
#include "../glb_writer.hpp"
#include "../leaf_atlas.hpp"
#include "../png_encoder.hpp"
#include "../tree_descriptor.hpp"
#include "../trivial_cylinder.hpp"

#include <stb_image.h>

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
    std::snprintf(tmp, sizeof(tmp), "treegen_tex_%d_%s.glb", std::rand(), suffix);
    return (fs::temp_directory_path() / tmp).string();
}

// Minimal mesh: 3 verts, 1 tri.
struct MiniMesh {
    std::vector<float>    positions = { 0,0,0, 1,0,0, 0,1,0 };
    std::vector<float>    normals   = { 0,0,1, 0,0,1, 0,0,1 };
    std::vector<float>    uvs       = { 0,0, 1,0, 0,1 };
    std::vector<uint16_t> indices   = { 0, 1, 2 };
};

treegen::PrimitiveData make_prim(const MiniMesh& m, const treegen::MaterialSpec& mat) {
    treegen::PrimitiveData p{};
    p.positions = std::span<const float>(m.positions.data(), m.positions.size());
    p.normals   = std::span<const float>(m.normals.data(), m.normals.size());
    p.uvs       = std::span<const float>(m.uvs.data(), m.uvs.size());
    p.indices   = std::span<const uint16_t>(m.indices.data(), m.indices.size());
    p.material  = mat;
    return p;
}

// Tiny 4x4 synthetic PNG for fast tests.
std::vector<uint8_t> make_tiny_png(uint8_t fill_r = 128) {
    std::vector<uint8_t> rgba(4 * 4 * 4, 0);
    for (size_t i = 0; i < 4 * 4; ++i) {
        rgba[i * 4 + 0] = fill_r;
        rgba[i * 4 + 1] = 64;
        rgba[i * 4 + 2] = 32;
        rgba[i * 4 + 3] = 255;
    }
    return treegen::encode_png_rgba8(4, 4, 4 * 4, rgba.data());
}

// Find a JSON key's value start offset. Returns string::npos if not found.
size_t find_json_key(const std::string& json, const char* key) {
    std::string needle = std::string("\"") + key + "\":";
    return json.find(needle);
}

// Extract the JSON chunk from a GLB byte vector.
std::string extract_json(const std::vector<char>& glb) {
    REQUIRE(glb.size() >= 20);
    uint32_t json_len = 0;
    std::memcpy(&json_len, glb.data() + 12, 4);
    uint32_t chunk_type = 0;
    std::memcpy(&chunk_type, glb.data() + 16, 4);
    REQUIRE(chunk_type == 0x4E4F534Au); // "JSON"
    return std::string(glb.data() + 20, json_len);
}

} // anonymous namespace


TEST_CASE("treegen: bark-only GLB embeds 4 images with sampler=0",
          "[treegen][treegen_glb_textures]") {
    namespace fs = std::filesystem;

    MiniMesh m;
    treegen::MaterialSpec mat;
    mat.base_color_tex_index         = 0;
    mat.normal_tex_index             = 1;
    mat.occlusion_tex_index          = 2;
    mat.metallic_roughness_tex_index = 3;
    mat.sampler_index                = 0;

    treegen::PrimitiveData prim = make_prim(m, mat);
    treegen::MeshData md;
    md.primitives = std::span<const treegen::PrimitiveData>(&prim, 1);

    std::vector<treegen::ImageData> images(4);
    for (int i = 0; i < 4; ++i) {
        images[i].png_bytes = make_tiny_png(static_cast<uint8_t>(100 + i * 30));
    }

    std::string path = tmp_path("bark_only");
    std::string err;
    REQUIRE(treegen::write_glb_multi_mesh(
        std::span<const treegen::MeshData>(&md, 1),
        std::span<const treegen::ImageData>(images.data(), images.size()),
        path, &err));

    auto glb = slurp_bytes(path);
    std::string json = extract_json(glb);

    // 4 images.
    {
        size_t pos = find_json_key(json, "images");
        REQUIRE(pos != std::string::npos);
        // Count "mimeType" occurrences — one per image.
        int count = 0;
        size_t s = pos;
        while ((s = json.find("mimeType", s + 1)) != std::string::npos) ++count;
        REQUIRE(count == 4);
    }

    // 4 textures, all sampler=0.
    {
        size_t pos = find_json_key(json, "textures");
        REQUIRE(pos != std::string::npos);
        // Count "sampler":0 — all 4 should be sampler 0.
        int count = 0;
        size_t s = pos;
        while ((s = json.find("\"sampler\":0", s + 1)) != std::string::npos) ++count;
        REQUIRE(count == 4);
    }

    // 2 samplers always emitted (bark + leaf).
    {
        size_t pos = find_json_key(json, "samplers");
        REQUIRE(pos != std::string::npos);
        // Two entries: count "magFilter" occurrences.
        int count = 0;
        size_t s = pos;
        while ((s = json.find("magFilter", s + 1)) != std::string::npos) ++count;
        REQUIRE(count == 2);
    }

    // Material refs correct.
    {
        REQUIRE(find_json_key(json, "baseColorTexture") != std::string::npos);
        REQUIRE(find_json_key(json, "normalTexture") != std::string::npos);
        REQUIRE(find_json_key(json, "occlusionTexture") != std::string::npos);
        REQUIRE(find_json_key(json, "metallicRoughnessTexture") != std::string::npos);
    }

    // extensionsUsed does NOT contain _RYNX_LEAF_TRANSLUCENCY.
    {
        REQUIRE(json.find("_RYNX_LEAF_TRANSLUCENCY") == std::string::npos);
    }

    std::error_code ec;
    fs::remove(path, ec);
}


TEST_CASE("treegen: leaf+bark GLB embeds 7 images with correct samplers",
          "[treegen][treegen_glb_textures]") {
    namespace fs = std::filesystem;

    MiniMesh m;

    // Bark primitive.
    treegen::MaterialSpec bark_mat;
    bark_mat.base_color_tex_index         = 0;
    bark_mat.normal_tex_index             = 1;
    bark_mat.occlusion_tex_index          = 2;
    bark_mat.metallic_roughness_tex_index = 3;
    bark_mat.sampler_index                = 0;

    // Leaf primitive.
    treegen::MaterialSpec leaf_mat;
    leaf_mat.alpha_mode              = "MASK";
    leaf_mat.alpha_cutoff            = 0.5f;
    leaf_mat.base_color_tex_index    = 4;
    leaf_mat.normal_tex_index        = 5;
    leaf_mat.translucency_tex_index  = 6;
    leaf_mat.sampler_index           = 1;

    treegen::PrimitiveData prims[2] = {
        make_prim(m, bark_mat),
        make_prim(m, leaf_mat),
    };
    treegen::MeshData md;
    md.primitives = std::span<const treegen::PrimitiveData>(prims, 2);

    std::vector<treegen::ImageData> images(7);
    for (int i = 0; i < 7; ++i) {
        images[i].png_bytes = make_tiny_png(static_cast<uint8_t>(50 + i * 20));
    }

    std::string path = tmp_path("leaf_bark");
    std::string err;
    REQUIRE(treegen::write_glb_multi_mesh(
        std::span<const treegen::MeshData>(&md, 1),
        std::span<const treegen::ImageData>(images.data(), images.size()),
        path, &err));

    auto glb = slurp_bytes(path);
    std::string json = extract_json(glb);

    // 7 images.
    {
        int count = 0;
        size_t s = 0;
        while ((s = json.find("mimeType", s + 1)) != std::string::npos) ++count;
        REQUIRE(count == 7);
    }

    // Bark textures (0-3) → sampler 0, leaf textures (4-6) → sampler 1.
    {
        size_t pos = find_json_key(json, "textures");
        REQUIRE(pos != std::string::npos);
        int s0 = 0, s1 = 0;
        size_t s = pos;
        while ((s = json.find("\"sampler\":", s + 1)) != std::string::npos) {
            if (json[s + 10] == '0') ++s0;
            else if (json[s + 10] == '1') ++s1;
        }
        REQUIRE(s0 == 4); // bark: 4 textures
        REQUIRE(s1 == 3); // leaf: 3 textures
    }

    // extensionsUsed contains _RYNX_LEAF_TRANSLUCENCY.
    REQUIRE(json.find("_RYNX_LEAF_TRANSLUCENCY") != std::string::npos);

    // Leaf material has translucencyTexture.index = 6.
    REQUIRE(json.find("\"translucencyTexture\":{\"index\":6}") != std::string::npos);

    std::error_code ec;
    fs::remove(path, ec);
}


TEST_CASE("treegen: image PNG bytes round-trip via stbi decode",
          "[treegen][treegen_glb_textures]") {
    namespace fs = std::filesystem;

    // Use real bark 1024x512 + leaf atlas 1024x1024 PNGs for fidelity.
    auto bark_png = treegen::encode_bark_png(treegen::LeafShape::OakLobed, 1u);
    auto leaf_png = treegen::encode_leaf_atlas_png(1u);

    MiniMesh m;
    treegen::MaterialSpec mat;
    mat.base_color_tex_index = 0;
    mat.normal_tex_index     = 1;
    mat.sampler_index        = 0;

    treegen::PrimitiveData prim = make_prim(m, mat);
    treegen::MeshData md;
    md.primitives = std::span<const treegen::PrimitiveData>(&prim, 1);

    std::vector<treegen::ImageData> images(2);
    images[0].png_bytes = bark_png;
    images[1].png_bytes = leaf_png;

    std::string path = tmp_path("roundtrip_img");
    std::string err;
    REQUIRE(treegen::write_glb_multi_mesh(
        std::span<const treegen::MeshData>(&md, 1),
        std::span<const treegen::ImageData>(images.data(), images.size()),
        path, &err));

    auto glb = slurp_bytes(path);
    REQUIRE(glb.size() >= 20);

    // Extract BIN chunk.
    uint32_t json_len = 0;
    std::memcpy(&json_len, glb.data() + 12, 4);
    const size_t bin_header_off = 20 + json_len;
    REQUIRE(glb.size() >= bin_header_off + 8);
    const char* bin_data = glb.data() + bin_header_off + 8;
    uint32_t bin_len = 0;
    std::memcpy(&bin_len, glb.data() + bin_header_off, 4);

    // Search BIN for the PNG magic bytes (0x89504E47) of each embedded image.
    // Image bufferViews are appended after geometry data; we find them by
    // scanning for the PNG signature within the BIN chunk.
    auto find_png_in_bin = [&](const std::vector<uint8_t>& original, size_t search_from) -> size_t {
        // Match first 8 bytes of original PNG in the BIN data.
        for (size_t off = search_from; off + original.size() <= bin_len; ++off) {
            if (std::memcmp(bin_data + off, original.data(),
                            std::min<size_t>(8, original.size())) == 0) {
                return off;
            }
        }
        return size_t(-1);
    };

    // Find bark PNG in BIN, decode from that offset.
    {
        size_t off = find_png_in_bin(bark_png, 0);
        REQUIRE(off != size_t(-1));
        int dw = 0, dh = 0, dch = 0;
        unsigned char* decoded = stbi_load_from_memory(
            reinterpret_cast<const unsigned char*>(bin_data + off),
            static_cast<int>(bark_png.size()), &dw, &dh, &dch, 4);
        REQUIRE(decoded != nullptr);
        INFO("bark decoded: " << dw << "x" << dh);
        REQUIRE(dw == 1024);
        REQUIRE(dh == 512);
        stbi_image_free(decoded);
    }

    // Find leaf PNG in BIN (after bark), decode.
    {
        // Search past the bark PNG location.
        size_t bark_off = find_png_in_bin(bark_png, 0);
        size_t off = find_png_in_bin(leaf_png, bark_off + bark_png.size());
        REQUIRE(off != size_t(-1));
        int dw = 0, dh = 0, dch = 0;
        unsigned char* decoded = stbi_load_from_memory(
            reinterpret_cast<const unsigned char*>(bin_data + off),
            static_cast<int>(leaf_png.size()), &dw, &dh, &dch, 4);
        REQUIRE(decoded != nullptr);
        INFO("leaf decoded: " << dw << "x" << dh);
        REQUIRE(dw == 1024);
        REQUIRE(dh == 1024);
        stbi_image_free(decoded);
    }

    std::error_code ec;
    fs::remove(path, ec);
}


TEST_CASE("treegen: textured GLB determinism byte-identical",
          "[treegen][treegen_glb_textures]") {
    namespace fs = std::filesystem;

    MiniMesh m;
    treegen::MaterialSpec mat;
    mat.base_color_tex_index = 0;
    mat.sampler_index        = 0;
    treegen::PrimitiveData prim = make_prim(m, mat);
    treegen::MeshData md;
    md.primitives = std::span<const treegen::PrimitiveData>(&prim, 1);

    std::vector<treegen::ImageData> images(1);
    images[0].png_bytes = make_tiny_png(200);

    std::string p1 = tmp_path("det_tex_a");
    std::string p2 = tmp_path("det_tex_b");
    std::string err;
    REQUIRE(treegen::write_glb_multi_mesh(
        std::span<const treegen::MeshData>(&md, 1),
        std::span<const treegen::ImageData>(images.data(), images.size()),
        p1, &err));
    REQUIRE(treegen::write_glb_multi_mesh(
        std::span<const treegen::MeshData>(&md, 1),
        std::span<const treegen::ImageData>(images.data(), images.size()),
        p2, &err));

    auto a = slurp_bytes(p1);
    auto b = slurp_bytes(p2);
    REQUIRE(a.size() == b.size());
    REQUIRE_FALSE(a.empty());
    REQUIRE(std::memcmp(a.data(), b.data(), a.size()) == 0);

    std::error_code ec;
    fs::remove(p1, ec);
    fs::remove(p2, ec);
}


TEST_CASE("treegen: no images = backwards compatible with 3-arg overload",
          "[treegen][treegen_glb_textures]") {
    namespace fs = std::filesystem;

    MiniMesh m;
    treegen::MaterialSpec mat;
    treegen::PrimitiveData prim = make_prim(m, mat);
    treegen::MeshData md;
    md.primitives = std::span<const treegen::PrimitiveData>(&prim, 1);

    std::string p_old = tmp_path("compat_old");
    std::string p_new = tmp_path("compat_new");
    std::string err;

    // Old 3-arg overload.
    REQUIRE(treegen::write_glb_multi_mesh(
        std::span<const treegen::MeshData>(&md, 1),
        p_old, &err));

    // New 4-arg overload with empty images.
    REQUIRE(treegen::write_glb_multi_mesh(
        std::span<const treegen::MeshData>(&md, 1),
        std::span<const treegen::ImageData>{},
        p_new, &err));

    auto a = slurp_bytes(p_old);
    auto b = slurp_bytes(p_new);
    REQUIRE(a.size() == b.size());
    REQUIRE_FALSE(a.empty());
    REQUIRE(std::memcmp(a.data(), b.data(), a.size()) == 0);

    // Verify no images/textures/samplers keys in JSON.
    std::string json = extract_json(a);
    REQUIRE(find_json_key(json, "images") == std::string::npos);
    REQUIRE(find_json_key(json, "textures") == std::string::npos);
    REQUIRE(find_json_key(json, "samplers") == std::string::npos);

    std::error_code ec;
    fs::remove(p_old, ec);
    fs::remove(p_new, ec);
}
