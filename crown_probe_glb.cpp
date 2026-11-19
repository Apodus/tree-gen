// The crown probe's one loader (K17 "Input"): GLB bytes -> CrownMesh, for both
// the scenario path (the bytes generate_tree returns) and --glb (a file
// treegen wrote). gltf_view.hpp is the engine's header copied byte for byte and
// is not edited; the image lookup it lacks lives here.
#include "crown_probe.hpp"

#include "external/gltf_view.hpp"

#include <stb_image.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace treegen::crown {

namespace gv = rynx::graphics::gltf_view;

namespace {

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error("crown probe: GLB: " + why); }

const gv::jval& at(const gv::jval& arr, int i, const char* what) {
    if (arr.kind != gv::jval::kind_t::arr_t || i < 0 || i >= int(arr.as_arr().size()))
        fail(std::string(what) + " index " + std::to_string(i) + " out of range");
    return arr.as_arr()[size_t(i)];
}

const gv::jval& field(const gv::jval& obj, const char* key) {
    const gv::jval* v = obj.find(key);
    if (!v) fail(std::string("missing '") + key + "'");
    return *v;
}

// The leaf material's base-colour image, decoded; its alpha channel kept.
AlphaAtlas decode_atlas(const gv::jval& root, const gv::glb_view& glb, int texture_index, float cutoff) {
    const gv::jval& tex = at(field(root, "textures"), texture_index, "texture");
    const int image_index = field(tex, "source").as_int(-1);
    const gv::jval& img = at(field(root, "images"), image_index, "image");
    const int bv_index = field(img, "bufferView").as_int(-1);
    const gv::jval& bv = at(field(root, "bufferViews"), bv_index, "bufferView");
    const gv::jval* off = bv.find("byteOffset");
    const size_t offset = off ? size_t(off->n) : 0;
    const size_t len    = size_t(field(bv, "byteLength").n);
    if (!glb.bin_begin || offset + len > glb.bin_len) fail("image bufferView outside the BIN chunk");

    int w = 0, h = 0, comp = 0;
    unsigned char* rgba = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(glb.bin_begin + offset),
                                                int(len), &w, &h, &comp, 4);
    if (!rgba) fail(std::string("leaf atlas decode failed: ") + stbi_failure_reason());
    AlphaAtlas a;
    a.width  = w;
    a.height = h;
    a.cutoff = cutoff;
    a.alpha.resize(size_t(w) * size_t(h));
    for (size_t i = 0; i < a.alpha.size(); ++i) a.alpha[i] = rgba[i * 4 + 3];
    stbi_image_free(rgba);
    return a;
}

}  // namespace

LoadedCrown load_crown_from_glb(std::span<const uint8_t> bytes, int lod) {
    const std::span<const char> data(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    const gv::glb_view glb = gv::parse_glb_header(data);
    if (!glb.valid) fail("not a GLB container");
    gv::jparser parser(glb.json_begin, glb.json_len);
    const gv::jval root = parser.parse_value();
    if (!parser.good() || root.kind != gv::jval::kind_t::obj_t) fail("JSON chunk does not parse");

    LoadedCrown out;
    out.meshes_in_glb = gv::count_meshes(data);
    if (lod < 0 || lod >= out.meshes_in_glb)
        fail("--probe-lod=" + std::to_string(lod) + " but the GLB has " + std::to_string(out.meshes_in_glb) + " meshes");
    out.lod = lod;

    std::vector<gv::cpu_mesh> prims;
    std::string err;
    if (!gv::extract_all_primitives(data, prims, &err, lod)) fail(err);

    int leaf = -1;
    std::vector<gv::pbr_material_desc> mats(prims.size());
    for (size_t i = 0; i < prims.size(); ++i) {
        if (prims[i].material_index < 0) continue;
        if (!gv::extract_pbr_material(root, prims[i].material_index, mats[i], &err)) fail(err);
        const bool is_leaf = mats[i].alpha_mode == 1 && mats[i].translucency_tex >= 0;
        if (!is_leaf) continue;
        if (leaf >= 0) fail("more than one leaf primitive (alphaMode MASK + _RYNX_LEAF_TRANSLUCENCY) in mesh " + std::to_string(lod));
        leaf = int(i);
    }

    for (size_t i = 0; i < prims.size(); ++i) {
        if (int(i) == leaf) continue;
        out.mesh.add_bark(prims[i].positions, prims[i].indices);
    }
    if (leaf >= 0) {
        const gv::cpu_mesh& lp = prims[size_t(leaf)];
        if (lp.texcoords.size() != lp.positions.size() / 3 * 2) fail("leaf primitive has no TEXCOORD_0");
        if (mats[size_t(leaf)].base_color_tex < 0) fail("leaf material has no baseColorTexture");
        out.alpha_cutoff = mats[size_t(leaf)].alpha_cutoff;
        out.mesh.atlas = decode_atlas(root, glb, mats[size_t(leaf)].base_color_tex, out.alpha_cutoff);
        out.mesh.add_leaves(lp.positions, lp.texcoords, lp.indices);
    }
    return out;
}

}  // namespace treegen::crown
