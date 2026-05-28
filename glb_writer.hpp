// Header-only spec-compliant glTF 2.0 binary (.glb) writer. Self-contained:
// no JSON library, no engine headers. Deterministic by construction —
// every byte is computed without iterating an unordered container, so
// re-runs produce byte-identical files (required for [treegen_determinism]).
//
// C1 scope: one mesh, one node, one primitive per .glb, single material,
// OPAQUE alpha mode, indices as UNSIGNED_SHORT.
// C2 P2:   multi-primitive emission (N primitives sharing one mesh + one
//          buffer + per-primitive material slot); index componentType
//          switches to UNSIGNED_INT (5125) when vcount > 65535; `_RYNX_WIND`
//          vendor extension (per-vertex VEC4 UNSIGNED_BYTE normalized=true)
//          when any primitive supplies wind_weights_packed.
//
// /fp:precise required for byte-identical min/max accessor metadata
// (treegen.sharpmake.cs enforces).
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace treegen {

struct MaterialSpec {
    float base_color[4] = { 0.6f, 0.45f, 0.3f, 1.0f }; // wood-ish default
    float metallic = 0.0f;
    float roughness = 1.0f;
    // C2 P6 — glTF 2.0 alphaMode emission. Default OPAQUE (omits JSON field
    // to keep the C1 byte-hash determinism pin valid). Set "MASK" + cutoff
    // when a primitive needs alpha-tested rasterization (leaf cards).
    // "BLEND" is reserved; the engine renders trees with leaf-cluster opaque
    // RT shadow + MASK rasterization, BLEND is not on the C2 path.
    const char* alpha_mode  = "OPAQUE";  // "OPAQUE" | "MASK" | "BLEND"
    float       alpha_cutoff = 0.5f;
    // C6 P4 — texture indices into the GLB images[] array. -1 = not set.
    int base_color_tex_index         = -1;  // pbrMetallicRoughness.baseColorTexture.index
    int normal_tex_index             = -1;  // normalTexture.index
    int occlusion_tex_index          = -1;  // occlusionTexture.index
    int metallic_roughness_tex_index = -1;  // pbrMetallicRoughness.metallicRoughnessTexture.index
    int translucency_tex_index       = -1;  // extensions._RYNX_LEAF_TRANSLUCENCY.translucencyTexture.index
    int sampler_index                = 0;   // 0=bark REPEAT/CLAMP, 1=leaf CLAMP/CLAMP
};

// C6 P4 — embedded image data (PNG bytes + MIME type).
struct ImageData {
    std::vector<uint8_t> png_bytes;
    const char* mime_type = "image/png";
};

struct PrimitiveData {
    std::span<const float>    positions;            // xyz packed
    std::span<const float>    normals;              // xyz packed; required
    std::span<const float>    uvs;                  // xy packed; empty span = no TEXCOORD_0
    std::span<const float>    tangents;             // C1 P6: xyzw packed; empty = no TANGENT
    std::span<const uint16_t> indices;              // UNSIGNED_SHORT path (legacy / C1)
    std::span<const uint32_t> indices_u32;          // UNSIGNED_INT  path (set when vcount > 65535)
    // C2 P2: `_RYNX_WIND` per-vertex 4-tier influences. Empty span = no extension
    // for this primitive. Length must equal positions.size()/3 * 4.
    std::span<const uint8_t>  wind_weights_packed;
    MaterialSpec              material;
};

// C2 P7 — per-mesh data for multi-mesh GLB writes (LOD chains). Each MeshData
// becomes a top-level meshes[i] entry. When lod_index >= 0, the mesh carries
// `extensions._RYNX_LOD = { lod_index, lod_max_distance_m }`.
struct MeshData {
    std::span<const PrimitiveData> primitives;
    int   lod_index          = -1;      // -1 = no _RYNX_LOD extension emitted
    float lod_max_distance_m = 0.0f;    // ignored unless lod_index >= 0
    float lod_screen_height_px = 0.0f;  // screen-height pixel threshold for LOD selection
    // _RYNX_COLLISION capsule (mesh[0] only). negative half_length = not set.
    float collision_half_length = -1.0f;
    float collision_radius      = 0.0f;
};

namespace glb_detail {

// 4-byte alignment is what the glTF 2.0 spec requires for chunks AND for
// per-bufferView alignment of the underlying scalar component (4 for float,
// 4 for uint32, 2 for uint16, 1 for u8). Padding to 4 between bufferViews
// satisfies every component's alignment.
constexpr size_t kAlign = 4;

inline size_t align_up(size_t n) {
    return (n + (kAlign - 1)) & ~(kAlign - 1);
}

inline void pad_bytes(std::vector<uint8_t>& buf, size_t n, uint8_t fill = 0x00) {
    buf.insert(buf.end(), n, fill);
}

inline void append_bytes(std::vector<uint8_t>& buf, const void* src, size_t n) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    buf.insert(buf.end(), s, s + n);
}

inline std::string f2s(double v) {
    char tmp[64];
    std::snprintf(tmp, sizeof(tmp), "%.9g", v);
    return tmp;
}

inline std::string i2s(int64_t v) {
    char tmp[32];
    std::snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(v));
    return tmp;
}

inline std::string u2s(uint64_t v) {
    char tmp[32];
    std::snprintf(tmp, sizeof(tmp), "%llu", static_cast<unsigned long long>(v));
    return tmp;
}

struct AccessorMeta {
    int bufferView;
    int byteOffset;
    int componentType;     // 5121 UB, 5123 US, 5125 UI, 5126 F
    int count;
    const char* type;      // "SCALAR" | "VEC2" | "VEC3" | "VEC4"
    bool normalized;       // emitted only when true
    bool emit_minmax;
    double mn[3];
    double mx[3];
    int    minmax_count;
};

struct BufferViewMeta {
    int byteOffset;
    int byteLength;
    int target;            // 34962 = ARRAY_BUFFER, 34963 = ELEMENT_ARRAY_BUFFER
};

// Per-primitive accessor indices computed during BIN layout.
struct PrimitiveAccessors {
    int positions_acc;
    int normals_acc;
    int uvs_acc;            // -1 if absent
    int tangents_acc;       // C1 P6: -1 if absent
    int indices_acc;
    int wind_weights_acc;   // -1 if absent
    bool material_emit;     // always true for now (one material per primitive)
};

// C2 P7 — per-mesh JSON shape metadata. Used by build_json_multi_mesh to
// group flat primitives into N top-level `meshes[]` entries and emit
// `extensions._RYNX_LOD` when the source MeshData carried lod_index >= 0.
struct MeshJsonMeta {
    int  first_prim;   // index into the flat prim_accs vector
    int  prim_count;
    int  lod_index;            // -1 = no _RYNX_LOD emitted
    float lod_max_distance_m;
    float lod_screen_height_px = 0.0f;
    float collision_half_length = -1.0f;  // negative = not set
    float collision_radius      = 0.0f;
};

// C6 P4 — forward-only material JSON emitter. Replaces the pop_back() hack
// with a single-pass builder: pbrMetallicRoughness block + optional texture
// refs + alphaMode + optional extensions, all emitted incrementally.
inline void emit_material_json(std::string& j, const MaterialSpec& mat, int mat_index,
                               bool has_textures) {
    j += R"({"name":"primary_)";
    j += i2s(int64_t(mat_index));
    j += R"(","pbrMetallicRoughness":{"baseColorFactor":[)";
    j += f2s(mat.base_color[0]); j += ',';
    j += f2s(mat.base_color[1]); j += ',';
    j += f2s(mat.base_color[2]); j += ',';
    j += f2s(mat.base_color[3]);
    j += R"(])";
    if (has_textures && mat.base_color_tex_index >= 0) {
        j += R"(,"baseColorTexture":{"index":)";
        j += i2s(mat.base_color_tex_index);
        j += '}';
    }
    j += R"(,"metallicFactor":)";
    j += f2s(mat.metallic);
    if (has_textures && mat.metallic_roughness_tex_index >= 0) {
        j += R"(,"metallicRoughnessTexture":{"index":)";
        j += i2s(mat.metallic_roughness_tex_index);
        j += '}';
    }
    j += R"(,"roughnessFactor":)";
    j += f2s(mat.roughness);
    j += '}';  // close pbrMetallicRoughness
    // alphaMode (OPAQUE is default — omit for byte-identity with pre-P4).
    const bool is_opaque = (mat.alpha_mode == nullptr)
                          || (mat.alpha_mode[0] == 'O');
    if (!is_opaque) {
        j += R"(,"alphaMode":")";
        j += mat.alpha_mode;
        j += '"';
        if (mat.alpha_mode[0] == 'M') {
            j += R"(,"alphaCutoff":)";
            j += f2s(mat.alpha_cutoff);
        }
    }
    if (has_textures && mat.normal_tex_index >= 0) {
        j += R"(,"normalTexture":{"index":)";
        j += i2s(mat.normal_tex_index);
        j += '}';
    }
    if (has_textures && mat.occlusion_tex_index >= 0) {
        j += R"(,"occlusionTexture":{"index":)";
        j += i2s(mat.occlusion_tex_index);
        j += '}';
    }
    if (has_textures && mat.translucency_tex_index >= 0) {
        j += R"(,"extensions":{"_RYNX_LEAF_TRANSLUCENCY":{"translucencyTexture":{"index":)";
        j += i2s(mat.translucency_tex_index);
        j += R"(}}})";
    }
    j += '}';  // close material
}

inline std::string build_json(
    const std::vector<BufferViewMeta>& bvs,
    const std::vector<AccessorMeta>&   accs,
    const std::vector<PrimitiveAccessors>& prim_accs,
    const std::vector<MaterialSpec>& materials,
    int total_buffer_bytes,
    bool any_wind)
{
    std::string j;
    j.reserve(4096);

    j += R"({"accessors":[)";
    for (size_t i = 0; i < accs.size(); ++i) {
        if (i) j += ',';
        const auto& a = accs[i];
        j += R"({"bufferView":)";
        j += i2s(a.bufferView);
        j += R"(,"byteOffset":)";
        j += i2s(a.byteOffset);
        j += R"(,"componentType":)";
        j += i2s(a.componentType);
        j += R"(,"count":)";
        j += i2s(a.count);
        if (a.emit_minmax) {
            j += R"(,"max":[)";
            for (int k = 0; k < a.minmax_count; ++k) {
                if (k) j += ',';
                j += f2s(a.mx[k]);
            }
            j += R"(],"min":[)";
            for (int k = 0; k < a.minmax_count; ++k) {
                if (k) j += ',';
                j += f2s(a.mn[k]);
            }
            j += ']';
        }
        if (a.normalized) {
            j += R"(,"normalized":true)";
        }
        j += R"(,"type":")";
        j += a.type;
        j += R"("})";
    }
    j += ']';

    j += R"(,"asset":{"generator":"rynx-treegen","version":"2.0"})";

    j += R"(,"bufferViews":[)";
    for (size_t i = 0; i < bvs.size(); ++i) {
        if (i) j += ',';
        const auto& b = bvs[i];
        j += R"({"buffer":0,"byteLength":)";
        j += i2s(b.byteLength);
        j += R"(,"byteOffset":)";
        j += i2s(b.byteOffset);
        if (b.target != 0) {
            j += R"(,"target":)";
            j += i2s(b.target);
        }
        j += '}';
    }
    j += ']';

    j += R"(,"buffers":[{"byteLength":)";
    j += i2s(total_buffer_bytes);
    j += "}]";

    // C2 P2: extensionsUsed only when any primitive carries `_RYNX_WIND`.
    if (any_wind) {
        j += R"(,"extensionsUsed":["_RYNX_WIND"])";
    }

    // Materials — forward-only via emit_material_json (no pop_back hack).
    j += R"(,"materials":[)";
    for (size_t i = 0; i < materials.size(); ++i) {
        if (i) j += ',';
        emit_material_json(j, materials[i], static_cast<int>(i), false);
    }
    j += ']';

    // One mesh containing N primitives. Attribute keys lexicographic.
    j += R"(,"meshes":[{"name":"mesh0","primitives":[)";
    for (size_t i = 0; i < prim_accs.size(); ++i) {
        if (i) j += ',';
        const auto& pa = prim_accs[i];
        j += R"({"attributes":{)";
        bool first_attr = true;
        auto emit_attr = [&](const char* k, int v) {
            if (!first_attr) j += ',';
            first_attr = false;
            j += '"'; j += k; j += R"(":)"; j += i2s(v);
        };
        emit_attr("NORMAL", pa.normals_acc);
        emit_attr("POSITION", pa.positions_acc);
        if (pa.tangents_acc >= 0) emit_attr("TANGENT", pa.tangents_acc);
        if (pa.uvs_acc >= 0) emit_attr("TEXCOORD_0", pa.uvs_acc);
        j += R"(},"indices":)";
        j += i2s(pa.indices_acc);
        j += R"(,"material":)";
        j += i2s(int64_t(i));
        j += R"(,"mode":4)";  // 4 = TRIANGLES
        if (pa.wind_weights_acc >= 0) {
            j += R"(,"extensions":{"_RYNX_WIND":{"weights":)";
            j += i2s(pa.wind_weights_acc);
            j += R"(}})";
        }
        j += '}';
    }
    j += ']';
    j += R"(}])";  // close meshes[0] + meshes array

    j += R"(,"nodes":[{"mesh":0,"name":"root"}])";
    j += R"(,"scene":0)";
    j += R"(,"scenes":[{"nodes":[0]}])";

    j += '}';
    return j;
}

// C2 P7 / C6 P4 — multi-mesh JSON build. Emits N top-level `meshes[]` entries
// (one per MeshJsonMeta), each with its slice of prim_accs. Per-mesh
// `_RYNX_LOD` extension when meta.lod_index >= 0. C6 P4 adds optional
// images/textures/samplers sections when image_count > 0.
inline std::string build_json_multi_mesh(
    const std::vector<BufferViewMeta>& bvs,
    const std::vector<AccessorMeta>&   accs,
    const std::vector<PrimitiveAccessors>& prim_accs,
    const std::vector<MaterialSpec>& materials,
    const std::vector<MeshJsonMeta>& meshes_meta,
    int total_buffer_bytes,
    bool any_wind,
    bool any_lod,
    int image_count = 0,
    const std::vector<int>* image_bv_indices = nullptr,
    bool any_translucency = false,
    bool any_collision = false)
{
    std::string j;
    j.reserve(4096 + meshes_meta.size() * 256);

    j += R"({"accessors":[)";
    for (size_t i = 0; i < accs.size(); ++i) {
        if (i) j += ',';
        const auto& a = accs[i];
        j += R"({"bufferView":)";
        j += i2s(a.bufferView);
        j += R"(,"byteOffset":)";
        j += i2s(a.byteOffset);
        j += R"(,"componentType":)";
        j += i2s(a.componentType);
        j += R"(,"count":)";
        j += i2s(a.count);
        if (a.emit_minmax) {
            j += R"(,"max":[)";
            for (int k = 0; k < a.minmax_count; ++k) {
                if (k) j += ',';
                j += f2s(a.mx[k]);
            }
            j += R"(],"min":[)";
            for (int k = 0; k < a.minmax_count; ++k) {
                if (k) j += ',';
                j += f2s(a.mn[k]);
            }
            j += ']';
        }
        if (a.normalized) {
            j += R"(,"normalized":true)";
        }
        j += R"(,"type":")";
        j += a.type;
        j += R"("})";
    }
    j += ']';

    j += R"(,"asset":{"generator":"rynx-treegen","version":"2.0"})";

    j += R"(,"bufferViews":[)";
    for (size_t i = 0; i < bvs.size(); ++i) {
        if (i) j += ',';
        const auto& b = bvs[i];
        j += R"({"buffer":0,"byteLength":)";
        j += i2s(b.byteLength);
        j += R"(,"byteOffset":)";
        j += i2s(b.byteOffset);
        if (b.target != 0) {
            j += R"(,"target":)";
            j += i2s(b.target);
        }
        j += '}';
    }
    j += ']';

    j += R"(,"buffers":[{"byteLength":)";
    j += i2s(total_buffer_bytes);
    j += "}]";

    // extensionsUsed — emit when any extension is in use.
    const bool has_textures = (image_count > 0);
    if (any_wind || any_lod || any_translucency || any_collision) {
        j += R"(,"extensionsUsed":[)";
        bool first = true;
        auto ext = [&](const char* name) {
            if (!first) j += ',';
            first = false;
            j += '"'; j += name; j += '"';
        };
        if (any_collision)    ext("_RYNX_COLLISION");
        if (any_translucency) ext("_RYNX_LEAF_TRANSLUCENCY");
        if (any_lod)          ext("_RYNX_LOD");
        if (any_wind)         ext("_RYNX_WIND");
        j += ']';
    }

    // C6 P4 — images/samplers/textures when images are embedded.
    if (has_textures) {
        j += R"(,"images":[)";
        for (int i = 0; i < image_count; ++i) {
            if (i) j += ',';
            j += R"({"bufferView":)";
            j += i2s((*image_bv_indices)[i]);
            j += R"(,"mimeType":"image/png"})";
        }
        j += ']';
    }

    // Materials — forward-only via emit_material_json (no pop_back hack).
    j += R"(,"materials":[)";
    for (size_t i = 0; i < materials.size(); ++i) {
        if (i) j += ',';
        emit_material_json(j, materials[i], static_cast<int>(i), has_textures);
    }
    j += ']';

    // C6 P4 — samplers + textures (after materials, before meshes).
    if (has_textures) {
        // Two samplers: 0 = bark (REPEAT U, CLAMP V), 1 = leaf (CLAMP U+V).
        j += R"(,"samplers":[)";
        j += R"({"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":33071})";
        j += R"(,{"magFilter":9729,"minFilter":9987,"wrapS":33071,"wrapT":33071})";
        j += ']';
        // textures[] — one entry per image, each referencing its sampler via
        // the material's sampler_index.
        j += R"(,"textures":[)";
        for (int i = 0; i < image_count; ++i) {
            if (i) j += ',';
            // Determine which sampler this texture uses by scanning materials
            // for any tex index matching i.
            int samp = 0;
            for (const auto& mat : materials) {
                if (mat.base_color_tex_index == i || mat.normal_tex_index == i
                    || mat.occlusion_tex_index == i || mat.metallic_roughness_tex_index == i
                    || mat.translucency_tex_index == i) {
                    samp = mat.sampler_index;
                    break;
                }
            }
            j += R"({"sampler":)";
            j += i2s(samp);
            j += R"(,"source":)";
            j += i2s(i);
            j += '}';
        }
        j += ']';
    }

    // Meshes — N top-level entries, each grouping a slice of prim_accs.
    j += R"(,"meshes":[)";
    for (size_t mi = 0; mi < meshes_meta.size(); ++mi) {
        if (mi) j += ',';
        const auto& mm = meshes_meta[mi];
        j += R"({"name":"mesh)";
        j += i2s(int64_t(mi));
        j += R"(","primitives":[)";
        for (int pi = 0; pi < mm.prim_count; ++pi) {
            if (pi) j += ',';
            const auto& pa = prim_accs[size_t(mm.first_prim + pi)];
            j += R"({"attributes":{)";
            bool first_attr = true;
            auto emit_attr = [&](const char* k, int v) {
                if (!first_attr) j += ',';
                first_attr = false;
                j += '"'; j += k; j += R"(":)"; j += i2s(v);
            };
            emit_attr("NORMAL", pa.normals_acc);
            emit_attr("POSITION", pa.positions_acc);
            if (pa.tangents_acc >= 0) emit_attr("TANGENT", pa.tangents_acc);
            if (pa.uvs_acc >= 0) emit_attr("TEXCOORD_0", pa.uvs_acc);
            j += R"(},"indices":)";
            j += i2s(pa.indices_acc);
            j += R"(,"material":)";
            // material index is the absolute (flat) primitive index — one
            // material per primitive across all meshes.
            j += i2s(int64_t(mm.first_prim + pi));
            j += R"(,"mode":4)";
            if (pa.wind_weights_acc >= 0) {
                j += R"(,"extensions":{"_RYNX_WIND":{"weights":)";
                j += i2s(pa.wind_weights_acc);
                j += R"(}})";
            }
            j += '}';
        }
        j += ']';
        // Per-mesh extensions (_RYNX_COLLISION, _RYNX_LOD).
        const bool has_lod = (mm.lod_index >= 0);
        const bool has_col = (mm.collision_half_length >= 0.0f);
        if (has_col || has_lod) {
            j += R"(,"extensions":{)";
            bool first_ext = true;
            if (has_col) {
                j += R"("_RYNX_COLLISION":{"type":"capsule","half_length":)";
                j += f2s(mm.collision_half_length);
                j += R"(,"radius":)";
                j += f2s(mm.collision_radius);
                j += '}';
                first_ext = false;
            }
            if (has_lod) {
                if (!first_ext) j += ',';
                j += R"("_RYNX_LOD":{"lod_index":)";
                j += i2s(mm.lod_index);
                j += R"(,"lod_max_distance_m":)";
                j += f2s(mm.lod_max_distance_m);
                j += R"(,"lod_screen_height_px":)";
                j += f2s(mm.lod_screen_height_px);
                j += '}';
            }
            j += '}';
        }
        j += '}';
    }
    j += ']';

    // Nodes — one per mesh, all children of a single root scene node. glTF
    // requires every mesh be reachable from the scene graph.
    j += R"(,"nodes":[)";
    for (size_t mi = 0; mi < meshes_meta.size(); ++mi) {
        if (mi) j += ',';
        j += R"({"mesh":)";
        j += i2s(int64_t(mi));
        j += R"(,"name":"mesh_node_)";
        j += i2s(int64_t(mi));
        j += R"("})";
    }
    j += ']';

    j += R"(,"scene":0)";
    j += R"(,"scenes":[{"nodes":[)";
    for (size_t mi = 0; mi < meshes_meta.size(); ++mi) {
        if (mi) j += ',';
        j += i2s(int64_t(mi));
    }
    j += R"(]}])";

    j += '}';
    return j;
}

} // namespace glb_detail

// Returns false on failure. `err_out` (optional) gets a short reason.
//
// Per-primitive index source preference:
//   - If `p.indices_u32` is non-empty, the u32 path is used (componentType
//     UNSIGNED_INT 5125). Required when vcount > 65535.
//   - Else if `p.indices` (u16) is non-empty AND vcount <= 65535, u16 path
//     is used (componentType UNSIGNED_SHORT 5123).
//   - Else an error is returned.
inline bool write_glb(std::span<const PrimitiveData> primitives,
                      const std::string& output_path,
                      std::string* err_out)
{
    auto set_err = [&](const char* m) {
        if (err_out) *err_out = m;
    };

    if (primitives.empty()) {
        set_err("write_glb: no primitives");
        return false;
    }

    // Per-primitive validation + index path resolution.
    struct PrimResolved {
        const PrimitiveData* src;
        uint32_t vcount;
        uint32_t icount;
        bool     use_u32_indices;
        bool     has_uvs;
        bool     has_tangents;
        bool     has_wind;
    };
    std::vector<PrimResolved> resolved;
    resolved.reserve(primitives.size());

    bool any_wind = false;
    for (const auto& p : primitives) {
        if (p.positions.empty()) { set_err("write_glb: positions empty"); return false; }
        if ((p.positions.size() % 3) != 0) { set_err("write_glb: positions not VEC3"); return false; }
        if (p.normals.size() != p.positions.size()) { set_err("write_glb: normals must match positions count"); return false; }
        if (!p.uvs.empty() && p.uvs.size() != (p.positions.size() / 3) * 2) {
            set_err("write_glb: uvs present but count mismatches positions");
            return false;
        }
        if (!p.tangents.empty() && p.tangents.size() != (p.positions.size() / 3) * 4) {
            set_err("write_glb: tangents present but count mismatches positions (need vcount*4)");
            return false;
        }

        const uint32_t vcount = static_cast<uint32_t>(p.positions.size() / 3);
        const bool has_u16 = !p.indices.empty();
        const bool has_u32 = !p.indices_u32.empty();

        if (!has_u16 && !has_u32) { set_err("write_glb: indices empty"); return false; }
        if (has_u16 && has_u32)   { set_err("write_glb: both u16 and u32 indices supplied"); return false; }

        const uint32_t icount = has_u32
            ? static_cast<uint32_t>(p.indices_u32.size())
            : static_cast<uint32_t>(p.indices.size());
        if (icount == 0 || (icount % 3) != 0) { set_err("write_glb: indices not TRIANGLES"); return false; }

        if (!has_u32 && vcount > 65535) {
            set_err("write_glb: vertex count exceeds UNSIGNED_SHORT range; supply indices_u32");
            return false;
        }

        if (!p.wind_weights_packed.empty() && p.wind_weights_packed.size() != size_t(vcount) * 4) {
            set_err("write_glb: wind_weights_packed length mismatch (need vcount*4)");
            return false;
        }
        if (!p.wind_weights_packed.empty()) any_wind = true;

        PrimResolved r;
        r.src = &p;
        r.vcount = vcount;
        r.icount = icount;
        r.use_u32_indices = has_u32;
        r.has_uvs = !p.uvs.empty();
        r.has_tangents = !p.tangents.empty();
        r.has_wind = !p.wind_weights_packed.empty();
        resolved.push_back(r);
    }

    // ---- Lay out BIN buffer. ----
    std::vector<uint8_t> bin;
    {
        size_t reserve_bytes = 64;
        for (const auto& r : resolved) {
            reserve_bytes += r.vcount * (3 + 3 + 2) * 4;
            if (r.has_tangents) reserve_bytes += r.vcount * 4 * 4;
            reserve_bytes += r.icount * (r.use_u32_indices ? 4 : 2);
            if (r.has_wind) reserve_bytes += r.vcount * 4;
        }
        bin.reserve(reserve_bytes);
    }

    std::vector<glb_detail::BufferViewMeta> bvs;
    std::vector<glb_detail::AccessorMeta>   accs;
    std::vector<glb_detail::PrimitiveAccessors> prim_accs;
    std::vector<MaterialSpec> materials;
    prim_accs.reserve(resolved.size());
    materials.reserve(resolved.size());

    auto add_bv = [&](int byte_length, int target) -> int {
        size_t cur = bin.size();
        size_t aligned = glb_detail::align_up(cur);
        glb_detail::pad_bytes(bin, aligned - cur, 0x00);
        glb_detail::BufferViewMeta bv;
        bv.byteOffset = static_cast<int>(bin.size());
        bv.byteLength = byte_length;
        bv.target     = target;
        bvs.push_back(bv);
        return static_cast<int>(bvs.size() - 1);
    };

    for (const auto& r : resolved) {
        const PrimitiveData& p = *r.src;
        materials.push_back(p.material);

        glb_detail::PrimitiveAccessors pa{};
        pa.uvs_acc = -1;
        pa.tangents_acc = -1;
        pa.wind_weights_acc = -1;

        // POSITION min/max under /fp:precise.
        double pmn[3] = { 1e300, 1e300, 1e300 };
        double pmx[3] = { -1e300, -1e300, -1e300 };
        for (uint32_t i = 0; i < r.vcount; ++i) {
            for (int c = 0; c < 3; ++c) {
                double v = static_cast<double>(p.positions[i * 3 + c]);
                if (v < pmn[c]) pmn[c] = v;
                if (v > pmx[c]) pmx[c] = v;
            }
        }

        // ---- Positions ----
        int pos_bv = add_bv(static_cast<int>(r.vcount * 3 * sizeof(float)), 34962);
        glb_detail::append_bytes(bin, p.positions.data(), p.positions.size() * sizeof(float));
        {
            glb_detail::AccessorMeta a{};
            a.bufferView = pos_bv;
            a.byteOffset = 0;
            a.componentType = 5126; // FLOAT
            a.count = static_cast<int>(r.vcount);
            a.type = "VEC3";
            a.emit_minmax = true;
            a.minmax_count = 3;
            for (int k = 0; k < 3; ++k) { a.mn[k] = pmn[k]; a.mx[k] = pmx[k]; }
            accs.push_back(a);
        }
        pa.positions_acc = static_cast<int>(accs.size() - 1);

        // ---- Normals ----
        int nrm_bv = add_bv(static_cast<int>(r.vcount * 3 * sizeof(float)), 34962);
        glb_detail::append_bytes(bin, p.normals.data(), p.normals.size() * sizeof(float));
        {
            glb_detail::AccessorMeta a{};
            a.bufferView = nrm_bv;
            a.byteOffset = 0;
            a.componentType = 5126;
            a.count = static_cast<int>(r.vcount);
            a.type = "VEC3";
            accs.push_back(a);
        }
        pa.normals_acc = static_cast<int>(accs.size() - 1);

        // ---- UVs (optional) ----
        if (r.has_uvs) {
            int uv_bv = add_bv(static_cast<int>(r.vcount * 2 * sizeof(float)), 34962);
            glb_detail::append_bytes(bin, p.uvs.data(), p.uvs.size() * sizeof(float));
            glb_detail::AccessorMeta a{};
            a.bufferView = uv_bv;
            a.byteOffset = 0;
            a.componentType = 5126;
            a.count = static_cast<int>(r.vcount);
            a.type = "VEC2";
            accs.push_back(a);
            pa.uvs_acc = static_cast<int>(accs.size() - 1);
        }

        // ---- TANGENT (optional) ----
        if (r.has_tangents) {
            int tan_bv = add_bv(static_cast<int>(r.vcount * 4 * sizeof(float)), 34962);
            glb_detail::append_bytes(bin, p.tangents.data(), p.tangents.size() * sizeof(float));
            glb_detail::AccessorMeta a{};
            a.bufferView = tan_bv;
            a.byteOffset = 0;
            a.componentType = 5126;
            a.count = static_cast<int>(r.vcount);
            a.type = "VEC4";
            accs.push_back(a);
            pa.tangents_acc = static_cast<int>(accs.size() - 1);
        }

        // ---- Indices ----
        if (r.use_u32_indices) {
            int idx_bv = add_bv(static_cast<int>(r.icount * sizeof(uint32_t)), 34963);
            glb_detail::append_bytes(bin, p.indices_u32.data(), p.indices_u32.size() * sizeof(uint32_t));
            glb_detail::AccessorMeta a{};
            a.bufferView = idx_bv;
            a.byteOffset = 0;
            a.componentType = 5125; // UNSIGNED_INT
            a.count = static_cast<int>(r.icount);
            a.type = "SCALAR";
            accs.push_back(a);
        }
        else {
            int idx_bv = add_bv(static_cast<int>(r.icount * sizeof(uint16_t)), 34963);
            glb_detail::append_bytes(bin, p.indices.data(), p.indices.size() * sizeof(uint16_t));
            glb_detail::AccessorMeta a{};
            a.bufferView = idx_bv;
            a.byteOffset = 0;
            a.componentType = 5123; // UNSIGNED_SHORT
            a.count = static_cast<int>(r.icount);
            a.type = "SCALAR";
            accs.push_back(a);
        }
        pa.indices_acc = static_cast<int>(accs.size() - 1);

        // ---- _RYNX_WIND (optional) ----
        if (r.has_wind) {
            int w_bv = add_bv(static_cast<int>(r.vcount * 4 * sizeof(uint8_t)), 34962);
            glb_detail::append_bytes(bin, p.wind_weights_packed.data(), p.wind_weights_packed.size());
            glb_detail::AccessorMeta a{};
            a.bufferView = w_bv;
            a.byteOffset = 0;
            a.componentType = 5121; // UNSIGNED_BYTE
            a.count = static_cast<int>(r.vcount);
            a.type = "VEC4";
            a.normalized = true;
            accs.push_back(a);
            pa.wind_weights_acc = static_cast<int>(accs.size() - 1);
        }

        prim_accs.push_back(pa);
    }

    const int total_buffer_bytes = static_cast<int>(bin.size());

    // ---- Build JSON. ----
    std::string json = glb_detail::build_json(bvs, accs, prim_accs, materials, total_buffer_bytes, any_wind);

    // ---- Pad JSON to 4 bytes with 0x20 (space). ----
    size_t json_pad = glb_detail::align_up(json.size()) - json.size();
    json.append(json_pad, ' ');

    // ---- Pad BIN to 4 bytes with 0x00. ----
    size_t bin_pad = glb_detail::align_up(bin.size()) - bin.size();
    glb_detail::pad_bytes(bin, bin_pad, 0x00);

    // ---- Compose GLB byte stream. ----
    const uint32_t magic   = 0x46546C67u;
    const uint32_t version = 2u;
    const uint32_t total_len =
        12 + 8 + static_cast<uint32_t>(json.size()) + 8 + static_cast<uint32_t>(bin.size());

    std::vector<uint8_t> out;
    out.reserve(total_len);

    auto put_u32 = [&](uint32_t v) {
        out.push_back(static_cast<uint8_t>(v & 0xFFu));
        out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
        out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
        out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
    };

    put_u32(magic);
    put_u32(version);
    put_u32(total_len);

    // JSON chunk
    put_u32(static_cast<uint32_t>(json.size()));
    put_u32(0x4E4F534Au); // "JSON"
    out.insert(out.end(), json.begin(), json.end());

    // BIN chunk
    put_u32(static_cast<uint32_t>(bin.size()));
    put_u32(0x004E4942u); // "BIN\0"
    out.insert(out.end(), bin.begin(), bin.end());

    // ---- Write file ----
    std::ofstream f(output_path, std::ios::binary | std::ios::trunc);
    if (!f) { set_err("write_glb: cannot open output file"); return false; }
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!f.good()) { set_err("write_glb: write failed"); return false; }
    return true;
}

// C6 P4 — multi-mesh GLB writer with embedded images. 4-arg overload adds
// per-image PNG bytes appended to BIN as target=0 bufferViews, plus
// images/textures/samplers JSON sections. When `images` is empty, output is
// byte-identical to the pre-P4 3-arg overload.
inline bool write_glb_multi_mesh(std::span<const MeshData> meshes,
                                 std::span<const ImageData> images,
                                 const std::string& output_path,
                                 std::string* err_out)
{
    auto set_err = [&](const char* m) {
        if (err_out) *err_out = m;
    };

    if (meshes.empty()) {
        set_err("write_glb_multi_mesh: no meshes");
        return false;
    }
    bool any_prim = false;
    for (const auto& mm : meshes) if (!mm.primitives.empty()) { any_prim = true; break; }
    if (!any_prim) {
        set_err("write_glb_multi_mesh: all meshes empty");
        return false;
    }

    // Flatten primitives across all meshes for shared accessor/bv/material
    // arrays; record per-mesh slice metadata for the JSON build.
    struct PrimResolved {
        const PrimitiveData* src;
        uint32_t vcount;
        uint32_t icount;
        bool     use_u32_indices;
        bool     has_uvs;
        bool     has_tangents;
        bool     has_wind;
    };
    std::vector<PrimResolved> resolved;
    std::vector<glb_detail::MeshJsonMeta> meshes_meta;
    meshes_meta.reserve(meshes.size());

    bool any_wind      = false;
    bool any_lod       = false;
    bool any_collision = false;
    for (const auto& mm : meshes) {
        glb_detail::MeshJsonMeta meta;
        meta.first_prim = static_cast<int>(resolved.size());
        meta.prim_count = static_cast<int>(mm.primitives.size());
        meta.lod_index  = mm.lod_index;
        meta.lod_max_distance_m     = mm.lod_max_distance_m;
        meta.lod_screen_height_px   = mm.lod_screen_height_px;
        meta.collision_half_length   = mm.collision_half_length;
        meta.collision_radius        = mm.collision_radius;
        if (mm.lod_index >= 0) any_lod = true;
        if (mm.collision_half_length >= 0.0f) any_collision = true;
        meshes_meta.push_back(meta);

        for (const auto& p : mm.primitives) {
            if (p.positions.empty()) { set_err("write_glb_multi_mesh: positions empty"); return false; }
            if ((p.positions.size() % 3) != 0) { set_err("write_glb_multi_mesh: positions not VEC3"); return false; }
            if (p.normals.size() != p.positions.size()) { set_err("write_glb_multi_mesh: normals must match positions"); return false; }
            if (!p.uvs.empty() && p.uvs.size() != (p.positions.size() / 3) * 2) {
                set_err("write_glb_multi_mesh: uvs/positions mismatch");
                return false;
            }
            if (!p.tangents.empty() && p.tangents.size() != (p.positions.size() / 3) * 4) {
                set_err("write_glb_multi_mesh: tangents present but count mismatches positions (need vcount*4)");
                return false;
            }
            const uint32_t vcount = static_cast<uint32_t>(p.positions.size() / 3);
            const bool has_u16 = !p.indices.empty();
            const bool has_u32 = !p.indices_u32.empty();
            if (!has_u16 && !has_u32) { set_err("write_glb_multi_mesh: indices empty"); return false; }
            if (has_u16 && has_u32)   { set_err("write_glb_multi_mesh: both u16 and u32 indices supplied"); return false; }
            const uint32_t icount = has_u32
                ? static_cast<uint32_t>(p.indices_u32.size())
                : static_cast<uint32_t>(p.indices.size());
            if (icount == 0 || (icount % 3) != 0) { set_err("write_glb_multi_mesh: indices not TRIANGLES"); return false; }
            if (!has_u32 && vcount > 65535) {
                set_err("write_glb_multi_mesh: vertex count exceeds UNSIGNED_SHORT range; supply indices_u32");
                return false;
            }
            if (!p.wind_weights_packed.empty() && p.wind_weights_packed.size() != size_t(vcount) * 4) {
                set_err("write_glb_multi_mesh: wind_weights_packed length mismatch");
                return false;
            }
            if (!p.wind_weights_packed.empty()) any_wind = true;
            PrimResolved r;
            r.src = &p;
            r.vcount = vcount;
            r.icount = icount;
            r.use_u32_indices = has_u32;
            r.has_uvs = !p.uvs.empty();
            r.has_tangents = !p.tangents.empty();
            r.has_wind = !p.wind_weights_packed.empty();
            resolved.push_back(r);
        }
    }

    // ---- Lay out BIN buffer (same algorithm as write_glb). ----
    std::vector<uint8_t> bin;
    {
        size_t reserve_bytes = 64;
        for (const auto& r : resolved) {
            reserve_bytes += r.vcount * (3 + 3 + 2) * 4;
            if (r.has_tangents) reserve_bytes += r.vcount * 4 * 4;
            reserve_bytes += r.icount * (r.use_u32_indices ? 4 : 2);
            if (r.has_wind) reserve_bytes += r.vcount * 4;
        }
        bin.reserve(reserve_bytes);
    }

    std::vector<glb_detail::BufferViewMeta> bvs;
    std::vector<glb_detail::AccessorMeta>   accs;
    std::vector<glb_detail::PrimitiveAccessors> prim_accs;
    std::vector<MaterialSpec> materials;
    prim_accs.reserve(resolved.size());
    materials.reserve(resolved.size());

    auto add_bv = [&](int byte_length, int target) -> int {
        size_t cur = bin.size();
        size_t aligned = glb_detail::align_up(cur);
        glb_detail::pad_bytes(bin, aligned - cur, 0x00);
        glb_detail::BufferViewMeta bv;
        bv.byteOffset = static_cast<int>(bin.size());
        bv.byteLength = byte_length;
        bv.target     = target;
        bvs.push_back(bv);
        return static_cast<int>(bvs.size() - 1);
    };

    for (const auto& r : resolved) {
        const PrimitiveData& p = *r.src;
        materials.push_back(p.material);

        glb_detail::PrimitiveAccessors pa{};
        pa.uvs_acc = -1;
        pa.tangents_acc = -1;
        pa.wind_weights_acc = -1;

        double pmn[3] = { 1e300, 1e300, 1e300 };
        double pmx[3] = { -1e300, -1e300, -1e300 };
        for (uint32_t i = 0; i < r.vcount; ++i) {
            for (int c = 0; c < 3; ++c) {
                double v = static_cast<double>(p.positions[i * 3 + c]);
                if (v < pmn[c]) pmn[c] = v;
                if (v > pmx[c]) pmx[c] = v;
            }
        }

        int pos_bv = add_bv(static_cast<int>(r.vcount * 3 * sizeof(float)), 34962);
        glb_detail::append_bytes(bin, p.positions.data(), p.positions.size() * sizeof(float));
        {
            glb_detail::AccessorMeta a{};
            a.bufferView = pos_bv; a.byteOffset = 0; a.componentType = 5126;
            a.count = static_cast<int>(r.vcount); a.type = "VEC3";
            a.emit_minmax = true; a.minmax_count = 3;
            for (int k = 0; k < 3; ++k) { a.mn[k] = pmn[k]; a.mx[k] = pmx[k]; }
            accs.push_back(a);
        }
        pa.positions_acc = static_cast<int>(accs.size() - 1);

        int nrm_bv = add_bv(static_cast<int>(r.vcount * 3 * sizeof(float)), 34962);
        glb_detail::append_bytes(bin, p.normals.data(), p.normals.size() * sizeof(float));
        {
            glb_detail::AccessorMeta a{};
            a.bufferView = nrm_bv; a.byteOffset = 0; a.componentType = 5126;
            a.count = static_cast<int>(r.vcount); a.type = "VEC3";
            accs.push_back(a);
        }
        pa.normals_acc = static_cast<int>(accs.size() - 1);

        if (r.has_uvs) {
            int uv_bv = add_bv(static_cast<int>(r.vcount * 2 * sizeof(float)), 34962);
            glb_detail::append_bytes(bin, p.uvs.data(), p.uvs.size() * sizeof(float));
            glb_detail::AccessorMeta a{};
            a.bufferView = uv_bv; a.byteOffset = 0; a.componentType = 5126;
            a.count = static_cast<int>(r.vcount); a.type = "VEC2";
            accs.push_back(a);
            pa.uvs_acc = static_cast<int>(accs.size() - 1);
        }

        if (r.has_tangents) {
            int tan_bv = add_bv(static_cast<int>(r.vcount * 4 * sizeof(float)), 34962);
            glb_detail::append_bytes(bin, p.tangents.data(), p.tangents.size() * sizeof(float));
            glb_detail::AccessorMeta a{};
            a.bufferView = tan_bv; a.byteOffset = 0; a.componentType = 5126;
            a.count = static_cast<int>(r.vcount); a.type = "VEC4";
            accs.push_back(a);
            pa.tangents_acc = static_cast<int>(accs.size() - 1);
        }

        if (r.use_u32_indices) {
            int idx_bv = add_bv(static_cast<int>(r.icount * sizeof(uint32_t)), 34963);
            glb_detail::append_bytes(bin, p.indices_u32.data(), p.indices_u32.size() * sizeof(uint32_t));
            glb_detail::AccessorMeta a{};
            a.bufferView = idx_bv; a.byteOffset = 0; a.componentType = 5125;
            a.count = static_cast<int>(r.icount); a.type = "SCALAR";
            accs.push_back(a);
        }
        else {
            int idx_bv = add_bv(static_cast<int>(r.icount * sizeof(uint16_t)), 34963);
            glb_detail::append_bytes(bin, p.indices.data(), p.indices.size() * sizeof(uint16_t));
            glb_detail::AccessorMeta a{};
            a.bufferView = idx_bv; a.byteOffset = 0; a.componentType = 5123;
            a.count = static_cast<int>(r.icount); a.type = "SCALAR";
            accs.push_back(a);
        }
        pa.indices_acc = static_cast<int>(accs.size() - 1);

        if (r.has_wind) {
            int w_bv = add_bv(static_cast<int>(r.vcount * 4 * sizeof(uint8_t)), 34962);
            glb_detail::append_bytes(bin, p.wind_weights_packed.data(), p.wind_weights_packed.size());
            glb_detail::AccessorMeta a{};
            a.bufferView = w_bv; a.byteOffset = 0; a.componentType = 5121;
            a.count = static_cast<int>(r.vcount); a.type = "VEC4";
            a.normalized = true;
            accs.push_back(a);
            pa.wind_weights_acc = static_cast<int>(accs.size() - 1);
        }

        prim_accs.push_back(pa);
    }

    // C6 P4 — append image bufferViews (target=0, 4-byte aligned).
    std::vector<int> image_bv_indices;
    bool any_translucency = false;
    if (!images.empty()) {
        image_bv_indices.reserve(images.size());
        for (const auto& img : images) {
            int bv_idx = add_bv(static_cast<int>(img.png_bytes.size()), 0);
            glb_detail::append_bytes(bin, img.png_bytes.data(), img.png_bytes.size());
            image_bv_indices.push_back(bv_idx);
        }
        for (const auto& mat : materials) {
            if (mat.translucency_tex_index >= 0) { any_translucency = true; break; }
        }
    }

    const int total_buffer_bytes = static_cast<int>(bin.size());

    std::string json = glb_detail::build_json_multi_mesh(
        bvs, accs, prim_accs, materials, meshes_meta,
        total_buffer_bytes, any_wind, any_lod,
        static_cast<int>(images.size()),
        images.empty() ? nullptr : &image_bv_indices,
        any_translucency,
        any_collision);

    size_t json_pad = glb_detail::align_up(json.size()) - json.size();
    json.append(json_pad, ' ');

    size_t bin_pad = glb_detail::align_up(bin.size()) - bin.size();
    glb_detail::pad_bytes(bin, bin_pad, 0x00);

    const uint32_t magic   = 0x46546C67u;
    const uint32_t version = 2u;
    const uint32_t total_len =
        12 + 8 + static_cast<uint32_t>(json.size()) + 8 + static_cast<uint32_t>(bin.size());

    std::vector<uint8_t> out;
    out.reserve(total_len);

    auto put_u32 = [&](uint32_t v) {
        out.push_back(static_cast<uint8_t>(v & 0xFFu));
        out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
        out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
        out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
    };

    put_u32(magic);
    put_u32(version);
    put_u32(total_len);

    put_u32(static_cast<uint32_t>(json.size()));
    put_u32(0x4E4F534Au);
    out.insert(out.end(), json.begin(), json.end());

    put_u32(static_cast<uint32_t>(bin.size()));
    put_u32(0x004E4942u);
    out.insert(out.end(), bin.begin(), bin.end());

    std::ofstream f(output_path, std::ios::binary | std::ios::trunc);
    if (!f) { set_err("write_glb_multi_mesh: cannot open output file"); return false; }
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!f.good()) { set_err("write_glb_multi_mesh: write failed"); return false; }
    return true;
}

// C2 P7 wrapper — 3-arg overload preserved for backwards compatibility.
inline bool write_glb_multi_mesh(std::span<const MeshData> meshes,
                                 const std::string& output_path,
                                 std::string* err_out)
{
    return write_glb_multi_mesh(meshes, std::span<const ImageData>{}, output_path, err_out);
}

} // namespace treegen
