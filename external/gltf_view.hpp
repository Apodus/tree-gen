// Header-only GLB **inspection** utilities — JSON DOM, accessor walking,
// extract_all_primitives (+ extract_first_primitive wrapper). Decoupled from
// GPU upload so test code (and the rynx-treegen pipeline) can parse a .glb
// byte buffer without dragging in `mesh`, `mesh_collection`, `GPUTextures`,
// or the Vulkan device. The full upload path lives in gltf_loader.cpp and is
// a thin wrapper over the utilities here.
//
// All functions are `inline` so this is a true header-only lift — no .cpp
// counterpart, no engine logger dependency. STL only.
//
// ---------------------------------------------------------------------------
// `_RYNX_WIND` vendor extension (multi-tier procedural wind weights)
// ---------------------------------------------------------------------------
// Per-primitive: extensions._RYNX_WIND.weights = <accessor index>.
// The referenced accessor is VEC4 UNSIGNED_BYTE normalized=true; each vertex
// carries 4 independent tier influences (trunk/branch/twig/leaf) in [0,1].
// The components are NOT required to sum to 255 — they are per-tier
// independent influence factors. Trees populate; other meshes leave empty.
//
// ---------------------------------------------------------------------------
// `_RYNX_LOD` vendor extension (per-mesh LOD chain metadata) — C2 P7
// ---------------------------------------------------------------------------
// Per-mesh (top-level meshes[i]): extensions._RYNX_LOD = { lod_index,
// lod_max_distance_m }. A multi-mesh GLB with N top-level `meshes[]` entries
// is the canonical LOD-chain shape: meshes[0] = LOD 0 (highest detail),
// meshes[1] = LOD 1, etc. Each mesh's `lod_max_distance_m` is the runtime
// selection upper bound (distance at which this LOD stops being used).
// Single-mesh GLBs leave the extension absent — engine defaults to
// lod_index = 0, lod_max_distance_m = 0 (always-used).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rynx::graphics::gltf_view {

// ============================================================================
// Minimal JSON parser — supports only what glTF 2.0 requires. Integers are
// parsed as double; the glTF spec caps integer fields at 2^32 which fits in
// IEEE-754 double without loss.
// ============================================================================

struct jval {
    enum class kind_t { null_t, bool_t, num_t, str_t, arr_t, obj_t };
    kind_t kind = kind_t::null_t;
    bool b = false;
    double n = 0.0;
    std::string s;
    std::vector<jval> a;
    std::vector<std::pair<std::string, jval>> o;

    const jval* find(const char* key) const {
        if (kind != kind_t::obj_t) return nullptr;
        for (auto& kv : o) if (kv.first == key) return &kv.second;
        return nullptr;
    }
    int as_int(int def = 0) const { return (kind == kind_t::num_t) ? int(n) : def; }
    const std::string& as_str() const {
        static const std::string empty;
        return (kind == kind_t::str_t) ? s : empty;
    }
    const std::vector<jval>& as_arr() const {
        static const std::vector<jval> empty;
        return (kind == kind_t::arr_t) ? a : empty;
    }
};

class jparser {
    const char* p;
    const char* e;
    bool ok = true;

    void skip_ws() {
        while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    jval parse_string() {
        jval v; v.kind = jval::kind_t::str_t;
        if (p >= e || *p != '"') { ok = false; return v; }
        ++p;
        while (p < e) {
            char c = *p++;
            if (c == '"') return v;
            if (c == '\\' && p < e) {
                char esc = *p++;
                switch (esc) {
                    case '"': v.s.push_back('"'); break;
                    case '\\': v.s.push_back('\\'); break;
                    case '/': v.s.push_back('/'); break;
                    case 'n': v.s.push_back('\n'); break;
                    case 't': v.s.push_back('\t'); break;
                    case 'r': v.s.push_back('\r'); break;
                    case 'b': v.s.push_back('\b'); break;
                    case 'f': v.s.push_back('\f'); break;
                    case 'u':
                        if (p + 4 <= e) p += 4;
                        v.s.push_back('?');
                        break;
                    default: v.s.push_back(esc); break;
                }
            }
            else {
                v.s.push_back(c);
            }
        }
        ok = false;
        return v;
    }

    jval parse_number() {
        jval v; v.kind = jval::kind_t::num_t;
        const char* start = p;
        if (p < e && (*p == '-' || *p == '+')) ++p;
        while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) ++p;
        v.n = std::strtod(std::string(start, p - start).c_str(), nullptr);
        return v;
    }

    jval parse_literal() {
        if (p + 4 <= e && std::memcmp(p, "true", 4) == 0) { p += 4; jval v; v.kind = jval::kind_t::bool_t; v.b = true; return v; }
        if (p + 5 <= e && std::memcmp(p, "false", 5) == 0) { p += 5; jval v; v.kind = jval::kind_t::bool_t; v.b = false; return v; }
        if (p + 4 <= e && std::memcmp(p, "null", 4) == 0) { p += 4; return jval(); }
        ok = false;
        return jval();
    }

    jval parse_array() {
        jval v; v.kind = jval::kind_t::arr_t;
        ++p;
        skip_ws();
        if (p < e && *p == ']') { ++p; return v; }
        while (p < e && ok) {
            v.a.push_back(parse_value());
            skip_ws();
            if (p < e && *p == ',') { ++p; skip_ws(); continue; }
            if (p < e && *p == ']') { ++p; return v; }
            ok = false;
            break;
        }
        return v;
    }

    jval parse_object() {
        jval v; v.kind = jval::kind_t::obj_t;
        ++p;
        skip_ws();
        if (p < e && *p == '}') { ++p; return v; }
        while (p < e && ok) {
            skip_ws();
            jval key = parse_string();
            skip_ws();
            if (p >= e || *p != ':') { ok = false; break; }
            ++p;
            jval val = parse_value();
            v.o.emplace_back(std::move(key.s), std::move(val));
            skip_ws();
            if (p < e && *p == ',') { ++p; continue; }
            if (p < e && *p == '}') { ++p; return v; }
            ok = false;
            break;
        }
        return v;
    }

public:
    jparser(const char* beg, size_t len) : p(beg), e(beg + len) {}

    jval parse_value() {
        skip_ws();
        if (p >= e) { ok = false; return jval(); }
        char c = *p;
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f' || c == 'n') return parse_literal();
        return parse_number();
    }

    bool good() const { return ok; }
};

// ============================================================================
// Accessor / bufferView utilities
// ============================================================================

inline constexpr uint32_t GLTF_BYTE           = 5120;
inline constexpr uint32_t GLTF_UNSIGNED_BYTE  = 5121;
inline constexpr uint32_t GLTF_SHORT          = 5122;
inline constexpr uint32_t GLTF_UNSIGNED_SHORT = 5123;
inline constexpr uint32_t GLTF_UNSIGNED_INT   = 5125;
inline constexpr uint32_t GLTF_FLOAT          = 5126;

inline int gltf_type_components(const std::string& t) {
    if (t == "SCALAR") return 1;
    if (t == "VEC2")   return 2;
    if (t == "VEC3")   return 3;
    if (t == "VEC4")   return 4;
    return 0;
}

inline int gltf_component_bytes(uint32_t ct) {
    switch (ct) {
        case GLTF_BYTE:
        case GLTF_UNSIGNED_BYTE:  return 1;
        case GLTF_SHORT:
        case GLTF_UNSIGNED_SHORT: return 2;
        case GLTF_UNSIGNED_INT:
        case GLTF_FLOAT:          return 4;
        default: return 0;
    }
}

struct accessor {
    uint32_t component_type = 0;
    uint32_t count = 0;
    uint32_t components = 0;
    uint32_t bv_byte_offset = 0;
    uint32_t bv_byte_stride = 0;
    uint32_t bv_byte_length = 0;
    uint32_t access_byte_offset = 0;
    uint32_t buffer_index = 0;

    uint32_t element_bytes() const { return components * gltf_component_bytes(component_type); }
    uint32_t stride() const { return bv_byte_stride ? bv_byte_stride : element_bytes(); }
    uint32_t first_byte() const { return bv_byte_offset + access_byte_offset; }
};

// Returns false if the accessor index is out of range or references malformed data.
inline bool resolve_accessor(const jval& root, int accessor_index, accessor& out) {
    auto* accessors = root.find("accessors");
    auto* buffer_views = root.find("bufferViews");
    if (!accessors || !buffer_views) return false;
    if (accessor_index < 0 || accessor_index >= int(accessors->as_arr().size())) return false;
    const jval& a = accessors->as_arr()[accessor_index];

    out.component_type = uint32_t(a.find("componentType") ? a.find("componentType")->as_int() : 0);
    out.count = uint32_t(a.find("count") ? a.find("count")->as_int() : 0);
    out.components = uint32_t(gltf_type_components(a.find("type") ? a.find("type")->as_str() : std::string()));
    out.access_byte_offset = uint32_t(a.find("byteOffset") ? a.find("byteOffset")->as_int() : 0);

    int bv_idx = a.find("bufferView") ? a.find("bufferView")->as_int(-1) : -1;
    if (bv_idx < 0 || bv_idx >= int(buffer_views->as_arr().size())) return false;
    const jval& bv = buffer_views->as_arr()[bv_idx];

    out.bv_byte_offset = uint32_t(bv.find("byteOffset") ? bv.find("byteOffset")->as_int() : 0);
    out.bv_byte_stride = uint32_t(bv.find("byteStride") ? bv.find("byteStride")->as_int() : 0);
    out.bv_byte_length = uint32_t(bv.find("byteLength") ? bv.find("byteLength")->as_int() : 0);
    out.buffer_index = uint32_t(bv.find("buffer") ? bv.find("buffer")->as_int() : 0);

    return out.count > 0 && out.components > 0 && gltf_component_bytes(out.component_type) > 0;
}

// Read one component as float, handling the 6 glTF numeric types.
inline float read_component_as_float(const char* src, uint32_t ct) {
    switch (ct) {
        case GLTF_FLOAT: {
            float v;
            std::memcpy(&v, src, 4);
            return v;
        }
        case GLTF_UNSIGNED_INT: {
            uint32_t v;
            std::memcpy(&v, src, 4);
            return float(v);
        }
        case GLTF_UNSIGNED_SHORT: {
            uint16_t v;
            std::memcpy(&v, src, 2);
            return float(v);
        }
        case GLTF_SHORT: {
            int16_t v;
            std::memcpy(&v, src, 2);
            return float(v);
        }
        case GLTF_UNSIGNED_BYTE:
            return float(uint8_t(*src));
        case GLTF_BYTE:
            return float(int8_t(*src));
    }
    return 0.0f;
}

inline uint32_t read_index(const char* src, uint32_t ct) {
    switch (ct) {
        case GLTF_UNSIGNED_INT: {
            uint32_t v;
            std::memcpy(&v, src, 4);
            return v;
        }
        case GLTF_UNSIGNED_SHORT: {
            uint16_t v;
            std::memcpy(&v, src, 2);
            return v;
        }
        case GLTF_UNSIGNED_BYTE:
            return uint32_t(uint8_t(*src));
    }
    return 0;
}

// ============================================================================
// GLB parsing
// ============================================================================

struct glb_view {
    const char* json_begin = nullptr;
    size_t json_len = 0;
    const char* bin_begin = nullptr;
    size_t bin_len = 0;
    bool valid = false;
};

inline glb_view parse_glb_header(std::span<const char> data) {
    glb_view v{};
    if (data.size() < 12) return v;
    uint32_t magic, version, total_len;
    std::memcpy(&magic, data.data() + 0, 4);
    std::memcpy(&version, data.data() + 4, 4);
    std::memcpy(&total_len, data.data() + 8, 4);
    if (magic != 0x46546C67u || version != 2u) return v;
    if (total_len > data.size()) return v;

    size_t pos = 12;
    while (pos + 8 <= total_len) {
        uint32_t chunk_len, chunk_type;
        std::memcpy(&chunk_len, data.data() + pos + 0, 4);
        std::memcpy(&chunk_type, data.data() + pos + 4, 4);
        pos += 8;
        if (pos + chunk_len > total_len) return v;
        if (chunk_type == 0x4E4F534Au) {          // "JSON"
            v.json_begin = data.data() + pos;
            v.json_len = chunk_len;
        }
        else if (chunk_type == 0x004E4942u) {     // "BIN\0"
            v.bin_begin = data.data() + pos;
            v.bin_len = chunk_len;
        }
        pos += chunk_len;
    }
    v.valid = (v.json_begin != nullptr);
    return v;
}

// ============================================================================
// Geometry extraction
// ============================================================================

struct cpu_mesh {
    std::vector<float> positions;   // xyz packed
    std::vector<float> normals;     // xyz packed (may be empty)
    std::vector<float> texcoords;   // uv packed (may be empty)
    std::vector<uint32_t> indices;

    // C2 P2: per-primitive material binding. -1 = no material assigned.
    int material_index = -1;

    // C1 P4: per-vertex tangent (VEC4 FLOAT, w=handedness). 4 floats/vert;
    // empty when the primitive doesn't carry TANGENT.
    std::vector<float> tangents;

    // C2 P2: `_RYNX_WIND` per-vertex 4-tier influences (UNSIGNED_BYTE VEC4
    // normalized). 4 bytes per vertex; empty when the primitive doesn't
    // carry the extension.
    std::vector<uint8_t> wind_weights_packed;
};

// Per-primitive material descriptor harvested from glTF 2.0 PBR material
// node. P2 reads + stashes; P3 wires into material_library SSBO.
struct pbr_material_desc {
    int  base_color_tex          = -1;  // pbrMetallicRoughness.baseColorTexture.index
    int  metallic_roughness_tex  = -1;  // pbrMetallicRoughness.metallicRoughnessTexture.index
    int  normal_tex              = -1;  // normalTexture.index
    int  occlusion_tex           = -1;  // occlusionTexture.index
    int  translucency_tex        = -1;  // _RYNX_LEAF_TRANSLUCENCY.translucencyTexture.index
    // alphaMode: 0=OPAQUE, 1=MASK, 2=BLEND. Spec default OPAQUE.
    int  alpha_mode       = 0;
    float alpha_cutoff    = 0.5f;
    bool  double_sided    = false;
    float base_color_factor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
};

inline bool read_vec3_accessor(const accessor& acc, const char* bin, size_t bin_len, std::vector<float>& out) {
    if (acc.components != 3) return false;
    if (acc.first_byte() + acc.stride() * (acc.count - 1) + acc.element_bytes() > bin_len) return false;
    out.resize(size_t(acc.count) * 3);
    uint32_t comp_bytes = gltf_component_bytes(acc.component_type);
    for (uint32_t i = 0; i < acc.count; ++i) {
        const char* e = bin + acc.first_byte() + i * acc.stride();
        out[i * 3 + 0] = read_component_as_float(e + comp_bytes * 0, acc.component_type);
        out[i * 3 + 1] = read_component_as_float(e + comp_bytes * 1, acc.component_type);
        out[i * 3 + 2] = read_component_as_float(e + comp_bytes * 2, acc.component_type);
    }
    return true;
}

inline bool read_vec4_accessor(const accessor& acc, const char* bin, size_t bin_len, std::vector<float>& out) {
    if (acc.components != 4) return false;
    if (acc.first_byte() + acc.stride() * (acc.count - 1) + acc.element_bytes() > bin_len) return false;
    out.resize(size_t(acc.count) * 4);
    uint32_t comp_bytes = gltf_component_bytes(acc.component_type);
    for (uint32_t i = 0; i < acc.count; ++i) {
        const char* e = bin + acc.first_byte() + i * acc.stride();
        out[i * 4 + 0] = read_component_as_float(e + comp_bytes * 0, acc.component_type);
        out[i * 4 + 1] = read_component_as_float(e + comp_bytes * 1, acc.component_type);
        out[i * 4 + 2] = read_component_as_float(e + comp_bytes * 2, acc.component_type);
        out[i * 4 + 3] = read_component_as_float(e + comp_bytes * 3, acc.component_type);
    }
    return true;
}

inline bool read_vec2_accessor(const accessor& acc, const char* bin, size_t bin_len, std::vector<float>& out) {
    if (acc.components != 2) return false;
    if (acc.first_byte() + acc.stride() * (acc.count - 1) + acc.element_bytes() > bin_len) return false;
    out.resize(size_t(acc.count) * 2);
    uint32_t comp_bytes = gltf_component_bytes(acc.component_type);
    for (uint32_t i = 0; i < acc.count; ++i) {
        const char* e = bin + acc.first_byte() + i * acc.stride();
        out[i * 2 + 0] = read_component_as_float(e + comp_bytes * 0, acc.component_type);
        out[i * 2 + 1] = read_component_as_float(e + comp_bytes * 1, acc.component_type);
    }
    return true;
}

// Read a VEC4 UNSIGNED_BYTE accessor into `out` as 4*count packed bytes.
// Used for the `_RYNX_WIND` extension: per-vertex 4 tier influences. The
// `normalized` accessor flag is metadata for the GPU sampler — bytes are
// copied verbatim (caller normalizes if needed).
inline bool read_vec4_byte_normalized_accessor(const accessor& acc, const char* bin, size_t bin_len, std::vector<uint8_t>& out) {
    if (acc.components != 4) return false;
    if (acc.component_type != GLTF_UNSIGNED_BYTE) return false;
    if (acc.first_byte() + acc.stride() * (acc.count - 1) + acc.element_bytes() > bin_len) return false;
    out.resize(size_t(acc.count) * 4);
    for (uint32_t i = 0; i < acc.count; ++i) {
        const char* e = bin + acc.first_byte() + i * acc.stride();
        out[i * 4 + 0] = uint8_t(e[0]);
        out[i * 4 + 1] = uint8_t(e[1]);
        out[i * 4 + 2] = uint8_t(e[2]);
        out[i * 4 + 3] = uint8_t(e[3]);
    }
    return true;
}

inline bool read_index_accessor(const accessor& acc, const char* bin, size_t bin_len, std::vector<uint32_t>& out) {
    if (acc.components != 1) return false;
    uint32_t comp_bytes = gltf_component_bytes(acc.component_type);
    if (comp_bytes == 0) return false;
    if (acc.first_byte() + acc.stride() * (acc.count - 1) + acc.element_bytes() > bin_len) return false;
    out.resize(acc.count);
    for (uint32_t i = 0; i < acc.count; ++i) {
        const char* e = bin + acc.first_byte() + i * acc.stride();
        out[i] = read_index(e, acc.component_type);
    }
    return true;
}

// Read one primitive into a cpu_mesh. Helper for extract_all_primitives.
inline bool extract_one_primitive(const jval& root, const jval& prim, const glb_view& glb,
                                  cpu_mesh& out, std::string* err_out) {
    auto set_err = [&](const char* m) {
        if (err_out) *err_out = m;
    };

    int mode = prim.find("mode") ? prim.find("mode")->as_int(4) : 4;
    if (mode != 4) {
        set_err("primitive mode not supported (need TRIANGLES)");
        return false;
    }

    auto* attrs = prim.find("attributes");
    if (!attrs) {
        set_err("primitive has no attributes");
        return false;
    }

    int pos_idx = attrs->find("POSITION")   ? attrs->find("POSITION")->as_int(-1)   : -1;
    int nrm_idx = attrs->find("NORMAL")     ? attrs->find("NORMAL")->as_int(-1)     : -1;
    int uv_idx  = attrs->find("TEXCOORD_0") ? attrs->find("TEXCOORD_0")->as_int(-1) : -1;
    int idx_idx = prim.find("indices")     ? prim.find("indices")->as_int(-1)     : -1;

    if (pos_idx < 0 || idx_idx < 0) {
        set_err("missing POSITION or indices");
        return false;
    }

    accessor pos_acc, nrm_acc, uv_acc, idx_acc;
    if (!resolve_accessor(root, pos_idx, pos_acc) || pos_acc.component_type != GLTF_FLOAT) {
        set_err("unsupported POSITION accessor");
        return false;
    }
    if (!resolve_accessor(root, idx_idx, idx_acc)) {
        set_err("unsupported index accessor");
        return false;
    }
    if (idx_acc.component_type != GLTF_UNSIGNED_BYTE &&
        idx_acc.component_type != GLTF_UNSIGNED_SHORT &&
        idx_acc.component_type != GLTF_UNSIGNED_INT) {
        set_err("index componentType unsupported");
        return false;
    }
    if (pos_acc.buffer_index != 0 || idx_acc.buffer_index != 0) {
        set_err("multi-buffer glTF not supported");
        return false;
    }

    out.positions.clear();
    out.normals.clear();
    out.texcoords.clear();
    out.indices.clear();
    out.wind_weights_packed.clear();
    out.material_index = -1;

    if (!read_vec3_accessor(pos_acc, glb.bin_begin, glb.bin_len, out.positions)) {
        set_err("POSITION read failed");
        return false;
    }
    if (!read_index_accessor(idx_acc, glb.bin_begin, glb.bin_len, out.indices)) {
        set_err("index read failed");
        return false;
    }

    if (nrm_idx >= 0) {
        if (resolve_accessor(root, nrm_idx, nrm_acc) && nrm_acc.component_type == GLTF_FLOAT) {
            (void)read_vec3_accessor(nrm_acc, glb.bin_begin, glb.bin_len, out.normals);
            if (out.normals.size() != out.positions.size()) {
                out.normals.clear();
            }
        }
    }

    if (uv_idx >= 0) {
        if (resolve_accessor(root, uv_idx, uv_acc) && uv_acc.component_type == GLTF_FLOAT) {
            (void)read_vec2_accessor(uv_acc, glb.bin_begin, glb.bin_len, out.texcoords);
        }
    }

    // C1 P4: TANGENT (VEC4 FLOAT, w=handedness).
    int tan_idx = attrs->find("TANGENT") ? attrs->find("TANGENT")->as_int(-1) : -1;
    if (tan_idx >= 0) {
        accessor tan_acc;
        if (resolve_accessor(root, tan_idx, tan_acc) && tan_acc.component_type == GLTF_FLOAT) {
            (void)read_vec4_accessor(tan_acc, glb.bin_begin, glb.bin_len, out.tangents);
            if (out.tangents.size() != (out.positions.size() / 3) * 4)
                out.tangents.clear();
        }
    }

    // C2 P2: per-primitive material binding.
    if (auto* mat = prim.find("material")) {
        out.material_index = mat->as_int(-1);
    }

    // C2 P2: `_RYNX_WIND` vendor extension. Accessor must be VEC4 UNSIGNED_BYTE.
    if (auto* exts = prim.find("extensions")) {
        if (auto* wind = exts->find("_RYNX_WIND")) {
            if (auto* w_acc = wind->find("weights")) {
                int wi = w_acc->as_int(-1);
                if (wi >= 0) {
                    accessor wacc;
                    if (resolve_accessor(root, wi, wacc)) {
                        (void)read_vec4_byte_normalized_accessor(wacc, glb.bin_begin, glb.bin_len, out.wind_weights_packed);
                        // Vertex-count mismatch is a producer bug; drop the payload.
                        const size_t vcount = out.positions.size() / 3;
                        if (out.wind_weights_packed.size() != vcount * 4) {
                            out.wind_weights_packed.clear();
                        }
                    }
                }
            }
        }
    }

    return true;
}

// Top-level: parse GLB header + JSON, iterate meshes[mesh_index].primitives,
// read each into a cpu_mesh appended to `out_meshes`. On error returns false
// (partial fill possible — caller treats `false` as terminal). No logger dep.
//
// `mesh_index` defaults to 0 for back-compat. Multi-mesh GLBs (e.g. LOD
// chains via `_RYNX_LOD`) call with mesh_index = 0..count_meshes()-1.
inline bool extract_all_primitives(std::span<const char> bytes,
                                   std::vector<cpu_mesh>& out_meshes,
                                   std::string* err_out,
                                   int mesh_index = 0) {
    auto set_err = [&](const char* m) {
        if (err_out) *err_out = m;
    };

    glb_view glb = parse_glb_header(bytes);
    if (!glb.valid || !glb.bin_begin) {
        set_err("invalid .glb container");
        return false;
    }

    jparser parser(glb.json_begin, glb.json_len);
    jval root = parser.parse_value();
    if (!parser.good() || root.kind != jval::kind_t::obj_t) {
        set_err("JSON parse failed");
        return false;
    }

    auto* meshes_node = root.find("meshes");
    if (!meshes_node || meshes_node->as_arr().empty()) {
        set_err("no meshes");
        return false;
    }
    if (mesh_index < 0 || mesh_index >= int(meshes_node->as_arr().size())) {
        set_err("mesh_index out of range");
        return false;
    }
    const jval& mesh_obj = meshes_node->as_arr()[mesh_index];
    auto* prims = mesh_obj.find("primitives");
    if (!prims || prims->as_arr().empty()) {
        set_err("no primitives in selected mesh");
        return false;
    }

    out_meshes.clear();
    out_meshes.reserve(prims->as_arr().size());
    for (const jval& prim : prims->as_arr()) {
        cpu_mesh cm;
        if (!extract_one_primitive(root, prim, glb, cm, err_out)) {
            return false;
        }
        out_meshes.push_back(std::move(cm));
    }
    return true;
}

// C2 P7 — count top-level meshes in a GLB. Returns 0 on parse failure or
// when the GLB has no meshes node.
inline int count_meshes(std::span<const char> bytes) {
    glb_view glb = parse_glb_header(bytes);
    if (!glb.valid) return 0;
    jparser parser(glb.json_begin, glb.json_len);
    jval root = parser.parse_value();
    if (!parser.good() || root.kind != jval::kind_t::obj_t) return 0;
    auto* meshes_node = root.find("meshes");
    if (!meshes_node) return 0;
    return int(meshes_node->as_arr().size());
}

// C2 P7 — read per-mesh `_RYNX_LOD` extension. `mesh_obj` is the JSON node
// from root.meshes[mesh_index]. Returns true if the extension is present and
// readable; false otherwise (caller assigns defaults).
inline bool read_rynx_lod_extension(const jval& mesh_obj,
                                    int& out_lod_index,
                                    float& out_max_distance_m,
                                    float& out_screen_height_px) {
    out_screen_height_px = 0.0f;
    auto* exts = mesh_obj.find("extensions");
    if (!exts) return false;
    auto* lod = exts->find("_RYNX_LOD");
    if (!lod) return false;
    bool any = false;
    if (auto* li = lod->find("lod_index")) {
        out_lod_index = li->as_int(0);
        any = true;
    }
    if (auto* md = lod->find("lod_max_distance_m")) {
        if (md->kind == jval::kind_t::num_t) {
            out_max_distance_m = float(md->n);
            any = true;
        }
    }
    if (auto* sh = lod->find("lod_screen_height_px")) {
        if (sh->kind == jval::kind_t::num_t) {
            out_screen_height_px = float(sh->n);
        }
    }
    return any;
}

// Thin wrapper: parse all primitives, return the first. Preserves the C1
// test surface; new callers should use extract_all_primitives.
inline bool extract_first_primitive(std::span<const char> bytes, cpu_mesh& out, std::string* err_out) {
    std::vector<cpu_mesh> all;
    if (!extract_all_primitives(bytes, all, err_out)) return false;
    if (all.empty()) {
        if (err_out) *err_out = "no primitives";
        return false;
    }
    out = std::move(all[0]);
    return true;
}

// C2 P2: glTF 2.0 PBR material extraction (single material node).
// `mat_idx` indexes into root.materials[]. Returns false on out-of-range.
inline bool extract_pbr_material(const jval& root, int mat_idx,
                                 pbr_material_desc& out, std::string* err_out) {
    auto set_err = [&](const char* m) {
        if (err_out) *err_out = m;
    };

    auto* materials = root.find("materials");
    if (!materials || mat_idx < 0 || mat_idx >= int(materials->as_arr().size())) {
        set_err("material index out of range");
        return false;
    }
    const jval& m = materials->as_arr()[mat_idx];

    if (auto* pbr = m.find("pbrMetallicRoughness")) {
        if (auto* bcf = pbr->find("baseColorFactor"); bcf && bcf->kind == jval::kind_t::arr_t) {
            for (int i = 0; i < 4 && i < int(bcf->as_arr().size()); ++i) {
                const jval& v = bcf->as_arr()[i];
                if (v.kind == jval::kind_t::num_t) out.base_color_factor[i] = float(v.n);
            }
        }
        if (auto* bct = pbr->find("baseColorTexture")) {
            out.base_color_tex = bct->find("index") ? bct->find("index")->as_int(-1) : -1;
        }
        if (auto* mrt = pbr->find("metallicRoughnessTexture")) {
            out.metallic_roughness_tex = mrt->find("index") ? mrt->find("index")->as_int(-1) : -1;
        }
    }

    if (auto* nt = m.find("normalTexture")) {
        out.normal_tex = nt->find("index") ? nt->find("index")->as_int(-1) : -1;
    }
    if (auto* ot = m.find("occlusionTexture")) {
        out.occlusion_tex = ot->find("index") ? ot->find("index")->as_int(-1) : -1;
    }

    if (auto* am = m.find("alphaMode")) {
        const std::string& s = am->as_str();
        if      (s == "MASK")  out.alpha_mode = 1;
        else if (s == "BLEND") out.alpha_mode = 2;
        else                   out.alpha_mode = 0;
    }
    if (auto* ac = m.find("alphaCutoff")) {
        if (ac->kind == jval::kind_t::num_t) out.alpha_cutoff = float(ac->n);
    }
    if (auto* ds = m.find("doubleSided")) {
        if (ds->kind == jval::kind_t::bool_t) out.double_sided = ds->b;
    }
    // _RYNX_LEAF_TRANSLUCENCY extension — per-material translucency texture.
    if (auto* ext = m.find("extensions")) {
        if (auto* lt = ext->find("_RYNX_LEAF_TRANSLUCENCY")) {
            if (auto* tt = lt->find("translucencyTexture")) {
                out.translucency_tex = tt->find("index") ? tt->find("index")->as_int(-1) : -1;
            }
        }
    }
    return true;
}

} // namespace rynx::graphics::gltf_view
