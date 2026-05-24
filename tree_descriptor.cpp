#include "tree_descriptor.hpp"

#include "json_reader.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace treegen {

namespace {

// Single source of truth for EnvelopeShape <-> string. Adding a new shape
// requires editing the enum + this table; switch exhaustion in
// envelope_shape_to_string makes mismatched parse/serialize compile-impossible
// (warning-as-error under MSVC default for unhandled enum).
struct shape_entry {
    EnvelopeShape shape;
    const char*   name;
};

constexpr std::array<shape_entry, 5> k_shapes = {{
    { EnvelopeShape::OblateSpheroid, "oblate_spheroid" },
    { EnvelopeShape::Conical,        "conical"         },
    { EnvelopeShape::Weeping,        "weeping"         },
    { EnvelopeShape::Fastigiate,     "fastigiate"      },
    { EnvelopeShape::Fan,            "fan"             },
}};

struct leaf_geometry_entry { LeafGeometryType g; const char* name; };
constexpr std::array<leaf_geometry_entry, 5> k_leaf_geometries = {{
    { LeafGeometryType::SingleCard,        "single_card"         },
    { LeafGeometryType::ProceduralVeined,  "procedural_veined"   },
    { LeafGeometryType::BentCard,          "bent_card"           },
    { LeafGeometryType::BentCrossCluster,  "bent_cross_cluster"  },
    { LeafGeometryType::BranchStrip,       "branch_strip"        },
}};

struct leaf_shape_entry { LeafShape s; const char* name; };
constexpr std::array<leaf_shape_entry, 4> k_leaf_shapes = {{
    { LeafShape::OakLobed,       "oak_lobed"       },
    { LeafShape::PineNeedle,     "pine_needle"     },
    { LeafShape::BirchSerrated,  "birch_serrated"  },
    { LeafShape::MapleStar,      "maple_star"      },
}};

// %.9g produces 1-9 significant digits, enough to round-trip float through
// double without loss but stripping trailing zeros. Used in serialize_*.
void emit_float(std::string& out, float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    out += buf;
}

void emit_int(std::string& out, long long v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", v);
    out += buf;
}

void emit_u64(std::string& out, uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(v));
    out += buf;
}

void emit_string(std::string& out, const std::string& s) {
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:   out.push_back(c); break;
        }
    }
    out.push_back('"');
}

void indent(std::string& out, int n) { out.append(static_cast<size_t>(n) * 2, ' '); }

} // anonymous namespace

EnvelopeShape parse_envelope_shape(std::string_view s) {
    for (const auto& e : k_shapes) {
        if (s == e.name) return e.shape;
    }
    throw std::runtime_error(std::string("tree_descriptor: unknown envelope shape '") +
                             std::string(s) + "'");
}

const char* envelope_shape_to_string(EnvelopeShape s) {
    switch (s) {
        case EnvelopeShape::OblateSpheroid: return "oblate_spheroid";
        case EnvelopeShape::Conical:        return "conical";
        case EnvelopeShape::Weeping:        return "weeping";
        case EnvelopeShape::Fastigiate:     return "fastigiate";
        case EnvelopeShape::Fan:            return "fan";
    }
    // Unreachable: enum is closed; switch is exhaustive. Throw to satisfy
    // non-void return path under MSVC without a default branch that would
    // suppress -Wswitch-enum coverage warnings.
    throw std::runtime_error("tree_descriptor: envelope_shape_to_string out-of-range enum");
}

LeafGeometryType parse_leaf_geometry_type(std::string_view s) {
    for (const auto& e : k_leaf_geometries) if (s == e.name) return e.g;
    throw std::runtime_error(std::string("tree_descriptor: unknown leaf geometry_type '") + std::string(s) + "'");
}
const char* leaf_geometry_type_to_string(LeafGeometryType g) {
    switch (g) {
        case LeafGeometryType::SingleCard:        return "single_card";
        case LeafGeometryType::ProceduralVeined:  return "procedural_veined";
        case LeafGeometryType::BentCard:          return "bent_card";
        case LeafGeometryType::BentCrossCluster:  return "bent_cross_cluster";
        case LeafGeometryType::BranchStrip:       return "branch_strip";
    }
    throw std::runtime_error("tree_descriptor: leaf_geometry_type_to_string out-of-range enum");
}

LeafShape parse_leaf_shape(std::string_view s) {
    for (const auto& e : k_leaf_shapes) if (s == e.name) return e.s;
    throw std::runtime_error(std::string("tree_descriptor: unknown leaf shape '") + std::string(s) + "'");
}
const char* leaf_shape_to_string(LeafShape s) {
    switch (s) {
        case LeafShape::OakLobed:      return "oak_lobed";
        case LeafShape::PineNeedle:    return "pine_needle";
        case LeafShape::BirchSerrated: return "birch_serrated";
        case LeafShape::MapleStar:     return "maple_star";
    }
    throw std::runtime_error("tree_descriptor: leaf_shape_to_string out-of-range enum");
}

TreeDescriptor parse_tree_descriptor(const json::value& obj) {
    if (!obj.is_object()) {
        throw std::runtime_error("tree_descriptor: top-level value must be an object");
    }

    TreeDescriptor td;

    // Top-level scalars. species is required for a tree descriptor (defaults
    // could mask a typo'd JSON file).
    if (auto v = json::get_string(obj, "species")) td.species = *v;
    if (auto v = json::get_number(obj, "seed"))    td.seed = static_cast<uint64_t>(*v);
    if (auto v = json::get_number(obj, "height_m"))            td.height_m = static_cast<float>(*v);
    if (auto v = json::get_number(obj, "trunk_base_radius_m")) td.trunk_base_radius_m = static_cast<float>(*v);
    if (auto v = json::get_number(obj, "taper_exponent"))      td.taper_exponent = static_cast<float>(*v);

    if (const auto* env = json::get_object(obj, "envelope")) {
        if (auto v = json::get_string(*env, "shape"))            td.envelope.shape = parse_envelope_shape(*v);
        if (auto v = json::get_number(*env, "width_m"))          td.envelope.width_m = static_cast<float>(*v);
        if (auto v = json::get_number(*env, "height_ratio"))     td.envelope.height_ratio = static_cast<float>(*v);
        if (auto v = json::get_number(*env, "top_height_ratio")) td.envelope.top_height_ratio = static_cast<float>(*v);
    }

    if (const auto* br = json::get_object(obj, "branching")) {
        if (auto v = json::get_int   (*br, "attractor_count"))          td.branching.attractor_count = *v;
        if (auto v = json::get_number(*br, "kill_distance"))            td.branching.kill_distance = static_cast<float>(*v);
        if (auto v = json::get_number(*br, "growth_distance"))          td.branching.growth_distance = static_cast<float>(*v);
        if (auto v = json::get_int   (*br, "max_iterations"))           td.branching.max_iterations = *v;
        if (auto v = json::get_number(*br, "branch_angle_jitter_deg"))  td.branching.branch_angle_jitter_deg = static_cast<float>(*v);
        if (auto v = json::get_int   (*br, "min_attractors_to_split"))  td.branching.min_attractors_to_split = *v;
        if (auto v = json::get_number(*br, "tropism_strength"))         td.branching.tropism_strength = static_cast<float>(*v);
        if (auto v = json::get_number(*br, "angular_spread_split_deg")) td.branching.angular_spread_split_deg = static_cast<float>(*v);
        if (auto v = json::get_number(*br, "crown_base_fraction"))      td.branching.crown_base_fraction = static_cast<float>(*v);
        if (auto v = json::get_number(*br, "min_branch_radius_m"))     td.branching.min_branch_radius_m = static_cast<float>(*v);
    }

    if (auto v = json::get_number(obj, "trunk_taper_rate"))   td.trunk_taper_rate   = static_cast<float>(*v);
    if (auto v = json::get_number(obj, "root_flare_factor")) td.root_flare_factor = static_cast<float>(*v);

    // C11 — trunk structure params
    if (auto v = json::get_number(obj, "crown_leader_fraction"))     td.crown_leader_fraction     = static_cast<float>(*v);
    if (auto v = json::get_number(obj, "crown_onset_ramp_fraction")) td.crown_onset_ramp_fraction = static_cast<float>(*v);
    if (auto v = json::get_number(obj, "trunk_sway_amplitude_m"))    td.trunk_sway_amplitude_m    = static_cast<float>(*v);

    if (const auto* tr = json::get_object(obj, "tropisms")) {
        if (auto v = json::get_number(*tr, "gravitropism")) td.tropisms.gravitropism = static_cast<float>(*v);
        if (auto v = json::get_number(*tr, "phototropism")) td.tropisms.phototropism = static_cast<float>(*v);
        if (auto v = json::get_vec3  (*tr, "light_dir"))    td.tropisms.light_dir = *v;
    }

    if (const auto* lv = json::get_object(obj, "leaves")) {
        if (auto v = json::get_string(*lv, "geometry_type"))        td.leaves.geometry_type        = parse_leaf_geometry_type(*v);
        if (auto v = json::get_string(*lv, "shape"))                td.leaves.shape                = parse_leaf_shape(*v);
        if (auto v = json::get_number(*lv, "leaf_size_m"))          td.leaves.leaf_size_m          = static_cast<float>(*v);
        if (auto v = json::get_int   (*lv, "cluster_count_per_tip")) td.leaves.cluster_count_per_tip = *v;
        if (auto v = json::get_int   (*lv, "leaf_min_branch_depth"))    td.leaves.leaf_min_branch_depth    = *v;
        if (auto v = json::get_number(*lv, "leaf_density_per_meter"))   td.leaves.leaf_density_per_meter   = static_cast<float>(*v);
        if (auto v = json::get_number(*lv, "leaf_depth_density_curve")) td.leaves.leaf_depth_density_curve = static_cast<float>(*v);
        if (auto v = json::get_number(*lv, "leaf_phototropic_bias"))    td.leaves.leaf_phototropic_bias    = static_cast<float>(*v);
        if (auto v = json::get_number(*lv, "leaf_bend_half_angle"))    td.leaves.leaf_bend_half_angle    = static_cast<float>(*v);
        if (auto v = json::get_number(*lv, "strip_width_m"))              td.leaves.strip_width_m              = static_cast<float>(*v);
        if (auto v = json::get_int   (*lv, "needle_count_per_strip"))     td.leaves.needle_count_per_strip     = *v;
        if (auto v = json::get_number(*lv, "needle_spacing_ratio"))       td.leaves.needle_spacing_ratio       = static_cast<float>(*v);
        if (auto v = json::get_number(*lv, "strip_droop_angle"))          td.leaves.strip_droop_angle          = static_cast<float>(*v);
        if (auto v = json::get_number(*lv, "strip_radius_threshold"))     td.leaves.strip_radius_threshold     = static_cast<float>(*v);
    }

    return td;
}

std::string serialize_tree_descriptor(const TreeDescriptor& td) {
    // Pretty-printed, 2-space indent, keys in schema order. Deterministic byte-
    // for-byte across runs given identical input.
    std::string out;
    out.reserve(1024);

    out += "{\n";

    indent(out, 1); out += "\"kind\": \"tree\",\n";
    indent(out, 1); out += "\"species\": "; emit_string(out, td.species); out += ",\n";
    indent(out, 1); out += "\"seed\": "; emit_u64(out, td.seed); out += ",\n";
    indent(out, 1); out += "\"height_m\": "; emit_float(out, td.height_m); out += ",\n";
    indent(out, 1); out += "\"trunk_base_radius_m\": "; emit_float(out, td.trunk_base_radius_m); out += ",\n";
    indent(out, 1); out += "\"taper_exponent\": "; emit_float(out, td.taper_exponent); out += ",\n";
    indent(out, 1); out += "\"trunk_taper_rate\": "; emit_float(out, td.trunk_taper_rate); out += ",\n";
    indent(out, 1); out += "\"root_flare_factor\": "; emit_float(out, td.root_flare_factor); out += ",\n";
    indent(out, 1); out += "\"crown_leader_fraction\": "; emit_float(out, td.crown_leader_fraction); out += ",\n";
    indent(out, 1); out += "\"crown_onset_ramp_fraction\": "; emit_float(out, td.crown_onset_ramp_fraction); out += ",\n";
    indent(out, 1); out += "\"trunk_sway_amplitude_m\": "; emit_float(out, td.trunk_sway_amplitude_m); out += ",\n";

    indent(out, 1); out += "\"envelope\": {\n";
    indent(out, 2); out += "\"shape\": \""; out += envelope_shape_to_string(td.envelope.shape); out += "\",\n";
    indent(out, 2); out += "\"width_m\": "; emit_float(out, td.envelope.width_m); out += ",\n";
    indent(out, 2); out += "\"height_ratio\": "; emit_float(out, td.envelope.height_ratio); out += ",\n";
    indent(out, 2); out += "\"top_height_ratio\": "; emit_float(out, td.envelope.top_height_ratio); out += "\n";
    indent(out, 1); out += "},\n";

    indent(out, 1); out += "\"branching\": {\n";
    indent(out, 2); out += "\"attractor_count\": "; emit_int(out, td.branching.attractor_count); out += ",\n";
    indent(out, 2); out += "\"kill_distance\": "; emit_float(out, td.branching.kill_distance); out += ",\n";
    indent(out, 2); out += "\"growth_distance\": "; emit_float(out, td.branching.growth_distance); out += ",\n";
    indent(out, 2); out += "\"max_iterations\": "; emit_int(out, td.branching.max_iterations); out += ",\n";
    indent(out, 2); out += "\"branch_angle_jitter_deg\": "; emit_float(out, td.branching.branch_angle_jitter_deg); out += ",\n";
    indent(out, 2); out += "\"min_attractors_to_split\": "; emit_int(out, td.branching.min_attractors_to_split); out += ",\n";
    indent(out, 2); out += "\"tropism_strength\": "; emit_float(out, td.branching.tropism_strength); out += ",\n";
    indent(out, 2); out += "\"angular_spread_split_deg\": "; emit_float(out, td.branching.angular_spread_split_deg); out += ",\n";
    indent(out, 2); out += "\"crown_base_fraction\": "; emit_float(out, td.branching.crown_base_fraction); out += ",\n";
    indent(out, 2); out += "\"min_branch_radius_m\": "; emit_float(out, td.branching.min_branch_radius_m); out += "\n";
    indent(out, 1); out += "},\n";

    indent(out, 1); out += "\"tropisms\": {\n";
    indent(out, 2); out += "\"gravitropism\": "; emit_float(out, td.tropisms.gravitropism); out += ",\n";
    indent(out, 2); out += "\"phototropism\": "; emit_float(out, td.tropisms.phototropism); out += ",\n";
    indent(out, 2); out += "\"light_dir\": [";
    emit_float(out, td.tropisms.light_dir.x); out += ", ";
    emit_float(out, td.tropisms.light_dir.y); out += ", ";
    emit_float(out, td.tropisms.light_dir.z); out += "]\n";
    indent(out, 1); out += "},\n";

    indent(out, 1); out += "\"leaves\": {\n";
    indent(out, 2); out += "\"geometry_type\": \""; out += leaf_geometry_type_to_string(td.leaves.geometry_type);  out += "\",\n";
    indent(out, 2); out += "\"shape\": \"";         out += leaf_shape_to_string(td.leaves.shape);                  out += "\",\n";
    indent(out, 2); out += "\"leaf_size_m\": ";             emit_float(out, td.leaves.leaf_size_m);             out += ",\n";
    indent(out, 2); out += "\"cluster_count_per_tip\": ";   emit_int  (out, td.leaves.cluster_count_per_tip);   out += ",\n";
    indent(out, 2); out += "\"leaf_min_branch_depth\": ";    emit_int  (out, td.leaves.leaf_min_branch_depth);    out += ",\n";
    indent(out, 2); out += "\"leaf_density_per_meter\": ";   emit_float(out, td.leaves.leaf_density_per_meter);   out += ",\n";
    indent(out, 2); out += "\"leaf_depth_density_curve\": "; emit_float(out, td.leaves.leaf_depth_density_curve); out += ",\n";
    indent(out, 2); out += "\"leaf_phototropic_bias\": ";    emit_float(out, td.leaves.leaf_phototropic_bias);    out += ",\n";
    indent(out, 2); out += "\"leaf_bend_half_angle\": ";   emit_float(out, td.leaves.leaf_bend_half_angle);   out += ",\n";
    indent(out, 2); out += "\"strip_width_m\": ";             emit_float(out, td.leaves.strip_width_m);             out += ",\n";
    indent(out, 2); out += "\"needle_count_per_strip\": ";    emit_int  (out, td.leaves.needle_count_per_strip);    out += ",\n";
    indent(out, 2); out += "\"needle_spacing_ratio\": ";      emit_float(out, td.leaves.needle_spacing_ratio);      out += ",\n";
    indent(out, 2); out += "\"strip_droop_angle\": ";         emit_float(out, td.leaves.strip_droop_angle);         out += ",\n";
    indent(out, 2); out += "\"strip_radius_threshold\": ";    emit_float(out, td.leaves.strip_radius_threshold);    out += "\n";
    indent(out, 1); out += "}\n";

    out += "}\n";
    return out;
}

} // namespace treegen
