// TreeDescriptor — POD IR for the procedural tree generator. Consumed by C3+
// space colonization (P2), C4 branch geometry, C5 leaves, etc. POD only; all
// behaviour (parse/serialize/envelope evaluation) lives in free functions.
//
// Defaults match c3_oak.json (the canonical reference fixture); fields missing
// from input JSON fall back to these.
#pragma once

#include "vec3.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace treegen {

namespace json { struct value; }

enum class EnvelopeShape {
    OblateSpheroid,
    Conical,
    Weeping,
    Fastigiate,
    Fan,
};

// C5 P1 — placeholder for per-site geometry emission (resolved in P2). The
// site itself only stores the type tag; the emitter dispatches in P2.
enum class LeafGeometryType {
    SingleCard,
    ProceduralVeined,
    BentCard,
    BentCrossCluster,
    BranchStrip,
};

// C5 P1 — species shape (mesh template + UV atlas slot resolved in P2).
enum class LeafShape {
    OakLobed,
    PineNeedle,
    BirchSerrated,
    MapleStar,
};

struct TreeDescriptor {
    std::string species;
    uint64_t    seed                  = 0;
    float       height_m              = 10.0f;
    float       trunk_base_radius_m   = 0.25f;
    float       taper_exponent        = 2.0f;

    struct Envelope {
        EnvelopeShape shape            = EnvelopeShape::OblateSpheroid;
        float         width_m          = 6.0f;
        float         height_ratio     = 1.0f;
        float         top_height_ratio = 1.0f;
    } envelope;

    struct Branching {
        int   attractor_count          = 600;
        float kill_distance            = 0.5f;
        float growth_distance          = 0.3f;
        int   max_iterations           = 200;
        float branch_angle_jitter_deg  = 12.0f;
        int   min_attractors_to_split  = 3;
        float tropism_strength         = 1.0f;
        float angular_spread_split_deg = 30.0f;
        float crown_base_fraction      = 0.0f; // [0,1] — fraction of height_m below which no attractors are placed; trunk grows straight through this zone
        float min_branch_radius_m      = 0.008f; // post-solve floor: every non-root node radius >= this value
    } branching;

    float trunk_taper_rate    = 0.7f; // linear taper on pre-seeded trunk chain: r(z) = trunk_base * lerp(1.0, trunk_taper_rate, z/crown_base_z)
    float root_flare_factor   = 1.3f; // C12 — widen trunk base rings (1.0 = no flare); cubic ease-in over 3 rings

    // C11 — trunk structure params
    float crown_leader_fraction       = 0.5f;  // fraction of crown height where depth-0 leader persists
    float crown_onset_ramp_fraction   = 0.15f; // linear attractor density ramp at crown base
    float trunk_sway_amplitude_m      = -1.0f; // <0 → auto = 0.03 * height_m; lateral trunk sway

    struct Tropisms {
        float gravitropism = 0.05f;
        float phototropism = 0.0f;
        vec3  light_dir    = {0.0f, 0.0f, 1.0f};
    } tropisms;

    // C5 P1 — leaf-site generation parameters consumed by leaf_placement.cpp.
    // Defaults match the canonical oak fixture. Per-species overrides live in
    // each scenarios/*.json `leaves` block.
    struct Leaves {
        LeafGeometryType geometry_type        = LeafGeometryType::BentCrossCluster;
        LeafShape        shape                = LeafShape::OakLobed;
        float leaf_size_m                     = 0.12f;
        int   cluster_count_per_tip           = 6;
        int   leaf_min_branch_depth           = 3;
        float leaf_density_per_meter          = 8.0f;
        float leaf_depth_density_curve        = 1.0f;
        float leaf_phototropic_bias           = 0.0f;
        float leaf_bend_half_angle            = 0.392699081698724f; // pi/8 radians (22.5 deg)

        // C3-needle-strips: BranchStrip geometry params
        float strip_width_m                   = 0.4f;
        int   needle_count_per_strip          = 8;
        float needle_spacing_ratio            = 0.07f;
        float strip_droop_angle               = 0.15f;   // radians
        float strip_radius_threshold          = 0.02f;    // branch radius above which extra strips appear
    } leaves;
};

// LeafGeometryType / LeafShape <-> string. parse_* throws on unknown.
LeafGeometryType parse_leaf_geometry_type(std::string_view s);
const char*      leaf_geometry_type_to_string(LeafGeometryType g);
LeafShape        parse_leaf_shape(std::string_view s);
const char*      leaf_shape_to_string(LeafShape s);

// EnvelopeShape <-> string. Single source of truth for the 5 names; switch
// exhaustion makes mismatched parse/serialize compile-impossible.
// `parse_envelope_shape` throws std::runtime_error on unknown name.
EnvelopeShape parse_envelope_shape(std::string_view s);
const char*   envelope_shape_to_string(EnvelopeShape s);

// JSON round-trip. parse_tree_descriptor reads from an object value (the
// top-level scenario object for kind:"tree"); missing fields fall back to the
// POD defaults above. serialize_tree_descriptor emits pretty-printed JSON in
// deterministic key order — byte-identical across runs given identical input.
TreeDescriptor parse_tree_descriptor(const json::value& obj);
std::string    serialize_tree_descriptor(const TreeDescriptor& td);

} // namespace treegen
