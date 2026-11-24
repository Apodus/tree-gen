// /fp:precise required — every angle/sin/cos contributes to GLB byte-hash
// determinism. See treegen.sharpmake.cs.
#include "leaf_geometry.hpp"

#include "det_rng.hpp"
#include "leaf_atlas.hpp"       // tile_uv — single source of truth atlas UV mapping
#include "leaf_shapes.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace treegen {

namespace {

constexpr uint64_t k_stream_leaf_geom = 0xC5'02'01ULL;

// Build orthonormal basis (right, forward) in the plane perpendicular to
// `up` (assumed unit). Right + forward span the leaf card plane.
void make_card_basis(vec3 up, vec3& out_right, vec3& out_forward) {
    const vec3 ref = std::abs(up.z) < 0.9f ? vec3{0.0f, 0.0f, 1.0f}
                                            : vec3{1.0f, 0.0f, 0.0f};
    out_right   = normalized(cross(up, ref));
    out_forward = normalized(cross(up, out_right));
}

// Rotate a vector `v` around unit axis `axis` by angle `a` (Rodrigues).
vec3 rotate_axis(vec3 v, vec3 axis, float a) {
    const float c = std::cos(a);
    const float s = std::sin(a);
    return v * c + cross(axis, v) * s + axis * (dot(axis, v) * (1.0f - c));
}

// Push xyz to a float vector.
inline void push_xyz(std::vector<float>& v, vec3 p) {
    v.push_back(p.x);
    v.push_back(p.y);
    v.push_back(p.z);
}

inline void push_uv(std::vector<float>& v, float u, float w) {
    v.push_back(u);
    v.push_back(w);
}

inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

inline uint8_t quantize_wind(float w) {
    return static_cast<uint8_t>(clamp01(w) * 255.0f + 0.5f);
}

// Per-vert metadata (normal / wind / material / tangent). All verts of one
// leaf share the same normal (= site.normal), tangent (= card right vector),
// material id, and depth-derived wind weights.
//
// C1 P3 — wind weights use the same 4-channel formula as bark (wind_weights.cpp)
// evaluated at branch_depth_fraction, with rooted_factor = 1 (leaves are above
// ground). Leaves on thick branches get trunk/branch influence; terminal-twig
// leaves stay near pure leaf tier.
void push_leaf_vert_meta(LeafMeshOutput& out, vec3 normal, vec3 tangent_dir,
                         float branch_depth_fraction, int branch_id, int n_verts,
                         float tangent_w = 1.0f) {
    // C8-wind P1 — host bone = site.branch_id (nearest skeleton node). Negative
    // branch_id (unbound site) clamps to 0 (root) so the index is always valid.
    const uint16_t bone = static_cast<uint16_t>(branch_id < 0 ? 0 : branch_id);
    const float d  = clamp01(branch_depth_fraction);
    const float od = 1.0f - d;
    const uint8_t trunk_b  = quantize_wind(od * od);            // (1-d)^2
    const uint8_t branch_b = quantize_wind(4.0f * d * od);      // 4d(1-d)
    const uint8_t twig_b   = quantize_wind(4.0f * d * d * od);  // 4d^2(1-d)
    const uint8_t leaf_b   = quantize_wind(d * d * d);           // d^3

    for (int i = 0; i < n_verts; ++i) {
        push_xyz(out.normals, normal);
        out.tangents.push_back(tangent_dir.x);
        out.tangents.push_back(tangent_dir.y);
        out.tangents.push_back(tangent_dir.z);
        out.tangents.push_back(tangent_w);
        out.wind_weights_packed.push_back(trunk_b);
        out.wind_weights_packed.push_back(branch_b);
        out.wind_weights_packed.push_back(twig_b);
        out.wind_weights_packed.push_back(leaf_b);
        out.material_slots.push_back(1);
        out.bone_index.push_back(bone);
    }
}

// C6 P3 — atlas UV mapping lives in `leaf_atlas.hpp::tile_uv`. The emitter
// and the rasterizer share the same function so per-vert UVs and per-pixel
// cell rectangles cannot drift apart.
vec2 uv_for_quad_corner(LeafShape shape, float lx, float ly) {
    return tile_uv(shape, lx, ly);
}

// ----- SingleCard -----------------------------------------------------------

void emit_single_card(LeafMeshOutput& out,
                      const LeafSite& site,
                      const LeafMeshOptions& opts,
                      uint32_t site_idx) {
    vec3 up = normalized(site.normal);
    vec3 right, forward;
    make_card_basis(up, right, forward);

    pcg32 rng;
    rng.seed(static_cast<uint64_t>(site_idx), k_stream_leaf_geom);
    const float spin = rng.next_float_01() * 6.28318530717958647692f;
    right   = rotate_axis(right,   up, spin);
    forward = rotate_axis(forward, up, spin);

    // 1mm jitter along the normal to break z-fight with bark.
    const float jitter = (rng.next_float_01() - 0.5f) * opts.wind_jitter_norm_eps_m;
    const vec3 center = site.position + up * jitter;
    const float s = opts.leaf_size_m;

    const uint32_t vbase = static_cast<uint32_t>(out.positions.size() / 3);

    // 4 corners CCW from (+right, -forward) (V-flip handled in uv_for_quad_corner).
    const vec2 corners[4] = {
        vec2{-1.0f, -1.0f}, // 0 bottom-left
        vec2{+1.0f, -1.0f}, // 1 bottom-right
        vec2{+1.0f, +1.0f}, // 2 top-right
        vec2{-1.0f, +1.0f}, // 3 top-left
    };
    for (int i = 0; i < 4; ++i) {
        const vec3 p = center + right * (corners[i].x * s) + forward * (corners[i].y * s);
        push_xyz(out.positions, p);
        const vec2 uv = uv_for_quad_corner(opts.shape, corners[i].x, corners[i].y);
        push_uv(out.uvs, uv.x, uv.y);
    }
    push_leaf_vert_meta(out, up, right, site.branch_depth_fraction, site.branch_id, 4);

    out.indices.push_back(vbase + 0);
    out.indices.push_back(vbase + 1);
    out.indices.push_back(vbase + 2);
    out.indices.push_back(vbase + 0);
    out.indices.push_back(vbase + 2);
    out.indices.push_back(vbase + 3);
}

// ----- BentCard / ClusterCard -------------------------------------------------
// Single-crease dihedral fold: 6 verts (4 corners + 2 center midpoints lifted
// along normal), 4 tris (2 per half-plane). The crease runs perpendicular to
// the card's forward axis — i.e. along the `right` direction. Parametrized on
// the UV functor so BentCard (single-leaf cell) and ClusterCard (multi-leaf
// cell) share ONE emitter — the only difference is which atlas cell the card
// samples. `uv_fn(lx, ly)` maps card-local [-1,1]² to atlas UV.

template <typename UvFn>
void emit_bent_card_uv(LeafMeshOutput& out,
                       const LeafSite& site,
                       const LeafMeshOptions& opts,
                       uint32_t site_idx,
                       UvFn uv_fn) {
    vec3 up = normalized(site.normal);
    vec3 right, forward;
    make_card_basis(up, right, forward);

    pcg32 rng;
    rng.seed(static_cast<uint64_t>(site_idx), k_stream_leaf_geom);
    const float spin = rng.next_float_01() * 6.28318530717958647692f;
    right   = rotate_axis(right,   up, spin);
    forward = rotate_axis(forward, up, spin);

    const float jitter = (rng.next_float_01() - 0.5f) * opts.wind_jitter_norm_eps_m;
    const vec3 center = site.position + up * jitter;
    const float s = opts.leaf_size_m;
    const float lift = s * 0.5f * std::sin(opts.bend_half_angle);

    const uint32_t vbase = static_cast<uint32_t>(out.positions.size() / 3);

    // 6 verts: bottom-left, bottom-mid (lifted), bottom-right,
    //          top-left,    top-mid (lifted),    top-right.
    // "bottom" = -forward, "top" = +forward, crease at forward=0.
    const vec3 verts[6] = {
        center + right * (-s) + forward * (-s),              // 0: BL
        center                + forward * (-s) + up * lift,  // 1: BM (crease)
        center + right * (+s) + forward * (-s),              // 2: BR
        center + right * (-s) + forward * (+s),              // 3: TL
        center                + forward * (+s) + up * lift,  // 4: TM (crease)
        center + right * (+s) + forward * (+s),              // 5: TR
    };

    const vec2 uvcoords[6] = {
        uv_fn(-1.0f, -1.0f), // 0: BL
        uv_fn( 0.0f, -1.0f), // 1: BM
        uv_fn(+1.0f, -1.0f), // 2: BR
        uv_fn(-1.0f, +1.0f), // 3: TL
        uv_fn( 0.0f, +1.0f), // 4: TM
        uv_fn(+1.0f, +1.0f), // 5: TR
    };

    for (int i = 0; i < 6; ++i) {
        push_xyz(out.positions, verts[i]);
        push_uv(out.uvs, uvcoords[i].x, uvcoords[i].y);
    }
    push_leaf_vert_meta(out, up, right, site.branch_depth_fraction, site.branch_id, 6);

    // 4 tris: left half (0-1-4, 0-4-3), right half (1-2-5, 1-5-4).
    out.indices.push_back(vbase + 0);
    out.indices.push_back(vbase + 1);
    out.indices.push_back(vbase + 4);

    out.indices.push_back(vbase + 0);
    out.indices.push_back(vbase + 4);
    out.indices.push_back(vbase + 3);

    out.indices.push_back(vbase + 1);
    out.indices.push_back(vbase + 2);
    out.indices.push_back(vbase + 5);

    out.indices.push_back(vbase + 1);
    out.indices.push_back(vbase + 5);
    out.indices.push_back(vbase + 4);
}

void emit_bent_card(LeafMeshOutput& out, const LeafSite& site,
                    const LeafMeshOptions& opts, uint32_t site_idx) {
    const LeafShape shape = opts.shape;
    emit_bent_card_uv(out, site, opts, site_idx,
                      [shape](float lx, float ly) { return tile_uv(shape, lx, ly); });
}

// ----- BentCrossCluster / ClusterCard -----------------------------------------
// N bent cards at 180/N degree angular offsets around site.normal. Bending +
// the crossed orientations present card faces tilted off `up` in every azimuth,
// so the cluster holds projected area from above (the single-card variant was
// edge-on from top-down → invisible in a top-down game). Parametrized on the UV
// functor: BentCrossCluster samples per-species single-leaf quad UVs;
// ClusterCard routes the SAME crossed geometry to the multi-leaf cluster cell,
// so each of the N cards reads as a whole leaf cluster.
template <typename UvFn>
void emit_bent_cross_cluster_uv(LeafMeshOutput& out,
                                const LeafSite& site,
                                const LeafMeshOptions& opts,
                                uint32_t site_idx,
                                UvFn uv_fn) {
    vec3 up = normalized(site.normal);
    vec3 right0, forward0;
    make_card_basis(up, right0, forward0);

    pcg32 rng;
    rng.seed(static_cast<uint64_t>(site_idx), k_stream_leaf_geom);
    const float spin = rng.next_float_01() * 6.28318530717958647692f;
    right0   = rotate_axis(right0,   up, spin);
    forward0 = rotate_axis(forward0, up, spin);

    const float jitter = (rng.next_float_01() - 0.5f) * opts.wind_jitter_norm_eps_m;
    const vec3 center = site.position + up * jitter;
    const float s = opts.leaf_size_m;
    const float lift = s * 0.5f * std::sin(opts.bend_half_angle);

    const int N = opts.cluster_count_per_tip > 0 ? opts.cluster_count_per_tip : 1;
    const float k_pi = 3.14159265358979323846f;

    for (int k = 0; k < N; ++k) {
        const float a = (k_pi * static_cast<float>(k)) / static_cast<float>(N);
        const vec3 right_k   = rotate_axis(right0,   up, a);
        const vec3 forward_k = rotate_axis(forward0, up, a);

        const uint32_t vbase = static_cast<uint32_t>(out.positions.size() / 3);

        const vec3 verts[6] = {
            center + right_k * (-s) + forward_k * (-s),
            center                  + forward_k * (-s) + up * lift,
            center + right_k * (+s) + forward_k * (-s),
            center + right_k * (-s) + forward_k * (+s),
            center                  + forward_k * (+s) + up * lift,
            center + right_k * (+s) + forward_k * (+s),
        };

        const vec2 uvcoords[6] = {
            uv_fn(-1.0f, -1.0f),
            uv_fn( 0.0f, -1.0f),
            uv_fn(+1.0f, -1.0f),
            uv_fn(-1.0f, +1.0f),
            uv_fn( 0.0f, +1.0f),
            uv_fn(+1.0f, +1.0f),
        };

        for (int i = 0; i < 6; ++i) {
            push_xyz(out.positions, verts[i]);
            push_uv(out.uvs, uvcoords[i].x, uvcoords[i].y);
        }
        push_leaf_vert_meta(out, up, right_k, site.branch_depth_fraction, site.branch_id, 6);

        out.indices.push_back(vbase + 0);
        out.indices.push_back(vbase + 1);
        out.indices.push_back(vbase + 4);

        out.indices.push_back(vbase + 0);
        out.indices.push_back(vbase + 4);
        out.indices.push_back(vbase + 3);

        out.indices.push_back(vbase + 1);
        out.indices.push_back(vbase + 2);
        out.indices.push_back(vbase + 5);

        out.indices.push_back(vbase + 1);
        out.indices.push_back(vbase + 5);
        out.indices.push_back(vbase + 4);
    }
}

void emit_bent_cross_cluster(LeafMeshOutput& out, const LeafSite& site,
                             const LeafMeshOptions& opts, uint32_t site_idx) {
    const LeafShape shape = opts.shape;
    emit_bent_cross_cluster_uv(out, site, opts, site_idx,
                               [shape](float lx, float ly) { return uv_for_quad_corner(shape, lx, ly); });
}

// ----- SprigBouquet (ClusterCard) -------------------------------------------
// 3 x 4-vert Volumetric Sprig Bouquet (12 verts / 6 tris when N = 3):
//   - Every card is a centered 4-vert quad around `center`, guaranteeing exact
//     0.0 mm K17 centroid agreement across all cards for any 3D rotation.
//   - Card normals n_k tilt outward from spray axis A = normalized(site.normal)
//     by cone_tilt <= 42 deg (hard cap keeps dot(n_k, outward) > 0 under
//     --render-glb deferred shader without !gl_FrontFacing flip).
//   - All 4 verts of card k share bit-identical vertex normal n_k (preserving
//     rigid wind flutter/tumble in tree.vs.glsl), while distinct n_0, n_1, n_2
//     decorrelate macro lighting and wind across sibling cards.
//   - Cycles through all 3 sprig variants (rows 1..3), applies ±12 deg azimuth
//     jitter, ±15 deg in-plane roll, ±12% scale jitter, and 50% U-mirror with
//     exact tangent handedness vec4(-right_k, -1.0f) (TRAP 3).
template <typename SprigUvFn>
void emit_sprig_bouquet_uv(LeafMeshOutput& out,
                           const LeafSite& site,
                           const LeafMeshOptions& opts,
                           uint32_t site_idx,
                           SprigUvFn uv_fn) {
    constexpr float k_pi     = 3.14159265358979323846f;
    constexpr float k_two_pi = 6.28318530717958647692f;
    constexpr float k_deg    = k_pi / 180.0f;

    const vec3 spray_axis = normalized(site.normal);
    vec3 right0, forward0;
    make_card_basis(spray_axis, right0, forward0);

    pcg32 rng;
    rng.seed(static_cast<uint64_t>(site_idx), k_stream_leaf_geom);
    const float spin = rng.next_float_01() * k_two_pi;

    // 1mm jitter along spray axis shared by all cards at this site -> 0.0mm centroid delta.
    const float jitter = (rng.next_float_01() - 0.5f) * opts.wind_jitter_norm_eps_m;
    const vec3 center = site.position + spray_axis * jitter;

    const float cone_tilt = std::max(0.0f, std::min(opts.bend_half_angle, K_MAX_SPRIG_CONE_TILT_RAD));
    const float cos_tilt  = std::cos(cone_tilt);
    const float sin_tilt  = std::sin(cone_tilt);

    const int N = opts.cluster_count_per_tip > 0 ? opts.cluster_count_per_tip : 1;
    const bool is_pine = (opts.shape == LeafShape::PineNeedle);

    static constexpr int k_variant_perms[6][3] = {
        {1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}
    };
    static constexpr int k_pine_combos[4][3] = {
        {2, 3, 2}, {3, 2, 3}, {2, 2, 3}, {3, 3, 2}
    };
    const int* perm = is_pine
        ? k_pine_combos[rng.next_u32() % 4u]
        : k_variant_perms[rng.next_u32() % 6u];

    const vec2 corners[4] = {
        vec2{-1.0f, -1.0f}, // 0 bottom-left
        vec2{+1.0f, -1.0f}, // 1 bottom-right
        vec2{+1.0f, +1.0f}, // 2 top-right
        vec2{-1.0f, +1.0f}, // 3 top-left
    };

    for (int k = 0; k < N; ++k) {
        const float az_span   = is_pine ? (32.0f * k_deg) : (24.0f * k_deg); // ±16 deg pine / ±12 deg broadleaf
        const float az_jitter = (rng.next_float_01() - 0.5f) * az_span;
        const float az_k      = spin + (k_two_pi * static_cast<float>(k)) / static_cast<float>(N) + az_jitter;
        const float ca = std::cos(az_k);
        const float sa = std::sin(az_k);

        // Radial and tangential unit vectors in the plane perpendicular to spray_axis.
        const vec3 rad_k = right0 * ca + forward0 * sa;
        const vec3 tan_k = right0 * (-sa) + forward0 * ca;

        // Outward-tilted card normal n_k (dot(n_k, spray_axis) = cos(tilt_k) >= cos(42 deg)).
        float ct_k = cos_tilt;
        float st_k = sin_tilt;
        if (is_pine) {
            // ±16 deg per-card tilt jitter, clamped cleanly under the 42 deg hard cap.
            const float tilt_jit = (rng.next_float_01() - 0.5f) * (32.0f * k_deg);
            const float tilt_k   = std::max(0.15f, std::min(cone_tilt + tilt_jit, K_MAX_SPRIG_CONE_TILT_RAD));
            ct_k = std::cos(tilt_k);
            st_k = std::sin(tilt_k);
        }
        const vec3 n_k = normalized(spray_axis * ct_k + rad_k * st_k);

        // Right-handed in-plane basis (u_axis, v_axis, n_k): cross(n_k, u_axis) = +v_axis.
        const vec3 u_axis = normalized(tan_k);
        const vec3 v_axis = normalized(cross(n_k, u_axis));

        // ±16 deg (pine) / ±15 deg (broadleaf) in-plane roll.
        const float roll_span = is_pine ? (32.0f * k_deg) : (30.0f * k_deg);
        const float roll_k    = (rng.next_float_01() - 0.5f) * roll_span;
        const float cr = std::cos(roll_k);
        const float sr = std::sin(roll_k);
        const vec3 right_k   = normalized(u_axis * cr + v_axis * sr);
        const vec3 forward_k = normalized(cross(n_k, right_k));

        // ±22% (pine) / ±12% (broadleaf) per-card scale jitter.
        const float s_k = is_pine
            ? opts.leaf_size_m * (0.78f + 0.44f * rng.next_float_01())
            : opts.leaf_size_m * (0.88f + 0.24f * rng.next_float_01());

        // Deterministic 50% U-mirror (doubles baked cells into 2x silhouettes).
        const bool  mirror_u = (rng.next_u32() & 1u) != 0u;
        const float u_sign   = mirror_u ? -1.0f : +1.0f;
        const int   var_row  = perm[k % 3];

        const uint32_t vbase = static_cast<uint32_t>(out.positions.size() / 3);
        for (int i = 0; i < 4; ++i) {
            const vec3 p = center + right_k * (corners[i].x * s_k) + forward_k * (corners[i].y * s_k);
            push_xyz(out.positions, p);
            const vec2 uv = uv_fn(var_row, corners[i].x * u_sign, corners[i].y);
            push_uv(out.uvs, uv.x, uv.y);
        }

        // TRAP 3: when U-mirroring (lx -> -lx), +dU flips to -right_k; set w = -1.0f
        // so B_w = cross(n_k, -right_k) * (-1.0f) = +forward_k matches -dV cleanly.
        const vec3  tangent_dir = mirror_u ? (right_k * -1.0f) : right_k;
        const float tangent_w   = mirror_u ? -1.0f : +1.0f;
        push_leaf_vert_meta(out, n_k, tangent_dir, site.branch_depth_fraction, site.branch_id, 4, tangent_w);

        out.indices.push_back(vbase + 0);
        out.indices.push_back(vbase + 1);
        out.indices.push_back(vbase + 2);
        out.indices.push_back(vbase + 0);
        out.indices.push_back(vbase + 2);
        out.indices.push_back(vbase + 3);
    }
}

// ClusterCard: volumetric 3 x 4-vert sprig bouquet sampling rows 1..3.
void emit_cluster_card(LeafMeshOutput& out, const LeafSite& site,
                       const LeafMeshOptions& opts, uint32_t site_idx) {
    const LeafShape shape = opts.shape;
    emit_sprig_bouquet_uv(out, site, opts, site_idx,
                          [shape](int row, float lx, float ly) { return sprig_tile_uv(shape, row, lx, ly); });
}

}  // namespace (anonymous — re-opened after the SoT helpers below)

// ----- ProceduralVeined -----------------------------------------------------
// `shape_mesh_for` lives in namespace scope (not the anon namespace) so the
// C5 P3 SoT helpers (`tris_per_leaf_shape`, `verts_per_leaf`) can share it
// without duplicating the switch — single source of truth seal.

LeafShapeMesh shape_mesh_for(LeafShape s) {
    switch (s) {
        case LeafShape::OakLobed:      return oak_lobed();
        case LeafShape::PineNeedle:    return pine_needle();
        case LeafShape::BirchSerrated: return birch_serrated();
        case LeafShape::MapleStar:     return maple_star();
    }
    return oak_lobed();
}

int tris_per_leaf_shape(LeafShape s) {
    const LeafShapeMesh mesh = shape_mesh_for(s);
    return static_cast<int>(mesh.tris.size() / 3);
}

int tris_per_leaf(LeafGeometryType g, LeafShape s, int cluster_count_per_tip) {
    switch (g) {
        case LeafGeometryType::SingleCard:        return K_TRIS_PER_LEAF_SINGLE_CARD;
        case LeafGeometryType::ProceduralVeined:  return tris_per_leaf_shape(s);
        case LeafGeometryType::BentCard:          return K_TRIS_PER_BENT_CARD;
        case LeafGeometryType::BentCrossCluster: {
            const int N = cluster_count_per_tip > 0 ? cluster_count_per_tip : 1;
            return K_TRIS_PER_BENT_CARD * N;
        }
        case LeafGeometryType::BranchStrip:
            return K_TRIS_PER_STRIP;
        case LeafGeometryType::ClusterCard: {          // N angled 4-vert sprig cards
            const int N = cluster_count_per_tip > 0 ? cluster_count_per_tip : 1;
            return K_TRIS_PER_SPRIG_CARD * N;
        }
    }
    return K_TRIS_PER_LEAF_SINGLE_CARD;
}

int verts_per_leaf(LeafGeometryType g, LeafShape s, int cluster_count_per_tip) {
    switch (g) {
        case LeafGeometryType::SingleCard:
            return K_VERTS_PER_LEAF_SINGLE_CARD;
        case LeafGeometryType::ProceduralVeined:
            return static_cast<int>(shape_mesh_for(s).verts.size());
        case LeafGeometryType::BentCard:
            return K_VERTS_PER_BENT_CARD;
        case LeafGeometryType::BentCrossCluster: {
            const int N = cluster_count_per_tip > 0 ? cluster_count_per_tip : 1;
            return K_VERTS_PER_BENT_CARD * N;
        }
        case LeafGeometryType::BranchStrip:
            return K_VERTS_PER_STRIP;
        case LeafGeometryType::ClusterCard: {          // N angled 4-vert sprig cards
            const int N = cluster_count_per_tip > 0 ? cluster_count_per_tip : 1;
            return K_VERTS_PER_SPRIG_CARD * N;
        }
    }
    return K_VERTS_PER_LEAF_SINGLE_CARD;
}

namespace {

void emit_procedural_veined(LeafMeshOutput& out,
                            const LeafSite& site,
                            const LeafMeshOptions& opts,
                            uint32_t site_idx,
                            const LeafShapeMesh& shape_mesh) {
    vec3 up = normalized(site.normal);
    vec3 right, forward;
    make_card_basis(up, right, forward);

    pcg32 rng;
    rng.seed(static_cast<uint64_t>(site_idx), k_stream_leaf_geom);
    const float spin = rng.next_float_01() * 6.28318530717958647692f;
    right   = rotate_axis(right,   up, spin);
    forward = rotate_axis(forward, up, spin);

    const float jitter = (rng.next_float_01() - 0.5f) * opts.wind_jitter_norm_eps_m;
    const vec3 center = site.position + up * jitter;
    const float s = opts.leaf_size_m;

    const uint32_t vbase = static_cast<uint32_t>(out.positions.size() / 3);
    for (const vec2& vert2d : shape_mesh.verts) {
        const vec3 p = center + right * (vert2d.x * s) + forward * (vert2d.y * s);
        push_xyz(out.positions, p);
        const vec2 uv = uv_for_quad_corner(opts.shape, vert2d.x, vert2d.y);
        push_uv(out.uvs, uv.x, uv.y);
    }
    push_leaf_vert_meta(out, up, right, site.branch_depth_fraction, site.branch_id,
                        static_cast<int>(shape_mesh.verts.size()));

    for (uint16_t i : shape_mesh.tris) {
        out.indices.push_back(vbase + static_cast<uint32_t>(i));
    }
}

}  // namespace

LeafMeshOutput build_leaf_mesh(const std::vector<LeafSite>& sites,
                               const LeafMeshOptions&       opts) {
    LeafMeshOutput out;

    // Pre-bake the procedural shape mesh once if needed; per-leaf work is just
    // local-to-world transform + index rebase.
    LeafShapeMesh shape_mesh;
    if (opts.geometry_type == LeafGeometryType::ProceduralVeined) {
        shape_mesh = shape_mesh_for(opts.shape);
    }

    size_t vpl_reserve = 0;
    size_t ipl_reserve = 0;
    switch (opts.geometry_type) {
        case LeafGeometryType::SingleCard:
            vpl_reserve = 4;
            ipl_reserve = 6;
            break;
        case LeafGeometryType::ProceduralVeined:
            vpl_reserve = shape_mesh.verts.size();
            ipl_reserve = shape_mesh.tris.size();
            break;
        case LeafGeometryType::BentCard:
            vpl_reserve = 6;
            ipl_reserve = 12;
            break;
        case LeafGeometryType::BentCrossCluster: {
            const int N = opts.cluster_count_per_tip > 0 ? opts.cluster_count_per_tip : 1;
            vpl_reserve = static_cast<size_t>(6 * N);
            ipl_reserve = static_cast<size_t>(12 * N);
            break;
        }
        case LeafGeometryType::ClusterCard: {
            const int N = opts.cluster_count_per_tip > 0 ? opts.cluster_count_per_tip : 1;
            vpl_reserve = static_cast<size_t>(4 * N);
            ipl_reserve = static_cast<size_t>(6 * N);
            break;
        }
        case LeafGeometryType::BranchStrip:
            // BranchStrip uses build_branch_strip_mesh, not build_leaf_mesh.
            // If reached here, treat as single card for fallback.
            vpl_reserve = 4;
            ipl_reserve = 6;
            break;
    }
    out.positions.reserve(sites.size() * vpl_reserve * 3);
    out.normals.reserve  (sites.size() * vpl_reserve * 3);
    out.uvs.reserve      (sites.size() * vpl_reserve * 2);
    out.tangents.reserve (sites.size() * vpl_reserve * 4);
    out.indices.reserve  (sites.size() * ipl_reserve);
    out.wind_weights_packed.reserve(sites.size() * vpl_reserve * 4);
    out.material_slots.reserve     (sites.size() * vpl_reserve);

    for (size_t i = 0; i < sites.size(); ++i) {
        const uint32_t idx = static_cast<uint32_t>(i);
        switch (opts.geometry_type) {
            case LeafGeometryType::SingleCard:
                emit_single_card(out, sites[i], opts, idx);
                break;
            case LeafGeometryType::ProceduralVeined:
                emit_procedural_veined(out, sites[i], opts, idx, shape_mesh);
                break;
            case LeafGeometryType::BentCard:
                emit_bent_card(out, sites[i], opts, idx);
                break;
            case LeafGeometryType::ClusterCard:
                emit_cluster_card(out, sites[i], opts, idx);
                break;
            case LeafGeometryType::BentCrossCluster:
                emit_bent_cross_cluster(out, sites[i], opts, idx);
                break;
            case LeafGeometryType::BranchStrip:
                // BranchStrip uses build_branch_strip_mesh, not this path.
                // Fallback: emit as single card.
                emit_single_card(out, sites[i], opts, idx);
                break;
        }
    }
    return out;
}

LeafMeshOutput build_branch_strip_mesh(const TreeSkeleton&      skel,
                                       const BranchStripOptions& opts,
                                       int                      min_branch_depth,
                                       float                    strip_radius_threshold) {
    LeafMeshOutput out;
    if (skel.nodes.size() < 2) return out;

    int max_depth = 0;
    for (const auto& n : skel.nodes) max_depth = std::max(max_depth, n.depth);
    if (max_depth < 4) max_depth = 4; // match wind_weights clamp

    const float inv_max_depth = 1.0f / static_cast<float>(max_depth);
    const float hw = opts.strip_width_m * 0.5f;

    // Count qualifying segments for reserve.
    size_t seg_count = 0;
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const auto& child = skel.nodes[i];
        if (child.depth < min_branch_depth) continue;
        if (child.parent_index < 0) continue;
        ++seg_count;
    }
    // Each strip = 4 verts, 6 indices. Max 3 strips per segment.
    out.positions.reserve(seg_count * 3 * 4 * 3);
    out.normals.reserve(seg_count * 3 * 4 * 3);
    out.uvs.reserve(seg_count * 3 * 4 * 2);
    out.tangents.reserve(seg_count * 3 * 4 * 4);
    out.indices.reserve(seg_count * 3 * 6);
    out.wind_weights_packed.reserve(seg_count * 3 * 4 * 4);
    out.material_slots.reserve(seg_count * 3 * 4);
    out.bone_index.reserve(seg_count * 3 * 4);

    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const auto& child  = skel.nodes[i];
        if (child.depth < min_branch_depth) continue;
        if (child.parent_index < 0) continue;
        const auto& parent = skel.nodes[static_cast<size_t>(child.parent_index)];

        const vec3 seg = child.position - parent.position;
        const float seg_len = std::sqrt(seg.x * seg.x + seg.y * seg.y + seg.z * seg.z);
        if (seg_len < 1e-5f) continue;

        const vec3 seg_dir = seg / seg_len;

        // Build perpendicular basis.
        const vec3 ref = std::abs(seg_dir.z) < 0.9f ? vec3{0.0f, 0.0f, 1.0f}
                                                     : vec3{1.0f, 0.0f, 0.0f};
        vec3 perp_u = normalized(cross(seg_dir, ref));
        vec3 perp_v = normalized(cross(seg_dir, perp_u));

        // Multi-strip: start from minimum, increase with branch radius.
        const float avg_radius = (parent.radius + child.radius) * 0.5f;
        int strip_count = opts.min_strips_per_segment;
        if (avg_radius >= strip_radius_threshold * 2.0f) strip_count = std::max(strip_count, 2);
        if (avg_radius >= strip_radius_threshold * 3.0f) strip_count = std::max(strip_count, 3);
        strip_count = std::min(strip_count, opts.max_strips_per_segment);

        const float depth_frac = static_cast<float>(child.depth) * inv_max_depth;

        // Wind weights (same formula as leaf wind).
        const float d  = clamp01(depth_frac);
        const float od = 1.0f - d;
        const uint8_t trunk_b  = quantize_wind(od * od);
        const uint8_t branch_b = quantize_wind(4.0f * d * od);
        const uint8_t twig_b   = quantize_wind(4.0f * d * d * od);
        const uint8_t leaf_b   = quantize_wind(d * d * d);

        // Deterministic per-segment angular jitter breaks rigid +-cross alignment
        // across successive segments while keeping exact pi/N spacing within a segment.
        constexpr float k_pi = 3.14159265358979323846f;
        pcg32 seg_rng;
        seg_rng.seed(static_cast<uint64_t>(i), 0xC3'05'57'1BULL);
        const float seg_jitter = (seg_rng.next_float_01() - 0.5f) * (k_pi * 0.65f);
        const float hw_start   = hw * 1.04f;
        const float hw_end     = hw * 0.90f;

        // Angular offsets: pi/N spacing (bilateral strips cover both sides).
        for (int s = 0; s < strip_count; ++s) {
            float angle = opts.strip_angular_offset + seg_jitter
                        + (k_pi / static_cast<float>(strip_count)) * static_cast<float>(s);

            // Perpendicular direction at this angular offset (quad width axis = +dP/dU).
            const float ca = std::cos(angle);
            const float sa = std::sin(angle);
            vec3 strip_perp = perp_u * ca + perp_v * sa;

            // Apply droop rotation around branch axis.
            if (opts.strip_droop_angle > 0.0f) {
                strip_perp = rotate_axis(strip_perp, seg_dir, opts.strip_droop_angle);
            }

            const vec3 strip_right  = normalized(strip_perp);
            // True face normal perpendicular to BOTH width (strip_right) and length (seg_dir).
            const vec3 strip_normal = normalized(cross(strip_right, seg_dir));
            const vec3 tangent_dir  = strip_right; // +dP/dU; B = cross(N, T) = +seg_dir

            // UV: U=[0,1] across width, V clamped safely inside [0, 1] of cell (1,1).
            const float v_tiles = clamp01(seg_len / std::max(0.05f, opts.strip_width_m));

            const uint32_t vbase = static_cast<uint32_t>(out.positions.size() / 3);

            // Centered on the branch axis (no sideways avg_radius drift) with slight axial taper.
            const vec3 v0 = parent.position - strip_right * hw_start; // start-left
            const vec3 v1 = parent.position + strip_right * hw_start; // start-right
            const vec3 v2 = child.position  + strip_right * hw_end;   // end-right
            const vec3 v3 = child.position  - strip_right * hw_end;   // end-left

            push_xyz(out.positions, v0);
            push_xyz(out.positions, v1);
            push_xyz(out.positions, v2);
            push_xyz(out.positions, v3);

            // UVs from needle-strip atlas cell (1,1).
            const vec2 uv0 = needle_strip_tile_uv(0.0f, 0.0f);
            const vec2 uv1 = needle_strip_tile_uv(1.0f, 0.0f);
            const vec2 uv2 = needle_strip_tile_uv(1.0f, v_tiles);
            const vec2 uv3 = needle_strip_tile_uv(0.0f, v_tiles);
            push_uv(out.uvs, uv0.x, uv0.y);
            push_uv(out.uvs, uv1.x, uv1.y);
            push_uv(out.uvs, uv2.x, uv2.y);
            push_uv(out.uvs, uv3.x, uv3.y);

            // Per-vertex normals, tangents, wind, material.
            assert(std::abs(dot(tangent_dir, strip_normal)) < 1e-4f);
            for (int vi = 0; vi < 4; ++vi) {
                push_xyz(out.normals, strip_normal);
                out.tangents.push_back(tangent_dir.x);
                out.tangents.push_back(tangent_dir.y);
                out.tangents.push_back(tangent_dir.z);
                out.tangents.push_back(1.0f);
                out.wind_weights_packed.push_back(trunk_b);
                out.wind_weights_packed.push_back(branch_b);
                out.wind_weights_packed.push_back(twig_b);
                out.wind_weights_packed.push_back(leaf_b);
                out.material_slots.push_back(1);
                // C8-wind P1 — host bone = the child node of this segment (i).
                out.bone_index.push_back(static_cast<uint16_t>(i));
            }

            // Two triangles (CCW).
            out.indices.push_back(vbase + 0);
            out.indices.push_back(vbase + 1);
            out.indices.push_back(vbase + 2);
            out.indices.push_back(vbase + 0);
            out.indices.push_back(vbase + 2);
            out.indices.push_back(vbase + 3);
        }
    }
    return out;
}

}  // namespace treegen
