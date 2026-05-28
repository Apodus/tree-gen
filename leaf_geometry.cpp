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
                         float branch_depth_fraction, int n_verts) {
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
        out.tangents.push_back(1.0f);
        out.wind_weights_packed.push_back(trunk_b);
        out.wind_weights_packed.push_back(branch_b);
        out.wind_weights_packed.push_back(twig_b);
        out.wind_weights_packed.push_back(leaf_b);
        out.material_slots.push_back(1);
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
    push_leaf_vert_meta(out, up, right, site.branch_depth_fraction, 4);

    out.indices.push_back(vbase + 0);
    out.indices.push_back(vbase + 1);
    out.indices.push_back(vbase + 2);
    out.indices.push_back(vbase + 0);
    out.indices.push_back(vbase + 2);
    out.indices.push_back(vbase + 3);
}

// ----- BentCard ---------------------------------------------------------------
// Single-crease dihedral fold: 6 verts (4 corners + 2 center midpoints lifted
// along normal), 4 tris (2 per half-plane). The crease runs perpendicular to
// the card's forward axis — i.e. along the `right` direction.

void emit_bent_card(LeafMeshOutput& out,
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
        uv_for_quad_corner(opts.shape, -1.0f, -1.0f), // 0: BL
        uv_for_quad_corner(opts.shape,  0.0f, -1.0f), // 1: BM
        uv_for_quad_corner(opts.shape, +1.0f, -1.0f), // 2: BR
        uv_for_quad_corner(opts.shape, -1.0f, +1.0f), // 3: TL
        uv_for_quad_corner(opts.shape,  0.0f, +1.0f), // 4: TM
        uv_for_quad_corner(opts.shape, +1.0f, +1.0f), // 5: TR
    };

    for (int i = 0; i < 6; ++i) {
        push_xyz(out.positions, verts[i]);
        push_uv(out.uvs, uvcoords[i].x, uvcoords[i].y);
    }
    push_leaf_vert_meta(out, up, right, site.branch_depth_fraction, 6);

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

// ----- BentCrossCluster -------------------------------------------------------
// N BentCard instances at 180/N degree angular offsets around site.normal.

void emit_bent_cross_cluster(LeafMeshOutput& out,
                             const LeafSite& site,
                             const LeafMeshOptions& opts,
                             uint32_t site_idx) {
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
            uv_for_quad_corner(opts.shape, -1.0f, -1.0f),
            uv_for_quad_corner(opts.shape,  0.0f, -1.0f),
            uv_for_quad_corner(opts.shape, +1.0f, -1.0f),
            uv_for_quad_corner(opts.shape, -1.0f, +1.0f),
            uv_for_quad_corner(opts.shape,  0.0f, +1.0f),
            uv_for_quad_corner(opts.shape, +1.0f, +1.0f),
        };

        for (int i = 0; i < 6; ++i) {
            push_xyz(out.positions, verts[i]);
            push_uv(out.uvs, uvcoords[i].x, uvcoords[i].y);
        }
        push_leaf_vert_meta(out, up, right_k, site.branch_depth_fraction, 6);

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
    push_leaf_vert_meta(out, up, right, site.branch_depth_fraction,
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

        // Angular offsets: pi/N spacing (bilateral strips cover both sides).
        constexpr float k_pi = 3.14159265358979323846f;
        for (int s = 0; s < strip_count; ++s) {
            float angle = opts.strip_angular_offset
                        + (k_pi / static_cast<float>(strip_count)) * static_cast<float>(s);

            // Perpendicular direction at this angular offset.
            const float ca = std::cos(angle);
            const float sa = std::sin(angle);
            vec3 strip_perp = perp_u * ca + perp_v * sa;

            // Apply droop: rotate strip_perp slightly downward (toward -Z).
            if (opts.strip_droop_angle > 0.0f) {
                strip_perp = rotate_axis(strip_perp, seg_dir, opts.strip_droop_angle);
            }

            const vec3 strip_normal = normalized(strip_perp);

            // 4 verts: start-left, start-right, end-right, end-left.
            const vec3 p0 = parent.position + strip_perp * avg_radius;
            const vec3 p1 = child.position  + strip_perp * avg_radius;
            const vec3 left  = strip_perp * (-hw);
            const vec3 right = strip_perp * hw;

            // UV: U=[0,1] across width, V tiled along branch.
            const float v_tiles = seg_len / opts.strip_width_m;

            const uint32_t vbase = static_cast<uint32_t>(out.positions.size() / 3);

            // Vertices offset perpendicular from the branch center.
            const vec3 v0 = parent.position + strip_perp * avg_radius + left;   // start-left
            const vec3 v1 = parent.position + strip_perp * avg_radius + right;  // start-right
            const vec3 v2 = child.position  + strip_perp * avg_radius + right;  // end-right
            const vec3 v3 = child.position  + strip_perp * avg_radius + left;   // end-left

            push_xyz(out.positions, v0);
            push_xyz(out.positions, v1);
            push_xyz(out.positions, v2);
            push_xyz(out.positions, v3);

            // UVs from needle-strip atlas cell.
            const vec2 uv0 = needle_strip_tile_uv(0.0f, 0.0f);
            const vec2 uv1 = needle_strip_tile_uv(1.0f, 0.0f);
            const vec2 uv2 = needle_strip_tile_uv(1.0f, v_tiles);
            const vec2 uv3 = needle_strip_tile_uv(0.0f, v_tiles);
            push_uv(out.uvs, uv0.x, uv0.y);
            push_uv(out.uvs, uv1.x, uv1.y);
            push_uv(out.uvs, uv2.x, uv2.y);
            push_uv(out.uvs, uv3.x, uv3.y);

            // Per-vertex normals, tangents, wind, material.
            const vec3 tangent_dir = seg_dir; // along-branch (dP/dV); atlas normal bake matches this convention
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
