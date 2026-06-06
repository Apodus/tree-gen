// C5 P2 — leaf geometry emitters. Given C5 P1's LeafSite stream + a
// LeafGeometryType + LeafShape, materialize CPU-side leaf mesh data:
// positions / normals / uvs / indices / wind-weight bytes / material slots.
//
// All emitters share:
//   - per-leaf deterministic rotation around site.normal (pcg32 keyed off the
//     site index, so re-orderings of the input vector change output — by
//     design; site index is the stable identity).
//   - per-vertex outward normal = site.normal (constant across all verts of
//     a single leaf, for backlit translucency).
//   - per-vertex wind weights = depth-interpolated (trunk, branch, twig, leaf)
//     per C1 P3 — same formula as bark wind weights evaluated at
//     site.branch_depth_fraction; leaves on thick branches get trunk/branch
//     influence, terminal-twig leaves are near pure leaf tier.
//   - per-vertex material slot = 1 (leaves).
//   - 1mm normal-direction z-fight jitter on the leaf center.
#pragma once

#include "leaf_shapes.hpp"       // LeafShapeMesh
#include "skeleton.hpp"
#include "tree_descriptor.hpp"

#include <cstdint>
#include <vector>

namespace treegen {

struct LeafMeshOptions {
    LeafGeometryType geometry_type           = LeafGeometryType::BentCrossCluster;
    LeafShape        shape                   = LeafShape::OakLobed;
    float            leaf_size_m             = 0.12f;  // half-extent
    int              cluster_count_per_tip   = 1;      // BentCrossCluster
    float            wind_jitter_norm_eps_m  = 0.001f; // 1mm z-fight breaker
    float            bend_half_angle         = 0.392699081698724f; // pi/8; BentCard / BentCrossCluster
};

struct LeafMeshOutput {
    std::vector<float>    positions;             // xyz packed
    std::vector<float>    normals;               // xyz packed (= site.normal)
    std::vector<float>    uvs;                   // xy packed (placeholder atlas tile)
    std::vector<uint32_t> indices;               // triangle list
    std::vector<uint8_t>  wind_weights_packed;   // 4 bytes/vert; depth-interpolated
    std::vector<int32_t>  material_slots;        // 1 per vert (leaves)
    // C1 P6 — per-vertex tangent (xyzw packed, 4 floats/vert). Card right
    // vector in UV space; w=+1 (right-handed TBN: B=cross(N,T)*w).
    std::vector<float>    tangents;
    // C8-wind P1 — host skeleton node (bone) per leaf vertex. For build_leaf_mesh
    // this is site.branch_id; for build_branch_strip_mesh it's the child node.
    // Length == positions.size() / 3. Leaves/strips are RIGID (bone_blend = 0
    // everywhere) — the rigid all-zero blend is supplied at LOD-assembly time.
    std::vector<uint16_t> bone_index;
};

// Per-emitter vertex / triangle counts. Exposed so the test can pin them
// without reaching into the .cpp constants.
inline constexpr int K_VERTS_PER_LEAF_SINGLE_CARD = 4;
inline constexpr int K_TRIS_PER_LEAF_SINGLE_CARD  = 2;

// BentCard: single-crease dihedral fold. 6 verts (4 corners + 2 center-crease
// midpoints), 4 tris (2 per half-plane). Guarantees nonzero visible area from
// every viewing direction.
inline constexpr int K_VERTS_PER_BENT_CARD = 6;
inline constexpr int K_TRIS_PER_BENT_CARD  = 4;

// BranchStrip: one quad per branch segment. 4 verts, 2 tris. Emission is
// per-segment (build_branch_strip_mesh), not per-site (build_leaf_mesh).
inline constexpr int K_VERTS_PER_STRIP = 4;
inline constexpr int K_TRIS_PER_STRIP  = 2;

// Single source of truth for shape → mesh lookup. Used by emitter, atlas bake,
// and the SoT helpers below.
LeafShapeMesh shape_mesh_for(LeafShape s);

// C5 P3 single-source-of-truth helpers. Both consult the same `shape_mesh_for`
// data that `build_leaf_mesh` loops over for ProceduralVeined, so the budget
// allocator and the emitter cannot drift on per-leaf tri/vert counts.
//
// `tris_per_leaf_shape(s)` — count of triangles in the analytic 2D polygon for
//   shape `s`. Independent of geometry_type; used by `verts_per_leaf` only for
//   the ProceduralVeined dispatch.
// `verts_per_leaf(g, s, cluster)` — exact vertex count one leaf contributes
//   under geometry type `g` (shape `s` matters only for ProceduralVeined;
//   `cluster` matters only for BentCrossCluster, in which case >=1 is enforced).
int tris_per_leaf_shape(LeafShape s);
int tris_per_leaf(LeafGeometryType g, LeafShape s, int cluster_count_per_tip);
int verts_per_leaf(LeafGeometryType g, LeafShape s, int cluster_count_per_tip);

LeafMeshOutput build_leaf_mesh(const std::vector<LeafSite>& sites,
                               const LeafMeshOptions&       opts);

// C3-needle-strips: per-segment strip mesh builder.
struct BranchStripOptions {
    float strip_width_m           = 0.4f;
    float strip_droop_angle       = 0.15f;   // radians
    float strip_angular_offset    = 0.0f;    // radians around branch axis
    float strip_radius_threshold  = 0.02f;   // multi-strip threshold
    int   min_strips_per_segment  = 3;
    int   max_strips_per_segment  = 3;
};

// Build strip mesh for qualifying skeleton segments. Returns LeafMeshOutput
// (same layout as build_leaf_mesh — positions/normals/uvs/indices/wind/
// tangents/material). Not per-site: walks skeleton edges directly.
LeafMeshOutput build_branch_strip_mesh(const TreeSkeleton&      skel,
                                       const BranchStripOptions& opts,
                                       int                      min_branch_depth,
                                       float                    strip_radius_threshold);

}  // namespace treegen
