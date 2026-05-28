// C4 P1+P2 — bark mesh cylinder extrusion from a TreeSkeleton.
//
// Per-branch (parent→child segment) cylindrical tube; per-vertex cylindrical
// UV (V along branch axis / bark_repeat_m, U = (θ + seam_offset)/2π with the
// seam-column duplicate from bark_uv.hpp). Radial segment count tapers with
// branch depth (trunk → twig). P2 closes T-junctions at forks via skin-rim
// blending + fan-stitch for unmatched verts (fork_blend.{hpp,cpp}).
//
// Output also carries `per_vertex_node_index` for the downstream wind-weight
// bake in P3 (each vert remembers which BranchNode it was extruded from) and
// `ring_metadata` so the fork blend can find rings by (branch, axial-index).
#pragma once

#include "skeleton.hpp"
#include "trivial_cylinder.hpp" // cpu_mesh_out

#include <cstdint>
#include <vector>

namespace treegen {

struct BarkMeshOptions {
    float bark_repeat_m          = 1.0f; // V = axial_distance_m / bark_repeat_m
    int   radial_seg_trunk       = 12;   // depth 0
    int   radial_seg_order1      = 8;    // depth 1
    int   radial_seg_order2      = 6;    // depth 2
    int   radial_seg_order3_plus = 4;    // depth ≥ 3 (twigs)
    float axial_segment_length_m = 0.5f; // per-branch axial subdivision target
    float seam_offset_rad        = 0.0f; // per-tree rotational offset (deterministic from seed)
    float fork_zone_factor       = 1.5f; // P2 skin-rim zone_length = factor * parent_radius
    bool  apply_fork_blend       = true; // P2 — set false to recover raw P1 output
    bool  emit_crotch_cap        = true; // P4 — centroid-fan at fork crotch gaps
    float tree_height_m          = 10.0f; // P3 — wind-weight bake rooted_factor edge1 scale
    float root_flare_factor      = 1.0f; // C12 P4 — widen trunk base rings (1.0 = no flare)
    float junction_shoulder_factor = 0.0f; // C1 P1 — parent-approach swell at fork junctions (0 = disabled)
    // Per-node merge mask (length skel.nodes.size(); 1 = merged into parent).
    // When non-empty, build_bark_mesh skips bark emission for merged branches.
    // Merged branches retain their skeleton node for leaf placement — the leaf
    // pipeline is independent of this mask. Source: face_budget's
    // merge_shortest_branches overflow rule. Merge-cascade: when all children
    // of a fork are merged, the fork's collar/crotch geometry is suppressed.
    std::vector<uint8_t> culled_node_mask;
};

// Per-ring layout record. Each "branch" (skeleton edge parent→child) emits
// (axial_segs+1) rings; each ring has (N+1) verts (seam duplicate at i=N).
//
// `branch_node_index` = the child node index of the parent→child edge
// (matches per_vertex_node_index). `axial_index` 0 = at parent.position,
// axial_segs = at child.position. `axial_distance_m` is along the branch
// from the parent end (axial_index 0).
struct RingMetadata {
    int   branch_node_index;
    int   axial_index;
    int   axial_segs;       // total axial subdivisions for this branch
    int   vert_start;       // start index in mesh.positions / 3
    int   N;                // radial segments (vert count is N+1 incl. seam dup)
    float axial_distance_m; // 0 at parent end
    float branch_length_m;  // full parent→child length
    int   collar_fork_node = -1; // P2+3: if >= 0, collar ring for this fork node
    int   collar_sequence  = -1; // P2+3: 0-based position in collar chain (0 = near parent rim)
};

// We carry u32 indices alongside cpu_mesh_out (which only has u16) because a
// non-toy tree (oak ~1700 nodes × ~30 verts/seg) easily blows past 65535
// verts. cpu_mesh_out is intentionally left unchanged — it pins the C1
// trivial cylinder counts and the writer path; bark uses u32 exclusively.
struct BarkMeshOutput {
    cpu_mesh_out              mesh;                  // positions/normals/uvs populated; mesh.indices left empty
    std::vector<uint32_t>     indices_u32;           // triangle list
    std::vector<int>          per_vertex_node_index; // length == mesh.positions.size() / 3
    std::vector<RingMetadata> ring_metadata;         // P2 fork-blend scaffolding
    // P3 — per-vertex (trunk, branch, twig, leaf) wind tier bytes for the
    // `_RYNX_WIND` GLB extension. Length == 4 * (mesh.positions.size() / 3).
    std::vector<uint8_t>      wind_weights_packed;
    // C1 P6 — per-vertex tangent (xyzw packed, 4 floats/vert). Circumferential
    // U-derivative of cylinder surface; w=+1 (right-handed TBN: B=cross(N,T)*w).
    std::vector<float>        tangents;
};

BarkMeshOutput build_bark_mesh(const TreeSkeleton& skel, const BarkMeshOptions& opts);

} // namespace treegen
