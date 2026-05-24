// C4 P2 — skin-rim fork blending.
//
// Closes T-junctions at branch forks by blending each child branch's first
// rings (those within `zone_length = zone_factor * parent.radius` axial
// distance from the fork point) toward the parent branch's last ring. At the
// fork point itself (axial_distance=0) the child ring verts coincide with the
// parent ring verts angularly-matched — closes the visual gap.
//
// Unmatched-vert stitch (verifier-r2 catch): when N_parent != N_child, some
// parent verts have no direct child match. Fan-stitch triangles connect them
// to the child's 2 angular-nearest verts so the surface is closed and no
// edge has > 2 adjacent triangles after blending.
#pragma once

#include "skeleton.hpp"
#include "branch_mesh.hpp"

#include <vector>

namespace treegen {

struct ForkZone {
    int              node_index;        // skeleton node at the fork point (has >1 child)
    std::vector<int> child_indices;     // skeleton node indices of children
    float            zone_length_m = 0.0f;  // zone_factor * skel.nodes[node_index].radius
};

// Walks the skeleton; for every node with >1 child emits a ForkZone.
// Deterministic order: parent nodes in ascending node-index; child_indices in
// ascending node-index.
std::vector<ForkZone> collect_fork_zones(const TreeSkeleton& skel, float zone_factor = 1.5f);

// Emits Hermite collar rings at each fork + stitches them to parent rim and
// child first ring. Replaces the old position-blend + degenerate-stitch path.
void apply_skin_rim_blend(BarkMeshOutput&              bark,
                          const TreeSkeleton&          skel,
                          const std::vector<ForkZone>& zones,
                          const BarkMeshOptions&       opts);

} // namespace treegen
