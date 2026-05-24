// Da Vinci's radius rule: parent_radius^n = sum(child_radius^n). For n=2.0
// this is exact area-conservation through bifurcations (matches xylem fluid
// transport, hence the visual realism). n < 2 produces thicker trunks (oak,
// maple); n > 2 produces whippier trees (birch, willow).
//
// Backward-postorder solver: node indices grow during space colonization
// (children always pushed AFTER their parent), so reverse iteration is
// postorder. Leaf-tip radius is explicitly seeded to 0.005 * trunk_base BEFORE
// reduction — without this the leaf loop reads uninitialized child radii.
//
// Root radius is force-set to trunk_base after the reduction (the descriptor
// IS the spec; the reduction's root output would only equal trunk_base when
// the entire tree just happens to match the user's input).
#pragma once

namespace treegen {

struct TreeSkeleton;
struct TreeDescriptor;

void solve_radii(TreeSkeleton& skel, const TreeDescriptor& desc);

} // namespace treegen
