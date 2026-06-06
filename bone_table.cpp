// C8-wind P1 — bone table bake (see bone_table.hpp).
// /fp:precise (treegen.sharpmake.cs).
#include "bone_table.hpp"

#include <cassert>

namespace treegen {

BoneTable build_bone_table(const TreeSkeleton& skel) {
    BoneTable out;
    const size_t node_count = skel.nodes.size();
    // Per-vertex bone indices are u16 (lod_emitter bark cast, leaf_geometry
    // host bone). A skeleton beyond u16 would silently truncate those indices,
    // so make that structurally impossible at the single bake site.
    assert(node_count <= 0xFFFFu && "tree skeleton exceeds u16 bone-index range");
    out.bone_count = static_cast<uint32_t>(node_count);
    out.records.resize(node_count * static_cast<size_t>(K_FLOATS_PER_BONE));

    for (size_t i = 0; i < node_count; ++i) {
        const BranchNode& node = skel.nodes[i];
        const int parent = node.parent_index;

        // Pivot = parent position (the joint node i rotates about). Root
        // (parent < 0) pivots about itself — i.e. it does not rotate.
        vec3 pivot = node.position;
        if (parent >= 0) {
            pivot = skel.nodes[static_cast<size_t>(parent)].position;
        }

        float* rec = &out.records[i * static_cast<size_t>(K_FLOATS_PER_BONE)];
        rec[0] = static_cast<float>(parent);
        rec[1] = static_cast<float>(node.depth);
        rec[2] = pivot.x;
        rec[3] = pivot.y;
        rec[4] = pivot.z;
        rec[5] = node.position.x;
        rec[6] = node.position.y;
        rec[7] = node.position.z;
    }

    return out;
}

}  // namespace treegen
