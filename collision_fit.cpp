#include "collision_fit.hpp"
#include "skeleton.hpp"
#include "tree_descriptor.hpp"

#include <algorithm>
#include <cassert>

namespace treegen {

TrunkCapsule compute_trunk_capsule(const TreeSkeleton& skel, const TreeDescriptor& desc) {
    assert(!skel.nodes.empty());
    assert(skel.nodes[0].radius > 0.0f);
    const float trunk_height = desc.height_m * desc.branching.crown_base_fraction;
    TrunkCapsule cap;
    cap.half_length = trunk_height * 0.5f;
    cap.radius      = std::max(skel.nodes[0].radius, 0.05f);
    return cap;
}

} // namespace treegen
