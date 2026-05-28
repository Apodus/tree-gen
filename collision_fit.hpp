#pragma once

namespace treegen {

struct TreeSkeleton;
struct TreeDescriptor;

struct TrunkCapsule {
    float half_length;
    float radius;
};

TrunkCapsule compute_trunk_capsule(const TreeSkeleton& skel, const TreeDescriptor& desc);

} // namespace treegen
