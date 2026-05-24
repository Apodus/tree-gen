// C5 P1 → C1 P4 — leaf-site generation. Single BranchWalk strategy; output
// is a deterministic vector of LeafSite (position authoritative). Geometry
// emission lives in leaf_geometry.cpp.
//
// Determinism: pcg32-seeded RNG; stable traversal order (input node index).
// Same (skel, opts, seed) -> byte-identical output across runs (/fp:precise).
#pragma once

#include "skeleton.hpp"
#include "tree_descriptor.hpp"

#include <cstdint>
#include <vector>

namespace treegen::leaf_placement {

struct Options {
    LeafGeometryType geometry_type        = LeafGeometryType::BentCrossCluster;
    LeafShape        shape                = LeafShape::OakLobed;
    float leaf_size_m                     = 0.12f;
    int   cluster_count_per_tip           = 6;
    int   leaf_min_branch_depth           = 3;
    float leaf_density_per_meter          = 8.0f;
    float leaf_depth_density_curve        = 1.0f;
    float leaf_phototropic_bias           = 0.0f;
};

// Copy descriptor.leaves into Options.
Options options_from_descriptor(const TreeDescriptor::Leaves& src);

// Build leaf sites via branch-walk placement.
std::vector<LeafSite> generate_leaf_sites(const TreeSkeleton& skel,
                                          const Options& opts,
                                          uint64_t seed_effective);

} // namespace treegen::leaf_placement
