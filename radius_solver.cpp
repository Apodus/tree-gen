#include "radius_solver.hpp"

#include "skeleton.hpp"
#include "tree_descriptor.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace treegen {

void solve_radii(TreeSkeleton& skel, const TreeDescriptor& desc) {
    const size_t n = skel.nodes.size();
    if (n == 0) return;

    const float trunk_base      = desc.trunk_base_radius_m;
    const float taper_exponent  = desc.taper_exponent;
    const float min_radius      = desc.branching.min_branch_radius_m;
    const float trunk_taper     = desc.trunk_taper_rate;
    const float crown_base_z    = desc.branching.crown_base_fraction * desc.height_m;

    // Step 1: identify leaves.
    std::vector<char> has_child(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const int p = skel.nodes[i].parent_index;
        if (p >= 0) has_child[static_cast<size_t>(p)] = 1;
    }

    // Step 2: seed leaf-tip radii.
    const float leaf_radius = 0.005f * trunk_base;
    for (size_t i = 0; i < n; ++i) {
        if (!has_child[i]) skel.nodes[i].radius = leaf_radius;
    }

    // Step 3: postorder pipe-model reduction (Da Vinci's rule).
    std::vector<double> sum_pow(n, 0.0);
    for (size_t i = n; i-- > 0; ) {
        const auto& node = skel.nodes[i];
        if (has_child[i]) {
            skel.nodes[i].radius = static_cast<float>(
                std::pow(sum_pow[i], 1.0 / static_cast<double>(taper_exponent)));
        }
        if (node.parent_index >= 0) {
            sum_pow[static_cast<size_t>(node.parent_index)] +=
                std::pow(static_cast<double>(skel.nodes[i].radius),
                         static_cast<double>(taper_exponent));
        }
    }

    // Step 4: proportional scale — the pipe model produces relative radii
    // from leaf_radius upward. Scale the entire tree so the root matches
    // trunk_base. This preserves the pipe model's proportional relationships
    // (first-order branches are almost trunk-width) instead of forcing the
    // root alone and creating a discontinuity at the first fork.
    {
        const float pipe_root = skel.nodes[0].radius;
        if (pipe_root > 1e-9f) {
            const float scale = trunk_base / pipe_root;
            for (size_t i = 0; i < n; ++i)
                skel.nodes[i].radius *= scale;
        } else {
            skel.nodes[0].radius = trunk_base;
        }
    }

    // Step 5: trunk taper — linear interpolation on pre-seeded trunk chain
    // (depth==0 nodes from z=0 to z=crown_base_z).
    if (crown_base_z > 0.0f) {
        for (size_t i = 0; i < n; ++i) {
            auto& node = skel.nodes[i];
            if (node.depth != 0) continue;
            const float z = node.position.z;
            if (z > crown_base_z) continue;
            const float t = z / crown_base_z;
            node.radius = trunk_base * (1.0f + (trunk_taper - 1.0f) * t);
        }
    }

    // Step 6: minimum radius floor — structural seal against wire-thin branches.
    for (size_t i = 1; i < n; ++i) {
        skel.nodes[i].radius = std::max(skel.nodes[i].radius, min_radius);
    }

    // Step 7: monotonicity enforcement — Steps 5+6 can violate Da Vinci
    // monotonicity (trunk taper increases parent; min-radius floor lifts child).
    // Top-down clamp: child.radius = min(child.radius, parent.radius).
    for (size_t i = 1; i < n; ++i) {
        const float pr = skel.nodes[static_cast<size_t>(skel.nodes[i].parent_index)].radius;
        skel.nodes[i].radius = std::min(skel.nodes[i].radius, pr);
    }
}

} // namespace treegen
