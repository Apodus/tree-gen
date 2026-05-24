#include "leaf_placement.hpp"

#include "det_rng.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace treegen::leaf_placement {

namespace {

constexpr uint64_t k_stream_branchwalk = 0xC5'01'06ULL;

float vlen(vec3 v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

void make_basis(vec3 axis, vec3& out_u, vec3& out_v) {
    const vec3 ref = std::abs(axis.z) < 0.9f ? vec3{0.0f, 0.0f, 1.0f}
                                              : vec3{1.0f, 0.0f, 0.0f};
    out_u = normalized(cross(axis, ref));
    out_v = normalized(cross(axis, out_u));
}

} // anonymous namespace

Options options_from_descriptor(const TreeDescriptor::Leaves& src) {
    Options o;
    o.geometry_type        = src.geometry_type;
    o.shape                = src.shape;
    o.leaf_size_m          = src.leaf_size_m;
    o.cluster_count_per_tip = src.cluster_count_per_tip;
    o.leaf_min_branch_depth    = src.leaf_min_branch_depth;
    o.leaf_density_per_meter   = src.leaf_density_per_meter;
    o.leaf_depth_density_curve = src.leaf_depth_density_curve;
    o.leaf_phototropic_bias    = src.leaf_phototropic_bias;
    return o;
}

std::vector<LeafSite> generate_leaf_sites(const TreeSkeleton& skel,
                                          const Options& opts,
                                          uint64_t seed_effective) {
    std::vector<LeafSite> sites;
    if (skel.nodes.empty()) return sites;

    int max_depth = 0;
    for (const auto& n : skel.nodes) max_depth = std::max(max_depth, n.depth);
    if (max_depth == 0) return sites;

    pcg32 rng;
    rng.seed(seed_effective, k_stream_branchwalk);

    constexpr float k_two_pi = 6.28318530717958647692f;
    constexpr vec3 k_world_up{0.0f, 0.0f, 1.0f};

    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const BranchNode& child  = skel.nodes[i];
        if (child.depth < opts.leaf_min_branch_depth) continue;
        const BranchNode& parent = skel.nodes[static_cast<size_t>(child.parent_index)];

        const vec3 seg = child.position - parent.position;
        const float seg_len = vlen(seg);
        if (seg_len <= 1e-5f) continue;

        const vec3 seg_dir = seg / seg_len;
        vec3 u, v;
        make_basis(seg_dir, u, v);

        const float depth_frac = static_cast<float>(child.depth) / static_cast<float>(max_depth);
        const float density = opts.leaf_density_per_meter * std::pow(depth_frac, opts.leaf_depth_density_curve);
        const int n_leaves = std::max(1, static_cast<int>(density * seg_len));

        for (int j = 0; j < n_leaves; ++j) {
            const float t_base = (static_cast<float>(j) + 0.5f) / static_cast<float>(n_leaves);
            const float t_jit  = (rng.next_float_01() - 0.5f) / static_cast<float>(n_leaves);
            const float t      = std::clamp(t_base + t_jit, 0.0f, 1.0f);

            const float az = rng.next_float_01() * k_two_pi;
            const float c  = std::cos(az);
            const float s  = std::sin(az);
            const vec3  radial_dir = normalized(u * c + v * s);

            const float interp_radius = parent.radius + (child.radius - parent.radius) * t;
            const float lateral = std::max(0.005f, interp_radius) + opts.leaf_size_m * 0.5f;
            const vec3 p = parent.position + seg * t + radial_dir * lateral;

            LeafSite ls;
            ls.position              = p;
            ls.normal                = normalized(radial_dir * (1.0f - opts.leaf_phototropic_bias)
                                                + k_world_up * opts.leaf_phototropic_bias);
            ls.branch_id             = static_cast<int>(i);
            ls.type                  = static_cast<int>(opts.shape);
            ls.age                   = rng.next_float_01();
            ls.branch_tangent        = seg_dir;
            ls.branch_normal         = radial_dir;
            ls.branch_depth_fraction = depth_frac;
            sites.push_back(ls);
        }
    }

    // Fill branch_depth_fraction from branch_id (single source of truth).
    {
        int md = max_depth;
        if (md < 4) md = 4;  // match wind_weights.cpp clamp
        const float inv_max = 1.0f / static_cast<float>(md);
        for (auto& s : sites) {
            const int bid = s.branch_id;
            if (bid >= 0 && bid < static_cast<int>(skel.nodes.size()))
                s.branch_depth_fraction = static_cast<float>(skel.nodes[static_cast<size_t>(bid)].depth) * inv_max;
        }
    }

    return sites;
}

} // namespace treegen::leaf_placement
