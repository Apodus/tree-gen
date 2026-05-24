#include "space_colonization.hpp"

#include "det_rng.hpp"
#include "envelopes.hpp"
#include "halton3d.hpp"
#include "radius_solver.hpp"
#include "skeleton.hpp"
#include "tree_descriptor.hpp"
#include "tropism.hpp"
#include "vec3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace treegen {

namespace {

struct Attractor {
    vec3     pos;
    uint32_t halton_idx;   // load-bearing for FP-deterministic tie-break
    bool     alive;
};

struct Pair {
    int      tip_index;    // index into skel.nodes
    float    dist_sq;
    uint32_t halton_idx;   // tie-break key
    int      attractor_index;
};

// Comparator pinned at the call site: (dist², halton_idx) — the FP-determinism
// seal. Equal distances tie-break on halton index, which is monotonic and
// integer (no FP drift). Then by tip_index for total ordering.
bool pair_less(const Pair& a, const Pair& b) {
    if (a.tip_index != b.tip_index) return a.tip_index < b.tip_index;
    if (a.dist_sq   != b.dist_sq)   return a.dist_sq   < b.dist_sq;
    return a.halton_idx < b.halton_idx;
}

// Build a basis (u, v, w) where w == axis (unit). u, v are unit orthogonal.
// Used to perturb a direction by an angle around `axis`.
void build_basis(vec3 w, vec3& u, vec3& v) {
    // pick an axis-aligned vector not parallel to w
    vec3 a = std::abs(w.z) < 0.9f ? vec3{0,0,1} : vec3{1,0,0};
    u = normalized(cross(a, w));
    v = cross(w, u);
}

vec3 perturb_direction(vec3 dir, float angle_deg, pcg32& rng) {
    if (angle_deg <= 0.0f) return dir;
    vec3 u, v;
    build_basis(dir, u, v);
    // Cone perturbation: uniform on the spherical cap of half-angle `angle_deg`.
    const float r1 = rng.next_float_01();
    const float r2 = rng.next_float_01();
    const float cos_max = std::cos(angle_deg * 3.14159265358979323846f / 180.0f);
    const float cos_t   = 1.0f - r1 * (1.0f - cos_max);
    const float sin_t   = std::sqrt(std::max(0.0f, 1.0f - cos_t * cos_t));
    const float phi     = 6.28318530717958647692f * r2;
    const float cx      = std::cos(phi) * sin_t;
    const float cy      = std::sin(phi) * sin_t;
    return normalized(u * cx + v * cy + dir * cos_t);
}

} // anonymous namespace

TreeSkeleton grow_skeleton(const TreeDescriptor& desc, uint64_t seed_effective) {
    TreeSkeleton skel;

    pcg32 rng;
    rng.seed(seed_effective, 0xC0FFEE);

    // Step 1: Halton-sample attractors inside the envelope AABB; reject
    // outside the envelope shape. 4× over-budget gives room for tight shapes
    // (cones, fastigiate columns) where most AABB samples fall outside.
    const Aabb aabb = envelope_aabb(desc.envelope.shape, desc.envelope, desc.height_m);
    const vec3 extent = aabb.max - aabb.min;
    const float crown_base_z = desc.branching.crown_base_fraction * desc.height_m;

    // C11: gradual branching onset — linear attractor density ramp near crown base.
    const float onset_ramp = desc.crown_onset_ramp_fraction;
    const float onset_ceil = crown_base_z + onset_ramp * desc.height_m;

    std::vector<Attractor> attractors;
    attractors.reserve(static_cast<size_t>(desc.branching.attractor_count));
    const uint32_t max_halton = static_cast<uint32_t>(std::max(
        4 * desc.branching.attractor_count, desc.branching.attractor_count + 1));
    for (uint32_t hi = 0; hi < max_halton &&
         static_cast<int>(attractors.size()) < desc.branching.attractor_count; ++hi) {
        const vec3 h = halton_3d(hi);
        const vec3 candidate = {
            aabb.min.x + extent.x * h.x,
            aabb.min.y + extent.y * h.y,
            aabb.min.z + extent.z * h.z,
        };
        if (candidate.z >= crown_base_z &&
            envelope_contains(desc.envelope.shape, desc.envelope, candidate, desc.height_m)) {
            // Onset ramp: accept with probability proportional to height in ramp zone.
            if (onset_ramp > 0.0f && candidate.z < onset_ceil) {
                const float p = (candidate.z - crown_base_z) / (onset_ceil - crown_base_z);
                // Use halton_idx bit-hash as deterministic uniform [0,1) source.
                const float u = static_cast<float>(hi * 2654435761u & 0xFFFFu) / 65536.0f;
                if (u >= p) continue;
            }
            attractors.push_back({candidate, hi, true});
        }
    }

    // Step 2: seed trunk. When crown_base_fraction > 0, pre-build a trunk
    // chain from z=0 to z=crown_base_z so the topmost tip can reach
    // attractors in the crown zone. Without this, the root at z=0 is outside
    // the influence_radius of the nearest attractor and the tree never grows.
    //
    // C11: trunk sway — small lateral sinusoidal displacement gives a natural
    // slight spiral instead of a perfectly straight stick.
    const float sway_amp = desc.trunk_sway_amplitude_m < 0.0f
                               ? 0.03f * desc.height_m
                               : desc.trunk_sway_amplitude_m;
    {
        BranchNode root;
        root.parent_index   = -1;
        root.position       = {0.0f, 0.0f, 0.0f};
        root.axis           = {0.0f, 0.0f, 1.0f};
        root.depth          = 0;
        root.t_along_parent = 1.0f;
        root.wind_tier      = 0;
        skel.nodes.push_back(root);

        const float trunk_step = desc.branching.growth_distance;
        const int trunk_segments = static_cast<int>(crown_base_z / trunk_step);
        const float pi2 = 6.28318530717958647692f;
        for (int s = 1; s <= trunk_segments; ++s) {
            const int parent = static_cast<int>(skel.nodes.size()) - 1;
            const float z = trunk_step * static_cast<float>(s);
            const float phase = z * pi2 / desc.height_m;
            const float dx = sway_amp * std::sin(phase);
            const float dy = sway_amp * std::cos(phase * 0.7f);
            BranchNode n;
            n.parent_index   = parent;
            n.position       = {dx, dy, z};
            n.axis           = {0.0f, 0.0f, 1.0f};
            n.depth          = 0;
            n.t_along_parent = 1.0f;
            n.wind_tier      = 0;
            skel.nodes.push_back(n);
        }
    }

    std::vector<int> tips;
    tips.push_back(static_cast<int>(skel.nodes.size()) - 1); // topmost trunk node is the first active tip

    const float influence_radius    = 4.0f * desc.branching.growth_distance;
    const float influence_radius_sq = influence_radius * influence_radius;
    const float kill_radius         = desc.branching.kill_distance;
    const float kill_radius_sq      = kill_radius * kill_radius;
    const float growth_distance     = desc.branching.growth_distance;

    int alive_count = static_cast<int>(attractors.size());

    // C11: central leader ceiling — depth-0 leader extends this far above crown base.
    const float crown_height = desc.height_m - crown_base_z;
    const float leader_ceiling = crown_base_z + desc.crown_leader_fraction * crown_height;

    // Step 3: growth iterations.
    for (int iter = 0; iter < desc.branching.max_iterations; ++iter) {
        if (alive_count == 0) break;

        // For each live attractor: find nearest tip within influence_radius.
        std::vector<Pair> pairs;
        pairs.reserve(static_cast<size_t>(alive_count));
        for (int ai = 0; ai < static_cast<int>(attractors.size()); ++ai) {
            const auto& a = attractors[ai];
            if (!a.alive) continue;
            int   best_tip = -1;
            float best_d2  = influence_radius_sq;
            for (int t : tips) {
                const vec3  d = skel.nodes[t].position - a.pos;
                const float d2 = length_squared(d);
                if (d2 < best_d2) {
                    best_d2  = d2;
                    best_tip = t;
                }
            }
            if (best_tip >= 0) {
                pairs.push_back({best_tip, best_d2, a.halton_idx, ai});
            }
        }

        if (pairs.empty()) break;

        // FP-determinism seal: stable_sort by (tip_index, dist², halton_idx).
        // Groups by tip; per-tip orders by distance with halton-tie-break.
        std::stable_sort(pairs.begin(), pairs.end(), pair_less);

        // Walk grouped-by-tip ranges. tips iterate in sorted-tip order →
        // deterministic regardless of `tips` insertion sequence.
        bool made_progress = false;
        std::vector<int> new_tips;
        new_tips.reserve(tips.size());

        size_t i = 0;
        while (i < pairs.size()) {
            const int tip_idx = pairs[i].tip_index;
            size_t j = i;
            while (j < pairs.size() && pairs[j].tip_index == tip_idx) ++j;

            // pairs[i..j) all target tip_idx.
            const int attr_count = static_cast<int>(j - i);
            const vec3 tip_pos = skel.nodes[tip_idx].position;

            // Mean direction toward attractors.
            vec3 mean_dir = {0,0,0};
            for (size_t k = i; k < j; ++k) {
                const vec3 d = normalized(attractors[pairs[k].attractor_index].pos - tip_pos);
                mean_dir = mean_dir + d;
            }
            const vec3 mean_unit = normalized(mean_dir);

            // Apply tropisms.
            vec3 grow_dir = apply_tropisms(mean_unit, desc.tropisms,
                                           desc.branching.tropism_strength);

            // Compute angular spread (max angle between any pair of attractor
            // directions). Used to decide if this tip should split.
            float max_cos = 1.0f;  // 1 = colinear (zero spread)
            if (attr_count >= 2) {
                std::vector<vec3> dirs;
                dirs.reserve(static_cast<size_t>(attr_count));
                for (size_t k = i; k < j; ++k) {
                    dirs.push_back(normalized(attractors[pairs[k].attractor_index].pos - tip_pos));
                }
                for (size_t a = 0; a < dirs.size(); ++a) {
                    for (size_t b = a + 1; b < dirs.size(); ++b) {
                        const float c = dot(dirs[a], dirs[b]);
                        if (c < max_cos) max_cos = c;
                    }
                }
            }
            const float max_angle_deg = std::acos(std::max(-1.0f, std::min(1.0f, max_cos))) *
                                        (180.0f / 3.14159265358979323846f);

            // Spawn primary child node.
            // C11 depth-inflation fix: continuation (primary) inherits parent depth;
            // only split children increment. Central leader: depth-0 primary
            // stays 0 while child z < leader_ceiling; above it, primary gets 1.
            const int parent_depth = skel.nodes[tip_idx].depth;
            const vec3 child_pos = tip_pos + grow_dir * growth_distance;

            int child_depth;
            if (parent_depth == 0 && child_pos.z >= leader_ceiling)
                child_depth = 1; // leader terminated
            else
                child_depth = parent_depth; // continuation — same depth

            const int new_idx = static_cast<int>(skel.nodes.size());
            BranchNode child;
            child.parent_index   = tip_idx;
            child.position       = child_pos;
            child.axis           = grow_dir;
            child.depth          = child_depth;
            child.t_along_parent = 1.0f;
            child.wind_tier      = std::min(3, child.depth / 4);
            skel.nodes.push_back(child);
            new_tips.push_back(new_idx);
            made_progress = true;

            // Optional split: if enough attractors AND wide angular spread.
            bool did_split = false;
            if (attr_count >= desc.branching.min_attractors_to_split &&
                max_angle_deg > desc.branching.angular_spread_split_deg) {
                const vec3 perturbed = perturb_direction(grow_dir,
                                          desc.branching.branch_angle_jitter_deg, rng);
                const vec3 split_pos = tip_pos + perturbed * growth_distance;
                const int  split_idx = static_cast<int>(skel.nodes.size());
                BranchNode split;
                split.parent_index   = tip_idx;
                split.position       = split_pos;
                split.axis           = perturbed;
                split.depth          = parent_depth + 1; // fork — increment depth
                split.t_along_parent = 1.0f;
                split.wind_tier      = std::min(3, split.depth / 4);
                skel.nodes.push_back(split);
                new_tips.push_back(split_idx);
                did_split = true;
            }

            // Kill attractors within kill_distance of the new node(s).
            for (size_t k = i; k < j; ++k) {
                Attractor& a = attractors[pairs[k].attractor_index];
                if (!a.alive) continue;
                const float d2 = length_squared(a.pos - skel.nodes[new_idx].position);
                if (d2 < kill_radius_sq) {
                    a.alive = false;
                    --alive_count;
                    continue;
                }
                if (did_split) {
                    const int split_idx = static_cast<int>(skel.nodes.size()) - 1;
                    const float d2s = length_squared(a.pos - skel.nodes[split_idx].position);
                    if (d2s < kill_radius_sq) {
                        a.alive = false;
                        --alive_count;
                    }
                }
            }

            i = j;
        }

        // Tips that didn't grow this iteration stay alive (they may pick up
        // attractors in later iters as other branches reshape the field).
        // Append new children. Cap tip count to avoid runaway with pathologically
        // dense attractor configurations.
        for (int t : new_tips) tips.push_back(t);

        ++skel.iterations_run;
        if (!made_progress) break;
    }

    skel.attractors_consumed = static_cast<int>(attractors.size()) - alive_count;

    // Step 4: solve radii.
    solve_radii(skel, desc);

    return skel;
}

} // namespace treegen
