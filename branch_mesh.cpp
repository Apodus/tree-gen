// C4 P1+P2 — bark mesh cylinder extrusion + fork blend wiring.
// /fp:precise (treegen.sharpmake.cs).
//
// Per parent→child segment: emit (axial_segs+1) rings of (N+1) verts (seam
// duplicate). Radius linearly tapers between parent.radius and child.radius
// along the segment. Per-vertex normals = outward radial; per-vertex UVs from
// bark_uv::ring_uv with the per-tree seam_offset.
//
// RingMetadata is captured per emitted ring so P2 fork blend (fork_blend.cpp)
// can find child rings near a fork by (branch_node_index, axial_distance_m).
// After all tubes are emitted, apply_skin_rim_blend snaps child first-rings
// to the parent's rim ring + emits a sector-partitioned bipartite stitch
// (manifold even when N_parent != N_child or > 2 children per fork).
#include "branch_mesh.hpp"
#include "bark_uv.hpp"
#include "fork_blend.hpp"
#include "vec3.hpp"
#include "wind_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace treegen {

namespace {

// Orthonormal basis (right, up) ⟂ axis. Mirrors space_colonization::build_basis
// (axis-aligned reference, prefer +Z unless axis is nearly ±Z).
void build_basis(vec3 axis, vec3& out_right, vec3& out_up) {
    axis = normalized(axis);
    vec3 ref = (std::abs(axis.z) < 0.9f) ? vec3{0.0f, 0.0f, 1.0f} : vec3{1.0f, 0.0f, 0.0f};
    out_right = normalized(cross(ref, axis));
    out_up    = cross(axis, out_right); // already unit
}

// Project prev_right onto the plane perpendicular to new_axis, preserving angular
// continuity across direction changes (Bishop frame / parallel transport).
vec3 parallel_transport(vec3 prev_right, vec3 new_axis) {
    new_axis = normalized(new_axis);
    const vec3 projected = prev_right - new_axis * dot(prev_right, new_axis);
    const float len = length(projected);
    if (len < 1e-6f) {
        vec3 r, u;
        build_basis(new_axis, r, u);
        return r;
    }
    return projected * (1.0f / len);
}

// Emit one ring of (N+1) verts (seam-duplicated) at `center` orthogonal to
// `axis`, radius `radius`, axial param `v_axial`. Returns starting vertex
// index.
// Quantize a [0,1] blend weight to a u8 — same idiom as wind_weights.cpp.
inline uint8_t quantize_blend(float w) {
    const float c = (w < 0.0f) ? 0.0f : (w > 1.0f ? 1.0f : w);
    return static_cast<uint8_t>(c * 255.0f + 0.5f);
}

int emit_ring(cpu_mesh_out& mesh,
              std::vector<float>& tangents,
              std::vector<int>& vert_node_idx,
              std::vector<uint8_t>& bone_blend,
              uint8_t ring_blend,
              int node_index,
              vec3 center,
              vec3 axis,
              vec3 right,
              vec3 up,
              float radius,
              int N,
              float v_axial,
              float seam_offset_rad)
{
    (void)axis; // basis supplied explicitly; axis retained for call-site readability.
    int start = static_cast<int>(mesh.positions.size() / 3);
    constexpr float k_2pi = 6.28318530717958647692f;

    for (int i = 0; i <= N; ++i) {
        // Seam duplicate: i=N shares angular position with i=0 (exact).
        const int   i_mod  = (i == N) ? 0 : i;
        const float theta  = (static_cast<float>(i_mod) / static_cast<float>(N)) * k_2pi + seam_offset_rad;
        const float cs     = std::cos(theta);
        const float sn     = std::sin(theta);
        const vec3 radial  = right * cs + up * sn;
        const vec3 pos     = center + radial * radius;

        mesh.positions.push_back(pos.x);
        mesh.positions.push_back(pos.y);
        mesh.positions.push_back(pos.z);

        mesh.normals.push_back(radial.x);
        mesh.normals.push_back(radial.y);
        mesh.normals.push_back(radial.z);

        auto [u, v] = bark_uv::ring_uv(i, N, v_axial, seam_offset_rad);
        mesh.uvs.push_back(u);
        mesh.uvs.push_back(v);

        // C1 P6 — tangent = circumferential U-derivative: dP/dtheta normalized.
        const vec3 tan = normalized(right * (-sn) + up * cs);
        tangents.push_back(tan.x);
        tangents.push_back(tan.y);
        tangents.push_back(tan.z);
        tangents.push_back(1.0f); // w = +1 right-handed

        vert_node_idx.push_back(node_index);
        bone_blend.push_back(ring_blend);
    }
    return start;
}

// Two CCW triangles per quad between two adjacent (N+1)-vert rings.
void emit_strip(std::vector<uint32_t>& indices, int ring_a_start, int ring_b_start, int N) {
    for (int i = 0; i < N; ++i) {
        const uint32_t a0 = static_cast<uint32_t>(ring_a_start + i);
        const uint32_t a1 = static_cast<uint32_t>(ring_a_start + i + 1);
        const uint32_t b0 = static_cast<uint32_t>(ring_b_start + i);
        const uint32_t b1 = static_cast<uint32_t>(ring_b_start + i + 1);
        // Quad (a0, a1, b1, b0) → (a0, a1, b1) + (a0, b1, b0).
        indices.push_back(a0); indices.push_back(a1); indices.push_back(b1);
        indices.push_back(a0); indices.push_back(b1); indices.push_back(b0);
    }
}

int radial_for_depth(int depth, const BarkMeshOptions& opts) {
    if (depth == 0) return opts.radial_seg_trunk;
    if (depth == 1) return opts.radial_seg_order1;
    if (depth == 2) return opts.radial_seg_order2;
    return opts.radial_seg_order3_plus;
}

} // anonymous namespace

BarkMeshOutput build_bark_mesh(const TreeSkeleton& skel, const BarkMeshOptions& opts) {
    BarkMeshOutput out;
    auto& mesh          = out.mesh;
    auto& indices       = out.indices_u32;
    auto& vert_node_idx = out.per_vertex_node_index;
    auto& rings         = out.ring_metadata;

    const int node_count = static_cast<int>(skel.nodes.size());

    // C12 P2 — child count per node for continuation merge + P3 tip caps.
    std::vector<int> child_count(static_cast<size_t>(node_count), 0);
    for (int i = 0; i < node_count; ++i) {
        const int p = skel.nodes[static_cast<size_t>(i)].parent_index;
        if (p >= 0) {
            // Respect cull mask: culled children don't count.
            if (!opts.culled_node_mask.empty() && opts.culled_node_mask[size_t(i)]) continue;
            ++child_count[static_cast<size_t>(p)];
        }
    }

    // C12 P2 — per-node: vert_start of the last ring emitted for the segment
    // terminating at that node + its radial segment count. -1 = not yet emitted.
    struct LastRing { int vert_start = -1; int N = 0; vec3 right = {0,0,0}; };
    std::vector<LastRing> last_ring_at_node(static_cast<size_t>(node_count));

    for (int i = 1; i < node_count; ++i) {
        // Per-node merge mask (face_budget's merge-by-length overflow rule).
        if (!opts.culled_node_mask.empty() && opts.culled_node_mask[size_t(i)]) continue;

        const auto& node = skel.nodes[i];
        if (node.parent_index < 0) continue;
        const auto& parent = skel.nodes[static_cast<size_t>(node.parent_index)];

        const vec3 raw = node.position - parent.position;
        const float seg_len = length(raw);
        if (seg_len < 1e-5f) continue; // skip degenerate segment
        const vec3 seg_dir = raw / seg_len;

        const int axial_segs = std::max(1, static_cast<int>(std::floor(seg_len / opts.axial_segment_length_m)));
        int N = radial_for_depth(static_cast<int>(node.depth), opts);
        if (N == 0) continue;       // P4 cull-this-depth-entirely signal (legacy)
        if (N < 3) N = 3;

        // C12 P2 — continuation merge: if the parent had exactly 1 child (this
        // node) and the parent's last ring has matching N, reuse it as s=0.
        const auto& parent_lr = last_ring_at_node[static_cast<size_t>(node.parent_index)];
        const bool can_merge = (child_count[static_cast<size_t>(node.parent_index)] == 1)
                               && (parent_lr.vert_start >= 0)
                               && (parent_lr.N == N);

        int prev_ring_start = -1;
        const int s_start = can_merge ? 1 : 0;
        if (can_merge) {
            prev_ring_start = parent_lr.vert_start;
        }

        // Compute basis for this segment. If continuation-merged, transport
        // from the parent's basis to maintain angular coherence across the
        // direction change. Otherwise, fresh basis via build_basis.
        vec3 seg_right, seg_up;
        if (can_merge) {
            const vec3& parent_right = last_ring_at_node[static_cast<size_t>(node.parent_index)].right;
            seg_right = parallel_transport(parent_right, seg_dir);
            seg_up = cross(seg_dir, seg_right);
        } else {
            build_basis(seg_dir, seg_right, seg_up);
        }

        for (int s = s_start; s <= axial_segs; ++s) {
            const float t = static_cast<float>(s) / static_cast<float>(axial_segs);
            const vec3  center = parent.position + seg_dir * (seg_len * t);
            const float radius = parent.radius + (node.radius - parent.radius) * t;
            const float v_axial = (seg_len * t) / opts.bark_repeat_m;

            // C12 P4 — root flare: widen the lowest rings of the trunk base.
            float flared_radius = radius;
            if (node.parent_index == 0 && opts.root_flare_factor > 1.0f) {
                // Cubic ease-in over the first 3 rings (or fewer if axial_segs < 3).
                const int flare_rings = std::min(3, axial_segs);
                if (s <= flare_rings) {
                    const float ease_t = static_cast<float>(s) / static_cast<float>(flare_rings);
                    const float cubic = ease_t * ease_t * ease_t;
                    flared_radius = radius * (opts.root_flare_factor + (1.0f - opts.root_flare_factor) * cubic);
                }
            }

            // C1 P1 — junction shoulder swell: widen the last ~30% of rings
            // approaching a fork node (child_count >= 2). Smoothstep taper
            // from normal radius to shoulder_radius, clamped to parent radius.
            if (opts.junction_shoulder_factor > 0.0f && child_count[static_cast<size_t>(i)] >= 2) {
                constexpr float k_swell_onset = 0.7f;
                if (t >= k_swell_onset) {
                    float shoulder_radius = node.radius * (1.0f + opts.junction_shoulder_factor);
                    shoulder_radius = std::min(shoulder_radius, parent.radius);
                    const float swell_t = (t - k_swell_onset) / (1.0f - k_swell_onset);
                    const float smooth = swell_t * swell_t * (3.0f - 2.0f * swell_t);
                    const float swelled = flared_radius + (shoulder_radius - flared_radius) * smooth;
                    if (swelled > flared_radius) {
                        flared_radius = swelled;
                    }
                }
            }

            // C8-wind P1 — parent-blend = quantize(1 - t): 255 at parent end
            // (t=0), 0 at child end (t=1). 2-bone LBS fork-continuity.
            const uint8_t ring_blend = quantize_blend(1.0f - t);
            const int ring_start = emit_ring(mesh, out.tangents, vert_node_idx, out.bone_blend,
                                             ring_blend, i, center, seg_dir,
                                             seg_right, seg_up, flared_radius, N, v_axial,
                                             opts.seam_offset_rad);

            RingMetadata rm;
            rm.branch_node_index = i;
            rm.axial_index       = s;
            rm.axial_segs        = axial_segs;
            rm.vert_start        = ring_start;
            rm.N                 = N;
            rm.axial_distance_m  = seg_len * t;
            rm.branch_length_m   = seg_len;
            rings.push_back(rm);

            if (prev_ring_start >= 0) {
                emit_strip(indices, prev_ring_start, ring_start, N);
            }
            prev_ring_start = ring_start;
        }

        // Record last ring for potential downstream continuation merge.
        last_ring_at_node[static_cast<size_t>(i)] = { prev_ring_start, N, seg_right };

        // C12 P3 — tip end-caps: triangle fan for leaf nodes (no children).
        if (child_count[static_cast<size_t>(i)] == 0 && prev_ring_start >= 0) {
            // Emit centroid vertex at node.position.
            const int centroid = static_cast<int>(mesh.positions.size() / 3);
            mesh.positions.push_back(node.position.x);
            mesh.positions.push_back(node.position.y);
            mesh.positions.push_back(node.position.z);
            // Normal points along the branch axis (outward from tip).
            const vec3 tip_n = seg_dir;
            mesh.normals.push_back(tip_n.x);
            mesh.normals.push_back(tip_n.y);
            mesh.normals.push_back(tip_n.z);
            mesh.uvs.push_back(0.5f);
            mesh.uvs.push_back(0.5f);
            // Tip-cap tangent: ring's right basis vector (segment basis still in scope).
            out.tangents.push_back(seg_right.x);
            out.tangents.push_back(seg_right.y);
            out.tangents.push_back(seg_right.z);
            out.tangents.push_back(1.0f);
            vert_node_idx.push_back(i);
            out.bone_blend.push_back(0u); // tip cap: t=1, fully host (no parent blend)

            // Fan from last ring to centroid.
            for (int j = 0; j < N; ++j) {
                const uint32_t a = static_cast<uint32_t>(prev_ring_start + j);
                const uint32_t b = static_cast<uint32_t>(prev_ring_start + j + 1);
                const uint32_t c = static_cast<uint32_t>(centroid);
                indices.push_back(a);
                indices.push_back(b);
                indices.push_back(c);
            }
        }
    }

    if (opts.apply_fork_blend) {
        auto zones = collect_fork_zones(skel, opts.fork_zone_factor);
        // Strip merged children from each fork's child_indices so the angular
        // Voronoi partition only allocates parent rim arc to children that
        // actually emit bark. Merge-cascade: a zone with 0 surviving children
        // is no longer a fork → no collar/crotch geometry emitted, saving tri
        // budget for forks with surviving children. A zone with 1 surviving
        // child remains valid (sector == full parent rim, manifold by
        // construction).
        if (!opts.culled_node_mask.empty()) {
            for (auto& z : zones) {
                z.child_indices.erase(
                    std::remove_if(z.child_indices.begin(), z.child_indices.end(),
                        [&](int ci) {
                            return ci >= 0 && size_t(ci) < opts.culled_node_mask.size()
                                && opts.culled_node_mask[size_t(ci)];
                        }),
                    z.child_indices.end());
            }
            zones.erase(
                std::remove_if(zones.begin(), zones.end(),
                    [](const ForkZone& z) { return z.child_indices.empty(); }),
                zones.end());
        }
        apply_skin_rim_blend(out, skel, zones, opts);
    }

    // P3 — wind weight bake. Runs AFTER fork blend so the appended collar-ring
    // and crotch-centroid vertices are covered: apply_skin_rim_blend mutates
    // existing positions AND appends new vertices (alongside per_vertex_node_index,
    // tangents, and bone_blend), so vcount here is the final count.
    const size_t vcount = mesh.positions.size() / 3;
    std::vector<float> z_per_vert;
    z_per_vert.reserve(vcount);
    for (size_t i = 0; i < vcount; ++i) {
        z_per_vert.push_back(mesh.positions[i * 3 + 2]);
    }
    out.wind_weights_packed = bake_wind_weights(skel, vert_node_idx, z_per_vert, opts.tree_height_m);

    return out;
}

} // namespace treegen
