// P2+3 — Hermite collar mesh at branch forks.
//
// Replaces the old skin-rim position-blend + degenerate bipartite stitch with
// N_collar intermediate rings along a cubic Hermite curve at each fork,
// connected by quad strips. The parent rim sector is stitched to the first
// collar ring; the last collar ring is stitched to the child's s=0 ring.
//
// Sector partition is unchanged: angular Voronoi assigns each parent rim vert
// to the nearest child — each child's stitch walks only its sector arc.
//
// /fp:precise (treegen.sharpmake.cs).
#include "fork_blend.hpp"

#include "bark_uv.hpp"
#include "branch_mesh.hpp"
#include "collar_curve.hpp"
#include "skeleton.hpp"
#include "vec3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace treegen {

namespace {

constexpr float k_2pi = 6.28318530717958647692f;
constexpr float k_pi  = 3.14159265358979323846f;

// Mirrors branch_mesh.cpp::build_basis (kept here to avoid leaking that
// internal symbol). Axis-aligned reference; prefer +Z unless near ±Z.
void build_basis(vec3 axis, vec3& out_right, vec3& out_up) {
    axis = normalized(axis);
    vec3 ref = (std::abs(axis.z) < 0.9f) ? vec3{0.0f, 0.0f, 1.0f} : vec3{1.0f, 0.0f, 0.0f};
    out_right = normalized(cross(ref, axis));
    out_up    = cross(axis, out_right);
}

inline vec3 get_pos(const cpu_mesh_out& mesh, int vert_idx) {
    return {
        mesh.positions[static_cast<size_t>(vert_idx) * 3 + 0],
        mesh.positions[static_cast<size_t>(vert_idx) * 3 + 1],
        mesh.positions[static_cast<size_t>(vert_idx) * 3 + 2],
    };
}

inline float vert_angle(const cpu_mesh_out& mesh, int vert_idx,
                        vec3 origin, vec3 right, vec3 up) {
    const vec3 d = get_pos(mesh, vert_idx) - origin;
    return std::atan2(dot(d, up), dot(d, right));
}

// Wrap an angle delta into (-pi, pi].
inline float angle_wrap_signed(float d) {
    while (d >  k_pi) d -= k_2pi;
    while (d <= -k_pi) d += k_2pi;
    return d;
}

// Shortest angular distance |a - b| in [0, pi].
inline float angle_dist(float a, float b) {
    return std::abs(angle_wrap_signed(a - b));
}

// Mirrors branch_mesh.cpp::emit_strip.
void emit_strip(std::vector<uint32_t>& indices, int ring_a_start, int ring_b_start, int N) {
    for (int i = 0; i < N; ++i) {
        const uint32_t a0 = static_cast<uint32_t>(ring_a_start + i);
        const uint32_t a1 = static_cast<uint32_t>(ring_a_start + i + 1);
        const uint32_t b0 = static_cast<uint32_t>(ring_b_start + i);
        const uint32_t b1 = static_cast<uint32_t>(ring_b_start + i + 1);
        indices.push_back(a0); indices.push_back(a1); indices.push_back(b1);
        indices.push_back(a0); indices.push_back(b1); indices.push_back(b0);
    }
}

struct BranchRingRange {
    int first; // first ring metadata index
    int count;
};

std::vector<BranchRingRange> index_by_branch(const std::vector<RingMetadata>& rings,
                                             int node_count) {
    std::vector<BranchRingRange> out(static_cast<size_t>(node_count), BranchRingRange{-1, 0});
    for (int r = 0; r < static_cast<int>(rings.size()); ++r) {
        const int b = rings[r].branch_node_index;
        if (b < 0 || b >= node_count) continue;
        if (out[static_cast<size_t>(b)].first < 0) {
            out[static_cast<size_t>(b)].first = r;
        }
        ++out[static_cast<size_t>(b)].count;
    }
    return out;
}

// Angular Voronoi: assign each parent rim vert to the nearest child.
std::vector<int> assign_parent_to_children(const std::vector<float>& parent_angles,
                                           const std::vector<float>& child_central_angles) {
    const int N_p = static_cast<int>(parent_angles.size());
    const int N_k = static_cast<int>(child_central_angles.size());
    std::vector<int> assign(static_cast<size_t>(N_p), 0);
    for (int j = 0; j < N_p; ++j) {
        const float theta_j = parent_angles[static_cast<size_t>(j)];
        std::vector<std::pair<float, int>> cands(static_cast<size_t>(N_k));
        for (int k = 0; k < N_k; ++k) {
            cands[static_cast<size_t>(k)] =
                { angle_dist(theta_j, child_central_angles[static_cast<size_t>(k)]), k };
        }
        std::stable_sort(cands.begin(), cands.end(),
            [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });
        assign[static_cast<size_t>(j)] = cands.front().second;
    }
    return assign;
}

std::vector<int> extract_sector(const std::vector<float>& parent_angles,
                                const std::vector<int>& parent_assignment,
                                int target_child) {
    const int N_p = static_cast<int>(parent_angles.size());

    std::vector<int> by_angle(static_cast<size_t>(N_p));
    for (int j = 0; j < N_p; ++j) by_angle[static_cast<size_t>(j)] = j;
    std::stable_sort(by_angle.begin(), by_angle.end(),
        [&](int a, int b) {
            const float ta = parent_angles[static_cast<size_t>(a)];
            const float tb = parent_angles[static_cast<size_t>(b)];
            if (ta != tb) return ta < tb;
            return a < b;
        });

    int start_in_sorted = -1;
    for (int k = 0; k < N_p; ++k) {
        const int prev_k = (k - 1 + N_p) % N_p;
        const int j_cur  = by_angle[static_cast<size_t>(k)];
        const int j_prev = by_angle[static_cast<size_t>(prev_k)];
        if (parent_assignment[static_cast<size_t>(j_cur)] == target_child &&
            parent_assignment[static_cast<size_t>(j_prev)] != target_child) {
            start_in_sorted = k;
            break;
        }
    }
    if (start_in_sorted < 0) start_in_sorted = 0;

    std::vector<int> sector;
    sector.reserve(static_cast<size_t>(N_p));
    for (int step = 0; step < N_p; ++step) {
        const int k = (start_in_sorted + step) % N_p;
        const int j = by_angle[static_cast<size_t>(k)];
        if (parent_assignment[static_cast<size_t>(j)] != target_child) break;
        sector.push_back(j);
    }
    return sector;
}

void emit_sector_stitch(std::vector<uint32_t>& indices,
                        int P0, const std::vector<int>& parent_sector,
                        int C0, const std::vector<int>& child_sweep,
                        const std::vector<float>& parent_angles,
                        const std::vector<float>& child_angles) {
    const int N_ps = static_cast<int>(parent_sector.size());
    const int N_cs = static_cast<int>(child_sweep.size());
    if (N_ps == 0 || N_cs == 0) return;

    auto unwrap = [&](float a, float base) {
        while (a < base - 1e-6f) a += k_2pi;
        return a;
    };
    const float p_base_raw = parent_angles[static_cast<size_t>(parent_sector[0])];
    const float c_base_raw = child_angles[static_cast<size_t>(child_sweep[0])];
    const float base = std::min(p_base_raw, c_base_raw);

    std::vector<float> p_sw(static_cast<size_t>(N_ps));
    for (int i = 0; i < N_ps; ++i) {
        p_sw[static_cast<size_t>(i)] = unwrap(
            parent_angles[static_cast<size_t>(parent_sector[static_cast<size_t>(i)])], base);
    }
    std::vector<float> c_sw(static_cast<size_t>(N_cs));
    for (int i = 0; i < N_cs; ++i) {
        c_sw[static_cast<size_t>(i)] = unwrap(
            child_angles[static_cast<size_t>(child_sweep[static_cast<size_t>(i)])], base);
    }

    int p = 0;
    int c = 0;
    while (p + 1 < N_ps || c + 1 < N_cs) {
        const uint32_t p_cur = static_cast<uint32_t>(P0 + parent_sector[static_cast<size_t>(p)]);
        const uint32_t c_cur = static_cast<uint32_t>(C0 + child_sweep[static_cast<size_t>(c)]);

        bool advance_parent;
        if (p + 1 >= N_ps)      advance_parent = false;
        else if (c + 1 >= N_cs) advance_parent = true;
        else {
            const float p_next = p_sw[static_cast<size_t>(p + 1)];
            const float c_next = c_sw[static_cast<size_t>(c + 1)];
            advance_parent = (p_next <= c_next);
        }

        if (advance_parent) {
            const uint32_t p_next_v =
                static_cast<uint32_t>(P0 + parent_sector[static_cast<size_t>(p + 1)]);
            indices.push_back(p_cur);
            indices.push_back(p_next_v);
            indices.push_back(c_cur);
            ++p;
        } else {
            const uint32_t c_next_v =
                static_cast<uint32_t>(C0 + child_sweep[static_cast<size_t>(c + 1)]);
            indices.push_back(p_cur);
            indices.push_back(c_next_v);
            indices.push_back(c_cur);
            ++c;
        }
    }
}

std::vector<int> sort_by_angle(const std::vector<float>& angles) {
    const int N = static_cast<int>(angles.size());
    std::vector<int> out(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) out[static_cast<size_t>(i)] = i;
    std::stable_sort(out.begin(), out.end(),
        [&](int a, int b) {
            const float ta = angles[static_cast<size_t>(a)];
            const float tb = angles[static_cast<size_t>(b)];
            if (ta != tb) return ta < tb;
            return a < b;
        });
    return out;
}

// Number of collar rings: angle-adaptive, 2..5.
int compute_n_collar(vec3 parent_axis, vec3 child_axis) {
    const float d = dot(normalized(parent_axis), normalized(child_axis));
    const float angle_rad = std::acos(std::max(-1.0f, std::min(1.0f, d)));
    const float angle_deg = angle_rad * (180.0f / k_pi);
    return std::min(5, std::max(2, static_cast<int>(std::ceil(angle_deg / 30.0f))));
}

// Emit one collar ring of (N+1) verts at `center` orthogonal to `axis`.
// Mirrors branch_mesh.cpp::emit_ring — emits positions, normals, UVs,
// tangents, per_vertex_node_index. Returns starting vert index.
int emit_collar_ring(BarkMeshOutput& bark,
                     int node_index,
                     vec3 center,
                     vec3 axis,
                     float radius,
                     int N,
                     float v_axial,
                     float seam_offset_rad) {
    vec3 right, up;
    build_basis(axis, right, up);

    auto& mesh = bark.mesh;
    const int start = static_cast<int>(mesh.positions.size() / 3);

    for (int i = 0; i <= N; ++i) {
        const int   i_mod = (i == N) ? 0 : i;
        const float theta = (static_cast<float>(i_mod) / static_cast<float>(N)) * k_2pi + seam_offset_rad;
        const float cs    = std::cos(theta);
        const float sn    = std::sin(theta);
        const vec3 radial = right * cs + up * sn;
        const vec3 pos    = center + radial * radius;

        mesh.positions.push_back(pos.x);
        mesh.positions.push_back(pos.y);
        mesh.positions.push_back(pos.z);

        mesh.normals.push_back(radial.x);
        mesh.normals.push_back(radial.y);
        mesh.normals.push_back(radial.z);

        auto [u, v] = bark_uv::ring_uv(i, N, v_axial, seam_offset_rad);
        mesh.uvs.push_back(u);
        mesh.uvs.push_back(v);

        // Tangent = circumferential U-derivative, matches branch_mesh.cpp.
        const vec3 tan = normalized(right * (-sn) + up * cs);
        bark.tangents.push_back(tan.x);
        bark.tangents.push_back(tan.y);
        bark.tangents.push_back(tan.z);
        bark.tangents.push_back(1.0f); // w = +1 right-handed

        bark.per_vertex_node_index.push_back(node_index);
    }
    return start;
}

} // anonymous namespace

std::vector<ForkZone> collect_fork_zones(const TreeSkeleton& skel, float zone_factor) {
    const int N = static_cast<int>(skel.nodes.size());

    std::vector<std::vector<int>> children(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        const int p = skel.nodes[static_cast<size_t>(i)].parent_index;
        if (p >= 0) children[static_cast<size_t>(p)].push_back(i);
    }

    std::vector<ForkZone> zones;
    for (int i = 0; i < N; ++i) {
        const auto& kids = children[static_cast<size_t>(i)];
        if (kids.size() < 2u) continue;
        ForkZone z;
        z.node_index    = i;
        z.child_indices = kids;
        z.zone_length_m = zone_factor * skel.nodes[static_cast<size_t>(i)].radius;
        zones.push_back(std::move(z));
    }
    return zones;
}

void apply_skin_rim_blend(BarkMeshOutput& bark,
                          const TreeSkeleton& skel,
                          const std::vector<ForkZone>& zones,
                          const BarkMeshOptions& opts) {
    const auto branch_idx = index_by_branch(bark.ring_metadata,
                                            static_cast<int>(skel.nodes.size()));

    for (const auto& zone : zones) {
        if (zone.child_indices.empty()) continue;

        const int fork_node = zone.node_index;
        const auto& fork_n  = skel.nodes[static_cast<size_t>(fork_node)];

        if (fork_n.parent_index < 0) continue;
        if (fork_node < 0 || fork_node >= static_cast<int>(branch_idx.size())) continue;
        const auto& parent_span = branch_idx[static_cast<size_t>(fork_node)];
        if (parent_span.first < 0 || parent_span.count <= 0) continue;
        const int parent_rim_r = parent_span.first + parent_span.count - 1;
        // Copy — not a reference — because collar ring push_back can reallocate ring_metadata.
        const RingMetadata parent_rim = bark.ring_metadata[static_cast<size_t>(parent_rim_r)];

        const float zone_length = zone.zone_length_m;
        if (zone_length <= 1e-6f) continue;

        const auto& fork_parent = skel.nodes[static_cast<size_t>(fork_n.parent_index)];
        const vec3 parent_axis = normalized(fork_n.position - fork_parent.position);
        vec3 plane_right, plane_up;
        build_basis(parent_axis, plane_right, plane_up);

        // Parent rim vert angles.
        const int N_p = parent_rim.N;
        std::vector<float> parent_angles(static_cast<size_t>(N_p));
        for (int j = 0; j < N_p; ++j) {
            parent_angles[static_cast<size_t>(j)] =
                vert_angle(bark.mesh, parent_rim.vert_start + j,
                           fork_n.position, plane_right, plane_up);
        }

        // Child central angles.
        const int N_kids = static_cast<int>(zone.child_indices.size());
        std::vector<float> child_central(static_cast<size_t>(N_kids));
        for (int k = 0; k < N_kids; ++k) {
            const auto& cn = skel.nodes[static_cast<size_t>(zone.child_indices[static_cast<size_t>(k)])];
            const vec3 cdir = cn.position - fork_n.position;
            child_central[static_cast<size_t>(k)] =
                std::atan2(dot(cdir, plane_up), dot(cdir, plane_right));
        }

        const std::vector<int> assignment = assign_parent_to_children(parent_angles, child_central);

        // Per-child collar data collected for the crotch cap pass.
        struct ChildCollarInfo {
            int  first_collar_vert_start = -1;
            int  child_index_k           = -1;
            std::vector<int> sector;  // parent rim vert local indices (angular order)
        };
        std::vector<ChildCollarInfo> child_collar_info(static_cast<size_t>(N_kids));

        // Per-child: emit collar rings + stitch.
        for (int k = 0; k < N_kids; ++k) {
            const int child_node = zone.child_indices[static_cast<size_t>(k)];
            if (child_node < 0 || child_node >= static_cast<int>(branch_idx.size())) continue;
            const auto& span = branch_idx[static_cast<size_t>(child_node)];
            if (span.first < 0 || span.count <= 0) continue;

            const std::vector<int> sector = extract_sector(parent_angles, assignment, k);
            if (sector.empty()) continue;

            const auto& child_n = skel.nodes[static_cast<size_t>(child_node)];
            const vec3 child_dir = normalized(child_n.position - fork_n.position);
            const float seg_len = length(child_n.position - fork_n.position);
            if (seg_len < 1e-6f) continue;

            const int n_collar = compute_n_collar(parent_axis, child_dir);
            const float collar_d = std::min(zone_length, seg_len);

            CollarCurve curve;
            curve.p0     = fork_n.position;
            curve.p1     = fork_n.position + child_dir * collar_d;
            curve.t0_dir = parent_axis;
            curve.t1_dir = child_dir;
            curve.r0     = fork_n.radius;
            curve.r1     = fork_n.radius + (child_n.radius - fork_n.radius) * (collar_d / seg_len);

            // Emit collar rings at t_k = (k+1) / (n_collar+1), k=0..n_collar-1.
            int prev_ring_start = -1;
            int first_collar_vert_start = -1;

            for (int ci = 0; ci < n_collar; ++ci) {
                const float t = static_cast<float>(ci + 1) / static_cast<float>(n_collar + 1);
                const CollarSample sample = eval_collar_curve(curve, t);
                const float v_axial = length(sample.position - curve.p0) / opts.bark_repeat_m;

                const int ring_start = emit_collar_ring(bark, child_node,
                    sample.position, sample.tangent, sample.radius,
                    N_p, v_axial, opts.seam_offset_rad);

                if (ci == 0) first_collar_vert_start = ring_start;

                RingMetadata rm;
                rm.branch_node_index = child_node;
                rm.axial_index       = -(n_collar - ci); // negative sentinel
                rm.axial_segs        = 0;
                rm.vert_start        = ring_start;
                rm.N                 = N_p;
                rm.axial_distance_m  = length(sample.position - curve.p0);
                rm.branch_length_m   = seg_len;
                rm.collar_fork_node  = fork_node;
                rm.collar_sequence   = ci;
                bark.ring_metadata.push_back(rm);

                if (prev_ring_start >= 0) {
                    emit_strip(bark.indices_u32, prev_ring_start, ring_start, N_p);
                }
                prev_ring_start = ring_start;
            }

            // Stitch parent rim sector -> first collar ring.
            if (first_collar_vert_start >= 0 && sector.size() >= 2u) {
                std::vector<float> collar_angles(static_cast<size_t>(N_p));
                for (int i = 0; i < N_p; ++i) {
                    collar_angles[static_cast<size_t>(i)] =
                        vert_angle(bark.mesh, first_collar_vert_start + i,
                                   fork_n.position, plane_right, plane_up);
                }
                const std::vector<int> collar_sweep = sort_by_angle(collar_angles);

                emit_sector_stitch(bark.indices_u32,
                                   parent_rim.vert_start, sector,
                                   first_collar_vert_start, collar_sweep,
                                   parent_angles, collar_angles);
            }

            // Stitch last collar ring -> child's first tube ring (s=0).
            if (prev_ring_start >= 0) {
                const auto& child_span = branch_idx[static_cast<size_t>(child_node)];
                if (child_span.first >= 0 && child_span.count > 0) {
                    const RingMetadata child_s0 = bark.ring_metadata[static_cast<size_t>(child_span.first)];
                    if (child_s0.N == N_p) {
                        emit_strip(bark.indices_u32, prev_ring_start, child_s0.vert_start, N_p);
                    } else {
                        // N mismatch: full-ring bipartite stitch.
                        std::vector<float> last_collar_angles(static_cast<size_t>(N_p));
                        for (int i = 0; i < N_p; ++i) {
                            last_collar_angles[static_cast<size_t>(i)] =
                                vert_angle(bark.mesh, prev_ring_start + i,
                                           fork_n.position, plane_right, plane_up);
                        }

                        std::vector<float> child_s0_angles(static_cast<size_t>(child_s0.N));
                        for (int i = 0; i < child_s0.N; ++i) {
                            child_s0_angles[static_cast<size_t>(i)] =
                                vert_angle(bark.mesh, child_s0.vert_start + i,
                                           fork_n.position, plane_right, plane_up);
                        }
                        const std::vector<int> child_s0_sweep = sort_by_angle(child_s0_angles);

                        // Treat full last-collar ring as a "sector" for the bipartite stitch.
                        std::vector<int> full_sector(static_cast<size_t>(N_p));
                        for (int i = 0; i < N_p; ++i)
                            full_sector[static_cast<size_t>(i)] = i;
                        std::stable_sort(full_sector.begin(), full_sector.end(),
                            [&](int a, int b) {
                                return last_collar_angles[static_cast<size_t>(a)]
                                     < last_collar_angles[static_cast<size_t>(b)];
                            });

                        emit_sector_stitch(bark.indices_u32,
                                           prev_ring_start, full_sector,
                                           child_s0.vert_start, child_s0_sweep,
                                           last_collar_angles, child_s0_angles);
                    }
                }
            }

            // Store collar info for crotch cap pass.
            child_collar_info[static_cast<size_t>(k)].first_collar_vert_start = first_collar_vert_start;
            child_collar_info[static_cast<size_t>(k)].child_index_k           = k;
            child_collar_info[static_cast<size_t>(k)].sector                  = sector;
        }

        // --- Crotch fill cap pass ---
        // For each pair of angularly-adjacent children, emit a centroid vertex
        // and a triangle fan bridging the diverging collar tubes. All edges
        // touch the new centroid vertex, so they cannot conflict with existing
        // sector stitch edges.
        if (N_kids < 2 || !opts.emit_crotch_cap) continue;

        // Sort children by central angle.
        std::vector<int> kids_by_angle(static_cast<size_t>(N_kids));
        for (int k = 0; k < N_kids; ++k) kids_by_angle[static_cast<size_t>(k)] = k;
        std::stable_sort(kids_by_angle.begin(), kids_by_angle.end(),
            [&](int a, int b) {
                return child_central[static_cast<size_t>(a)]
                     < child_central[static_cast<size_t>(b)];
            });

        for (int adj = 0; adj < N_kids; ++adj) {
            const int kA = kids_by_angle[static_cast<size_t>(adj)];
            const int kB = kids_by_angle[static_cast<size_t>((adj + 1) % N_kids)];

            const auto& infoA = child_collar_info[static_cast<size_t>(kA)];
            const auto& infoB = child_collar_info[static_cast<size_t>(kB)];
            if (infoA.first_collar_vert_start < 0 || infoB.first_collar_vert_start < 0) continue;
            if (infoA.sector.empty() || infoB.sector.empty()) continue;

            // Sector boundary: angular midpoint between A's last sector vert
            // and B's first sector vert.
            const float angle_A_end = parent_angles[static_cast<size_t>(infoA.sector.back())];
            const float angle_B_beg = parent_angles[static_cast<size_t>(infoB.sector.front())];
            const float gap_mid = angle_A_end + angle_wrap_signed(angle_B_beg - angle_A_end) * 0.5f;

            // Find collar A vert closest to gap midpoint.
            int bestA = 0;
            float bestA_dist = 1e9f;
            for (int i = 0; i < N_p; ++i) {
                const float ca = vert_angle(bark.mesh, infoA.first_collar_vert_start + i,
                                            fork_n.position, plane_right, plane_up);
                const float d = angle_dist(ca, gap_mid);
                if (d < bestA_dist) { bestA_dist = d; bestA = i; }
            }

            // Find collar B vert closest to gap midpoint.
            int bestB = 0;
            float bestB_dist = 1e9f;
            for (int i = 0; i < N_p; ++i) {
                const float cb = vert_angle(bark.mesh, infoB.first_collar_vert_start + i,
                                            fork_n.position, plane_right, plane_up);
                const float d = angle_dist(cb, gap_mid);
                if (d < bestB_dist) { bestB_dist = d; bestB = i; }
            }

            const uint32_t cA = static_cast<uint32_t>(infoA.first_collar_vert_start + bestA);
            const uint32_t cB = static_cast<uint32_t>(infoB.first_collar_vert_start + bestB);

            // Emit centroid vertex at fork position, extruded slightly along
            // the mean of the two children's directions.
            const auto& cnA = skel.nodes[static_cast<size_t>(zone.child_indices[static_cast<size_t>(kA)])];
            const auto& cnB = skel.nodes[static_cast<size_t>(zone.child_indices[static_cast<size_t>(kB)])];
            const vec3 dirA = normalized(cnA.position - fork_n.position);
            const vec3 dirB = normalized(cnB.position - fork_n.position);
            const vec3 mean_dir = normalized(dirA + dirB);
            const vec3 centroid_pos = fork_n.position + mean_dir * (fork_n.radius * 0.5f);
            const vec3 centroid_normal = mean_dir;

            auto& mesh = bark.mesh;
            const int centroid_idx = static_cast<int>(mesh.positions.size() / 3);
            mesh.positions.push_back(centroid_pos.x);
            mesh.positions.push_back(centroid_pos.y);
            mesh.positions.push_back(centroid_pos.z);
            mesh.normals.push_back(centroid_normal.x);
            mesh.normals.push_back(centroid_normal.y);
            mesh.normals.push_back(centroid_normal.z);
            mesh.uvs.push_back(0.5f);
            mesh.uvs.push_back(0.0f);
            bark.tangents.push_back(1.0f);
            bark.tangents.push_back(0.0f);
            bark.tangents.push_back(0.0f);
            bark.tangents.push_back(1.0f);
            bark.per_vertex_node_index.push_back(fork_node);

            const uint32_t cIdx = static_cast<uint32_t>(centroid_idx);

            // Triangle fan from centroid to collar verts.
            // All edges involve the new centroid vertex, so no existing edge
            // can exceed 2 adjacent triangles.
            bark.indices_u32.push_back(cIdx);
            bark.indices_u32.push_back(cA);
            bark.indices_u32.push_back(cB);
        }
    }
}

} // namespace treegen
