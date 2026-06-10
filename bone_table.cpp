// Decimated wind rig bake (see bone_table.hpp).
// /fp:precise (treegen.sharpmake.cs).
#include "bone_table.hpp"

#include <cassert>
#include <cmath>

namespace treegen {

namespace {

inline float seg_length(const TreeSkeleton& skel, int node) {
    const BranchNode& n = skel.nodes[static_cast<size_t>(node)];
    assert(n.parent_index >= 0);
    const BranchNode& p = skel.nodes[static_cast<size_t>(n.parent_index)];
    const float dx = n.position.x - p.position.x;
    const float dy = n.position.y - p.position.y;
    const float dz = n.position.z - p.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// CSR children adjacency, child order = node-index order (deterministic).
struct Children {
    std::vector<int> start;  // size N+1
    std::vector<int> list;   // size N-1 (every non-root has one parent edge)
};

Children build_children(const TreeSkeleton& skel) {
    const int N = static_cast<int>(skel.nodes.size());
    Children c;
    c.start.assign(static_cast<size_t>(N) + 1, 0);
    for (int i = 1; i < N; ++i) {
        ++c.start[static_cast<size_t>(skel.nodes[static_cast<size_t>(i)].parent_index) + 1];
    }
    for (int i = 0; i < N; ++i) c.start[static_cast<size_t>(i) + 1] += c.start[static_cast<size_t>(i)];
    c.list.resize(static_cast<size_t>(N > 0 ? N - 1 : 0));
    std::vector<int> cursor(c.start.begin(), c.start.end() - 1);
    for (int i = 1; i < N; ++i) {
        const int p = skel.nodes[static_cast<size_t>(i)].parent_index;
        c.list[static_cast<size_t>(cursor[static_cast<size_t>(p)]++)] = i;
    }
    return c;
}

}  // anonymous namespace

uint8_t BoneTable::vertex_parent_blend(int node, float t_seg) const {
    const size_t n = static_cast<size_t>(node);
    if (!node_in_run[n]) return 0;  // rigid rider: full host transform
    const float t0 = node_seg_t0[n];
    const float t1 = node_run_t[n];
    float t = t0 + (t1 - t0) * (t_seg < 0.0f ? 0.0f : (t_seg > 1.0f ? 1.0f : t_seg));
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return static_cast<uint8_t>((1.0f - t) * 255.0f + 0.5f);
}

BoneTable build_bone_table(const TreeSkeleton& skel,
                           const BoneDecimateParams& params) {
    BoneTable out;
    const int N = static_cast<int>(skel.nodes.size());
    if (N == 0) return out;

    const Children ch = build_children(skel);
    auto child_count = [&](int i) {
        return ch.start[static_cast<size_t>(i) + 1] - ch.start[static_cast<size_t>(i)];
    };

    // --- Kept set: root + forks within order_cap. -------------------------
    std::vector<uint8_t> kept(static_cast<size_t>(N), 0);
    kept[0] = 1;
    for (int i = 1; i < N; ++i) {
        if (child_count(i) >= 2 &&
            skel.nodes[static_cast<size_t>(i)].depth <= params.order_cap) {
            kept[static_cast<size_t>(i)] = 1;
        }
    }

    // --- Subdivision pass: split over-long runs at original nodes near
    // equal arc spacing. Anchors = the pre-subdivision kept set, walked in
    // index order; each chain is visited exactly once (from its anchor), so
    // marks made here never affect another chain's walk.
    {
        std::vector<int>   chain;   // reused scratch
        std::vector<float> cum;     // cumulative arc at each chain node
        for (int k = 0; k < N; ++k) {
            if (!kept[static_cast<size_t>(k)]) continue;
            for (int ci = ch.start[static_cast<size_t>(k)];
                 ci < ch.start[static_cast<size_t>(k) + 1]; ++ci) {
                int cur = ch.list[static_cast<size_t>(ci)];
                if (skel.nodes[static_cast<size_t>(cur)].depth > params.order_cap) continue;
                chain.clear();
                cum.clear();
                float arc = 0.0f;
                // Collect the chain up to (exclusive of) the next kept node.
                while (true) {
                    arc += seg_length(skel, cur);
                    if (kept[static_cast<size_t>(cur)]) break;  // run ends at a fork
                    chain.push_back(cur);
                    cum.push_back(arc);
                    if (child_count(cur) != 1) break;           // tip
                    const int nxt = ch.list[static_cast<size_t>(ch.start[static_cast<size_t>(cur)])];
                    if (skel.nodes[static_cast<size_t>(nxt)].depth > params.order_cap) break;
                    cur = nxt;
                }
                if (chain.empty() || arc <= params.max_segment_m) continue;
                const int n_sub = static_cast<int>(arc / params.max_segment_m);
                const float spacing = arc / static_cast<float>(n_sub + 1);
                size_t lo = 0;
                for (int s = 1; s <= n_sub; ++s) {
                    const float target = spacing * static_cast<float>(s);
                    // First chain node at/after target (cum is increasing).
                    while (lo + 1 < chain.size() && cum[lo] < target) ++lo;
                    kept[static_cast<size_t>(chain[lo])] = 1;
                    if (lo + 1 < chain.size()) ++lo;  // distinct node per joint
                }
            }
        }
    }

    // --- Bone ids: kept nodes in index order (parent bone < own bone). ----
    std::vector<int> bone_of(static_cast<size_t>(N), -1);
    std::vector<int> bone_node;  // bone -> node
    for (int i = 0; i < N; ++i) {
        if (kept[static_cast<size_t>(i)]) {
            bone_of[static_cast<size_t>(i)] = static_cast<int>(bone_node.size());
            bone_node.push_back(i);
        }
    }
    const int B = static_cast<int>(bone_node.size());
    assert(B <= 0xFFFF && "decimated rig exceeds u16 bone-index range");

    // --- Records: parent bone, compliance aggregation, pivot, node pos. ---
    out.bone_count = static_cast<uint32_t>(B);
    out.records.assign(static_cast<size_t>(B) * K_FLOATS_PER_BONE, 0.0f);
    std::vector<int> bone_chain_depth(static_cast<size_t>(B), 0);
    for (int b = 0; b < B; ++b) {
        const int node = bone_node[static_cast<size_t>(b)];
        // Walk original ancestors to the parent kept joint, aggregating the
        // collapsed run's compliance (count + depth sum).
        int   n_collapsed = 0;
        int   sum_depth   = 0;
        int   anchor      = -1;
        for (int j = node; j >= 0;) {
            if (j != node && kept[static_cast<size_t>(j)]) { anchor = j; break; }
            ++n_collapsed;
            sum_depth += skel.nodes[static_cast<size_t>(j)].depth;
            j = skel.nodes[static_cast<size_t>(j)].parent_index;
        }
        float* rec = &out.records[static_cast<size_t>(b) * K_FLOATS_PER_BONE];
        const vec3 node_pos = skel.nodes[static_cast<size_t>(node)].position;
        if (anchor < 0) {
            // Root bone: does not rotate.
            rec[0] = -1.0f;
            rec[1] = 0.0f;
            rec[2] = 0.0f;
            rec[4] = node_pos.x; rec[5] = node_pos.y; rec[6] = node_pos.z;
        } else {
            const int pb = bone_of[static_cast<size_t>(anchor)];
            assert(pb >= 0 && pb < b);
            rec[0] = static_cast<float>(pb);
            rec[1] = static_cast<float>(n_collapsed);
            rec[2] = static_cast<float>(sum_depth);
            const vec3 pivot = skel.nodes[static_cast<size_t>(anchor)].position;
            rec[4] = pivot.x; rec[5] = pivot.y; rec[6] = pivot.z;
            bone_chain_depth[static_cast<size_t>(b)] =
                bone_chain_depth[static_cast<size_t>(pb)] + 1;
        }
        rec[8] = node_pos.x; rec[9] = node_pos.y; rec[10] = node_pos.z;
        // Shader FK ancestor cap is 16; the bake must stay under it so the
        // cap is a degenerate-data guard, never a silent truncation.
        assert(bone_chain_depth[static_cast<size_t>(b)] < 16 &&
               "decimated bone chain exceeds shader MAX_BONE_DEPTH");
    }

    // --- Node remap: runs first, then rigid riders inherit. ---------------
    out.node_to_bone.assign(static_cast<size_t>(N), 0);
    out.node_run_t.assign(static_cast<size_t>(N), 1.0f);
    out.node_in_run.assign(static_cast<size_t>(N), 0);
    out.node_seg_t0.assign(static_cast<size_t>(N), 0.0f);
    std::vector<uint8_t> assigned(static_cast<size_t>(N), 0);

    // Kept nodes host themselves. Root stays a rigid rider (no segment, no
    // rotation): vertex_parent_blend(0, *) == 0 == identity-root FK.
    for (int i = 0; i < N; ++i) {
        if (kept[static_cast<size_t>(i)]) {
            out.node_to_bone[static_cast<size_t>(i)] =
                static_cast<uint16_t>(bone_of[static_cast<size_t>(i)]);
            assigned[static_cast<size_t>(i)] = 1;
            out.node_in_run[static_cast<size_t>(i)] = (i != 0) ? uint8_t(1) : uint8_t(0);
        }
    }

    {
        std::vector<int>   chain;
        std::vector<float> cum;
        for (int k = 0; k < N; ++k) {
            if (!kept[static_cast<size_t>(k)]) continue;
            for (int ci = ch.start[static_cast<size_t>(k)];
                 ci < ch.start[static_cast<size_t>(k) + 1]; ++ci) {
                int cur = ch.list[static_cast<size_t>(ci)];
                if (skel.nodes[static_cast<size_t>(cur)].depth > params.order_cap) continue;
                chain.clear();
                cum.clear();
                float arc = 0.0f;
                bool ends_kept = false;
                while (true) {
                    arc += seg_length(skel, cur);
                    chain.push_back(cur);
                    cum.push_back(arc);
                    if (kept[static_cast<size_t>(cur)]) { ends_kept = true; break; }
                    if (child_count(cur) != 1) break;
                    const int nxt = ch.list[static_cast<size_t>(ch.start[static_cast<size_t>(cur)])];
                    if (skel.nodes[static_cast<size_t>(nxt)].depth > params.order_cap) break;
                    cur = nxt;
                }
                if (ends_kept) {
                    // Run k -> e: interior nodes host bone(e) at arc fraction.
                    const int e = chain.back();
                    const uint16_t host =
                        static_cast<uint16_t>(bone_of[static_cast<size_t>(e)]);
                    const float inv_len = 1.0f / arc;
                    float t_prev = 0.0f;
                    for (size_t m = 0; m < chain.size(); ++m) {
                        const size_t node = static_cast<size_t>(chain[m]);
                        const float  t    = cum[m] * inv_len;
                        out.node_seg_t0[node] = t_prev;
                        if (chain[m] != e) {
                            out.node_to_bone[node] = host;
                            out.node_run_t[node]   = t;
                            out.node_in_run[node]  = 1;
                            assigned[node] = 1;
                        }
                        // e itself is kept (assigned above, t = 1); only its
                        // seg-t0 is set here so its rings interpolate.
                        t_prev = t;
                    }
                }
                // Tail chains fall through to the rigid-rider inherit pass.
            }
        }
    }

    // Rigid riders: tails past the last kept joint and beyond-cap subtrees
    // ride their nearest kept ancestor's bone whole. Forward pass is enough:
    // parent_index < own index, and the parent is always assigned first.
    for (int i = 1; i < N; ++i) {
        if (assigned[static_cast<size_t>(i)]) continue;
        out.node_to_bone[static_cast<size_t>(i)] =
            out.node_to_bone[static_cast<size_t>(skel.nodes[static_cast<size_t>(i)].parent_index)];
        assigned[static_cast<size_t>(i)] = 1;
        // node_run_t stays 1, node_in_run stays 0 -> blend 0, full host.
    }

    return out;
}

}  // namespace treegen
