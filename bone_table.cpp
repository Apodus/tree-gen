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
    if (!node_in_run[n]) return 0;  // rigid rider of a kept joint: full host
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
    assert(N <= 0xFFFF && "skeleton exceeds u16 node-index range (per-vertex casts)");

    const Children ch = build_children(skel);
    auto child_count = [&](int i) {
        return ch.start[static_cast<size_t>(i) + 1] - ch.start[static_cast<size_t>(i)];
    };
    auto node_depth = [&](int i) { return skel.nodes[static_cast<size_t>(i)].depth; };

    // Primary child = the continuation of a node's own limb: prefer the
    // same-depth child (forks split off at depth+1), else the lowest index.
    // -1 for tips. Index order = deterministic.
    std::vector<int> primary(static_cast<size_t>(N), -1);
    for (int i = 0; i < N; ++i) {
        int pick = -1;
        for (int ci = ch.start[static_cast<size_t>(i)]; ci < ch.start[static_cast<size_t>(i) + 1]; ++ci) {
            const int c = ch.list[static_cast<size_t>(ci)];
            if (pick < 0) pick = c;
            if (node_depth(c) == node_depth(i)) { pick = c; break; }
        }
        primary[static_cast<size_t>(i)] = pick;
    }

    // --- Kept set pass 1: root + forks, gated by order cap AND a minimum
    // arc spacing since the nearest kept ancestor. The spacing gate is what
    // bounds FK chain length: without it every fork on a limb becomes a
    // chain link and deep crowns ride the shader's ancestor cap (16).
    // Forward pass: parent_index < own index, so arc_since_kept is ready.
    std::vector<uint8_t> kept(static_cast<size_t>(N), 0);
    kept[0] = 1;
    {
        std::vector<float> arc_since_kept(static_cast<size_t>(N), 0.0f);
        for (int i = 1; i < N; ++i) {
            const int p = skel.nodes[static_cast<size_t>(i)].parent_index;
            const float a = (kept[static_cast<size_t>(p)] ? 0.0f
                             : arc_since_kept[static_cast<size_t>(p)]) + seg_length(skel, i);
            arc_since_kept[static_cast<size_t>(i)] = a;
            if (child_count(i) >= 2 && node_depth(i) <= params.order_cap &&
                a >= params.min_fork_spacing_m) {
                kept[static_cast<size_t>(i)] = 1;
            }
        }
    }

    // --- Walk machinery. A "run" is the original-tree path from a kept
    // anchor to the next kept node along primary edges, passing THROUGH
    // dropped forks; each dropped fork's split children spawn their own run
    // sharing the same anchor (arc offset = distance anchor->fork). Each
    // node is owned by exactly one run (a split run owns only nodes past its
    // fork). fn(owned_chain, cum_arcs, offset, ends_kept) is invoked per
    // run; chains end at a kept node (inclusive) or at a tail (tip / beyond-
    // cap edge). Deterministic: anchors in index order, children in index
    // order, owned sets disjoint.
    struct WalkSeed { int start; float offset; };
    auto for_each_run = [&](auto&& fn) {
        std::vector<WalkSeed> stack;
        std::vector<int>   chain;
        std::vector<float> cum;
        for (int k = 0; k < N; ++k) {
            if (!kept[static_cast<size_t>(k)]) continue;
            for (int ci = ch.start[static_cast<size_t>(k)]; ci < ch.start[static_cast<size_t>(k) + 1]; ++ci) {
                const int c = ch.list[static_cast<size_t>(ci)];
                if (node_depth(c) > params.order_cap) continue;  // rigid subtree
                stack.push_back({c, 0.0f});
                while (!stack.empty()) {
                    WalkSeed w = stack.back();
                    stack.pop_back();
                    chain.clear();
                    cum.clear();
                    float arc = w.offset;
                    int cur = w.start;
                    bool ends_kept = false;
                    while (true) {
                        arc += seg_length(skel, cur);
                        chain.push_back(cur);
                        cum.push_back(arc);
                        if (kept[static_cast<size_t>(cur)]) { ends_kept = true; break; }
                        // Dropped fork: split children spawn runs sharing
                        // this anchor; the walk continues down the primary.
                        int nxt = -1;
                        for (int cj = ch.start[static_cast<size_t>(cur)]; cj < ch.start[static_cast<size_t>(cur) + 1]; ++cj) {
                            const int cc = ch.list[static_cast<size_t>(cj)];
                            if (node_depth(cc) > params.order_cap) continue;
                            if (cc == primary[static_cast<size_t>(cur)]) { nxt = cc; continue; }
                            stack.push_back({cc, arc});
                        }
                        if (nxt < 0) break;  // tip (or primary beyond cap)
                        cur = nxt;
                    }
                    fn(chain, cum, w.offset, ends_kept);
                }
            }
        }
    };

    // --- Kept set pass 2: arc subdivision. Over-long runs gain joints at
    // owned nodes nearest to equal spacing over the FULL anchor->end arc
    // (tails included, so long unforked limbs still bend). Marks land
    // strictly inside the current run's owned chain; each node is owned by
    // exactly one run, so no other walk in this same traversal re-reads a
    // mark as a run boundary.
    for_each_run([&](const std::vector<int>& chain, const std::vector<float>& cum,
                     float /*offset*/, bool ends_kept) {
        if (chain.empty()) return;
        const float total = cum.back();
        if (total <= params.max_segment_m) return;
        // Never re-mark the kept end; tails may mark their last node.
        const size_t last = ends_kept ? chain.size() - 1 : chain.size();
        if (last == 0) return;
        const int n_sub = static_cast<int>(total / params.max_segment_m);
        const float spacing = total / static_cast<float>(n_sub + 1);
        size_t lo = 0;
        for (int s = 1; s <= n_sub; ++s) {
            const float target = spacing * static_cast<float>(s);
            while (lo + 1 < last && cum[lo] < target) ++lo;
            if (lo >= last) break;
            kept[static_cast<size_t>(chain[lo])] = 1;
            if (lo + 1 < last) ++lo;  // distinct node per joint
        }
    });

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
        // Walk original ancestors to the parent kept joint (through any
        // dropped forks), aggregating the collapsed compliance (count +
        // depth sum) so total root-to-tip bend is preserved exactly.
        int   n_collapsed = 0;
        int   sum_depth   = 0;
        int   anchor      = -1;
        for (int j = node; j >= 0;) {
            if (j != node && kept[static_cast<size_t>(j)]) { anchor = j; break; }
            ++n_collapsed;
            sum_depth += node_depth(j);
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
        // Shader FK ancestor cap is 16 (MAX_BONE_DEPTH); the bake must stay
        // strictly under it so the cap is a degenerate-data guard, never a
        // silent truncation. min_fork_spacing_m is the knob that bounds this.
        assert(bone_chain_depth[static_cast<size_t>(b)] < 16 &&
               "decimated bone chain exceeds shader MAX_BONE_DEPTH");
    }

    // --- Node remap. Runs assign run-arc t; everything else rides rigidly,
    // copying its PARENT's binding so a twig attached mid-run keeps exactly
    // its attachment point's transform (no over- or under-bend kink).
    out.node_to_bone.assign(static_cast<size_t>(N), 0);
    out.node_run_t.assign(static_cast<size_t>(N), 1.0f);
    out.node_in_run.assign(static_cast<size_t>(N), 0);
    out.node_seg_t0.assign(static_cast<size_t>(N), 1.0f);
    std::vector<uint8_t> assigned(static_cast<size_t>(N), 0);

    // Kept nodes host themselves. Root stays a rigid rider (no segment, no
    // rotation): vertex_parent_blend(0, *) == 0 == identity-root FK.
    for (int i = 0; i < N; ++i) {
        if (kept[static_cast<size_t>(i)]) {
            out.node_to_bone[static_cast<size_t>(i)] =
                static_cast<uint16_t>(bone_of[static_cast<size_t>(i)]);
            assigned[static_cast<size_t>(i)] = 1;
            if (i != 0) out.node_in_run[static_cast<size_t>(i)] = 1;
        }
    }

    for_each_run([&](const std::vector<int>& chain, const std::vector<float>& cum,
                     float offset, bool ends_kept) {
        if (!ends_kept || chain.empty()) return;  // tails ride the inherit pass
        const int e = chain.back();
        const uint16_t host = static_cast<uint16_t>(bone_of[static_cast<size_t>(e)]);
        const float inv_total = 1.0f / cum.back();
        float t_prev = offset * inv_total;
        for (size_t m = 0; m < chain.size(); ++m) {
            const size_t node = static_cast<size_t>(chain[m]);
            const float  t    = cum[m] * inv_total;
            out.node_seg_t0[node] = t_prev;
            if (chain[m] != e) {
                out.node_to_bone[node] = host;
                out.node_run_t[node]   = t;
                out.node_in_run[node]  = 1;
                assigned[node] = 1;
            }
            // e itself is kept (assigned above, t = 1); only its seg-t0 is
            // set here so its rings interpolate within this run.
            t_prev = t;
        }
    });

    // Rigid riders (tails past the last kept joint, beyond-cap subtrees):
    // copy the parent's binding verbatim — bone, in_run, run-t, and seg-t0
    // pinned to the parent's END value so vertex_parent_blend degenerates to
    // the parent's CONSTANT blend. Forward pass suffices: parent_index <
    // own index, so the parent is always assigned first.
    for (int i = 1; i < N; ++i) {
        const size_t n = static_cast<size_t>(i);
        if (assigned[n]) continue;
        const size_t p = static_cast<size_t>(skel.nodes[n].parent_index);
        out.node_to_bone[n] = out.node_to_bone[p];
        out.node_in_run[n]  = out.node_in_run[p];
        out.node_run_t[n]   = out.node_run_t[p];
        out.node_seg_t0[n]  = out.node_run_t[p];  // constant blend == parent's
        assigned[n] = 1;
    }

    return out;
}

}  // namespace treegen
