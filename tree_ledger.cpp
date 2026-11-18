// T0 geometry ledger — see tree_ledger.hpp.
#include "tree_ledger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace treegen {

namespace {

template <size_t N>
using Key = std::array<uint32_t, N>;

struct KeyHash {
    template <size_t N>
    size_t operator()(const Key<N>& k) const noexcept {
        uint64_t h = 1469598103934665603ull;
        for (uint32_t w : k) { h ^= w; h *= 1099511628211ull; }
        return static_cast<size_t>(h ^ (h >> 32));
    }
};

uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, sizeof u); return u; }
Key<3> pos_key(vec3 p) { return {bits(p.x), bits(p.y), bits(p.z)}; }
Key<3> pos_key(const float* p) { return {bits(p[0]), bits(p[1]), bits(p[2])}; }

void require(bool ok, const char* what) {
    if (!ok) throw std::logic_error(std::string("tree_ledger: ") + what);
}

int diameter_class(float radius) {
    const float d = 2.0f * radius;
    if (d < 0.02f) return 0;
    if (d < 0.04f) return 1;
    if (d < 0.08f) return 2;
    if (d < 0.16f) return 3;
    return 4;
}

void bark_section(LevelLedger& L, const TreeSkeleton& skel, const LodOutput& lod) {
    const auto& pos = lod.mesh.positions;
    const auto& idx = lod.indices_u32;
    require(idx.size() % 3 == 0, "bark index count not a multiple of 3");
    const size_t tris = idx.size() / 3;
    require(lod.bark_triangle_class.size() == tris, "bark triangle_class does not cover every triangle");
    require(lod.bark_triangle_node.size() == tris, "bark triangle_node does not cover every triangle");

    L.bark_triangles = static_cast<int>(tris);
    L.bark_vertices  = static_cast<int>(pos.size() / 3);

    std::unordered_set<Key<9>, KeyHash>      seen_tris;
    std::unordered_map<Key<6>, int, KeyHash> edges;
    seen_tris.reserve(tris);
    edges.reserve(tris * 2);

    for (size_t t = 0; t < tris; ++t) {
        const uint8_t c = lod.bark_triangle_class[t];
        require(c < 4, "bark triangle class out of range");
        ++L.by_class[c];
        if (c == static_cast<uint8_t>(BarkTriClass::tube) || c == static_cast<uint8_t>(BarkTriClass::tip)) {
            const int n = lod.bark_triangle_node[t];
            require(n >= 0 && size_t(n) < skel.nodes.size(), "bark triangle node out of range");
            auto& bins = (c == static_cast<uint8_t>(BarkTriClass::tube)) ? L.tube_by_diameter : L.tip_by_diameter;
            ++bins[size_t(diameter_class(skel.nodes[size_t(n)].radius))];
        }

        Key<3> v[3];
        for (int k = 0; k < 3; ++k) {
            const uint32_t vi = idx[t * 3 + size_t(k)];
            require(size_t(vi) * 3 + 2 < pos.size(), "bark index out of range");
            v[k] = pos_key(&pos[size_t(vi) * 3]);
        }

        std::array<Key<3>, 3> s = {v[0], v[1], v[2]};
        std::sort(s.begin(), s.end());
        Key<9> tk;
        for (int k = 0; k < 3; ++k) std::copy(s[size_t(k)].begin(), s[size_t(k)].end(), tk.begin() + 3 * k);
        if (!seen_tris.insert(tk).second) ++L.coincident_triangles[c];

        // Each distinct position-welded edge counts once per triangle: a triangle with two
        // bitwise-equal corners has one real edge (twice in its corner order) and one of zero length.
        const bool degenerate = v[0] == v[1] || v[1] == v[2] || v[2] == v[0];
        if (degenerate) ++L.degenerate_triangles;
        Key<6> tri_edges[3];
        int n_edges = 0;
        for (int k = 0; k < 3; ++k) {
            Key<3> a = v[k], b = v[(k + 1) % 3];
            if (a == b) continue;  // zero length: not an edge
            if (b < a) std::swap(a, b);
            Key<6> ek;
            std::copy(a.begin(), a.end(), ek.begin());
            std::copy(b.begin(), b.end(), ek.begin() + 3);
            if (std::find(tri_edges, tri_edges + n_edges, ek) != tri_edges + n_edges) continue;
            tri_edges[n_edges++] = ek;
            ++edges[ek];
        }
    }
    for (const auto& [k, n] : edges) {
        if (n == 1) ++L.open_edges;
        else if (n > 2) ++L.nonmanifold_edges;
    }
}

void leaf_section(LevelLedger& L, const LodOutput& lod) {
    if (!lod.has_leaves) return;
    const auto& pos = lod.leaf_positions;
    const auto& idx = lod.leaf_indices_u32;
    require(idx.size() % 3 == 0, "leaf index count not a multiple of 3");
    const size_t verts = pos.size() / 3;
    require(lod.leaf_pivots.size() == verts * 3, "leaf_pivots not one triple per leaf vertex");

    L.leaf_triangles = static_cast<int>(idx.size() / 3);
    L.leaf_vertices  = static_cast<int>(verts);

    double area = 0.0;
    for (size_t t = 0; t + 2 < idx.size(); t += 3) {
        const float* a = &pos[size_t(idx[t + 0]) * 3];
        const float* b = &pos[size_t(idx[t + 1]) * 3];
        const float* c = &pos[size_t(idx[t + 2]) * 3];
        const double ux = double(b[0]) - a[0], uy = double(b[1]) - a[1], uz = double(b[2]) - a[2];
        const double vx = double(c[0]) - a[0], vy = double(c[1]) - a[1], vz = double(c[2]) - a[2];
        const double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
        area += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
    }
    L.leaf_card_area_m2 = area;

    std::unordered_map<Key<3>, int, KeyHash> hosts;
    std::unordered_set<Key<3>, KeyHash>      distinct;
    for (size_t v = 0; v < verts; ++v) {
        const int n = ++hosts[pos_key(&lod.leaf_pivots[v * 3])];
        L.leaf_max_vertices_on_one_host = std::max(L.leaf_max_vertices_on_one_host, n);
        distinct.insert(pos_key(&pos[v * 3]));
    }
    L.leaf_hosts = static_cast<int>(hosts.size());
    L.leaf_distinct_position_ratio = verts ? double(distinct.size()) / double(verts) : 0.0;
}

void put(std::string& o, const char* key, int v, bool comma = true) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "\"%s\": %d%s", key, v, comma ? ", " : "");
    o += buf;
}

void put(std::string& o, const char* key, double v, bool comma = true) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "\"%s\": %.17g%s", key, v, comma ? ", " : "");
    o += buf;
}

void put_classes(std::string& o, const char* key, const BarkClassCounts& c) {
    o += "\""; o += key; o += "\": {";
    put(o, "tube", c[0]); put(o, "tip", c[1]); put(o, "collar", c[2]); put(o, "crotch", c[3], false);
    o += "}, ";
}

void put_bins(std::string& o, const char* key, const std::array<int, k_ledger_diameter_classes>& b) {
    o += "\""; o += key; o += "\": [";
    for (size_t i = 0; i < b.size(); ++i) {
        o += std::to_string(b[i]);
        if (i + 1 < b.size()) o += ", ";
    }
    o += "]";
}

}  // namespace

SkeletonLedger compute_skeleton_ledger(const TreeDescriptor& desc, const TreeSkeleton& skel) {
    SkeletonLedger S;
    S.nodes               = static_cast<int>(skel.nodes.size());
    S.iterations_run      = skel.iterations_run;
    S.max_iterations      = desc.branching.max_iterations;
    S.attractors_total    = desc.branching.attractor_count;
    S.attractors_consumed = skel.attractors_consumed;
    S.attractors_stalled  = skel.attractors_stalled;
    S.stall_side_branches = skel.stall_side_branches;
    S.stall_guard_retirements = skel.stall_guard_retirements;

    const size_t N = skel.nodes.size();
    std::vector<std::vector<int>> children(N);
    for (size_t i = 0; i < N; ++i) {
        const int p = skel.nodes[i].parent_index;
        if (p >= 0) {
            require(size_t(p) < i, "parent_index not below own index");
            children[size_t(p)].push_back(static_cast<int>(i));
        }
    }

    std::unordered_set<Key<3>, KeyHash> mm;
    mm.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        const vec3 p = skel.nodes[i].position;
        const Key<3> k = {uint32_t(std::lround(p.x * 1000.0f)), uint32_t(std::lround(p.y * 1000.0f)),
                          uint32_t(std::lround(p.z * 1000.0f))};
        if (!mm.insert(k).second) ++S.coincident_nodes_1mm;
    }

    require(skel.stall_side_nodes.size() == size_t(skel.stall_side_branches), "stall_side_nodes does not match stall_side_branches");
    std::vector<uint8_t> is_side(N, 0);
    for (int n : skel.stall_side_nodes) {
        require(n > 0 && size_t(n) < N, "stall side node out of range");
        is_side[size_t(n)] = 1;
    }

    const float near_dist = 0.25f * desc.branching.growth_distance;
    const float near_sq   = near_dist * near_dist;
    for (const auto& kids : children) {
        S.largest_child_count = std::max(S.largest_child_count, static_cast<int>(kids.size()));
        std::unordered_map<Key<3>, int, KeyHash> stack;
        for (int c : kids) {
            const int n = ++stack[pos_key(skel.nodes[size_t(c)].position)];
            if (n > 1) ++S.coincident_siblings_exact;
            S.largest_coincident_stack = std::max(S.largest_coincident_stack, n);
        }
        for (size_t a = 0; a < kids.size(); ++a) {
            const vec3 pa = skel.nodes[size_t(kids[a])].position;
            for (size_t b = a + 1; b < kids.size(); ++b) {
                const vec3 pb = skel.nodes[size_t(kids[b])].position;
                if (pos_key(pa) == pos_key(pb)) continue;
                if (length_squared(pa - pb) >= near_sq) continue;
                ++S.near_sibling_pairs;
                // kids ascend in node index = emission order: kids[a] was emitted first.
                if (is_side[size_t(kids[b])]) ++S.near_pairs_side_vs_earlier;
                else if (is_side[size_t(kids[a])]) ++S.near_pairs_side_vs_later;
            }
        }
    }
    return S;
}

TreeLedger compute_ledger(const TreeDescriptor& desc, const TreeSkeleton& skel,
                          std::span<const LodOutput> lods) {
    TreeLedger T;
    T.skeleton = compute_skeleton_ledger(desc, skel);
    T.levels.reserve(lods.size());
    for (const LodOutput& lod : lods) {
        LevelLedger L;
        bark_section(L, skel, lod);
        leaf_section(L, lod);
        T.levels.push_back(L);
    }
    return T;
}

std::string ledger_json(const TreeLedger& T) {
    std::string o;
    o += "{";
    put(o, "schema", T.schema);
    const SkeletonLedger& S = T.skeleton;
    o += "\"skeleton\": {";
    put(o, "nodes", S.nodes);
    put(o, "iterations_run", S.iterations_run);
    put(o, "max_iterations", S.max_iterations);
    put(o, "attractors_total", S.attractors_total);
    put(o, "attractors_consumed", S.attractors_consumed);
    put(o, "coincident_siblings_exact", S.coincident_siblings_exact);
    put(o, "coincident_nodes_1mm", S.coincident_nodes_1mm);
    put(o, "largest_child_count", S.largest_child_count);
    put(o, "largest_coincident_stack", S.largest_coincident_stack);
    put(o, "near_sibling_pairs", S.near_sibling_pairs);
    put(o, "attractors_stalled", S.attractors_stalled);
    put(o, "stall_side_branches", S.stall_side_branches);
    put(o, "near_pairs_side_vs_earlier", S.near_pairs_side_vs_earlier);
    put(o, "near_pairs_side_vs_later", S.near_pairs_side_vs_later);
    put(o, "stall_guard_retirements", S.stall_guard_retirements, false);
    o += "},\n\"levels\": [";
    for (size_t i = 0; i < T.levels.size(); ++i) {
        const LevelLedger& L = T.levels[i];
        o += "\n  {";
        put(o, "level", static_cast<int>(i));
        put(o, "bark_triangles", L.bark_triangles);
        put(o, "bark_vertices", L.bark_vertices);
        put(o, "tube", L.by_class[0]);
        put(o, "tip", L.by_class[1]);
        put(o, "collar", L.by_class[2]);
        put(o, "crotch", L.by_class[3]);
        o += "\"tube_tip_by_diameter\": {";
        put_bins(o, "tube", L.tube_by_diameter);
        o += ", ";
        put_bins(o, "tip", L.tip_by_diameter);
        o += "}, ";
        put_classes(o, "coincident_triangles", L.coincident_triangles);
        put(o, "open_edges", L.open_edges);
        put(o, "nonmanifold_edges", L.nonmanifold_edges);
        put(o, "degenerate_triangles", L.degenerate_triangles);
        put(o, "leaf_triangles", L.leaf_triangles);
        put(o, "leaf_vertices", L.leaf_vertices);
        put(o, "leaf_card_area_m2", L.leaf_card_area_m2);
        put(o, "leaf_hosts", L.leaf_hosts);
        put(o, "leaf_max_vertices_on_one_host", L.leaf_max_vertices_on_one_host);
        put(o, "leaf_distinct_position_ratio", L.leaf_distinct_position_ratio, false);
        o += "}";
        if (i + 1 < T.levels.size()) o += ",";
    }
    o += "\n]}\n";
    return o;
}

}  // namespace treegen
