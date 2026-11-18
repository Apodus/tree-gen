// [ledger] / [skeleton] — T0 geometry ledger (tree-budget plan, phase T0).
//
// G0a classes are counted right; G0b the stall detector; G0c the premise check
// over the 16 shipped trees; G0d the 36-seed baselines; G0e fingerprints and
// vertex sums; G0f the schema. The shipped trees are built in-process exactly as
// main.cpp builds them (a case below byte-compares this mirror's ledger with the
// tool's --ledger output). Baselines: shipped_tree_table.hpp.

#include "../external/catch2/catch.hpp"
#include "test_env.hpp"
#include "test_support_paths.hpp"

#include "shipped_tree_table.hpp"

#include "../branch_mesh.hpp"
#include "../det_rng.hpp"
#include "../face_budget.hpp"
#include "../leaf_budget.hpp"
#include "../leaf_geometry.hpp"
#include "../leaf_placement.hpp"
#include "../lod_emitter.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"
#include "../tree_descriptor.hpp"
#include "../tree_ledger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace ts  = treegen;
namespace tbl = treegen::shipped_table;

namespace {

// FNV-1a 64 over each node's parent_index, position, axis, depth, radius,
// wind_tier and t_along_parent bytes, field by field, in that order.
uint64_t skeleton_fnv(const ts::TreeSkeleton& skel) {
    uint64_t h = 14695981039346656037ull;
    auto mix = [&](const void* p, size_t n) {
        const auto* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    for (const auto& n : skel.nodes) {
        mix(&n.parent_index, sizeof n.parent_index);
        mix(&n.position.x, 4); mix(&n.position.y, 4); mix(&n.position.z, 4);
        mix(&n.axis.x, 4);     mix(&n.axis.y, 4);     mix(&n.axis.z, 4);
        mix(&n.depth, sizeof n.depth);
        mix(&n.radius, sizeof n.radius);
        mix(&n.wind_tier, sizeof n.wind_tier);
        mix(&n.t_along_parent, sizeof n.t_along_parent);
    }
    return h;
}

ts::Scenario load_species(const std::string& species) {
    const std::string path = rynx::test_support::treegen_file("scenarios/c3_" + species + ".json");
    REQUIRE_FALSE(path.empty());
    ts::Scenario s = ts::load_scenario(path);
    REQUIRE(s.kind == "tree");
    return s;
}

// The tool's tree branch (main.cpp), step for step: skeleton, leaf sites, bark
// options, budgets, emit_all_lods.
std::vector<ts::LodOutput> build_levels_as_tool(const ts::Scenario& s, ts::TreeSkeleton& skel,
                                                uint64_t seed_effective) {
    auto lp = ts::leaf_placement::options_from_descriptor(s.tree.leaves);
    skel.leaf_sites = ts::leaf_placement::generate_leaf_sites(skel, lp, seed_effective);

    ts::BarkMeshOptions bopts;
    bopts.tree_height_m            = s.tree.height_m;
    bopts.root_flare_factor        = s.tree.root_flare_factor;
    bopts.junction_shoulder_factor = s.tree.junction_shoulder_factor;
    ts::pcg32 rng;
    rng.seed(seed_effective, 0x5EA00FF5E7u);
    bopts.seam_offset_rad = rng.next_float_01() * 6.28318530717958647692f;

    const ts::LodBudget budget = ts::lod_budget_for_species(s.tree.leaves.shape);
    const bool is_strip   = (s.tree.leaves.geometry_type == ts::LeafGeometryType::BranchStrip);
    const bool has_leaves = is_strip || (!skel.leaf_sites.empty() && s.tree.leaves.leaf_density_per_meter > 0.0f);
    if (!has_leaves) return ts::emit_all_lods(skel, bopts, budget, s.tree.height_m, seed_effective);

    ts::LeafBudget leaf_budget;
    ts::LeafMeshOptions g;
    g.geometry_type         = s.tree.leaves.geometry_type;
    g.shape                 = s.tree.leaves.shape;
    g.leaf_size_m           = s.tree.leaves.leaf_size_m;
    g.cluster_count_per_tip = s.tree.leaves.cluster_count_per_tip;
    g.bend_half_angle       = s.tree.leaves.leaf_bend_half_angle;
    return ts::emit_all_lods(skel, skel.leaf_sites, bopts, budget, leaf_budget, g, s.tree.height_m, seed_effective);
}

struct ShippedTree {
    std::string    species;
    uint64_t       seed = 0;
    uint64_t       fnv  = 0;
    bool           strip = false;
    ts::TreeLedger ledger;
};

ShippedTree build_shipped(const std::string& species, uint64_t seed) {
    const ts::Scenario s = load_species(species);
    const uint64_t seed_effective = seed ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
    const auto lods = build_levels_as_tool(s, skel, seed_effective);
    ShippedTree t;
    t.species = species;
    t.seed    = seed;
    t.fnv     = skeleton_fnv(skel);
    t.strip   = (s.tree.leaves.geometry_type == ts::LeafGeometryType::BranchStrip);
    t.ledger  = ts::compute_ledger(s.tree, skel, std::span<const ts::LodOutput>(lods.data(), lods.size()));
    return t;
}

// Built once per process; the shipped cases share it.
const std::vector<ShippedTree>& shipped() {
    static const std::vector<ShippedTree> trees = [] {
        std::vector<ShippedTree> v;
        for (const auto& r : tbl::k_rows) v.push_back(build_shipped(r.species, r.seed));
        return v;
    }();
    return trees;
}

struct SeedRow { uint64_t seed; uint64_t fnv; ts::SkeletonLedger led; };

const std::vector<SeedRow>& seed_set(const std::string& species) {
    static std::vector<std::pair<std::string, std::vector<SeedRow>>> cache;
    for (const auto& [name, rows] : cache) if (name == species) return rows;
    const ts::Scenario s = load_species(species);
    std::vector<SeedRow> rows;
    for (uint64_t seed : tbl::k_seed_set) {
        const ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed ^ s.scenario_fnv);
        rows.push_back({seed, skeleton_fnv(skel), ts::compute_skeleton_ledger(s.tree, skel)});
    }
    cache.emplace_back(species, std::move(rows));
    return cache.back().second;
}

bool is_clean(const ShippedTree& t) { return t.ledger.skeleton.coincident_siblings_exact == 0; }

// One LodOutput from a hand-meshed bark (the ledger's input type).
ts::LodOutput as_level(ts::BarkMeshOutput b) {
    ts::LodOutput lod;
    lod.mesh                = std::move(b.mesh);
    lod.indices_u32         = std::move(b.indices_u32);
    lod.bark_triangle_class = std::move(b.triangle_class);
    lod.bark_triangle_node  = std::move(b.triangle_node);
    return lod;
}

ts::BranchNode node(int parent, ts::vec3 p, float r, int depth) {
    ts::BranchNode n;
    n.parent_index = parent;
    n.position     = p;
    n.radius       = r;
    n.depth        = depth;
    return n;
}

// Keys of a ledger document, in order ("key": ...).
std::vector<std::string> json_keys(const std::string& doc) {
    std::vector<std::string> keys;
    for (size_t i = 0; i < doc.size(); ++i) {
        if (doc[i] != '"') continue;
        const size_t end = doc.find('"', i + 1);
        REQUIRE(end != std::string::npos);
        if (end + 1 < doc.size() && doc[end + 1] == ':') keys.push_back(doc.substr(i + 1, end - i - 1));
        i = end;
    }
    return keys;
}

std::vector<std::string> expected_schema_keys(size_t levels) {
    std::vector<std::string> k = {
        "schema", "skeleton",
        "nodes", "iterations_run", "max_iterations", "attractors_total", "attractors_consumed",
        "coincident_siblings_exact", "coincident_nodes_1mm", "largest_child_count",
        "largest_coincident_stack", "near_sibling_pairs",
        "levels",
    };
    for (size_t i = 0; i < levels; ++i) {
        for (const char* s : {"level", "bark_triangles", "bark_vertices", "tube", "tip", "collar", "crotch",
                              "tube_tip_by_diameter", "tube", "tip",
                              "coincident_triangles", "tube", "tip", "collar", "crotch",
                              "open_edges", "nonmanifold_edges",
                              "leaf_triangles", "leaf_vertices", "leaf_card_area_m2", "leaf_hosts",
                              "leaf_max_vertices_on_one_host", "leaf_distinct_position_ratio"})
            k.push_back(s);
    }
    return k;
}

int level_bark_vertex_sum(int level) {
    int v = 0;
    for (const auto& t : shipped()) v += t.ledger.levels[size_t(level)].bark_vertices;
    return v;
}
int level_leaf_vertex_sum(int level) {
    int v = 0;
    for (const auto& t : shipped()) v += t.ledger.levels[size_t(level)].leaf_vertices;
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// G0a — classes are counted right (asserts).
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0a bark triangle classes on a hand-built fork", "[treegen][ledger]") {
    // Root, a trunk of 3 nodes (the last is the fork), two children of 2 nodes each.
    ts::TreeSkeleton skel;
    skel.nodes = {
        node(-1, {0, 0, 0}, 0.20f, 0),
        node(0, {0, 0, 1}, 0.18f, 0),
        node(1, {0, 0, 2}, 0.16f, 0),
        node(2, {0, 0, 3}, 0.14f, 0),
        node(3, {0.4f, 0, 3.4f}, 0.08f, 1),
        node(4, {0.8f, 0, 3.8f}, 0.06f, 1),
        node(3, {-0.4f, 0, 3.4f}, 0.08f, 1),
        node(6, {-0.8f, 0, 3.8f}, 0.06f, 1),
    };
    ts::BarkMeshOptions opts;
    opts.radial_seg_trunk = 8;
    opts.radial_seg_order1 = 5;
    opts.radial_seg_order2 = 4;
    opts.radial_seg_order3_plus = 3;
    const int N[2] = {8, 5};  // by depth

    // Tube closed form: every emitted segment contributes axial_segs strips of N quads
    // (a continuation-merged segment reuses the parent's ring as its s=0 ring and still
    // emits axial_segs strips), 2 triangles per quad.
    int tube_expected = 0;
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const auto& n = skel.nodes[i];
        const float len = ts::length(n.position - skel.nodes[size_t(n.parent_index)].position);
        const int axial = std::max(1, int(std::floor(len / opts.axial_segment_length_m)));
        tube_expected += 2 * N[n.depth] * axial;
    }
    REQUIRE(tube_expected == 3 * 2 * 8 * 2 + 4 * 2 * 5 * 1);
    const int tip_expected = N[1] + N[1];  // leaf nodes 5 and 7

    SECTION("fork blend on") {
        const ts::LodOutput lv[1] = {as_level(ts::build_bark_mesh(skel, opts))};
        const int tris = int(lv[0].indices_u32.size() / 3);
        REQUIRE(tris > 0);
        const ts::TreeLedger L = ts::compute_ledger(ts::TreeDescriptor{}, skel, lv);
        const auto& c = L.levels[0].by_class;
        INFO("tube=" << c[0] << " tip=" << c[1] << " collar=" << c[2] << " crotch=" << c[3] << " total=" << tris);
        CHECK(c[0] + c[1] + c[2] + c[3] == tris);
        CHECK(c[0] == tube_expected);
        CHECK(c[1] == tip_expected);
        CHECK(c[3] == 2);  // a two-child fork emits its crotch pair twice
        CHECK(c[2] == tris - c[0] - c[1] - c[3]);
        CHECK(c[2] > 0);
    }
    SECTION("fork blend off") {
        opts.apply_fork_blend = false;
        const ts::LodOutput lv[1] = {as_level(ts::build_bark_mesh(skel, opts))};
        const int tris = int(lv[0].indices_u32.size() / 3);
        REQUIRE(tris > 0);
        const ts::TreeLedger L = ts::compute_ledger(ts::TreeDescriptor{}, skel, lv);
        const auto& c = L.levels[0].by_class;
        INFO("tube=" << c[0] << " tip=" << c[1] << " collar=" << c[2] << " crotch=" << c[3] << " total=" << tris);
        CHECK(c[0] + c[1] + c[2] + c[3] == tris);
        CHECK(c[0] == tube_expected);
        CHECK(c[1] == tip_expected);
        CHECK(c[2] == 0);
        CHECK(c[3] == 0);
    }
}

// ---------------------------------------------------------------------------
// G0b — the stall detector (asserts).
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0b coincident siblings and stacked triangles", "[treegen][ledger]") {
    ts::TreeDescriptor desc;
    desc.branching.growth_distance = 0.3f;

    // Fork node 2 with child 3 (depth 1, a 1.17 m segment: 2 axial segments) and,
    // in the duplicate arms, node 4 = node 3 under the same parent.
    const ts::BranchNode child = node(2, {0.6f, 0, 3.0f}, 0.05f, 1);
    auto make = [&](bool dup, float offset_y) {
        ts::TreeSkeleton s;
        s.nodes = {node(-1, {0, 0, 0}, 0.20f, 0), node(0, {0, 0, 1}, 0.15f, 0), node(1, {0, 0, 2}, 0.12f, 0), child};
        if (dup) {
            ts::BranchNode d = child;
            d.position.y += offset_y;
            s.nodes.push_back(d);
        }
        return s;
    };
    ts::BarkMeshOptions opts;
    opts.apply_fork_blend = false;  // exact counts
    const int N = opts.radial_seg_order1;
    const float len = ts::length(child.position - ts::vec3{0, 0, 2});
    const int axial = std::max(1, int(std::floor(len / opts.axial_segment_length_m)));
    REQUIRE(axial == 2);

    auto ledger_of = [&](const ts::TreeSkeleton& s) {
        const ts::LodOutput lv[1] = {as_level(ts::build_bark_mesh(s, opts))};
        REQUIRE(lv[0].indices_u32.size() / 3 > 0);
        return ts::compute_ledger(desc, s, lv);
    };

    SECTION("bitwise duplicate") {
        const auto L = ledger_of(make(true, 0.0f));
        CHECK(L.skeleton.coincident_siblings_exact == 1);
        CHECK(L.skeleton.largest_coincident_stack == 2);
        CHECK(L.skeleton.near_sibling_pairs == 0);
        CHECK(L.levels[0].coincident_triangles[0] == 2 * N * axial);
        CHECK(L.levels[0].coincident_triangles[1] == N);
    }
    SECTION("no duplicate") {
        const auto L = ledger_of(make(false, 0.0f));
        CHECK(L.skeleton.coincident_siblings_exact == 0);
        CHECK(L.skeleton.largest_coincident_stack == 1);
        CHECK(L.skeleton.near_sibling_pairs == 0);
        CHECK(L.levels[0].coincident_triangles == ts::BarkClassCounts{0, 0, 0, 0});
    }
    SECTION("duplicate moved by 0.5 mm") {
        const auto L = ledger_of(make(true, 0.0005f));
        CHECK(L.skeleton.coincident_siblings_exact == 0);
        CHECK(L.skeleton.largest_coincident_stack == 1);
        CHECK(L.skeleton.near_sibling_pairs == 1);
    }
}

// ---------------------------------------------------------------------------
// The test's tool mirror is the tool: ledger bytes from `rynx-treegen --ledger`
// equal the in-process ledger (one card species, one strip species).
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] tool --ledger equals the in-process mirror", "[ledger]") {  // spawns the tool
    namespace fs = std::filesystem;
    const fs::path exe = treegen::test_env::treegen_exe();
    for (const char* species : {"oak", "pine"}) {
        INFO("species=" << species);
        const ShippedTree t = build_shipped(species, 42);
        const fs::path glb = treegen::test_env::scratch(std::string("ledger_") + species + ".glb");
        const fs::path out = treegen::test_env::scratch(std::string("ledger_") + species + ".json");
        std::error_code ec;
        fs::remove(out, ec);
        std::ostringstream args;
        args << "--scenario=\"" << rynx::test_support::treegen_file(std::string("scenarios/c3_") + species + ".json")
             << "\" --seed=42 --out=\"" << glb.string() << "\" --ledger=\"" << out.string() << "\"";
        REQUIRE(rynx::test_support::run_exe_in_dir(exe.string(), out.parent_path().string(), args.str()) == 0);
        std::ifstream f(out, std::ios::binary);
        REQUIRE(f.good());
        const std::string doc((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        CHECK(doc == ts::ledger_json(t.ledger));
    }
}

TEST_CASE("[ledger] tool --ledger without --out exits 2", "[ledger]") {  // spawns the tool
    namespace fs = std::filesystem;
    const fs::path exe = treegen::test_env::treegen_exe();
    const fs::path out = treegen::test_env::scratch("ledger_no_out.json");
    std::ostringstream args;
    args << "--species oak --ledger=\"" << out.string() << "\"";
    CHECK(rynx::test_support::run_exe_in_dir(exe.string(), out.parent_path().string(), args.str()) == 2);
    CHECK_FALSE(fs::exists(out));
}

// ---------------------------------------------------------------------------
// G0c — premise check and stop rule over the 16 shipped trees.
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0c shipped trees: stall premise", "[treegen][ledger]") {
    const auto& trees = shipped();
    REQUIRE(trees.size() == 16u);

    std::printf("G0c species   seed  nodes iters/max  exact  1mm stack kids near consumed | per level: bark(tube/tip/collar/crotch) coinc(tube/tip/collar/crotch) open nonmanifold leaf_t leaf_v hosts maxhost area distinct\n");
    for (const auto& t : trees) {
        const auto& S = t.ledger.skeleton;
        std::printf("G0c %-6s %5llu %6d %5d/%-3d %6d %4d %5d %4d %4d %8d\n", t.species.c_str(),
                    (unsigned long long)t.seed, S.nodes, S.iterations_run, S.max_iterations,
                    S.coincident_siblings_exact, S.coincident_nodes_1mm, S.largest_coincident_stack,
                    S.largest_child_count, S.near_sibling_pairs, S.attractors_consumed);
        for (size_t l = 0; l < t.ledger.levels.size(); ++l) {
            const auto& L = t.ledger.levels[l];
            std::printf("G0c     L%zu bark %6d (%d/%d/%d/%d) v %6d coinc (%d/%d/%d/%d) open %5d nonman %4d "
                        "leaf_t %5d leaf_v %5d hosts %5d maxhost %5d area %.3f distinct %.4f "
                        "tube_by_d [%d %d %d %d %d] tip_by_d [%d %d %d %d %d]\n",
                        l, L.bark_triangles, L.by_class[0], L.by_class[1], L.by_class[2], L.by_class[3],
                        L.bark_vertices, L.coincident_triangles[0], L.coincident_triangles[1],
                        L.coincident_triangles[2], L.coincident_triangles[3], L.open_edges, L.nonmanifold_edges,
                        L.leaf_triangles, L.leaf_vertices, L.leaf_hosts, L.leaf_max_vertices_on_one_host,
                        L.leaf_card_area_m2, L.leaf_distinct_position_ratio,
                        L.tube_by_diameter[0], L.tube_by_diameter[1], L.tube_by_diameter[2],
                        L.tube_by_diameter[3], L.tube_by_diameter[4],
                        L.tip_by_diameter[0], L.tip_by_diameter[1], L.tip_by_diameter[2],
                        L.tip_by_diameter[3], L.tip_by_diameter[4]);
        }
    }

    // (i) asserts: the stall exists on a shipped tree. If this is red, T1 is dropped.
    int worst = 0;
    for (const auto& t : trees) worst = std::max(worst, t.ledger.skeleton.coincident_siblings_exact);
    INFO("largest coincident_siblings_exact on a shipped tree: " << worst);
    CHECK(worst > 100);

    // (ii) prints: the clean set (expected birch 7/42/123/999, maple 42, pine 7/999).
    std::printf("G0c (ii) clean:");
    for (const auto& t : trees) if (is_clean(t)) std::printf(" %s_%llu", t.species.c_str(), (unsigned long long)t.seed);
    std::printf("\nG0c (ii) stalled:");
    for (const auto& t : trees) if (!is_clean(t)) std::printf(" %s_%llu", t.species.c_str(), (unsigned long long)t.seed);
    std::printf("\n");

    // (iii) prints: collar + crotch share of L0 bark on every clean tree.
    for (const auto& t : trees) {
        if (!is_clean(t)) continue;
        const auto& L = t.ledger.levels[0];
        std::printf("G0c (iii) %s_%llu L0 collar+crotch %d of %d = %.1f%%\n", t.species.c_str(),
                    (unsigned long long)t.seed, L.by_class[2] + L.by_class[3], L.bark_triangles,
                    100.0 * (L.by_class[2] + L.by_class[3]) / std::max(1, L.bark_triangles));
    }

    // (iv) asserts the proven direction: a run that hit max_iterations re-emitted a
    // bit-identical child. Prints transient stalls (coincident siblings, ended early).
    for (const auto& t : trees) {
        const auto& S = t.ledger.skeleton;
        INFO(t.species << "_" << t.seed << " iterations " << S.iterations_run << "/" << S.max_iterations
                       << " exact " << S.coincident_siblings_exact);
        if (S.iterations_run == S.max_iterations) CHECK(S.coincident_siblings_exact > 0);
        if (S.coincident_siblings_exact > 0 && S.iterations_run < S.max_iterations)
            std::printf("G0c (iv) transient: %s_%llu\n", t.species.c_str(), (unsigned long long)t.seed);
    }
}

// ---------------------------------------------------------------------------
// G0d — 36-seed baselines for T1 ([pin]).
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0d seed-set skeleton baselines", "[treegen][ledger]") {
    for (const auto& sp : tbl::k_species) {
        const auto& rows = seed_set(sp.species);
        REQUIRE(rows.size() == tbl::k_seed_set.size());
        int stalled = 0, max_children = 0, max_near = 0;
        long long consumed = 0;
        bool any_clean = false;
        std::printf("G0d %s stalled seeds:", sp.species);
        for (const auto& r : rows) {
            consumed += r.led.attractors_consumed;
            if (r.led.coincident_siblings_exact > 0) {
                ++stalled;
                std::printf(" %llu", (unsigned long long)r.seed);
            } else {
                any_clean = true;
                max_children = std::max(max_children, r.led.largest_child_count);
                max_near     = std::max(max_near, r.led.near_sibling_pairs);
            }
        }
        std::printf("\n");
        for (const auto& r : rows)
            std::printf("G0d %s seed %3llu nodes %5d iters %3d/%d exact %4d stack %4d kids %4d near %3d consumed %4d/%d\n",
                        sp.species, (unsigned long long)r.seed, r.led.nodes, r.led.iterations_run,
                        r.led.max_iterations, r.led.coincident_siblings_exact, r.led.largest_coincident_stack,
                        r.led.largest_child_count, r.led.near_sibling_pairs, r.led.attractors_consumed,
                        r.led.attractors_total);
        std::printf("G0d TABLE {\"%s\", %d, %d, %d, %lld}%s\n", sp.species, stalled, max_children, max_near, consumed,
                    any_clean ? "" : "  (no stall-free seed: maxima taken from the other species)");
        INFO("species " << sp.species);
        CHECK(stalled == sp.k_t0_stalled_seeds);
        CHECK(consumed == sp.k_t0_consumed_sum);
        if (any_clean) {
            CHECK(max_children == sp.k_t0_max_children);
            CHECK(max_near == sp.k_t0_max_near_pairs);
        } else {
            // The plan's rule: the largest value of the other three species.
            int kids = 0, near = 0;
            for (const auto& o : tbl::k_species) {
                if (std::strcmp(o.species, sp.species) == 0) continue;
                kids = std::max(kids, o.k_t0_max_children);
                near = std::max(near, o.k_t0_max_near_pairs);
            }
            CHECK(sp.k_t0_max_children == kids);
            CHECK(sp.k_t0_max_near_pairs == near);
        }
    }
}

// ---------------------------------------------------------------------------
// G0e — fingerprints and sums ([pin]; hold by construction in T0).
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0e shipped fingerprints and vertex sums", "[treegen][ledger]") {
    const auto& trees = shipped();
    REQUIRE(trees.size() == tbl::k_rows.size());
    for (size_t i = 0; i < trees.size(); ++i) {
        const auto& t = trees[i];
        const auto& r = tbl::k_rows[i];
        std::printf("G0e TABLE {\"%s\", %llu, 0x%016llxull, 0x%016llxull, %d, %d},\n", t.species.c_str(),
                    (unsigned long long)t.seed, (unsigned long long)t.fnv, (unsigned long long)t.fnv,
                    t.ledger.skeleton.coincident_siblings_exact, t.ledger.skeleton.iterations_run);
        INFO(t.species << "_" << t.seed);
        REQUIRE(t.species == r.species);
        REQUIRE(t.seed == r.seed);
        CHECK(t.fnv == r.skeleton_fnv);
        CHECK(t.ledger.skeleton.iterations_run == r.t0_iterations_run);
        CHECK(t.ledger.skeleton.coincident_siblings_exact == r.t0_coincident_siblings_exact);
    }
    for (int l = 0; l < 3; ++l) {
        const long long bark = level_bark_vertex_sum(l), leaf = level_leaf_vertex_sum(l);
        std::printf("G0e TABLE L%d vertex_sum %lld bark %lld leaf %lld\n", l, bark + leaf, bark, leaf);
        INFO("level " << l);
        CHECK(bark + leaf == tbl::k_t0_vertex_sum[size_t(l)]);
        CHECK(bark == tbl::k_t0_bark_vertex_sum[size_t(l)]);
        CHECK(leaf == tbl::k_t0_leaf_vertex_sum[size_t(l)]);
    }
    int clean_max[2] = {0, 0};
    for (const auto& t : trees) {
        if (!is_clean(t)) continue;
        for (const auto& L : t.ledger.levels)
            clean_max[t.strip ? 1 : 0] = std::max(clean_max[t.strip ? 1 : 0], L.leaf_max_vertices_on_one_host);
    }
    std::printf("G0e TABLE clean_max_host_vertices {%d, %d}\n", clean_max[0], clean_max[1]);
    CHECK(clean_max[0] == tbl::k_t0_clean_max_host_vertices[0]);
    CHECK(clean_max[1] == tbl::k_t0_clean_max_host_vertices[1]);
}

// ---------------------------------------------------------------------------
// G0f — schema (asserts), with its positive control.
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0f ledger JSON field list", "[treegen][ledger]") {
    const auto& t = shipped().front();
    const std::string doc = ts::ledger_json(t.ledger);
    const auto expected = expected_schema_keys(t.ledger.levels.size());
    CHECK(json_keys(doc) == expected);

    // Positive control: the same comparison fails on a document with one field removed.
    const std::string field = "\"open_edges\": ";
    const size_t at = doc.find(field);
    REQUIRE(at != std::string::npos);
    const size_t end = doc.find(", ", at);
    REQUIRE(end != std::string::npos);
    std::string cut = doc;
    cut.erase(at, end + 2 - at);
    CHECK(json_keys(cut) != expected);
    CHECK(json_keys(cut).size() + 1 == expected.size());
}
