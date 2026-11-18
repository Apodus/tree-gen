// [ledger] — T0 geometry ledger (tree-budget plan, phase T0).
//
// G0a classes, nodes and diameter bins are counted right (plus hand-placed edge
// and leaf-host arms); G0b the stall detector; G0c the premise record
// over the 16 shipped trees; G0d the 36-seed record; G0f the schema. T1 (grower
// stall): G1a-G1f; T1 deleted G0e (its pins moved to G1d and G1f) and the
// generator-matching arms of G0c and G0d. The shipped trees are built in-process exactly as
// main.cpp builds them (a case below byte-compares this mirror's ledger with the
// tool's --ledger output). Baselines: shipped_tree_table.hpp.

#include "../external/catch2/catch.hpp"
#include "test_env.hpp"
#include "test_support_paths.hpp"

#include "face_budget_overshoot.hpp"
#include "shipped_tree_table.hpp"

#include "../branch_mesh.hpp"
#include "../collision_fit.hpp"
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
        "largest_coincident_stack", "near_sibling_pairs", "attractors_stalled", "stall_side_branches",
        "near_pairs_side_vs_earlier", "near_pairs_side_vs_later", "stall_guard_retirements",
        "levels",
    };
    for (size_t i = 0; i < levels; ++i) {
        for (const char* s : {"level", "bark_triangles", "bark_vertices", "tube", "tip", "collar", "crotch",
                              "tube_tip_by_diameter", "tube", "tip",
                              "coincident_triangles", "tube", "tip", "collar", "crotch",
                              "open_edges", "nonmanifold_edges", "degenerate_triangles",
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
// G0a — classes, nodes and diameter bins are counted right (asserts). Every count
// is a closed form from the hand-built tree; none is a remainder.
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0a bark triangle classes on a hand-built fork", "[treegen][ledger]") {
    // Root, a trunk of 3 nodes (the last is the fork), two children of 2 nodes each.
    // Radii put the tubes in four diameter classes, two of them exactly on a class
    // boundary (2 * 0.04f == 0.08f and 2 * 0.02f == 0.04f exactly: the upper class).
    ts::TreeSkeleton skel;
    skel.nodes = {
        node(-1, {0, 0, 0}, 0.20f, 0),
        node(0, {0, 0, 1}, 0.10f, 0),           // d 0.20: >= 16 cm
        node(1, {0, 0, 2}, 0.06f, 0),           // d 0.12: 8-16
        node(2, {0, 0, 3}, 0.04f, 0),           // d 0.08: 8-16 (boundary)
        node(3, {0.4f, 0, 3.4f}, 0.015f, 1),    // d 0.03: 2-4
        node(4, {0.8f, 0, 3.8f}, 0.005f, 1),    // d 0.01: < 2
        node(3, {-0.4f, 0, 3.4f}, 0.02f, 1),    // d 0.04: 4-8 (boundary)
        node(6, {-0.8f, 0, 3.8f}, 0.009f, 1),   // d 0.018: < 2
    };
    ts::BarkMeshOptions opts;
    opts.radial_seg_trunk = 8;
    opts.radial_seg_order1 = 5;
    opts.radial_seg_order2 = 4;
    opts.radial_seg_order3_plus = 3;
    const int N[2] = {8, 5};  // by depth

    // Tube closed form: every emitted segment contributes axial_segs strips of N quads
    // (a continuation-merged segment reuses the parent's ring as its s=0 ring and still
    // emits axial_segs strips), 2 triangles per quad. Trunk segments are 1 m (2 axial),
    // the children's 0.57 m (1 axial).
    int tube_expected = 0;
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const auto& n = skel.nodes[i];
        const float len = ts::length(n.position - skel.nodes[size_t(n.parent_index)].position);
        const int axial = std::max(1, int(std::floor(len / opts.axial_segment_length_m)));
        tube_expected += 2 * N[n.depth] * axial;
    }
    REQUIRE(tube_expected == 3 * 2 * 8 * 2 + 4 * 2 * 5 * 1);
    const int tip_expected = N[1] + N[1];  // leaf nodes 5 and 7
    // Diameter classes < 2, 2-4, 4-8, 8-16, >= 16 cm. Tubes: nodes 5, 7 (10 each) | node 4 |
    // node 6 | nodes 2, 3 (32 each) | node 1 (32). Tips: the fans of nodes 5 and 7.
    const std::array<int, 5> tube_bins = {20, 10, 10, 64, 32};
    const std::array<int, 5> tip_bins  = {10, 0, 0, 0, 0};

    SECTION("fork blend on") {
        // Collar closed form per child k (fork_blend.cpp): n_collar = clamp(ceil(45 deg / 30), 2, 5)
        // = 2 rings of the parent's N_p = 8 sides -> one strip (2 * 8 triangles); the sector stitch
        // from the child's parent-rim sector s_k to the first collar ring, (|s_k| - 1) + (8 - 1);
        // the full-ring stitch from the last collar ring to the child's 5-sided first ring,
        // (8 - 1) + (5 - 1). The two sectors partition the 8 rim vertices.
        const int collar_expected = 2 * (2 * 8) + (8 - 2) + 2 * (8 - 1) + 2 * ((8 - 1) + (5 - 1));
        REQUIRE(collar_expected == 74);
        const ts::LodOutput lv[1] = {as_level(ts::build_bark_mesh(skel, opts))};
        const int tris = int(lv[0].indices_u32.size() / 3);
        REQUIRE(tris > 0);
        const ts::TreeLedger L = ts::compute_ledger(ts::TreeDescriptor{}, skel, lv);
        const auto& c = L.levels[0].by_class;
        INFO("tube=" << c[0] << " tip=" << c[1] << " collar=" << c[2] << " crotch=" << c[3] << " total=" << tris);
        CHECK(c[0] == tube_expected);
        CHECK(c[1] == tip_expected);
        CHECK(c[2] == collar_expected);
        CHECK(c[3] == 2);  // a two-child fork emits its crotch pair twice
        CHECK(c[0] + c[1] + c[2] + c[3] == tris);
        CHECK(L.levels[0].tube_by_diameter == tube_bins);
        CHECK(L.levels[0].tip_by_diameter == tip_bins);
    }
    SECTION("fork blend off") {
        opts.apply_fork_blend = false;
        const ts::LodOutput lv[1] = {as_level(ts::build_bark_mesh(skel, opts))};
        const int tris = int(lv[0].indices_u32.size() / 3);
        REQUIRE(tris > 0);
        const ts::TreeLedger L = ts::compute_ledger(ts::TreeDescriptor{}, skel, lv);
        const auto& c = L.levels[0].by_class;
        INFO("tube=" << c[0] << " tip=" << c[1] << " collar=" << c[2] << " crotch=" << c[3] << " total=" << tris);
        CHECK(c[0] == tube_expected);
        CHECK(c[1] == tip_expected);
        CHECK(c[2] == 0);
        CHECK(c[3] == 0);
        CHECK(c[0] + c[1] == tris);
        CHECK(L.levels[0].tube_by_diameter == tube_bins);
        CHECK(L.levels[0].tip_by_diameter == tip_bins);
    }
}

// ---------------------------------------------------------------------------
// Edges and degenerate triangles on hand-placed triangles (asserts; F4/F5).
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] edge counts on hand-placed triangles", "[treegen][ledger]") {
    ts::TreeSkeleton skel;
    skel.nodes = {node(-1, {0, 0, 0}, 0.1f, 0)};
    ts::LodOutput lod;
    // Vertices: a unit quad 0-1-2-3, vertex 4 at vertex 1's position, vertex 5 off the quad.
    lod.mesh.positions = {0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,  1, 0, 0,  0.5f, 0.5f, 1};
    auto add = [&](uint32_t a, uint32_t b, uint32_t c) {
        lod.indices_u32.insert(lod.indices_u32.end(), {a, b, c});
        lod.bark_triangle_class.push_back(uint8_t(ts::BarkTriClass::tube));
        lod.bark_triangle_node.push_back(0);
    };
    auto level = [&] {
        const ts::LodOutput lv[1] = {lod};
        return ts::compute_ledger(ts::TreeDescriptor{}, skel, lv).levels[0];
    };
    add(0, 1, 2);
    add(0, 2, 3);
    SECTION("a quad: four open edges, the diagonal shared") {
        const auto L = level();
        CHECK(L.open_edges == 4);
        CHECK(L.nonmanifold_edges == 0);
        CHECK(L.degenerate_triangles == 0);
    }
    SECTION("a triangle with two equal corners closes edge 0-1 once, not twice") {
        add(0, 1, 4);  // corners 1 and 4 are bitwise equal: one real edge, 0-1
        const auto L = level();
        CHECK(L.degenerate_triangles == 1);
        CHECK(L.open_edges == 3);       // 1-2, 2-3, 3-0
        CHECK(L.nonmanifold_edges == 0);
    }
    SECTION("a third triangle on the diagonal makes it non-manifold") {
        add(0, 2, 5);
        const auto L = level();
        CHECK(L.nonmanifold_edges == 1);  // 0-2
        CHECK(L.open_edges == 4 + 2);     // the quad's four, plus 2-5 and 5-0
        CHECK(L.degenerate_triangles == 0);
    }
}

// ---------------------------------------------------------------------------
// Leaf hosts and card area on hand-placed cards (asserts; F5).
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] leaf hosts on hand-placed cards", "[treegen][ledger]") {
    ts::TreeSkeleton skel;
    skel.nodes = {node(-1, {0, 0, 0}, 0.1f, 0)};
    ts::LodOutput lod;
    lod.has_leaves = true;
    // Three hosts: A carries one card (4 vertices), B one card (4), C two cards (8).
    const float host[3][3] = {{0, 0, 1}, {0, 0, 2}, {0, 0, 3}};
    const int cards_on[3] = {1, 1, 2};
    float x = 0.0f;
    for (int h = 0; h < 3; ++h) {
        for (int c = 0; c < cards_on[h]; ++c) {
            const uint32_t base = uint32_t(lod.leaf_positions.size() / 3);
            // A 1 x 0.5 m card in the z = h plane, each card at its own x: area 0.5 m2.
            const float q[4][3] = {{x, 0, float(h)}, {x + 1, 0, float(h)}, {x + 1, 0.5f, float(h)}, {x, 0.5f, float(h)}};
            for (const auto& p : q) {
                lod.leaf_positions.insert(lod.leaf_positions.end(), {p[0], p[1], p[2]});
                lod.leaf_pivots.insert(lod.leaf_pivots.end(), {host[h][0], host[h][1], host[h][2]});
            }
            lod.leaf_indices_u32.insert(lod.leaf_indices_u32.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
            x += 2.0f;
        }
    }
    const ts::LodOutput lv[1] = {lod};
    const auto L = ts::compute_ledger(ts::TreeDescriptor{}, skel, lv).levels[0];
    CHECK(L.leaf_vertices == 16);
    CHECK(L.leaf_triangles == 8);
    CHECK(L.leaf_hosts == 3);
    CHECK(L.leaf_max_vertices_on_one_host == 8);
    CHECK(L.leaf_card_area_m2 == Approx(4 * 0.5));
    CHECK(L.leaf_distinct_position_ratio == 1.0);
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
        // Fork blend off: the open edges are the rings no strip closes. Root ring of node 1's
        // tube (12 trunk sides); node 2's last ring (12: its only child has 8 sides, so no
        // continuation merge); node 3's first ring (8). Node 3's tip fan closes its end; the
        // seam duplicate welds by position. Nothing is shared by three triangles.
        CHECK(L.levels[0].open_edges == opts.radial_seg_trunk + opts.radial_seg_trunk + N);
        CHECK(L.levels[0].nonmanifold_edges == 0);
        CHECK(L.levels[0].degenerate_triangles == 0);
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
// G0c — premise record over the 16 shipped trees. T1 rewrote it: arm (i) reads
// the frozen column (the stall existed on main); the live (i) and arm (iv)
// ("max_iterations implies a stall", true only of the stalled grower) are gone,
// replaced by G1a.
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0c shipped trees: stall premise", "[treegen][ledger]") {
    const auto& trees = shipped();
    REQUIRE(trees.size() == 16u);

    std::printf("G0c species   seed  nodes iters/max  exact  1mm stack kids near consumed side retired | per level: bark(tube/tip/collar/crotch) coinc(tube/tip/collar/crotch) open nonmanifold leaf_t leaf_v hosts maxhost area distinct\n");
    for (const auto& t : trees) {
        const auto& S = t.ledger.skeleton;
        std::printf("G0c %-6s %5llu %6d %5d/%-3d %6d %4d %5d %4d %4d %8d %4d %4d\n", t.species.c_str(),
                    (unsigned long long)t.seed, S.nodes, S.iterations_run, S.max_iterations,
                    S.coincident_siblings_exact, S.coincident_nodes_1mm, S.largest_coincident_stack,
                    S.largest_child_count, S.near_sibling_pairs, S.attractors_consumed,
                    S.stall_side_branches, S.attractors_stalled);
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

    // (i) asserts on the frozen column: a shipped tree had more than 100 coincident siblings on main.
    int worst = 0;
    for (const auto& r : tbl::k_rows) worst = std::max(worst, r.t0_coincident_siblings_exact);
    CHECK(worst > 100);

    // (ii) prints: trees with a coincident sibling (none, after T1).
    std::printf("G0c (ii) trees with a coincident sibling:");
    int with_sibling = 0;
    for (const auto& t : trees)
        if (!is_clean(t)) { ++with_sibling; std::printf(" %s_%llu", t.species.c_str(), (unsigned long long)t.seed); }
    std::printf(with_sibling ? "\n" : " none\n");

    // (iii) prints: collar + crotch share of L0 bark.
    for (const auto& t : trees) {
        const auto& L = t.ledger.levels[0];
        std::printf("G0c (iii) %s_%llu L0 collar+crotch %d of %d = %.1f%%\n", t.species.c_str(),
                    (unsigned long long)t.seed, L.by_class[2] + L.by_class[3], L.bark_triangles,
                    100.0 * (L.by_class[2] + L.by_class[3]) / std::max(1, L.bark_triangles));
    }
}

// ---------------------------------------------------------------------------
// G0d — the 36-seed record. T1 deleted its "the generator matches" arms: the
// k_t0_ constants stay frozen as what G1a and G1c compare against. What is
// left asserts the table's own rule for a species with no stall-free seed.
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0d seed-set skeleton baselines", "[treegen][ledger]") {
    for (const auto& sp : tbl::k_species) {
        const auto& rows = seed_set(sp.species);
        REQUIRE(rows.size() == tbl::k_seed_set.size());
        for (const auto& r : rows)
            std::printf("G0d %s seed %3llu nodes %5d iters %3d/%d exact %4d stack %4d kids %4d near %3d consumed %4d/%d side %3d retired %3d\n",
                        sp.species, (unsigned long long)r.seed, r.led.nodes, r.led.iterations_run,
                        r.led.max_iterations, r.led.coincident_siblings_exact, r.led.largest_coincident_stack,
                        r.led.largest_child_count, r.led.near_sibling_pairs, r.led.attractors_consumed,
                        r.led.attractors_total, r.led.stall_side_branches, r.led.attractors_stalled);
        if (sp.k_t0_stalled_seeds == int(tbl::k_seed_set.size())) {
            // No stall-free seed on main: the plan's rule, the largest value of the other three species.
            int kids = 0, near_pairs = 0;
            for (const auto& o : tbl::k_species) {
                if (std::strcmp(o.species, sp.species) == 0) continue;
                kids = std::max(kids, o.k_t0_max_children);
                near_pairs = std::max(near_pairs, o.k_t0_max_near_pairs);
            }
            INFO("species " << sp.species);
            CHECK(sp.k_t0_max_children == kids);
            CHECK(sp.k_t0_max_near_pairs == near_pairs);
        }
    }
}

// ---------------------------------------------------------------------------
// T1 — grower stall. "Clean" and "stalled" are the frozen T0 rows
// (t0_coincident_siblings_exact == 0 or not).
// ---------------------------------------------------------------------------
namespace {
bool t0_clean(const tbl::Row& r) { return r.t0_coincident_siblings_exact == 0; }

const tbl::SpeciesRow& species_row(const std::string& species) {
    for (const auto& sp : tbl::k_species) if (species == sp.species) return sp;
    FAIL("no species row " << species);
    return tbl::k_species[0];
}
}  // namespace

// G1a — no stall anywhere: 4 species x the 36 seeds.
TEST_CASE("[skeleton] G1a no stall on the seed set", "[treegen][skeleton]") {
    long long side_total = 0;
    int trees_with_side = 0;
    for (const auto& sp : tbl::k_species) {
        const ts::Scenario s = load_species(sp.species);
        const float g = s.tree.branching.growth_distance, kill = s.tree.branching.kill_distance;
        // Side-branch separation from EARLIER siblings (ruling R-T1-near). A side branch S grows
        // toward attractor a at distance d > g/2 from the tip: |S - a| = |d - g|. An earlier
        // sibling X was emitted while a was in this tip's group (groups only shrink), did not
        // kill a (|X - a| >= kill) and did not take it from the tip (|X - a| >= d). So
        // |S - X| >= |X - a| - |S - a| >= g when d >= g, and > kill - g/2 when d < g. With
        // kill - g/2 above the near threshold g/4 that is never a near pair.
        INFO(sp.species << " kill " << kill << " step " << g);
        REQUIRE(kill - 0.5f * g > 0.25f * g);
        const int m = s.tree.branching.min_attractors_to_split;
        // A stuck group of n members adds at most n side branches and n - 1 regular children
        // to one tip; below the split threshold m that is at most 2m - 3.
        const int kids_bound = sp.k_t0_max_children + 2 * m - 3;
        for (const auto& r : seed_set(sp.species)) {
            INFO(sp.species << " seed " << r.seed << " kids " << r.led.largest_child_count << "/" << kids_bound
                            << " near " << r.led.near_sibling_pairs
                            << " side_vs_earlier " << r.led.near_pairs_side_vs_earlier
                            << " side_vs_later " << r.led.near_pairs_side_vs_later
                            << " iterations " << r.led.iterations_run << "/" << r.led.max_iterations
                            << " side " << r.led.stall_side_branches << " retired " << r.led.attractors_stalled);
            CHECK(r.led.coincident_siblings_exact == 0);
            CHECK(r.led.iterations_run < r.led.max_iterations);
            CHECK(r.led.largest_child_count <= kids_bound);
            CHECK(r.led.near_pairs_side_vs_earlier == 0);
            CHECK(r.led.stall_guard_retirements == 0);   // the F-A guard never fires on the seed set
            std::printf("G1a %s seed %3llu near %3d side %2d side_vs_earlier %d side_vs_later %d\n", sp.species,
                        (unsigned long long)r.seed, r.led.near_sibling_pairs, r.led.stall_side_branches,
                        r.led.near_pairs_side_vs_earlier, r.led.near_pairs_side_vs_later);
            side_total += r.led.stall_side_branches;
            trees_with_side += r.led.stall_side_branches > 0 ? 1 : 0;
        }
        // Seeds that were stall-free on main grow the same skeleton: their total near pairs are T0's.
        // The table lists every seed that was stall-free on main (review F-C).
        const auto listed = std::count_if(std::begin(tbl::k_t0_stall_free_near_pairs),
            std::end(tbl::k_t0_stall_free_near_pairs),
            [&](const tbl::StallFreeNearPairs& c) { return std::strcmp(c.species, sp.species) == 0; });
        CHECK(int(listed) == int(tbl::k_seed_set.size()) - sp.k_t0_stalled_seeds);
        for (const auto& c : tbl::k_t0_stall_free_near_pairs) {
            if (std::strcmp(c.species, sp.species) != 0) continue;
            const auto& rows = seed_set(sp.species);
            const auto it = std::find_if(rows.begin(), rows.end(), [&](const SeedRow& r) { return r.seed == c.seed; });
            REQUIRE(it != rows.end());
            INFO(sp.species << " stall-free seed " << c.seed);
            CHECK(it->led.near_sibling_pairs == c.near_sibling_pairs);
            CHECK(it->led.stall_side_branches == 0);
        }
    }
    // The side-branch arm has a subject.
    std::printf("G1a side branches over the seed set: %lld on %d trees\n", side_total, trees_with_side);
    CHECK(side_total > 0);
    CHECK(trees_with_side > 0);
}

// G1g (tool options) — the shipped trees as the stage builds them, every level, through the
// overshoot table's one rule.
TEST_CASE("[ledger] G1g shipped bark within budget or listed", "[treegen][ledger]") {
    treegen::face_budget_overshoot::BarkBudgetCheck budget_check("tool");
    for (const auto& t : shipped()) {
        const ts::Scenario s = load_species(t.species);
        const ts::LodBudget budget = ts::lod_budget_for_species(s.tree.leaves.shape);
        const int per_level[3] = {budget.l0_tris, budget.l1_tris, budget.l2_tris};
        for (int l = 0; l < 3; ++l)
            budget_check.check(t.species, t.seed, l, t.ledger.levels[size_t(l)].bark_triangles, per_level[l]);
    }
    budget_check.require_every_row_visited();
}

// G1h (T1 review F-A) — the integer progress guard ends a stall even when the binder and the
// stall branch disagree. The test seam keeps every stall side branch out of the tip list, so
// its attractor cannot leave the group although the branch believes it did; only the guard
// (a second entry at the same group size retires unconditionally) can end the stall.
TEST_CASE("[skeleton] G1h progress guard survives a binder disagreement", "[treegen][skeleton]") {
    struct hook_on {
        hook_on() { ts::test_hooks::stall_side_branch_unbindable = true; }
        ~hook_on() { ts::test_hooks::stall_side_branch_unbindable = false; }
    };
    // kill 0.1 m: a side branch kills nothing at its own distance, so its attractor stays in the
    // tip's group whenever the seam hides the side branch from the binder (the shipped kill radii
    // kill it, which is why the shipped trees never need the guard).
    ts::Scenario s = load_species("oak");
    s.tree.branching.kill_distance = 0.1f;
    int guard_total = 0;
    for (const uint64_t seed : {7ull, 42ull, 123ull, 999ull}) {
        ts::TreeSkeleton skel;
        {
            hook_on h;
            skel = ts::grow_skeleton(s.tree, seed ^ s.scenario_fnv);
        }
        const ts::SkeletonLedger L = ts::compute_skeleton_ledger(s.tree, skel);
        std::printf("G1h oak kill 0.1 seed %llu iterations %d/%d side %d guard %d retired %d exact %d\n",
                    (unsigned long long)seed, L.iterations_run, L.max_iterations, L.stall_side_branches,
                    L.stall_guard_retirements, L.attractors_stalled, L.coincident_siblings_exact);
        INFO("oak seed " << seed);
        CHECK(L.coincident_siblings_exact == 0);
        CHECK(L.iterations_run < L.max_iterations / 4);   // well under: 23-27 on the correct grower
        guard_total += L.stall_guard_retirements;
    }
    CHECK(guard_total > 0);   // the guard is what ended them
    CHECK_FALSE(ts::test_hooks::stall_side_branch_unbindable);
}

// G1i (T1 review F-B) — the retire paths run, on descriptors where kill no longer covers them:
// an attractor within half a step of a tip that no child kills, and low split thresholds.
TEST_CASE("[skeleton] G1i retire paths on perturbed descriptors", "[treegen][skeleton]") {
    struct variant { const char* name; float kill; int split; };
    for (const variant v : {variant{"oak kill 0.1", 0.1f, -1}, variant{"oak kill 0.1 split 2", 0.1f, 2},
                            variant{"oak split 2", -1.0f, 2}}) {
        ts::Scenario s = load_species("oak");
        if (v.kill > 0.0f) s.tree.branching.kill_distance = v.kill;
        if (v.split > 0) s.tree.branching.min_attractors_to_split = v.split;
        int retired = 0;
        for (const uint64_t seed : {0ull, 1ull, 2ull, 3ull}) {
            const ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed ^ s.scenario_fnv);
            const ts::SkeletonLedger L = ts::compute_skeleton_ledger(s.tree, skel);
            std::printf("G1i %s seed %llu iterations %d/%d side %d retired %d guard %d exact %d\n", v.name,
                        (unsigned long long)seed, L.iterations_run, L.max_iterations, L.stall_side_branches,
                        L.attractors_stalled, L.stall_guard_retirements, L.coincident_siblings_exact);
            INFO(v.name << " seed " << seed);
            CHECK(L.coincident_siblings_exact == 0);
            CHECK(L.iterations_run < L.max_iterations);
            CHECK(L.stall_guard_retirements == 0);   // resolved by the branch itself, not the guard
            retired += L.attractors_stalled;
        }
        INFO(v.name);
        if (v.kill > 0.0f) CHECK(retired > 0);   // the half-step retire path ran
    }
}

// Positive control for the side-branch near-pair attribution: one parent, a regular child, a
// side branch, a later regular child; each regular child within the near threshold of the side
// branch, the two regular children apart.
TEST_CASE("[ledger] side-branch near pairs on a hand-built skeleton", "[treegen][ledger]") {
    ts::TreeDescriptor desc;
    desc.branching.growth_distance = 0.3f;   // near: < 0.075 m
    ts::TreeSkeleton skel;
    skel.nodes = {
        node(-1, {0, 0, 0}, 0.2f, 0),
        node(0, {0, 0, 1}, 0.1f, 0),
        node(1, {0.05f, 0, 1.3f}, 0.05f, 0),    // regular, emitted before the side branch
        node(1, {0.0f, 0, 1.3f}, 0.05f, 1),     // the side branch
        node(1, {-0.05f, 0, 1.3f}, 0.05f, 0),   // regular, emitted after it
    };
    skel.stall_side_nodes    = {3};
    skel.stall_side_branches = 1;
    const ts::SkeletonLedger L = ts::compute_skeleton_ledger(desc, skel);
    CHECK(L.near_sibling_pairs == 2);           // 2-3 and 3-4; 2-4 are 0.1 m apart
    CHECK(L.near_pairs_side_vs_earlier == 1);
    CHECK(L.near_pairs_side_vs_later == 1);
    skel.stall_side_nodes = {};
    skel.stall_side_branches = 0;
    const ts::SkeletonLedger N = ts::compute_skeleton_ledger(desc, skel);
    CHECK(N.near_sibling_pairs == 2);
    CHECK(N.near_pairs_side_vs_earlier == 0);
    CHECK(N.near_pairs_side_vs_later == 0);
}

// G1b — no stacked geometry on the 16 shipped trees, every level.
TEST_CASE("[ledger] G1b no stacked geometry on the shipped trees", "[treegen][ledger]") {
    for (const auto& t : shipped()) {
        for (size_t l = 0; l < t.ledger.levels.size(); ++l) {
            const auto& L = t.ledger.levels[l];
            const int host_bound = tbl::k_t0_clean_max_host_vertices[t.strip ? 1 : 0];
            INFO(t.species << "_" << t.seed << " L" << l);
            CHECK(L.coincident_triangles[0] == 0);
            CHECK(L.coincident_triangles[1] == 0);
            CHECK(L.leaf_max_vertices_on_one_host <= host_bound);
            // Printed, not asserted: a two-child fork can emit its crotch triangle twice on the
            // same collar vertices (a fork-blend defect unrelated to the stall).
            std::printf("G1b %s_%llu L%zu coincident collar %d crotch %d\n", t.species.c_str(),
                        (unsigned long long)t.seed, l, L.coincident_triangles[2], L.coincident_triangles[3]);
        }
    }
}

// G1c — the fix grows rather than gives up (sums over the seed set).
TEST_CASE("[skeleton] G1c stalls resolved by growing", "[treegen][skeleton]") {
    int stalled_seeds = 0;
    for (const auto& sp : tbl::k_species) stalled_seeds += sp.k_t0_stalled_seeds;
    REQUIRE(stalled_seeds >= 9);
    for (const auto& sp : tbl::k_species) {
        long long side = 0, retired = 0, consumed = 0, total = 0;
        for (const auto& r : seed_set(sp.species)) {
            side     += r.led.stall_side_branches;
            retired  += r.led.attractors_stalled;
            consumed += r.led.attractors_consumed;
            total    += r.led.attractors_total;
        }
        std::printf("G1c %s side_branches %lld retired %lld (%.4f of attractors) consumed %lld vs t0 %lld\n",
                    sp.species, side, retired, double(retired) / double(total), consumed, sp.k_t0_consumed_sum);
        INFO("species " << sp.species);
        if (sp.k_t0_stalled_seeds > 0) CHECK(side > retired);
        CHECK(consumed >= sp.k_t0_consumed_sum);
    }
}

// G1d — clean trees byte-identical skeletons, stalled trees changed.
TEST_CASE("[ledger] G1d clean skeletons unchanged, stalled skeletons changed", "[treegen][ledger]") {
    const auto& trees = shipped();
    REQUIRE(trees.size() == tbl::k_rows.size());
    int clean = 0, stalled = 0;
    for (size_t i = 0; i < trees.size(); ++i) {
        const auto& t = trees[i];
        const auto& r = tbl::k_rows[i];
        REQUIRE(t.species == r.species);
        REQUIRE(t.seed == r.seed);
        std::printf("G1d TABLE {\"%s\", %llu, 0x%016llxull, 0x%016llxull, %d, %d},\n", r.species,
                    (unsigned long long)r.seed, (unsigned long long)r.t0_skeleton_fnv, (unsigned long long)t.fnv,
                    r.t0_coincident_siblings_exact, r.t0_iterations_run);
        INFO(t.species << "_" << t.seed);
        CHECK(t.fnv == r.skeleton_fnv);
        if (t0_clean(r)) {
            ++clean;
            CHECK(r.skeleton_fnv == r.t0_skeleton_fnv);
        } else {
            ++stalled;
            CHECK(r.skeleton_fnv != r.t0_skeleton_fnv);
            CHECK(t.fnv != r.t0_skeleton_fnv);   // the generator's value, not only the table's two constants
        }
    }
    CHECK(clean > 0);
    CHECK(stalled > 0);
}

// G1e — collision capsule unchanged ([pin]; holds by construction: scenario constants).
TEST_CASE("[ledger] G1e collision capsule pinned", "[treegen][ledger]") {
    for (const auto& r : tbl::k_rows) {
        const ts::Scenario s = load_species(r.species);
        const ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, r.seed ^ s.scenario_fnv);
        const ts::TrunkCapsule cap = ts::compute_trunk_capsule(skel, s.tree);
        const auto& k = species_row(r.species);
        INFO(r.species << "_" << r.seed);
        CHECK(cap.half_length == k.t0_capsule_half_length);
        CHECK(cap.radius == k.t0_capsule_radius);
    }
}

// G1f — not more expensive (R20): per level, summed vertices over the 16 at most main's.
TEST_CASE("[ledger] G1f shipped vertex sums at most main's", "[treegen][ledger]") {
    for (int l = 0; l < 3; ++l) {
        const long long bark = level_bark_vertex_sum(l), leaf = level_leaf_vertex_sum(l);
        std::printf("G1f L%d vertex_sum %lld (bark %lld leaf %lld) vs t0 %lld (bark %lld leaf %lld)\n", l,
                    bark + leaf, bark, leaf, tbl::k_t0_vertex_sum[size_t(l)], tbl::k_t0_bark_vertex_sum[size_t(l)],
                    tbl::k_t0_leaf_vertex_sum[size_t(l)]);
        INFO("level " << l);
        CHECK(bark + leaf <= tbl::k_t0_vertex_sum[size_t(l)]);
        // Positive control: the arm is red against the constant halved.
        CHECK_FALSE(bark + leaf <= tbl::k_t0_vertex_sum[size_t(l)] / 2);
        // [pin] the before-state for T2.
        CHECK(bark + leaf == tbl::k_t1_vertex_sum[size_t(l)]);
    }
    for (const auto& t : shipped())
        for (size_t l = 0; l < t.ledger.levels.size(); ++l)
            std::printf("G1f %s_%llu L%zu bark_t %d bark_v %d leaf_t %d leaf_v %d\n", t.species.c_str(),
                        (unsigned long long)t.seed, l, t.ledger.levels[l].bark_triangles,
                        t.ledger.levels[l].bark_vertices, t.ledger.levels[l].leaf_triangles,
                        t.ledger.levels[l].leaf_vertices);
}

// ---------------------------------------------------------------------------
// G0f — schema (asserts), with its positive control.
// ---------------------------------------------------------------------------
TEST_CASE("[ledger] G0f ledger JSON field list", "[treegen][ledger]") {
    const auto& t = shipped().front();
    const std::string doc = ts::ledger_json(t.ledger);
    const auto expected = expected_schema_keys(t.ledger.levels.size());
    CHECK(json_keys(doc) == expected);
    CHECK(t.ledger.schema == 1);
    CHECK(doc.rfind("{\"schema\": 1, ", 0) == 0);

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
