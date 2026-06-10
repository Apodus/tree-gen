// [treegen_bone_table] — pins the DECIMATED analytic-rotation wind rig:
// kept-set construction (root + forks-within-cap + arc subdivision), the
// 12-float bone records (parent / compliance aggregation / pivot / node_pos),
// the original-node remap (run blend + rigid riders), compliance conservation
// (total root-to-tip bend angle preserved exactly under runtime knobs), leaf
// pivot pinning, and the _RYNX_BONE GLB emission. Tool sources link into
// TestTech via rynx_tests.sharpmake.cs so this pins exactly the code shipped
// in rynx-treegen.exe.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../bone_table.hpp"
#include "../branch_mesh.hpp"
#include "../face_budget.hpp"
#include "../glb_writer.hpp"
#include "../leaf_budget.hpp"
#include "../leaf_geometry.hpp"
#include "../leaf_placement.hpp"
#include "../lod_emitter.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace ts = treegen;

namespace {

constexpr int kStride = ts::K_FLOATS_PER_BONE;

ts::TreeSkeleton grow_oak() {
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    ts::Scenario s = ts::load_scenario(fixture);
    REQUIRE(s.kind == "tree");
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
    REQUIRE(skel.nodes.size() > 100u);
    return skel;
}

// Synthetic skeleton with hand-computable decimation:
//   root(0) — trunk chain 1..10 (0.5 m steps, depth 0, 5.0 m total)
//   node 10 forks:
//     branch A: 11..16 (+x, 0.5 m, depth 1) + tail node 21 off 16 (0.5 m)
//     branch B: 17..20 (-x, 0.5 m, depth 1) — 2.0 m, below subdivision
//   node 1 forks (0.5 m from root — under min_fork_spacing → DROPPED):
//     side branch: 22, 23 (+x at z=0.5, depth 1) — rides the trunk run.
ts::TreeSkeleton synthetic_skel() {
    ts::TreeSkeleton s;
    auto add = [&](int parent, ts::vec3 pos, int depth) {
        ts::BranchNode n;
        n.parent_index = parent;
        n.position     = pos;
        n.depth        = depth;
        s.nodes.push_back(n);
        return static_cast<int>(s.nodes.size()) - 1;
    };
    add(-1, {0, 0, 0}, 0);                                       // 0 root
    for (int i = 1; i <= 10; ++i) add(i - 1, {0, 0, 0.5f * i}, 0);  // 1..10
    int prev = 10;
    for (int i = 0; i < 6; ++i) prev = add(prev, {0.5f * (i + 1), 0, 5.0f}, 1);  // 11..16
    int b = 10;
    for (int i = 0; i < 4; ++i) b = add(b, {-0.5f * (i + 1), 0, 5.0f}, 1);       // 17..20
    add(16, {3.5f, 0, 5.0f}, 1);                                 // 21 tail
    int sb = add(1, {0.5f, 0, 0.5f}, 1);                         // 22 (dropped-fork side)
    add(sb, {1.0f, 0, 0.5f}, 1);                                 // 23
    return s;
}

// Decimated-chain compliance sums for the bone hosting `kept_node`.
void decimated_compliance(const ts::BoneTable& bt, int bone,
                          long& n_total, long& depth_total) {
    n_total = 0;
    depth_total = 0;
    int cur = bone;
    int guard = 0;
    while (cur >= 0) {
        const float* rec = &bt.records[static_cast<size_t>(cur) * kStride];
        n_total     += static_cast<long>(rec[1]);
        depth_total += static_cast<long>(rec[2]);
        cur = static_cast<int>(rec[0]);
        REQUIRE(++guard <= 64);
    }
}

// Original-chain compliance for node e: every ancestor-or-self with a parent
// (the root does not rotate in the 1:1 rig either).
void original_compliance(const ts::TreeSkeleton& skel, int e,
                         long& n_total, long& depth_total) {
    n_total = 0;
    depth_total = 0;
    for (int j = e; j >= 0; j = skel.nodes[static_cast<size_t>(j)].parent_index) {
        if (skel.nodes[static_cast<size_t>(j)].parent_index < 0) break;  // root
        ++n_total;
        depth_total += skel.nodes[static_cast<size_t>(j)].depth;
    }
}

// --- GLB JSON helpers (local copies; mirror tests/test_treegen_glb_textures.cpp;
// anon-namespace so no link clash with that TU). ---

std::vector<char> slurp_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    f.seekg(0, std::ios::end);
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(n));
    if (n > 0) f.read(buf.data(), n);
    return buf;
}

std::string tmp_path(const char* suffix) {
    namespace fs = std::filesystem;
    char buf[80];
    std::snprintf(buf, sizeof(buf), "treegen_bone_%d_%s.glb", std::rand(), suffix);
    return (fs::temp_directory_path() / buf).string();
}

size_t find_json_key(const std::string& json, const char* key) {
    std::string needle = std::string("\"") + key + "\":";
    return json.find(needle);
}

std::string extract_json(const std::vector<char>& glb) {
    REQUIRE(glb.size() >= 20);
    uint32_t json_len = 0;
    std::memcpy(&json_len, glb.data() + 12, 4);
    uint32_t chunk_type = 0;
    std::memcpy(&chunk_type, glb.data() + 16, 4);
    REQUIRE(chunk_type == 0x4E4F534Au); // "JSON"
    return std::string(glb.data() + 20, json_len);
}

// Grow oak + full C5 P1 leaf sites + run leaf-aware emit_all_lods (the exact
// production path main.cpp drives). Stashes the grown skeleton out-param.
std::vector<ts::LodOutput> emit_oak_lods(ts::TreeSkeleton& skel_out) {
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    ts::Scenario s = ts::load_scenario(fixture);
    REQUIRE(s.kind == "tree");
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    skel_out = ts::grow_skeleton(s.tree, seed_effective);
    REQUIRE(skel_out.nodes.size() > 100u);

    auto lp_opts = ts::leaf_placement::options_from_descriptor(s.tree.leaves);
    skel_out.leaf_sites = ts::leaf_placement::generate_leaf_sites(skel_out, lp_opts, seed_effective);
    REQUIRE(skel_out.leaf_sites.size() > 100u);

    ts::BarkMeshOptions base_opts;
    base_opts.tree_height_m   = s.tree.height_m;
    base_opts.seam_offset_rad = 0.0f;

    ts::LodBudget  bark_budget;
    ts::LeafBudget leaf_budget;

    ts::LeafMeshOptions leaf_geom_opts;
    leaf_geom_opts.geometry_type         = s.tree.leaves.geometry_type;
    leaf_geom_opts.shape                 = s.tree.leaves.shape;
    leaf_geom_opts.leaf_size_m           = s.tree.leaves.leaf_size_m;
    leaf_geom_opts.cluster_count_per_tip = s.tree.leaves.cluster_count_per_tip;
    leaf_geom_opts.bend_half_angle       = s.tree.leaves.leaf_bend_half_angle;

    auto lods = ts::emit_all_lods(skel_out, skel_out.leaf_sites, base_opts, bark_budget,
                                  leaf_budget, leaf_geom_opts,
                                  s.tree.height_m, seed_effective);
    REQUIRE(lods.size() == 3u);
    return lods;
}

// Pack a LOD chain into a multi-mesh GLB exactly like main.cpp: bark = prim[0]
// (_RYNX_BONE index+blend), leaves = prim[1] (index+blend+pivot), and the
// per-LOD decimated bone table on the MeshData.
std::string write_bone_lod_glb(const std::vector<ts::LodOutput>& lods, const char* suffix) {
    std::vector<std::vector<ts::PrimitiveData>> per_lod_prims(lods.size());
    std::vector<ts::MeshData>                    meshes;
    meshes.reserve(lods.size());

    for (size_t li = 0; li < lods.size(); ++li) {
        const auto& lod = lods[li];
        ts::PrimitiveData bark_prim{};
        bark_prim.positions   = std::span<const float>(lod.mesh.positions.data(), lod.mesh.positions.size());
        bark_prim.normals     = std::span<const float>(lod.mesh.normals.data(),   lod.mesh.normals.size());
        bark_prim.uvs         = std::span<const float>(lod.mesh.uvs.data(),       lod.mesh.uvs.size());
        bark_prim.indices_u32 = std::span<const uint32_t>(lod.indices_u32.data(), lod.indices_u32.size());
        bark_prim.wind_weights_packed = std::span<const uint8_t>(
            lod.wind_weights_packed.data(), lod.wind_weights_packed.size());
        bark_prim.bone_index = std::span<const uint16_t>(
            lod.bark_bone_index.data(), lod.bark_bone_index.size());
        bark_prim.bone_blend = std::span<const uint8_t>(
            lod.bark_bone_blend.data(), lod.bark_bone_blend.size());
        per_lod_prims[li].push_back(bark_prim);

        if (lod.has_leaves) {
            ts::PrimitiveData leaf_prim{};
            leaf_prim.positions   = std::span<const float>(
                lod.leaf_positions.data(), lod.leaf_positions.size());
            leaf_prim.normals     = std::span<const float>(
                lod.leaf_normals.data(), lod.leaf_normals.size());
            leaf_prim.uvs         = std::span<const float>(
                lod.leaf_uvs.data(), lod.leaf_uvs.size());
            leaf_prim.indices_u32 = std::span<const uint32_t>(
                lod.leaf_indices_u32.data(), lod.leaf_indices_u32.size());
            leaf_prim.wind_weights_packed = std::span<const uint8_t>(
                lod.leaf_wind_weights_packed.data(), lod.leaf_wind_weights_packed.size());
            leaf_prim.bone_index = std::span<const uint16_t>(
                lod.leaf_bone_index.data(), lod.leaf_bone_index.size());
            leaf_prim.bone_blend = std::span<const uint8_t>(
                lod.leaf_bone_blend.data(), lod.leaf_bone_blend.size());
            leaf_prim.bone_pivot = std::span<const float>(
                lod.leaf_pivots.data(), lod.leaf_pivots.size());
            leaf_prim.material.alpha_mode   = "MASK";
            leaf_prim.material.alpha_cutoff = 0.5f;
            per_lod_prims[li].push_back(leaf_prim);
        }

        ts::MeshData md{};
        md.primitives            = std::span<const ts::PrimitiveData>(
            per_lod_prims[li].data(), per_lod_prims[li].size());
        md.lod_index             = lod.lod_index;
        md.lod_max_distance_m    = lod.lod_max_distance_m;
        md.lod_screen_height_px  = lod.lod_screen_height_px;
        md.bone_table            = std::span<const float>(
            lod.bone_table_records.data(), lod.bone_table_records.size());
        md.bone_count            = lod.bone_count;
        meshes.push_back(md);
    }

    const std::string path = tmp_path(suffix);
    std::string err;
    REQUIRE(ts::write_glb_multi_mesh(
        std::span<const ts::MeshData>(meshes.data(), meshes.size()), path, &err));
    REQUIRE(std::filesystem::exists(path));
    return path;
}

}  // namespace

TEST_CASE("[treegen_bone_table] decimated rig: exact structure on synthetic skeleton",
          "[treegen][treegen_bone_table]") {
    const ts::TreeSkeleton skel = synthetic_skel();
    ts::BoneDecimateParams params;  // order_cap 3, max_segment_m 2.0
    const ts::BoneTable bt = ts::build_bone_table(skel, params);

    // Kept set: root 0; trunk subdivision 4 + 7 (5.0 m run, 2 joints);
    // fork 10 (5.0 m >= min_fork_spacing); branch-A subdivision 14 (3.5 m
    // run). The fork at node 1 (0.5 m < min_fork_spacing) is DROPPED — its
    // side branch (22,23) rides the trunk run. Branch B (2.0 m) and the
    // tails (15,16,21) stay rigid.
    REQUIRE(bt.bone_count == 5u);
    REQUIRE(bt.records.size() == 5u * kStride);

    auto rec = [&](int b) { return &bt.records[static_cast<size_t>(b) * kStride]; };

    // Root bone.
    REQUIRE(rec(0)[0] == -1.0f);
    REQUIRE(rec(0)[1] == 0.0f);
    REQUIRE(rec(0)[2] == 0.0f);

    // bone 1 = node 4: run 1..4 under root. n=4, sum_depth=0, pivot=root pos.
    REQUIRE(rec(1)[0] == 0.0f);
    REQUIRE(rec(1)[1] == 4.0f);
    REQUIRE(rec(1)[2] == 0.0f);
    REQUIRE(rec(1)[4] == 0.0f);  // pivot = root (0,0,0)
    REQUIRE(rec(1)[6] == 0.0f);
    REQUIRE(rec(1)[10] == 2.0f); // node 4 z

    // bone 2 = node 7: nodes 5..7. bone 3 = node 10: nodes 8..10.
    REQUIRE(rec(2)[0] == 1.0f);
    REQUIRE(rec(2)[1] == 3.0f);
    REQUIRE(rec(3)[0] == 2.0f);
    REQUIRE(rec(3)[1] == 3.0f);
    REQUIRE(rec(3)[10] == 5.0f); // node 10 z

    // bone 4 = node 14: branch-A run 11..14, depth 1 each.
    REQUIRE(rec(4)[0] == 3.0f);
    REQUIRE(rec(4)[1] == 4.0f);
    REQUIRE(rec(4)[2] == 4.0f);
    REQUIRE(rec(4)[4] == 0.0f);  // pivot = node 10 (0,0,5)
    REQUIRE(rec(4)[6] == 5.0f);
    REQUIRE(rec(4)[8] == 2.0f);  // node 14 x

    // Remap: trunk interior node 2 -> bone 1 at t = 1.0/2.0.
    REQUIRE(bt.node_to_bone[2] == 1u);
    REQUIRE(bt.node_in_run[2] == 1u);
    REQUIRE(bt.node_run_t[2] == Approx(0.5f));
    // Run-start vertex blends to pure parent: t_seg=0 on node 1's segment.
    REQUIRE(bt.vertex_parent_blend(1, 0.0f) == 255u);
    // Kept node at its own ring: pure host.
    REQUIRE(bt.vertex_parent_blend(4, 1.0f) == 0u);
    // Rigid riders: branch B + branch-A tail ride their kept ancestor whole
    // (parent is a kept joint → constant blend 0 across the whole rider).
    for (int n : {17, 18, 19, 20}) {
        REQUIRE(bt.node_to_bone[static_cast<size_t>(n)] == 3u);
        REQUIRE(bt.vertex_parent_blend(n, 0.0f) == 0u);
        REQUIRE(bt.vertex_parent_blend(n, 1.0f) == 0u);
    }
    for (int n : {15, 16, 21}) {
        REQUIRE(bt.node_to_bone[static_cast<size_t>(n)] == 4u);
        REQUIRE(bt.vertex_parent_blend(n, 0.5f) == 0u);
    }
    // Dropped-fork side branch (22,23): rides the trunk run at its
    // attachment point — bone 1 (node 4's run) at node 1's CONSTANT blend
    // (t = 0.25 → blend = 191). No kink at the attach, no over-bend.
    for (int n : {22, 23}) {
        REQUIRE(bt.node_to_bone[static_cast<size_t>(n)] == 1u);
        REQUIRE(bt.node_run_t[static_cast<size_t>(n)] == Approx(0.25f));
        REQUIRE(bt.vertex_parent_blend(n, 0.0f) == 191u);
        REQUIRE(bt.vertex_parent_blend(n, 1.0f) == 191u);
    }

    // Compliance conservation at every kept node: decimated chain sums equal
    // the original chain exactly (integers, so equality is exact).
    for (int e : {4, 7, 10, 14}) {
        long dn = 0, dd = 0, on = 0, od = 0;
        decimated_compliance(bt, static_cast<int>(bt.node_to_bone[static_cast<size_t>(e)]), dn, dd);
        original_compliance(skel, e, on, od);
        REQUIRE(dn == on);
        REQUIRE(dd == od);
    }
}

TEST_CASE("[treegen_bone_table] decimated rig on oak: shape, conservation, chain cap",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel = grow_oak();
    const ts::BoneTable bt = ts::build_bone_table(skel);

    // The whole point: tens of bones, not thousands.
    REQUIRE(bt.bone_count >= 8u);
    REQUIRE(bt.bone_count <= 300u);
    REQUIRE(bt.bone_count < skel.nodes.size() / 4);
    REQUIRE(bt.records.size() == static_cast<size_t>(bt.bone_count) * kStride);
    WARN("oak decimated bone count: " << bt.bone_count
         << " (from " << skel.nodes.size() << " skeleton nodes)");

    // Remap arrays cover every node; every binding is a valid bone.
    REQUIRE(bt.node_to_bone.size() == skel.nodes.size());
    REQUIRE(bt.node_run_t.size() == skel.nodes.size());
    REQUIRE(bt.node_in_run.size() == skel.nodes.size());
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        REQUIRE(bt.node_to_bone[i] < bt.bone_count);
        REQUIRE(bt.node_run_t[i] > 0.0f);
        REQUIRE(bt.node_run_t[i] <= 1.0f);
    }

    // Structure: DAG order, pivot = parent bone's node_pos (bitwise), and the
    // FK ancestor chain stays under the shader cap (16) with margin.
    std::vector<int> chain_depth(bt.bone_count, 0);
    int max_chain = 0;
    for (uint32_t b = 0; b < bt.bone_count; ++b) {
        const float* rec = &bt.records[static_cast<size_t>(b) * kStride];
        const int parent = static_cast<int>(rec[0]);
        if (b == 0) {
            REQUIRE(parent == -1);
            continue;
        }
        REQUIRE(parent >= 0);
        REQUIRE(parent < static_cast<int>(b));
        const float* prec = &bt.records[static_cast<size_t>(parent) * kStride];
        REQUIRE(rec[4] == prec[8]);
        REQUIRE(rec[5] == prec[9]);
        REQUIRE(rec[6] == prec[10]);
        chain_depth[b] = chain_depth[static_cast<size_t>(parent)] + 1;
        if (chain_depth[b] > max_chain) max_chain = chain_depth[b];
    }
    // Strict margin under the shader's MAX_BONE_DEPTH (16): the cap must be
    // a degenerate-data guard, never load-bearing. min_fork_spacing_m is the
    // knob that enforces this; 12 leaves room for taller species.
    REQUIRE(max_chain + 1 <= 12);
    WARN("oak max FK chain length: " << (max_chain + 1));

    // Compliance conservation for EVERY kept bone (records hold its node via
    // node_pos; recover kept nodes through the remap: node hosts itself iff
    // node_run_t == 1 and in_run... root excepted — instead walk all nodes
    // and check the ones whose bone's node_pos matches their own position).
    long checked = 0;
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        const uint16_t b = bt.node_to_bone[i];
        const float* rec = &bt.records[static_cast<size_t>(b) * kStride];
        const bool is_kept_self = (rec[8] == skel.nodes[i].position.x &&
                                   rec[9] == skel.nodes[i].position.y &&
                                   rec[10] == skel.nodes[i].position.z);
        if (!is_kept_self) continue;
        long dn = 0, dd = 0, on = 0, od = 0;
        decimated_compliance(bt, b, dn, dd);
        original_compliance(skel, static_cast<int>(i), on, od);
        REQUIRE(dn == on);
        REQUIRE(dd == od);
        ++checked;
    }
    REQUIRE(checked >= static_cast<long>(bt.bone_count) - 1);
}

TEST_CASE("[treegen_bone_table] bark blend gradient survives the remap",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel;
    std::vector<ts::LodOutput> lods = emit_oak_lods(skel);
    const auto& lod = lods[0];

    REQUIRE(lod.bark_bone_blend.size() == lod.bark_bone_index.size());
    REQUIRE_FALSE(lod.bark_bone_index.empty());
    for (uint16_t h : lod.bark_bone_index) REQUIRE(h < lod.bone_count);

    // The run blend must still be a gradient (smooth curvature over collapsed
    // runs), realizing both extremes: pure-parent rings at run anchors and
    // pure-host rings at kept joints / rigid riders.
    std::set<uint8_t> distinct(lod.bark_bone_blend.begin(), lod.bark_bone_blend.end());
    REQUIRE(distinct.size() >= 8u);
    REQUIRE(distinct.count(0u) == 1u);
    REQUIRE(distinct.count(255u) == 1u);
}

TEST_CASE("[treegen_bone_table] leaf emitter hosts original nodes (pre-remap)",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel = grow_oak();

    // Synthesize leaf sites at every non-root skeleton node (branch_id = node).
    std::vector<ts::LeafSite> sites;
    sites.reserve(skel.nodes.size());
    for (size_t i = 1; i < skel.nodes.size(); ++i) {
        ts::LeafSite s;
        s.position = skel.nodes[i].position;
        s.normal   = skel.nodes[i].axis;
        s.branch_id = static_cast<int>(i);
        s.branch_depth_fraction = 0.5f;
        sites.push_back(s);
    }
    REQUIRE_FALSE(sites.empty());

    ts::LeafMeshOptions lopts;
    lopts.geometry_type = ts::LeafGeometryType::SingleCard;
    ts::LeafMeshOutput leaves = ts::build_leaf_mesh(sites, lopts);

    const size_t lvcount = leaves.positions.size() / 3;
    REQUIRE(lvcount > 0u);
    REQUIRE(leaves.bone_index.size() == lvcount);

    // 4 verts/leaf for SingleCard; the emitter binds ORIGINAL node ids — the
    // decimated-rig rebind happens in lod_emitter (bind_leaves_to_rig).
    REQUIRE(lvcount == sites.size() * 4u);
    for (size_t i = 0; i < lvcount; ++i) {
        const uint16_t host = leaves.bone_index[i];
        REQUIRE(host < skel.nodes.size());
        const uint16_t expected = static_cast<uint16_t>(sites[i / 4].branch_id);
        REQUIRE(host == expected);
    }
}

TEST_CASE("[treegen_bone_table] leaf-aware emit: bones valid, pivots pinned to twigs",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel;
    std::vector<ts::LodOutput> lods = emit_oak_lods(skel);

    // Bit-exact node-position set for pivot membership checks.
    auto key = [](float x, float y, float z) {
        uint32_t bx, by, bz;
        std::memcpy(&bx, &x, 4);
        std::memcpy(&by, &y, 4);
        std::memcpy(&bz, &z, 4);
        return (uint64_t(bx) << 32) ^ (uint64_t(by) << 13) ^ uint64_t(bz);
    };
    std::unordered_set<uint64_t> node_pos_keys;
    node_pos_keys.reserve(skel.nodes.size() * 2);
    for (const auto& n : skel.nodes) {
        node_pos_keys.insert(key(n.position.x, n.position.y, n.position.z));
    }

    const ts::BoneTable bt = ts::build_bone_table(skel);

    bool saw_leaves = false;
    bool pivot_differs_from_bone = false;
    for (const auto& lod : lods) {
        // Per-LOD bone table is the decimated rig, identical across LODs.
        REQUIRE(lod.bone_count == bt.bone_count);
        REQUIRE(lod.bone_table_records == bt.records);

        if (!lod.has_leaves) continue;
        saw_leaves = true;
        const size_t n = lod.leaf_bone_index.size();
        REQUIRE(lod.leaf_bone_blend.size() == n);
        REQUIRE(lod.leaf_pivots.size() == n * 3);
        REQUIRE(n > 0u);
        for (size_t v = 0; v < n; ++v) {
            const uint16_t h = lod.leaf_bone_index[v];
            REQUIRE(h < lod.bone_count);
            // Pivot must be bit-exactly an ORIGINAL skeleton node position —
            // the twig attachment, not the surviving joint.
            const float px = lod.leaf_pivots[v * 3 + 0];
            const float py = lod.leaf_pivots[v * 3 + 1];
            const float pz = lod.leaf_pivots[v * 3 + 2];
            REQUIRE(node_pos_keys.count(key(px, py, pz)) == 1u);
            const float* rec = &bt.records[static_cast<size_t>(h) * kStride];
            if (px != rec[8] || py != rec[9] || pz != rec[10]) {
                pivot_differs_from_bone = true;
            }
        }
    }
    REQUIRE(saw_leaves);
    // Decimation must have actually moved at least one host away from its
    // attachment — otherwise the pivot stream is dead weight.
    REQUIRE(pivot_differs_from_bone);
}

TEST_CASE("[treegen_bone_table] decimated bake is deterministic",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel = grow_oak();
    ts::BoneTable a = ts::build_bone_table(skel);
    ts::BoneTable b = ts::build_bone_table(skel);
    REQUIRE(a.bone_count == b.bone_count);
    REQUIRE(a.records == b.records);  // byte-identical float vectors
    REQUIRE(a.node_to_bone == b.node_to_bone);
    REQUIRE(a.node_run_t == b.node_run_t);
    REQUIRE(a.node_in_run == b.node_in_run);
}

TEST_CASE("[treegen_bone_table] write_glb_multi_mesh emits decimated _RYNX_BONE JSON",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel;
    std::vector<ts::LodOutput> lods = emit_oak_lods(skel);

    const std::string path = write_bone_lod_glb(lods, "oak");
    const std::vector<char> bytes = slurp_bytes(path);
    const std::string json = extract_json(bytes);

    // (1) _RYNX_BONE registered in extensionsUsed.
    {
        const size_t eu = find_json_key(json, "extensionsUsed");
        REQUIRE(eu != std::string::npos);
        const size_t arr_end = json.find(']', eu);
        REQUIRE(arr_end != std::string::npos);
        const std::string arr = json.substr(eu, arr_end - eu);
        REQUIRE(arr.find("\"_RYNX_BONE\"") != std::string::npos);
    }

    // (2) A primitive carries extensions._RYNX_BONE with int "index" + "blend".
    {
        const size_t p = json.find("\"_RYNX_BONE\":{\"index\":");
        REQUIRE(p != std::string::npos);
        size_t cur = p + std::strlen("\"_RYNX_BONE\":{\"index\":");
        REQUIRE(cur < json.size());
        REQUIRE(std::isdigit(static_cast<unsigned char>(json[cur])));
        while (cur < json.size() && std::isdigit(static_cast<unsigned char>(json[cur]))) ++cur;
        REQUIRE(json.compare(cur, std::strlen(",\"blend\":"), ",\"blend\":") == 0);
        cur += std::strlen(",\"blend\":");
        REQUIRE(std::isdigit(static_cast<unsigned char>(json[cur])));
    }

    // (3) The leaf primitive additionally carries the tumble pivot stream.
    REQUIRE(json.find(",\"pivot\":") != std::string::npos);

    // (4) Mesh-level _RYNX_BONE: count == DECIMATED bone count, and the
    //     record stride is declared so the loader can validate.
    {
        const size_t b = json.find("\"_RYNX_BONE\":{\"bones\":");
        REQUIRE(b != std::string::npos);
        const size_t c = json.find("\"count\":", b);
        REQUIRE(c != std::string::npos);
        const size_t v = c + std::strlen("\"count\":");
        const long parsed = std::strtol(json.c_str() + v, nullptr, 10);
        REQUIRE(parsed == static_cast<long>(lods[0].bone_count));
        REQUIRE(parsed < static_cast<long>(skel.nodes.size()) / 4);

        const size_t fpb = json.find("\"floats_per_bone\":", b);
        REQUIRE(fpb != std::string::npos);
        const long stride = std::strtol(json.c_str() + fpb + std::strlen("\"floats_per_bone\":"),
                                        nullptr, 10);
        REQUIRE(stride == kStride);
    }

    std::filesystem::remove(path);
}
