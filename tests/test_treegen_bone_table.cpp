// [treegen_bone_table] — pins C8-wind P1 analytic-rotation bone data-model:
// the bone table bake, the per-vertex bark parent-blend gradient, and the
// rigid leaf bone binding. Same link rationale as test_treegen_branch_mesh:
// tool sources (bone_table.cpp / branch_mesh.cpp / leaf_geometry.cpp / …) link
// into TestTech via rynx_tests.sharpmake.cs so this pins exactly the code
// shipped in rynx-treegen.exe.

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
#include <vector>

namespace ts = treegen;

namespace {

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
// production path main.cpp drives). Stashes the skeleton node count out-param.
std::vector<ts::LodOutput> emit_oak_lods(size_t& node_count_out) {
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    ts::Scenario s = ts::load_scenario(fixture);
    REQUIRE(s.kind == "tree");
    const uint64_t seed_effective = 42ull ^ s.scenario_fnv;
    ts::TreeSkeleton skel = ts::grow_skeleton(s.tree, seed_effective);
    REQUIRE(skel.nodes.size() > 100u);
    node_count_out = skel.nodes.size();

    auto lp_opts = ts::leaf_placement::options_from_descriptor(s.tree.leaves);
    skel.leaf_sites = ts::leaf_placement::generate_leaf_sites(skel, lp_opts, seed_effective);
    REQUIRE(skel.leaf_sites.size() > 100u);

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

    auto lods = ts::emit_all_lods(skel, skel.leaf_sites, base_opts, bark_budget,
                                  leaf_budget, leaf_geom_opts,
                                  s.tree.height_m, seed_effective);
    REQUIRE(lods.size() == 3u);
    return lods;
}

// Pack a LOD chain into a multi-mesh GLB exactly like main.cpp: bark = prim[0]
// (with _RYNX_BONE index+blend), leaves = prim[1] (rigid bone binding), and the
// per-LOD full-skeleton bone table on the MeshData.
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

TEST_CASE("[treegen_bone_table] bone table shape + rest-pose fidelity",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel = grow_oak();

    ts::BoneTable bt = ts::build_bone_table(skel);

    // Shape: one bone per node, 8 floats each.
    REQUIRE(bt.bone_count == static_cast<uint32_t>(skel.nodes.size()));
    REQUIRE(bt.records.size() == static_cast<size_t>(bt.bone_count) * 8u);

    // Rest-pose identity / table fidelity: for every bone the stored node_pos
    // is bitwise-equal to skel.nodes[i].position; for non-root bones the stored
    // pivot is bitwise-equal to the parent's position; parent index round-trips.
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        const ts::BranchNode& node = skel.nodes[i];
        const float* rec = &bt.records[i * 8u];

        const int parent_round = static_cast<int>(rec[0]);
        REQUIRE(parent_round == node.parent_index);
        REQUIRE(static_cast<int>(rec[1]) == node.depth);

        // node_pos exact.
        REQUIRE(rec[5] == node.position.x);
        REQUIRE(rec[6] == node.position.y);
        REQUIRE(rec[7] == node.position.z);

        if (node.parent_index < 0) {
            // Root: pivot == own position (does not rotate).
            REQUIRE(rec[2] == node.position.x);
            REQUIRE(rec[3] == node.position.y);
            REQUIRE(rec[4] == node.position.z);
        } else {
            const ts::vec3 ppos = skel.nodes[static_cast<size_t>(node.parent_index)].position;
            REQUIRE(rec[2] == ppos.x);
            REQUIRE(rec[3] == ppos.y);
            REQUIRE(rec[4] == ppos.z);
        }
    }
}

TEST_CASE("[treegen_bone_table] bark per-vertex bone binding + blend gradient",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel = grow_oak();
    const ts::BoneTable bt = ts::build_bone_table(skel);

    ts::BarkMeshOptions opts;
    opts.seam_offset_rad = 0.0f;
    ts::BarkMeshOutput bark = ts::build_bark_mesh(skel, opts);

    const size_t vcount = bark.mesh.positions.size() / 3;

    // Length parity: blend byte per vertex, matching the host-node-index stream.
    REQUIRE(bark.bone_blend.size() == vcount);
    REQUIRE(bark.per_vertex_node_index.size() == vcount);

    // Every host node index references a real bone.
    for (int n : bark.per_vertex_node_index) {
        REQUIRE(n >= 0);
        REQUIRE(n < static_cast<int>(bt.bone_count));
    }

    // Axial gradient present: the parent-blend takes at least two distinct
    // values across the mesh (t=0 → 255 at fork ends, t=1 → 0 at tips). All
    // values are trivially valid u8 (the type guarantees [0,255]); the real
    // assertion is that the binding is NOT a constant collapse.
    std::set<uint8_t> distinct(bark.bone_blend.begin(), bark.bone_blend.end());
    REQUIRE(distinct.size() >= 2u);
    // Sanity: both ends of the gradient are actually realized somewhere.
    REQUIRE(distinct.count(0u) == 1u);     // tip / child-end host
    REQUIRE(distinct.count(255u) == 1u);   // parent-end (ring t=0)
}

TEST_CASE("[treegen_bone_table] leaves are rigid (valid host bone, blend 0)",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel = grow_oak();
    const ts::BoneTable bt = ts::build_bone_table(skel);

    // Synthesize leaf sites at every non-root skeleton node (branch_id = node).
    // grow_skeleton does not populate leaf_sites; this exercises the same
    // build_leaf_mesh path the leaf-aware LOD emit uses.
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

    // 4 verts/leaf for SingleCard; each leaf's verts share its site's host node.
    REQUIRE(lvcount == sites.size() * 4u);
    for (size_t i = 0; i < lvcount; ++i) {
        const uint16_t host = leaves.bone_index[i];
        REQUIRE(host < bt.bone_count);
        // Block of 4 consecutive verts maps to one site (in emission order).
        const uint16_t expected = static_cast<uint16_t>(sites[i / 4].branch_id);
        REQUIRE(host == expected);
    }

    // Leaf rigidity: blend is supplied as all-zero at LOD assembly. The leaf
    // emitter itself carries no blend; the rigid all-zero vector is the
    // contract. Assert the bone_index is the only per-vertex binding the
    // emitter produces (no spurious blend stream sneaking in).
    // (bone_blend lives on LodOutput, not LeafMeshOutput — verified there.)
}

TEST_CASE("[treegen_bone_table] bone table bake is deterministic",
          "[treegen][treegen_bone_table]") {
    ts::TreeSkeleton skel = grow_oak();
    ts::BoneTable a = ts::build_bone_table(skel);
    ts::BoneTable b = ts::build_bone_table(skel);
    REQUIRE(a.bone_count == b.bone_count);
    REQUIRE(a.records == b.records);  // byte-identical float vectors

    // Bark blend stream is also deterministic across rebuilds.
    ts::BarkMeshOptions opts;
    opts.seam_offset_rad = 0.0f;
    ts::BarkMeshOutput m1 = ts::build_bark_mesh(skel, opts);
    ts::BarkMeshOutput m2 = ts::build_bark_mesh(skel, opts);
    REQUIRE(m1.bone_blend == m2.bone_blend);
    REQUIRE(m1.per_vertex_node_index == m2.per_vertex_node_index);
}

TEST_CASE("[treegen_bone_table] leaf-aware emit_all_lods carries rigid leaf binding",
          "[treegen][treegen_bone_table]") {
    size_t node_count = 0;
    std::vector<ts::LodOutput> lods = emit_oak_lods(node_count);

    bool saw_leaves = false;
    for (const auto& lod : lods) {
        // Per-LOD bone table is the full skeleton (no decimation).
        REQUIRE(lod.bone_count == static_cast<uint32_t>(node_count));
        REQUIRE(lod.bone_table_records.size() == static_cast<size_t>(lod.bone_count) * 8u);

        // Bark binding parity.
        REQUIRE(lod.bark_bone_blend.size() == lod.bark_bone_index.size());

        if (!lod.has_leaves) continue;
        saw_leaves = true;
        // Leaf-rigidity contract: blend == index in length, and every blend byte
        // is zero (leaves rotate fully with their host node, no parent blend).
        REQUIRE(lod.leaf_bone_blend.size() == lod.leaf_bone_index.size());
        REQUIRE_FALSE(lod.leaf_bone_index.empty());
        for (uint8_t b : lod.leaf_bone_blend) REQUIRE(b == 0u);
        // Host indices in range.
        for (uint16_t h : lod.leaf_bone_index) REQUIRE(h < lod.bone_count);
    }
    REQUIRE(saw_leaves);
}

TEST_CASE("[treegen_bone_table] write_glb_multi_mesh emits _RYNX_BONE JSON",
          "[treegen][treegen_bone_table]") {
    size_t node_count = 0;
    std::vector<ts::LodOutput> lods = emit_oak_lods(node_count);

    const std::string path = write_bone_lod_glb(lods, "oak");
    const std::vector<char> bytes = slurp_bytes(path);
    const std::string json = extract_json(bytes);

    // (1) _RYNX_BONE registered in extensionsUsed.
    {
        const size_t eu = find_json_key(json, "extensionsUsed");
        REQUIRE(eu != std::string::npos);
        // The array text after the key must contain "_RYNX_BONE".
        const size_t arr_end = json.find(']', eu);
        REQUIRE(arr_end != std::string::npos);
        const std::string arr = json.substr(eu, arr_end - eu);
        REQUIRE(arr.find("\"_RYNX_BONE\"") != std::string::npos);
    }

    // (2) A primitive carries extensions._RYNX_BONE with int "index" + "blend".
    //     Match the exact emitted shape: "_RYNX_BONE":{"index":<n>,"blend":<n>}.
    {
        const size_t p = json.find("\"_RYNX_BONE\":{\"index\":");
        REQUIRE(p != std::string::npos);
        // Parse the index value, then require ",\"blend\":" follows.
        size_t cur = p + std::strlen("\"_RYNX_BONE\":{\"index\":");
        REQUIRE(cur < json.size());
        REQUIRE(std::isdigit(static_cast<unsigned char>(json[cur])));
        while (cur < json.size() && std::isdigit(static_cast<unsigned char>(json[cur]))) ++cur;
        REQUIRE(json.compare(cur, std::strlen(",\"blend\":"), ",\"blend\":") == 0);
        cur += std::strlen(",\"blend\":");
        REQUIRE(std::isdigit(static_cast<unsigned char>(json[cur])));
    }

    // (3) Mesh-level _RYNX_BONE "count" equals the skeleton node count.
    {
        const size_t b = json.find("\"_RYNX_BONE\":{\"bones\":");
        REQUIRE(b != std::string::npos);
        const size_t c = json.find("\"count\":", b);
        REQUIRE(c != std::string::npos);
        const size_t v = c + std::strlen("\"count\":");
        const long parsed = std::strtol(json.c_str() + v, nullptr, 10);
        REQUIRE(parsed == static_cast<long>(node_count));
    }

    // (4) The blend accessor is componentType 5121 with normalized:true. We
    //     locate the blend accessor index from the primitive extension, then
    //     read that accessor's JSON object.
    {
        const size_t p = json.find("\"_RYNX_BONE\":{\"index\":");
        REQUIRE(p != std::string::npos);
        const size_t bk = json.find("\"blend\":", p);
        REQUIRE(bk != std::string::npos);
        const size_t bv = bk + std::strlen("\"blend\":");
        const long blend_acc = std::strtol(json.c_str() + bv, nullptr, 10);
        REQUIRE(blend_acc >= 0);

        // Walk the accessors array to the blend_acc-th element. Accessors are
        // emitted in order, each as a {...} object; count opening braces inside
        // the "accessors":[ ... ] array.
        const size_t accs = find_json_key(json, "accessors");
        REQUIRE(accs != std::string::npos);
        size_t scan = json.find('[', accs);
        REQUIRE(scan != std::string::npos);
        ++scan;
        long idx = 0;
        size_t obj_start = std::string::npos;
        int depth = 0;
        for (size_t k = scan; k < json.size(); ++k) {
            const char ch = json[k];
            if (ch == '{') {
                if (depth == 0 && idx == blend_acc) obj_start = k;
                ++depth;
            } else if (ch == '}') {
                --depth;
                if (depth == 0) {
                    if (idx == blend_acc) {
                        const std::string obj = json.substr(obj_start, k - obj_start + 1);
                        REQUIRE(obj.find("\"componentType\":5121") != std::string::npos);
                        REQUIRE(obj.find("\"normalized\":true") != std::string::npos);
                        break;
                    }
                    ++idx;
                }
            } else if (ch == ']' && depth == 0) {
                break;  // end of accessors array
            }
        }
        REQUIRE(obj_start != std::string::npos);
    }

    std::filesystem::remove(path);
}
