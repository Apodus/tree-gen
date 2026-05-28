// NOTE: this exe must stay /fp:precise — byte-hash determinism depends on it.
// See rynx/generate/treegen.sharpmake.cs.

#include "bark_texture.hpp"
#include "branch_mesh.hpp"
#include "collision_fit.hpp"
#include "det_rng.hpp"
#include "envelopes.hpp"
#include "face_budget.hpp"
#include "glb_writer.hpp"
#include "impostor_bake.hpp"
#include "leaf_atlas.hpp"
#include "leaf_budget.hpp"
#include "leaf_geometry.hpp"
#include "leaf_placement.hpp"
#include "lod_emitter.hpp"
#include "png_encoder.hpp"
#include "scenario.hpp"
#include "skeleton.hpp"
#include "skeleton_json.hpp"
#include "space_colonization.hpp"
#include "species_registry.hpp"
#include "tree_descriptor.hpp"
#include "trivial_cylinder.hpp"
#include "vec3.hpp"

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Single source of truth for shape names enumerated in --list-envelope-shapes.
// EnvelopeShape <-> string conversion lives in tree_descriptor.cpp; this is
// the enumeration of the 5 values, deliberately sorted alphabetically for
// stable script-parseable output.
constexpr std::array<treegen::EnvelopeShape, 5> k_all_envelope_shapes = {
    treegen::EnvelopeShape::Conical,
    treegen::EnvelopeShape::Fan,
    treegen::EnvelopeShape::Fastigiate,
    treegen::EnvelopeShape::OblateSpheroid,
    treegen::EnvelopeShape::Weeping,
};

void print_usage() {
    std::fprintf(stderr,
        "usage: rynx-treegen --scenario=<path> [--seed=<u64>] [--out=<path>] [--dump-skeleton-json=<path>] [--bake-impostor]\n"
        "       rynx-treegen --species=<name> [--seed=<u64>] [--out=<path>] [--dump-skeleton-json=<path>] [--bake-impostor]\n"
        "       rynx-treegen --list-species\n"
        "       rynx-treegen --list-envelope-shapes\n"
        "       rynx-treegen --help\n");
}

// Enumerate scenarios/*.json next to the exe and (best-effort) the repo source
// tree. Best-effort — failure is silently swallowed; --help still prints the
// flag list.
std::vector<std::filesystem::path> find_scenario_files() {
    std::vector<std::filesystem::path> hits;
    namespace fs = std::filesystem;

    auto scan = [&](const fs::path& dir) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) return;
        for (const auto& e : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!e.is_regular_file()) continue;
            if (e.path().extension() == ".json") hits.push_back(e.path());
        }
    };

    // 1) next to the running exe (./scenarios/), 2) source-tree fallback for
    // when the exe is invoked from a build output dir.
    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (!ec) {
        scan(cwd / "scenarios");
        scan(cwd / "tools" / "rynx-treegen" / "scenarios");
    }

    std::sort(hits.begin(), hits.end());
    hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
    return hits;
}

void print_help() {
    std::fprintf(stderr,
        "rynx-treegen — procedural tree skeleton generator.\n\n"
        "flags:\n"
        "  --scenario=<path>             Scenario JSON (required unless --species).\n"
        "  --species=<name>              Species preset (resolves to c3_<name>.json).\n"
        "  --seed=<u64>                  CLI seed XOR'd with scenario_fnv (default 0).\n"
        "  --out=<path>                  Mesh output (.glb).\n"
        "  --dump-skeleton-json=<path>   Write TreeSkeleton JSON dump (tree scenarios).\n"
        "  --list-species                Print available species names + exit.\n"
        "  --list-envelope-shapes        Print the 5 envelope shape names + exit.\n"
        "  --bake-impostor               Bake 8-azimuth impostor atlas into L3 GLB image.\n"
        "  -h, --help                    Print this message + scenario index + exit.\n"
        "\n"
        "Both --key=value and --key value forms accepted.\n"
        "--species and --scenario are mutually exclusive (exit 2 if both given).\n");

    const auto scenarios = find_scenario_files();
    if (scenarios.empty()) {
        std::fprintf(stderr, "\nno scenarios/*.json found near cwd.\n");
    } else {
        std::fprintf(stderr, "\navailable scenarios (%zu):\n", scenarios.size());
        for (const auto& p : scenarios) {
            std::fprintf(stderr, "  %s\n", p.string().c_str());
        }
    }
}

// Accept either `--key=value` or `--key value`. `--key` matches keys with
// matching trailing nul (no false-positive matches like --seeds vs --seed).
// Returns nullptr if not matched. Advances `argi` past the value form when
// the space-separated form is consumed.
const char* match_arg(const char* arg, const char* key, int argc, char** argv, int& argi) {
    size_t key_len = std::strlen(key);
    if (std::strncmp(arg, key, key_len) != 0) return nullptr;
    if (arg[key_len] == '=') return arg + key_len + 1;
    if (arg[key_len] == '\0') {
        if (argi + 1 >= argc) return nullptr;
        ++argi;
        return argv[argi];
    }
    return nullptr;
}

uint64_t parse_u64(const char* s) {
    return static_cast<uint64_t>(std::strtoull(s, nullptr, 10));
}

float vlen(treegen::vec3 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

struct SkeletonStats {
    size_t nodes              = 0;
    size_t leaf_tips          = 0;   // nodes with no child
    int    max_depth          = 0;
    float  total_branch_len_m = 0.0f;
    int    attractors_used    = 0;
    int    attractors_total   = 0;
};

SkeletonStats compute_stats(const treegen::TreeSkeleton& skel, int attractors_total) {
    SkeletonStats s;
    s.nodes              = skel.nodes.size();
    s.attractors_used    = skel.attractors_consumed;
    s.attractors_total   = attractors_total;

    // Count children per node; a node with 0 children is a leaf tip. Depth +
    // segment length from parent are O(N) over the dense node array.
    std::vector<int> child_count(skel.nodes.size(), 0);
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        const auto& n = skel.nodes[i];
        if (n.depth > s.max_depth) s.max_depth = n.depth;
        if (n.parent_index >= 0) {
            const auto& parent = skel.nodes[static_cast<size_t>(n.parent_index)];
            s.total_branch_len_m += vlen(n.position - parent.position);
            ++child_count[static_cast<size_t>(n.parent_index)];
        }
    }
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        if (child_count[i] == 0) ++s.leaf_tips;
    }
    return s;
}

} // anonymous namespace

int main(int argc, char** argv) {
    try {
        std::string scenario_path;
        std::string species_name;
        std::string out_path;
        std::string skeleton_json_path;
        uint64_t    cli_seed = 0;
        bool        bake_impostor = false;

        // --dump-skeleton-json MUST be matched before --dump (prefix-wise the
        // longer key first), and before --scenario (no prefix conflict but
        // order is the seal).
        for (int i = 1; i < argc; ++i) {
            const char* arg = argv[i];
            if (std::strcmp(arg, "--list-envelope-shapes") == 0) {
                for (const auto s : k_all_envelope_shapes) {
                    std::printf("%s\n", treegen::envelope_shape_to_string(s));
                }
                return 0;
            }
            if (std::strcmp(arg, "--list-species") == 0) {
                for (const auto& name : treegen::get_species_names()) {
                    std::printf("%s\n", name.c_str());
                }
                return 0;
            }
            if (const char* v = match_arg(arg, "--dump-skeleton-json", argc, argv, i)) { skeleton_json_path = v; continue; }
            if (const char* v = match_arg(arg, "--scenario", argc, argv, i)) { scenario_path = v; continue; }
            if (const char* v = match_arg(arg, "--species",  argc, argv, i)) { species_name = v; continue; }
            if (const char* v = match_arg(arg, "--seed",     argc, argv, i)) { cli_seed = parse_u64(v); continue; }
            if (const char* v = match_arg(arg, "--out",      argc, argv, i)) { out_path = v; continue; }
            if (std::strcmp(arg, "--bake-impostor") == 0) { bake_impostor = true; continue; }
            if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) { print_help(); return 0; }
            std::fprintf(stderr, "rynx-treegen: unknown argument: %s\n", arg);
            print_usage();
            return 2;
        }

        // --species and --scenario are mutually exclusive.
        if (!species_name.empty() && !scenario_path.empty()) {
            std::fprintf(stderr, "rynx-treegen: --species and --scenario are mutually exclusive\n");
            print_usage();
            return 2;
        }

        // Resolve --species to a scenario path.
        if (!species_name.empty()) {
            scenario_path = treegen::get_scenario_path(species_name);
            if (scenario_path.empty()) {
                std::fprintf(stderr, "rynx-treegen: unknown species '%s'\n", species_name.c_str());
                return 2;
            }
        }

        if (scenario_path.empty()) {
            std::fprintf(stderr, "rynx-treegen: --scenario or --species is required\n");
            print_usage();
            return 2;
        }

        treegen::Scenario s = treegen::load_scenario(scenario_path);
        uint64_t seed_effective = cli_seed ^ s.scenario_fnv;

        // %g for floats: deterministic shortest round-trippable form across
        // MSVC stdlib versions. %016llx pins width so seed lines align byte-
        // for-byte across runs.
        if (s.kind == "trivial_cylinder") {
            std::printf(
                "treegen: scenario=%s seed=%llu seed_effective=%016llx radius=%g height=%g radial=%d axial=%d\n",
                s.name.c_str(),
                static_cast<unsigned long long>(cli_seed),
                static_cast<unsigned long long>(seed_effective),
                static_cast<double>(s.cyl.radius),
                static_cast<double>(s.cyl.height),
                s.cyl.radial_segments,
                s.cyl.axial_segments);
        }
        else if (s.kind == "tree") {
            // C3 P2 — grow the skeleton; print summary stats; optional JSON dump.
            treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed_effective);
            const treegen::TrunkCapsule capsule = treegen::compute_trunk_capsule(skel, s.tree);

            // C5 P1 — leaf-site generation (BranchWalk).
            {
                auto opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
                skel.leaf_sites = treegen::leaf_placement::generate_leaf_sites(
                    skel, opts, seed_effective);
            }

            int max_depth = 0;
            for (const auto& n : skel.nodes) {
                if (n.depth > max_depth) max_depth = n.depth;
            }
            std::printf(
                "treegen: scenario=%s seed=%llu seed_effective=%016llx kind=tree species=%s height_m=%g shape=%s width_m=%g attractors=%d nodes=%zu max_depth=%d attractors_consumed=%d iterations=%d leaves=%zu\n",
                s.name.c_str(),
                static_cast<unsigned long long>(cli_seed),
                static_cast<unsigned long long>(seed_effective),
                s.tree.species.c_str(),
                static_cast<double>(s.tree.height_m),
                treegen::envelope_shape_to_string(s.tree.envelope.shape),
                static_cast<double>(s.tree.envelope.width_m),
                s.tree.branching.attractor_count,
                skel.nodes.size(),
                max_depth,
                skel.attractors_consumed,
                skel.iterations_run,
                skel.leaf_sites.size());
            std::printf("treegen: capsule half_length=%g radius=%g\n",
                        static_cast<double>(capsule.half_length),
                        static_cast<double>(capsule.radius));

            if (!skeleton_json_path.empty()) {
                const std::string dump = treegen::dump_skeleton_json(skel);
                std::ofstream f(skeleton_json_path, std::ios::binary);
                if (!f) {
                    std::fprintf(stderr, "rynx-treegen: cannot open dump path: %s\n",
                                 skeleton_json_path.c_str());
                    return 1;
                }
                f.write(dump.data(), static_cast<std::streamsize>(dump.size()));
                if (!f) {
                    std::fprintf(stderr, "rynx-treegen: write failed: %s\n",
                                 skeleton_json_path.c_str());
                    return 1;
                }
                std::printf("treegen: wrote skeleton JSON to %s (%zu bytes)\n",
                            skeleton_json_path.c_str(), dump.size());

                // C3 P3 polish: per-dump summary stats for visual eyeball-debug.
                const SkeletonStats st = compute_stats(skel, s.tree.branching.attractor_count);
                std::printf(
                    "treegen: skeleton stats:\n"
                    "  nodes:              %zu\n"
                    "  leaf_tips:          %zu\n"
                    "  max_depth:          %d\n"
                    "  total_branch_len_m: %g\n"
                    "  attractors_used:    %d/%d\n",
                    st.nodes, st.leaf_tips, st.max_depth,
                    static_cast<double>(st.total_branch_len_m),
                    st.attractors_used, st.attractors_total);
            }

            if (!out_path.empty()) {
                // C4 P4 + C10 — emit 3 LODs (L0, L1, L2) into a single
                // multi-mesh GLB. Per-LOD radial counts allocated by
                // face_budget; per-LOD wind weights baked by lod_emitter.
                // C5 P3 — when the scenario has leaves (density > 0 + non-empty
                // leaf_sites), also allocate per-LOD leaf budget and emit them
                // as primitive[1] in each LOD's MeshData. Bark-only fallback
                // for scenarios with no leaves keeps the 5-arg overload.
                treegen::BarkMeshOptions bopts;
                bopts.tree_height_m      = s.tree.height_m;
                bopts.root_flare_factor  = s.tree.root_flare_factor; // C12 P4
                bopts.junction_shoulder_factor = s.tree.junction_shoulder_factor; // C1 P1
                // Per-tree seam rotation: deterministic from seed_effective.
                treegen::pcg32 rng;
                rng.seed(seed_effective, 0x5EA00FF5E7u /* "seam offset" stream */);
                bopts.seam_offset_rad = rng.next_float_01() * 6.28318530717958647692f;

                treegen::LodBudget budget;  // P5 defaults: L0=12000/L1=4000/L2=1500/L3=4

                const bool is_strip = (s.tree.leaves.geometry_type == treegen::LeafGeometryType::BranchStrip);
                const bool has_leaves = is_strip
                                       || (!skel.leaf_sites.empty()
                                           && s.tree.leaves.leaf_density_per_meter > 0.0f);

                std::vector<treegen::LodOutput> lods;
                if (has_leaves) {
                    treegen::LeafBudget leaf_budget;  // defaults: 4000/1200/400/0
                    treegen::LeafMeshOptions leaf_geom_opts;
                    leaf_geom_opts.geometry_type         = s.tree.leaves.geometry_type;
                    leaf_geom_opts.shape                 = s.tree.leaves.shape;
                    leaf_geom_opts.leaf_size_m           = s.tree.leaves.leaf_size_m;
                    leaf_geom_opts.cluster_count_per_tip = s.tree.leaves.cluster_count_per_tip;
                    leaf_geom_opts.bend_half_angle       = s.tree.leaves.leaf_bend_half_angle;

                    lods = treegen::emit_all_lods(skel, skel.leaf_sites, bopts, budget,
                                                  leaf_budget, leaf_geom_opts,
                                                  s.tree.height_m, seed_effective);
                } else {
                    lods = treegen::emit_all_lods(skel, bopts, budget,
                                                  s.tree.height_m, seed_effective);
                }

                // C6 P4 — bark textures (per-species, shared across LODs).
                const treegen::LeafShape bark_species = s.tree.leaves.shape;
                std::vector<treegen::ImageData> glb_images;
                {
                    treegen::ImageData d, n, ao, mr;
                    d.png_bytes  = treegen::encode_bark_png(bark_species, seed_effective);
                    n.png_bytes  = treegen::encode_bark_normal_png(bark_species, seed_effective);
                    ao.png_bytes = treegen::encode_bark_ao_png(bark_species, seed_effective);
                    mr.png_bytes = treegen::encode_bark_roughness_png(bark_species, seed_effective);
                    glb_images.push_back(std::move(d));   // index 0
                    glb_images.push_back(std::move(n));   // index 1
                    glb_images.push_back(std::move(ao));  // index 2
                    glb_images.push_back(std::move(mr));  // index 3
                }

                // C6 P4 — leaf textures (when has_leaves).
                // C3 P2 — adds roughness atlas at index 7.
                if (has_leaves) {
                    treegen::ImageData ld, ln, lt, lr;
                    ld.png_bytes = treegen::encode_leaf_atlas_png(seed_effective);
                    ln.png_bytes = treegen::encode_leaf_normal_png(seed_effective);
                    lt.png_bytes = treegen::encode_leaf_translucency_png(seed_effective);
                    lr.png_bytes = treegen::encode_leaf_roughness_png(seed_effective);
                    glb_images.push_back(std::move(ld));  // index 4
                    glb_images.push_back(std::move(ln));  // index 5
                    glb_images.push_back(std::move(lt));  // index 6
                    glb_images.push_back(std::move(lr));  // index 7
                }

                // C9 P3 — impostor atlas bake. Decodes bark + leaf diffuse PNGs,
                // rasterizes L0 from 8 azimuths, encodes the atlas as a new
                // GLB image. L3 billboard material references this image.
                int impostor_tex_index = -1;
                if (bake_impostor && lods.size() >= 4) {
                    treegen::ImpostorInput imp_in{};
                    // L0 bark mesh.
                    const auto& l0 = lods[0];
                    imp_in.positions    = l0.mesh.positions.data();
                    imp_in.vertex_count = int(l0.mesh.positions.size() / 3);
                    imp_in.indices      = l0.indices_u32.data();
                    imp_in.index_count  = int(l0.indices_u32.size());
                    imp_in.uvs          = l0.mesh.uvs.data();

                    // Decode bark diffuse PNG.
                    int bw = 0, bh = 0, bc = 0;
                    stbi_uc* bark_px = stbi_load_from_memory(
                        glb_images[0].png_bytes.data(), int(glb_images[0].png_bytes.size()),
                        &bw, &bh, &bc, 4);
                    imp_in.bark_rgba = bark_px;
                    imp_in.bark_w    = bw;
                    imp_in.bark_h    = bh;

                    // Decode leaf diffuse PNG (if present).
                    stbi_uc* leaf_px = nullptr;
                    int lw = 0, lh = 0, lc = 0;
                    if (has_leaves && glb_images.size() > 4) {
                        leaf_px = stbi_load_from_memory(
                            glb_images[4].png_bytes.data(), int(glb_images[4].png_bytes.size()),
                            &lw, &lh, &lc, 4);
                        imp_in.leaf_rgba = leaf_px;
                        imp_in.leaf_w    = lw;
                        imp_in.leaf_h    = lh;
                    }

                    // L0 leaf mesh (if present).
                    if (l0.has_leaves) {
                        imp_in.leaf_positions    = l0.leaf_positions.data();
                        imp_in.leaf_vertex_count = int(l0.leaf_positions.size() / 3);
                        imp_in.leaf_indices      = l0.leaf_indices_u32.data();
                        imp_in.leaf_index_count  = int(l0.leaf_indices_u32.size());
                        imp_in.leaf_uvs          = l0.leaf_uvs.data();
                    }

                    auto atlas = treegen::bake_impostor_atlas(imp_in);
                    if (bark_px) stbi_image_free(bark_px);
                    if (leaf_px) stbi_image_free(leaf_px);

                    // Encode atlas to PNG and append as GLB image.
                    treegen::ImageData atlas_img;
                    atlas_img.png_bytes = treegen::encode_png_rgba8(
                        atlas.width, atlas.height, atlas.width * 4, atlas.rgba.data());
                    impostor_tex_index = int(glb_images.size());
                    glb_images.push_back(std::move(atlas_img));
                    std::printf("treegen: impostor atlas baked (%dx%d, image index %d)\n",
                                atlas.width, atlas.height, impostor_tex_index);
                }

                // Build flat per-LOD PrimitiveData arrays + MeshData wrappers.
                // Storage lives in vectors so the spans inside MeshData remain
                // valid until write_glb_multi_mesh returns.
                std::vector<std::vector<treegen::PrimitiveData>> per_lod_prims(lods.size());
                std::vector<treegen::MeshData>                    meshes;
                meshes.reserve(lods.size());

                for (size_t li = 0; li < lods.size(); ++li) {
                    const auto& lod = lods[li];
                    treegen::PrimitiveData bark_prim{};
                    bark_prim.positions   = std::span<const float>(lod.mesh.positions.data(), lod.mesh.positions.size());
                    bark_prim.normals     = std::span<const float>(lod.mesh.normals.data(),   lod.mesh.normals.size());
                    bark_prim.uvs         = std::span<const float>(lod.mesh.uvs.data(),       lod.mesh.uvs.size());
                    bark_prim.tangents    = std::span<const float>(lod.bark_tangents.data(),   lod.bark_tangents.size());
                    bark_prim.indices_u32 = std::span<const uint32_t>(lod.indices_u32.data(), lod.indices_u32.size());
                    bark_prim.wind_weights_packed = std::span<const uint8_t>(
                        lod.wind_weights_packed.data(), lod.wind_weights_packed.size());
                    // C9 P3 — L3 billboard uses impostor atlas when baked.
                    if (lod.lod_index == 3 && impostor_tex_index >= 0) {
                        bark_prim.material.base_color_tex_index = impostor_tex_index;
                        bark_prim.material.alpha_mode   = "MASK";
                        bark_prim.material.alpha_cutoff  = 0.5f;
                        bark_prim.material.sampler_index = 1;  // CLAMP
                    } else {
                        bark_prim.material.base_color_tex_index         = 0;
                        bark_prim.material.normal_tex_index             = 1;
                        bark_prim.material.occlusion_tex_index          = 2;
                        bark_prim.material.metallic_roughness_tex_index = 3;
                        bark_prim.material.sampler_index                = 0;
                    }
                    per_lod_prims[li].push_back(bark_prim);

                    if (lod.has_leaves) {
                        treegen::PrimitiveData leaf_prim{};
                        leaf_prim.positions   = std::span<const float>(
                            lod.leaf_positions.data(), lod.leaf_positions.size());
                        leaf_prim.normals     = std::span<const float>(
                            lod.leaf_normals.data(), lod.leaf_normals.size());
                        leaf_prim.uvs         = std::span<const float>(
                            lod.leaf_uvs.data(), lod.leaf_uvs.size());
                        leaf_prim.tangents    = std::span<const float>(
                            lod.leaf_tangents.data(), lod.leaf_tangents.size());
                        leaf_prim.indices_u32 = std::span<const uint32_t>(
                            lod.leaf_indices_u32.data(), lod.leaf_indices_u32.size());
                        leaf_prim.wind_weights_packed = std::span<const uint8_t>(
                            lod.leaf_wind_weights_packed.data(),
                            lod.leaf_wind_weights_packed.size());
                        leaf_prim.material.alpha_mode   = "MASK";
                        leaf_prim.material.alpha_cutoff = 0.5f;
                        leaf_prim.material.base_color_tex_index         = 4;
                        leaf_prim.material.normal_tex_index             = 5;
                        leaf_prim.material.translucency_tex_index       = 6;
                        leaf_prim.material.metallic_roughness_tex_index = 7;
                        leaf_prim.material.sampler_index                = 1;
                        per_lod_prims[li].push_back(leaf_prim);
                    }

                    treegen::MeshData md{};
                    md.primitives            = std::span<const treegen::PrimitiveData>(
                        per_lod_prims[li].data(), per_lod_prims[li].size());
                    md.lod_index             = lod.lod_index;
                    md.lod_max_distance_m    = lod.lod_max_distance_m;
                    md.lod_screen_height_px  = lod.lod_screen_height_px;
                    if (li == 0) {
                        md.collision_half_length = capsule.half_length;
                        md.collision_radius      = capsule.radius;
                    }
                    meshes.push_back(md);
                }

                std::string err;
                if (!treegen::write_glb_multi_mesh(
                        std::span<const treegen::MeshData>(meshes.data(), meshes.size()),
                        std::span<const treegen::ImageData>(glb_images.data(), glb_images.size()),
                        out_path, &err)) {
                    std::fprintf(stderr, "rynx-treegen: write_glb_multi_mesh failed: %s\n", err.c_str());
                    return 1;
                }
                std::error_code ec;
                auto fsize = std::filesystem::file_size(out_path, ec);
                auto leaf_vcount_for = [](const treegen::LodOutput& lod) {
                    // 4 bytes/vert wind weights → vcount = wind/4. Robust against
                    // any future change to per-leaf vert math.
                    return lod.has_leaves ? lod.leaf_wind_weights_packed.size() / 4 : size_t{0};
                };
                auto leaf_tris_for = [](const treegen::LodOutput& lod) {
                    return lod.has_leaves ? lod.leaf_indices_u32.size() / 3 : size_t{0};
                };
                std::printf(
                    "treegen: wrote %s (%zu LODs: "
                    "L0=%zu bark_t + %zu leaf_t (%zu leaf_v), "
                    "L1=%zu bark_t + %zu leaf_t (%zu leaf_v), "
                    "L2=%zu bark_t + %zu leaf_t (%zu leaf_v), "
                    "%llu bytes)\n",
                    out_path.c_str(),
                    lods.size(),
                    lods[0].indices_u32.size() / 3, leaf_tris_for(lods[0]), leaf_vcount_for(lods[0]),
                    lods[1].indices_u32.size() / 3, leaf_tris_for(lods[1]), leaf_vcount_for(lods[1]),
                    lods[2].indices_u32.size() / 3, leaf_tris_for(lods[2]), leaf_vcount_for(lods[2]),
                    static_cast<unsigned long long>(ec ? 0 : fsize));
                return 0;
            }
            return 0;
        }

        if (!out_path.empty()) {
            if (s.kind != "trivial_cylinder") {
                std::fprintf(stderr, "rynx-treegen: --out only implemented for trivial_cylinder in C1\n");
                return 1;
            }

            treegen::cpu_mesh_out cm = treegen::build_trivial_cylinder(
                s.cyl.radius, s.cyl.height,
                s.cyl.radial_segments, s.cyl.axial_segments);

            treegen::PrimitiveData prim{};
            prim.positions = std::span<const float>(cm.positions.data(), cm.positions.size());
            prim.normals   = std::span<const float>(cm.normals.data(), cm.normals.size());
            prim.uvs       = std::span<const float>(cm.uvs.data(), cm.uvs.size());
            prim.indices   = std::span<const uint16_t>(cm.indices.data(), cm.indices.size());

            std::string err;
            treegen::PrimitiveData prims_arr[1] = { prim };
            if (!treegen::write_glb(std::span<const treegen::PrimitiveData>(prims_arr, 1),
                                    out_path, &err)) {
                std::fprintf(stderr, "rynx-treegen: write_glb failed: %s\n", err.c_str());
                return 1;
            }

            std::error_code ec;
            auto size = std::filesystem::file_size(out_path, ec);
            std::printf("treegen: wrote %s (%zu verts, %zu tris, %llu bytes)\n",
                out_path.c_str(),
                cm.positions.size() / 3,
                cm.indices.size() / 3,
                static_cast<unsigned long long>(ec ? 0 : size));
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::fprintf(stderr, "rynx-treegen: error: %s\n", ex.what());
        return 1;
    }
}
