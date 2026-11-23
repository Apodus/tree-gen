// NOTE: /fp:precise like the rest of TreegenCore — byte-hash determinism
// depends on it (rynx/generate/treegen.sharpmake.cs).
#include "generate_tree.hpp"

#include "bark_texture.hpp"
#include "branch_mesh.hpp"
#include "crown_probe.hpp"
#include "crown_volume.hpp"
#include "det_rng.hpp"
#include "face_budget.hpp"
#include "leaf_atlas.hpp"
#include "leaf_budget.hpp"
#include "leaf_geometry.hpp"
#include "leaf_placement.hpp"
#include "space_colonization.hpp"

#include <stb_image.h>

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace treegen {

namespace {
bool scenario_has_leaves(const Scenario& s, const TreeSkeleton& skel) {
    return s.tree.leaves.geometry_type == LeafGeometryType::BranchStrip
           || (!skel.leaf_sites.empty() && s.tree.leaves.leaf_density_per_meter > 0.0f);
}
}  // namespace

GrownTree grow_tree(const Scenario& s, uint64_t cli_seed) {
    if (s.kind != "tree")
        throw std::runtime_error("generate_tree: scenario '" + s.name + "' is kind '" + s.kind + "', not tree");

    GrownTree g;
    g.scenario_fnv = s.scenario_fnv;
    g.seed_effective = cli_seed ^ s.scenario_fnv;
    const uint64_t seed_effective = g.seed_effective;

    // C3 P2 — grow the skeleton.
    g.skeleton = grow_skeleton(s.tree, seed_effective);
    TreeSkeleton& skel = g.skeleton;
    g.capsule = compute_trunk_capsule(skel, s.tree);

    // C5 P1 — leaf-site generation (BranchWalk).
    {
        auto opts = leaf_placement::options_from_descriptor(s.tree.leaves);
        skel.leaf_sites = leaf_placement::generate_leaf_sites(skel, opts, seed_effective);
    }
    return g;
}

TreeLevels emit_levels(const Scenario& s, GrownTree grown) {
    if (grown.scenario_fnv != s.scenario_fnv)
        throw std::logic_error("emit_levels: GrownTree was grown from a different scenario (scenario_fnv mismatch)");
    TreeLevels g{std::move(grown)};
    const uint64_t seed_effective = g.seed_effective;
    const TreeSkeleton& skel = g.skeleton;

    // C4 P4 + C10 — emit 3 LODs (L0, L1, L2) into a single multi-mesh GLB.
    // Per-LOD radial counts allocated by face_budget; per-LOD wind weights
    // baked by lod_emitter. C5 P3 — when the scenario has leaves (density > 0 +
    // non-empty leaf_sites), also allocate per-LOD leaf budget and emit them as
    // primitive[1] in each LOD's MeshData. Bark-only fallback for scenarios
    // with no leaves keeps the 5-arg overload.
    BarkMeshOptions bopts;
    bopts.tree_height_m            = s.tree.height_m;
    bopts.root_flare_factor        = s.tree.root_flare_factor;         // C12 P4
    bopts.junction_shoulder_factor = s.tree.junction_shoulder_factor;  // C1 P1
    // Per-tree seam rotation: deterministic from seed_effective.
    pcg32 rng;
    rng.seed(seed_effective, 0x5EA00FF5E7u /* "seam offset" stream */);
    bopts.seam_offset_rad = rng.next_float_01() * 6.28318530717958647692f;

    // C3 P2 bark diet — per-species L0 (oak/pine 10k; broadleaf ≈natural).
    const LodBudget budget = lod_budget_for_species(s.tree.leaves.shape);

    if (scenario_has_leaves(s, skel)) {
        const LeafBudget leaf_budget = leaf_budget_from(s.tree.leaves.lod_leaf_triangle_budget);
        LeafMeshOptions leaf_geom_opts;
        leaf_geom_opts.geometry_type         = s.tree.leaves.geometry_type;
        leaf_geom_opts.shape                 = s.tree.leaves.shape;
        leaf_geom_opts.leaf_size_m           = s.tree.leaves.leaf_size_m;
        leaf_geom_opts.cluster_count_per_tip = s.tree.leaves.cluster_count_per_tip;
        leaf_geom_opts.bend_half_angle       = s.tree.leaves.leaf_bend_half_angle;

        g.lods = emit_all_lods(skel, skel.leaf_sites, bopts, budget, leaf_budget, leaf_geom_opts,
                               s.tree.height_m, seed_effective);
    } else {
        g.lods = emit_all_lods(skel, bopts, budget, s.tree.height_m, seed_effective);
    }
    return g;
}

GeneratedTree generate_tree(const Scenario& s, uint64_t cli_seed) {
    GeneratedTree g{emit_levels(s, grow_tree(s, cli_seed))};
    const uint64_t seed_effective = g.seed_effective;
    const bool has_leaves = scenario_has_leaves(s, g.skeleton);

    // Bark B-1 — per-species canonical bark texture bundle (single-pass bake,
    // keyed deterministically on bark_species for VRAM dedup across seeds; per-tree
    // rotational variety comes from bopts.seam_offset_rad).
    const LeafShape bark_species = s.tree.leaves.shape;
    {
        BarkPngBundle bark = encode_bark_bundle_pngs(bark_species, species_fnv_name(bark_species));
        ImageData d, n, ao, mr;
        d.png_bytes  = std::move(bark.diffuse_png);
        n.png_bytes  = std::move(bark.normal_png);
        ao.png_bytes = std::move(bark.ao_png);
        mr.png_bytes = std::move(bark.roughness_png);
        g.images.push_back(std::move(d));   // index 0
        g.images.push_back(std::move(n));   // index 1
        g.images.push_back(std::move(ao));  // index 2
        g.images.push_back(std::move(mr));  // index 3
    }

    // C6 P4 — leaf textures (when has_leaves). C3 P2 — roughness atlas at index 7.
    if (has_leaves) {
        const auto& grid = s.tree.leaves.cluster_cell_grid;
        ImageData ld, ln, lt, lr;
        ld.png_bytes = encode_leaf_atlas_png(seed_effective, grid);
        ln.png_bytes = encode_leaf_normal_png(seed_effective, grid);
        lt.png_bytes = encode_leaf_translucency_png(seed_effective, grid);
        lr.png_bytes = encode_leaf_roughness_png(seed_effective, grid);
        g.images.push_back(std::move(ld));  // index 4
        g.images.push_back(std::move(ln));  // index 5
        g.images.push_back(std::move(lt));  // index 6
        g.images.push_back(std::move(lr));  // index 7
    }

    // Flat per-LOD PrimitiveData arrays + MeshData wrappers. Storage lives in
    // vectors so the spans inside MeshData stay valid through the GLB build.
    const auto& lods = g.lods;
    std::vector<std::vector<PrimitiveData>> per_lod_prims(lods.size());
    std::vector<MeshData>                   meshes;
    meshes.reserve(lods.size());

    for (size_t li = 0; li < lods.size(); ++li) {
        const auto& lod = lods[li];
        PrimitiveData bark_prim{};
        bark_prim.positions   = std::span<const float>(lod.mesh.positions.data(), lod.mesh.positions.size());
        bark_prim.normals     = std::span<const float>(lod.mesh.normals.data(),   lod.mesh.normals.size());
        bark_prim.uvs         = std::span<const float>(lod.mesh.uvs.data(),       lod.mesh.uvs.size());
        bark_prim.tangents    = std::span<const float>(lod.bark_tangents.data(),   lod.bark_tangents.size());
        bark_prim.indices_u32 = std::span<const uint32_t>(lod.indices_u32.data(), lod.indices_u32.size());
        bark_prim.wind_weights_packed = std::span<const uint8_t>(
            lod.wind_weights_packed.data(), lod.wind_weights_packed.size());
        // C8-wind P1 — bark _RYNX_BONE per-vertex binding.
        bark_prim.bone_index = std::span<const uint16_t>(lod.bark_bone_index.data(), lod.bark_bone_index.size());
        bark_prim.bone_blend = std::span<const uint8_t>(lod.bark_bone_blend.data(), lod.bark_bone_blend.size());
        bark_prim.material.base_color_tex_index         = 0;
        bark_prim.material.normal_tex_index             = 1;
        bark_prim.material.occlusion_tex_index          = 2;
        bark_prim.material.metallic_roughness_tex_index = 3;
        bark_prim.material.sampler_index                = 0;
        per_lod_prims[li].push_back(bark_prim);

        if (lod.has_leaves) {
            PrimitiveData leaf_prim{};
            leaf_prim.positions   = std::span<const float>(lod.leaf_positions.data(), lod.leaf_positions.size());
            leaf_prim.normals     = std::span<const float>(lod.leaf_normals.data(), lod.leaf_normals.size());
            leaf_prim.uvs         = std::span<const float>(lod.leaf_uvs.data(), lod.leaf_uvs.size());
            leaf_prim.tangents    = std::span<const float>(lod.leaf_tangents.data(), lod.leaf_tangents.size());
            leaf_prim.indices_u32 = std::span<const uint32_t>(lod.leaf_indices_u32.data(), lod.leaf_indices_u32.size());
            leaf_prim.wind_weights_packed = std::span<const uint8_t>(
                lod.leaf_wind_weights_packed.data(), lod.leaf_wind_weights_packed.size());
            // Leaf _RYNX_BONE binding: decimated-rig bone + run blend + per-vertex
            // rest attachment pivot (tumble).
            leaf_prim.bone_index = std::span<const uint16_t>(lod.leaf_bone_index.data(), lod.leaf_bone_index.size());
            leaf_prim.bone_blend = std::span<const uint8_t>(lod.leaf_bone_blend.data(), lod.leaf_bone_blend.size());
            leaf_prim.bone_pivot = std::span<const float>(lod.leaf_pivots.data(), lod.leaf_pivots.size());
            leaf_prim.material.alpha_mode   = "MASK";
            leaf_prim.material.alpha_cutoff = 0.5f;
            leaf_prim.material.base_color_tex_index         = 4;
            leaf_prim.material.normal_tex_index             = 5;
            leaf_prim.material.translucency_tex_index       = 6;
            leaf_prim.material.metallic_roughness_tex_index = 7;
            leaf_prim.material.sampler_index                = 1;
            per_lod_prims[li].push_back(leaf_prim);
        }

        MeshData md{};
        md.primitives           = std::span<const PrimitiveData>(per_lod_prims[li].data(), per_lod_prims[li].size());
        md.lod_index            = lod.lod_index;
        md.lod_max_distance_m   = lod.lod_max_distance_m;
        md.lod_screen_height_px = lod.lod_screen_height_px;
        // Decimated bone table for this LOD (identical per LOD).
        md.bone_table = std::span<const float>(lod.bone_table_records.data(), lod.bone_table_records.size());
        md.bone_count = lod.bone_count;
        if (li == 0) {
            md.collision_half_length = g.capsule.half_length;
            md.collision_radius      = g.capsule.radius;
        }
        meshes.push_back(md);
    }

    // S-P1: build and pack F2_tensor Option 2A crown density volume from L0 leaf
    // triangles + mip-0 leaf atlas alpha coverage, padded outward with zero voxels
    // to enclose union(L0..L2) leaf AABBs, and attach to meshes[0].
    if (has_leaves && !lods.empty() && lods[0].has_leaves && g.images.size() > 4u) {
        int w = 0, h = 0, comp = 0;
        const auto& png = g.images[4].png_bytes;
        unsigned char* rgba = stbi_load_from_memory(
            reinterpret_cast<const stbi_uc*>(png.data()), static_cast<int>(png.size()), &w, &h, &comp, 4);
        if (!rgba)
            throw std::runtime_error(std::string("generate_tree: failed to decode leaf atlas PNG for crown volume: ")
                                     + stbi_failure_reason());
        crown::CrownMesh cm;
        cm.atlas.width  = w;
        cm.atlas.height = h;
        cm.atlas.cutoff = 0.5f;  // matches leaf_prim.material.alpha_cutoff = 0.5f
        cm.atlas.alpha.resize(size_t(w) * size_t(h));
        for (size_t i = 0; i < cm.atlas.alpha.size(); ++i) cm.atlas.alpha[i] = rgba[i * 4u + 3u];
        stbi_image_free(rgba);

        cm.add_leaves(lods[0].leaf_positions, lods[0].leaf_uvs, lods[0].leaf_indices_u32);
        const crown::Probe l0_probe(std::move(cm));

        vec3 union_min{1e30f, 1e30f, 1e30f};
        vec3 union_max{-1e30f, -1e30f, -1e30f};
        for (const auto& lod : lods) {
            if (!lod.has_leaves) continue;
            for (size_t i = 0; i + 2 < lod.leaf_positions.size(); i += 3) {
                const float x = lod.leaf_positions[i + 0];
                const float y = lod.leaf_positions[i + 1];
                const float z = lod.leaf_positions[i + 2];
                union_min = {std::min(union_min.x, x), std::min(union_min.y, y), std::min(union_min.z, z)};
                union_max = {std::max(union_max.x, x), std::max(union_max.y, y), std::max(union_max.z, z)};
            }
        }

        const crown::CrownVolume vol = crown::CrownVolume::build_shipped(l0_probe, union_min, union_max);
        const float clumping = (s.tree.leaves.geometry_type == LeafGeometryType::BranchStrip)
                             ? crown::k_strip_crown_clumping
                             : crown::k_default_crown_clumping;
        g.crown_volume = vol.pack_option2a(clumping);

        CrownVolumeSpec spec{};
        for (int a = 0; a < 3; ++a) {
            spec.dims[a]    = g.crown_volume->dims[a];
        }
        spec.box_min[0] = g.crown_volume->box_min.x;
        spec.box_min[1] = g.crown_volume->box_min.y;
        spec.box_min[2] = g.crown_volume->box_min.z;
        spec.box_max[0] = g.crown_volume->box_max.x;
        spec.box_max[1] = g.crown_volume->box_max.y;
        spec.box_max[2] = g.crown_volume->box_max.z;
        spec.cell_m     = g.crown_volume->cell_m;
        spec.rho_cap    = g.crown_volume->rho_cap;
        spec.clumping   = g.crown_volume->clumping;
        spec.tex0       = std::span<const uint8_t>(g.crown_volume->tex0.data(), g.crown_volume->tex0.size());
        spec.tex1       = std::span<const int8_t>(g.crown_volume->tex1.data(), g.crown_volume->tex1.size());
        meshes[0].crown_volume = spec;
    }

    std::string err;
    if (!build_glb_multi_mesh(std::span<const MeshData>(meshes.data(), meshes.size()),
                              std::span<const ImageData>(g.images.data(), g.images.size()), g.glb, &err))
        throw std::runtime_error("write_glb_multi_mesh failed: " + err);
    return g;
}

}  // namespace treegen
