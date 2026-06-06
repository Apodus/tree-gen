// C4 P4 + C5 P3 — see lod_emitter.hpp.
#include "lod_emitter.hpp"

#include "bone_table.hpp"
#include "branch_mesh.hpp"
#include "det_rng.hpp"
#include "face_budget.hpp"
#include "leaf_budget.hpp"
#include "leaf_geometry.hpp"

#include <algorithm>
#include <cmath>

namespace treegen {

namespace {

// Screen-height pixel thresholds: L0 used when tree subtends >= 120px;
// L1 when >= 40px; L2 when >= 15px. Indexed by lod_index 0..2.
constexpr float k_lod_screen_height_px[3] = { 120.0f, 40.0f, 15.0f };

// Fallback lod_max_distance_m for back-compat with old runtimes that don't
// read lod_screen_height_px. Reference: 10m tree, 1080p, proj_y ~ 1.0.
// distance = (ref_h * ref_proj_y * ref_vp_h) / (2 * screen_px)
constexpr float k_lod_fallback_distance_m[3] = { 45.0f, 135.0f, 360.0f };

// C3-LOD-quality: per-LOD leaf scale factor. Fewer leaves at lower LODs are
// slightly larger to maintain canopy coverage. Tuned so count*area is approx
// constant: L1 30%*1.4^2 ~ 59%, L2 10%*2.0^2 ~ 40%.
constexpr float k_lod_leaf_scale[3] = { 1.0f, 1.4f, 2.0f };

// Per-LOD radial targets — L0 is the base; L1 halves; L2 quarters. L3 ignores
// radial (it's a billboard stub).
PerOrderRadial radial_target_for_lod(int lod_index) {
    PerOrderRadial r;  // defaults = {12, 8, 6, 4}
    if (lod_index == 1) {
        r.trunk = 6; r.order1 = 4; r.order2 = 3; r.order3plus = 3;
    } else if (lod_index == 2) {
        r.trunk = 3; r.order1 = 3; r.order2 = 3; r.order3plus = 3;
    }
    return r;
}

int budget_for_lod(const LodBudget& b, int lod_index) {
    switch (lod_index) {
        case 0: return b.l0_tris;
        case 1: return b.l1_tris;
        case 2: return b.l2_tris;
        default: return b.l3_tris;
    }
}

LodOutput emit_one_bark_lod(const TreeSkeleton&    skel,
                            const BarkMeshOptions& base_opts,
                            const LodBudget&       budget,
                            float                  tree_height_m,
                            int                    lod_index)
{
    const PerOrderRadial target = radial_target_for_lod(lod_index);
    // Request merge-by-length mask when allocator can't fit budget via
    // radial-cut alone. Merged branches suppress bark but retain skeleton
    // nodes for leaf placement. Mask is empty if radial-cut suffices.
    std::vector<uint8_t> culled_node_mask;
    const PerOrderRadial actual = allocate_radial_for_lod(
        skel, budget_for_lod(budget, lod_index), target, &culled_node_mask);

    BarkMeshOptions opts = base_opts;
    opts.radial_seg_trunk       = actual.trunk;
    opts.radial_seg_order1      = actual.order1;
    opts.radial_seg_order2      = actual.order2;
    opts.radial_seg_order3_plus = actual.order3plus;
    opts.tree_height_m          = tree_height_m;
    opts.emit_crotch_cap        = true;
    opts.culled_node_mask       = std::move(culled_node_mask);

    BarkMeshOutput bark = build_bark_mesh(skel, opts);

    LodOutput lod;
    // C8-wind P1 — bark bone binding: host node (u16) + parent-blend byte.
    // Build before the move so bark.per_vertex_node_index is still populated.
    lod.bark_bone_index.reserve(bark.per_vertex_node_index.size());
    for (int n : bark.per_vertex_node_index) {
        lod.bark_bone_index.push_back(static_cast<uint16_t>(n));
    }
    lod.bark_bone_blend     = std::move(bark.bone_blend);
    // Full-skeleton bone table — identical for every LOD (no per-LOD bone
    // decimation in this campaign; every vertex binds to its full-skeleton node).
    {
        BoneTable bt = build_bone_table(skel);
        lod.bone_table_records = std::move(bt.records);
        lod.bone_count         = bt.bone_count;
    }
    lod.mesh                = std::move(bark.mesh);
    lod.indices_u32         = std::move(bark.indices_u32);
    lod.wind_weights_packed = std::move(bark.wind_weights_packed);
    lod.bark_tangents       = std::move(bark.tangents);
    lod.lod_index            = lod_index;
    lod.lod_max_distance_m   = k_lod_fallback_distance_m[lod_index];
    lod.lod_screen_height_px = k_lod_screen_height_px[lod_index];
    return lod;
}

// L3 = 4-tri vertical billboard quad at (0, 0, height/2), sized 1m × height_m.
// 4 verts, 4 triangles (front + back, each as 2 tris) so the impostor renders
// from either side. UV [0,1]. Wind weights all zero (impostor doesn't animate).
LodOutput emit_billboard_stub_lod(float tree_height_m) {
    LodOutput lod;
    lod.lod_index          = 3;
    lod.lod_max_distance_m   = 1000.0f; // billboard far distance fallback
    lod.lod_screen_height_px = 0.0f;    // billboard = coarsest

    const float h = std::max(0.1f, tree_height_m);
    const float half_w = 0.5f;        // 1m wide
    const float z0 = 0.0f;
    const float z1 = h;

    // 4 verts: (-w, 0, 0), (+w, 0, 0), (+w, 0, h), (-w, 0, h). Y-axis is the
    // billboard normal (camera-faceable later).
    lod.mesh.positions = {
        -half_w, 0.0f, z0,
         half_w, 0.0f, z0,
         half_w, 0.0f, z1,
        -half_w, 0.0f, z1,
    };
    lod.mesh.normals = {
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    // C9 P4 — UV maps to cell 0 of the 8-azimuth impostor atlas
    // (u=[0, 1/8], v=[0, 1]). When no impostor atlas is baked, the billboard
    // material still references the bark texture and the narrow UV just
    // samples a vertical bark stripe — acceptable at L3 distance.
    constexpr float k_cell_u = 1.0f / 8.0f;
    lod.mesh.uvs = {
        0.0f,     0.0f,
        k_cell_u, 0.0f,
        k_cell_u, 1.0f,
        0.0f,     1.0f,
    };
    // 4 triangles: front (CCW from +Y) + back (CCW from -Y) so visible
    // from either side without face-culling concerns.
    lod.indices_u32 = {
        0, 1, 2,  0, 2, 3,    // front
        0, 2, 1,  0, 3, 2,    // back
    };
    // Wind weights = all zeros (4 bytes/vert × 4 verts = 16). Keeps the
    // `_RYNX_WIND` accessor present so the runtime doesn't fork its vertex-
    // attribute layout per LOD. Trunk/branch/twig/leaf all 0 → no animation.
    lod.wind_weights_packed.assign(4u * 4u, 0u);

    // C1 P6 — billboard tangent: +X for all 4 verts (UV-right of the billboard).
    lod.bark_tangents = {
        1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f,
    };

    return lod;
}

} // anonymous namespace

std::vector<LodOutput> emit_all_lods(const TreeSkeleton&    skel,
                                     const BarkMeshOptions& base_opts,
                                     const LodBudget&       budget,
                                     float                  tree_height_m,
                                     uint64_t               /*seed_effective*/)
{
    std::vector<LodOutput> out;
    out.reserve(3);
    for (int i = 0; i < 3; ++i) {
        out.push_back(emit_one_bark_lod(skel, base_opts, budget, tree_height_m, i));
    }
    // L3 billboard stub suppressed (C10).
    return out;
}

std::vector<LodOutput> emit_all_lods(const TreeSkeleton&            skel,
                                     const std::vector<LeafSite>&   all_leaf_sites,
                                     const BarkMeshOptions&         bark_opts,
                                     const LodBudget&               bark_budget,
                                     const LeafBudget&              leaf_budget_in,
                                     const LeafMeshOptions&         leaf_geom_opts,
                                     float                          tree_height_m,
                                     uint64_t                       seed_effective)
{
    const bool is_strip = (leaf_geom_opts.geometry_type == LeafGeometryType::BranchStrip);

    // Site-based allocation (non-strip path).
    std::array<LeafBudgetResult, 4> leaf_alloc{};
    if (!is_strip) {
        leaf_alloc = allocate_leaves_all_lods(all_leaf_sites,
                                              leaf_geom_opts.geometry_type,
                                              leaf_geom_opts.shape,
                                              leaf_geom_opts.cluster_count_per_tip,
                                              leaf_budget_in,
                                              seed_effective);
    }

    std::vector<LodOutput> out;
    out.reserve(3);

    for (int i = 0; i < 3; ++i) {
        LodOutput lod = emit_one_bark_lod(skel, bark_opts, bark_budget, tree_height_m, i);

        if (is_strip) {
            // Segment-based strip emission. LOD controls strip_radius_threshold:
            // higher threshold at lower LODs = fewer qualifying segments.
            BranchStripOptions strip_opts;
            strip_opts.strip_width_m          = leaf_geom_opts.leaf_size_m > 0.01f
                                                ? leaf_geom_opts.leaf_size_m * 5.0f : 0.4f;
            strip_opts.strip_droop_angle      = 0.15f;

            // P4: per-tree angular jitter — breaks visual uniformity across
            // same-species trees by rotating the strip fan per seed.
            constexpr float k_pi = 3.14159265358979323846f;
            pcg32 strip_rng;
            strip_rng.seed(seed_effective, 0xC3'04'01ULL);
            strip_opts.strip_angular_offset   = strip_rng.next_float_01() * k_pi;

            // Per-LOD: L0 = full quality, L1/L2 = single strip only.
            float threshold = 0.02f;
            int max_strips = 3;
            int min_strips = 2;
            if (i == 1) { min_strips = 1; max_strips = 1; threshold = 0.03f; }
            if (i == 2) { min_strips = 1; max_strips = 1; threshold = 0.05f; }

            strip_opts.strip_radius_threshold = threshold;
            strip_opts.min_strips_per_segment = min_strips;
            strip_opts.max_strips_per_segment = max_strips;

            int min_depth = 2; // same as pine scenario default
            LeafMeshOutput leaf_out = build_branch_strip_mesh(skel, strip_opts, min_depth, threshold);

            if (!leaf_out.positions.empty()) {
                // C8-wind P1 — leaves/strips are rigid: bone_index from emitter,
                // bone_blend all-zero (full host rotation, no parent blend).
                lod.leaf_bone_index            = std::move(leaf_out.bone_index);
                lod.leaf_bone_blend.assign(lod.leaf_bone_index.size(), 0u);
                lod.leaf_positions             = std::move(leaf_out.positions);
                lod.leaf_normals               = std::move(leaf_out.normals);
                lod.leaf_uvs                   = std::move(leaf_out.uvs);
                lod.leaf_indices_u32           = std::move(leaf_out.indices);
                lod.leaf_wind_weights_packed   = std::move(leaf_out.wind_weights_packed);
                lod.leaf_material_slots        = std::move(leaf_out.material_slots);
                lod.leaf_tangents              = std::move(leaf_out.tangents);
                lod.has_leaves                 = true;
            }
        } else {
            const LeafBudgetResult& alloc = leaf_alloc[static_cast<size_t>(i)];
            if (!alloc.kept_indices.empty()) {
                std::vector<LeafSite> kept_sites;
                kept_sites.reserve(alloc.kept_indices.size());
                for (uint32_t idx : alloc.kept_indices) {
                    kept_sites.push_back(all_leaf_sites[static_cast<size_t>(idx)]);
                }

                LeafMeshOptions leaf_opts        = leaf_geom_opts;
                leaf_opts.geometry_type          = alloc.emitted_type;
                leaf_opts.cluster_count_per_tip  = alloc.effective_cluster;
                leaf_opts.leaf_size_m           *= k_lod_leaf_scale[i];  // C3-LOD-quality
                LeafMeshOutput leaf_out = build_leaf_mesh(kept_sites, leaf_opts);

                // C8-wind P1 — rigid leaf binding (see strip path above).
                lod.leaf_bone_index            = std::move(leaf_out.bone_index);
                lod.leaf_bone_blend.assign(lod.leaf_bone_index.size(), 0u);
                lod.leaf_positions             = std::move(leaf_out.positions);
                lod.leaf_normals               = std::move(leaf_out.normals);
                lod.leaf_uvs                   = std::move(leaf_out.uvs);
                lod.leaf_indices_u32           = std::move(leaf_out.indices);
                lod.leaf_wind_weights_packed   = std::move(leaf_out.wind_weights_packed);
                lod.leaf_material_slots        = std::move(leaf_out.material_slots);
                lod.leaf_tangents              = std::move(leaf_out.tangents);
                lod.has_leaves                 = true;
            }
        }

        out.push_back(std::move(lod));
    }
    // L3 billboard stub suppressed (C10).
    return out;
}

}  // namespace treegen
