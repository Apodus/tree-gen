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

// C3-LOD-quality -> Botanical Sprig Uplift: moderated per-LOD leaf scale factor
// avoids giant 2.0x plates at L2 while preserving canopy fill across L0/L1/L2.
constexpr float k_lod_leaf_scale[3] = { 1.0f, 1.35f, 1.75f };

// Per-LOD radial targets — L0 is the base; L1 halves; L2 quarters.
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
                            const BoneTable&       rig,
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
    // Bark bone binding against the decimated rig: host = run bone of the
    // vertex's original node; blend = run-arc parent blend. The mesher's old
    // per-internode blend byte encodes the INTRA-segment axial fraction
    // (255 at the parent end, 0 at the node end) — recover t_seg from it so
    // the run blend interpolates at vertex (not just node) granularity.
    // Build before the move so bark.per_vertex_node_index is still populated.
    lod.bark_bone_index.reserve(bark.per_vertex_node_index.size());
    lod.bark_bone_blend.reserve(bark.per_vertex_node_index.size());
    constexpr float inv255 = 1.0f / 255.0f;
    for (size_t v = 0; v < bark.per_vertex_node_index.size(); ++v) {
        const int   n     = bark.per_vertex_node_index[v];
        const float t_seg = 1.0f - static_cast<float>(bark.bone_blend[v]) * inv255;
        lod.bark_bone_index.push_back(rig.node_to_bone[static_cast<size_t>(n)]);
        lod.bark_bone_blend.push_back(rig.vertex_parent_blend(n, t_seg));
    }
    // Decimated bone table — identical for every LOD (built once by caller).
    lod.bone_table_records = rig.records;
    lod.bone_count         = rig.bone_count;
    lod.mesh                = std::move(bark.mesh);
    lod.indices_u32         = std::move(bark.indices_u32);
    lod.bark_triangle_class = std::move(bark.triangle_class);
    lod.bark_triangle_node  = std::move(bark.triangle_node);
    lod.wind_weights_packed = std::move(bark.wind_weights_packed);
    lod.bark_tangents       = std::move(bark.tangents);
    lod.lod_index            = lod_index;
    lod.lod_max_distance_m   = k_lod_fallback_distance_m[lod_index];
    lod.lod_screen_height_px = k_lod_screen_height_px[lod_index];
    return lod;
}

// Leaf binding against the decimated rig. leaf_out.bone_index arrives holding
// ORIGINAL skeleton node ids; rebind to run bones, derive the run blend at the
// attachment point (t_seg = 1), and pin the tumble pivot to the original host
// node position so cards keep hanging from their twig, not from the (possibly
// distant) surviving joint.
void bind_leaves_to_rig(LodOutput& lod, const LeafMeshOutput& leaf_out,
                        const TreeSkeleton& skel, const BoneTable& rig) {
    const size_t n = leaf_out.bone_index.size();
    lod.leaf_bone_index.resize(n);
    lod.leaf_bone_blend.resize(n);
    lod.leaf_pivots.reserve(n * 3);
    for (size_t v = 0; v < n; ++v) {
        const int h = leaf_out.bone_index[v];
        lod.leaf_bone_index[v] = rig.node_to_bone[static_cast<size_t>(h)];
        lod.leaf_bone_blend[v] = rig.vertex_parent_blend(h, 1.0f);
        const vec3 p = skel.nodes[static_cast<size_t>(h)].position;
        lod.leaf_pivots.push_back(p.x);
        lod.leaf_pivots.push_back(p.y);
        lod.leaf_pivots.push_back(p.z);
    }
}

} // anonymous namespace

std::vector<LodOutput> emit_all_lods(const TreeSkeleton&    skel,
                                     const BarkMeshOptions& base_opts,
                                     const LodBudget&       budget,
                                     float                  tree_height_m,
                                     uint64_t               /*seed_effective*/)
{
    const BoneTable rig = build_bone_table(skel);
    std::vector<LodOutput> out;
    out.reserve(3);
    for (int i = 0; i < 3; ++i) {
        out.push_back(emit_one_bark_lod(skel, rig, base_opts, budget, tree_height_m, i));
    }
    // Three LODs (C10).
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

    const BoneTable rig = build_bone_table(skel);
    std::vector<LodOutput> out;
    out.reserve(3);

    for (int i = 0; i < 3; ++i) {
        LodOutput lod = emit_one_bark_lod(skel, rig, bark_opts, bark_budget, tree_height_m, i);

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
                bind_leaves_to_rig(lod, leaf_out, skel, rig);
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

                bind_leaves_to_rig(lod, leaf_out, skel, rig);
                lod.leaf_positions             = std::move(leaf_out.positions);
                lod.leaf_normals               = std::move(leaf_out.normals);
                lod.leaf_uvs                   = std::move(leaf_out.uvs);
                lod.leaf_indices_u32           = std::move(leaf_out.indices);
                lod.leaf_wind_weights_packed   = std::move(leaf_out.wind_weights_packed);
                lod.leaf_material_slots        = std::move(leaf_out.material_slots);
                lod.leaf_tangents              = std::move(leaf_out.tangents);
                lod.leaf_kept_sites            = alloc.kept_indices;
                lod.has_leaves                 = true;
            }
        }

        out.push_back(std::move(lod));
    }
    // Three LODs (C10).
    return out;
}

}  // namespace treegen
