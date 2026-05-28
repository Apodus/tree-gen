// [treegen_leaf_geometry] — pins C5 P2's leaf geometry emitters. Three
// geometry types × four species shapes. Tool sources (leaf_geometry.cpp +
// leaf_shapes.cpp + their P1 deps) link into TestTech via rynx_tests.sharpmake.cs.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../leaf_geometry.hpp"
#include "../leaf_placement.hpp"
#include "../leaf_shapes.hpp"
#include "../scenario.hpp"
#include "../skeleton.hpp"
#include "../space_colonization.hpp"
#include "../tree_descriptor.hpp"
#include "../vec3.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

float vlen(treegen::vec3 v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

// Build a tiny synthetic skeleton + 4 leaf sites for shape-independent tests.
std::vector<treegen::LeafSite> make_synthetic_sites() {
    using treegen::vec3;
    std::vector<treegen::LeafSite> sites;
    auto add = [&](vec3 p, vec3 n) {
        treegen::LeafSite s;
        s.position = p;
        s.normal   = n;
        s.branch_id = 0;
        s.type      = 0;
        sites.push_back(s);
    };
    add(vec3{1.0f, 0.0f, 5.0f}, vec3{1.0f, 0.0f, 0.0f});
    add(vec3{0.0f, 1.0f, 5.5f}, vec3{0.0f, 1.0f, 0.0f});
    add(vec3{-1.0f, 0.0f, 6.0f}, vec3{-1.0f, 0.0f, 0.0f});
    add(vec3{0.0f, -1.0f, 5.5f}, vec3{0.0f, -1.0f, 0.0f});
    return sites;
}

}  // namespace

TEST_CASE("treegen_leaf_geometry: shape polygon tri counts pinned",
          "[treegen][treegen_leaf_geometry][treegen_leaf_shapes]") {
    const auto oak  = treegen::oak_lobed();
    const auto pine = treegen::pine_needle();
    const auto bir  = treegen::birch_serrated();
    const auto map  = treegen::maple_star();

    INFO("oak verts="  << oak.verts.size()  << " tris=" << oak.tris.size()  / 3);
    INFO("pine verts=" << pine.verts.size() << " tris=" << pine.tris.size() / 3);
    INFO("birch verts=" << bir.verts.size() << " tris=" << bir.tris.size()  / 3);
    INFO("maple verts=" << map.verts.size() << " tris=" << map.tris.size()  / 3);

    // Pinned per spec (P2 targets — exact counts).
    REQUIRE(oak.verts.size() == 15u);
    REQUIRE(oak.tris.size()  == 14u * 3u);

    REQUIRE(pine.verts.size() == 6u);
    REQUIRE(pine.tris.size()  == 4u * 3u);

    // Birch lands at 15 perimeter (1 center + 15 verts = 16 total → 15 fan tris)
    // due to the asymmetric serrated/smooth side split.
    REQUIRE(bir.verts.size() == 16u);
    REQUIRE(bir.tris.size()  == 15u * 3u);

    REQUIRE(map.verts.size() == 16u);
    REQUIRE(map.tris.size()  == 20u * 3u);

    // Every shape's vertices stay in [-1, 1].
    auto check_bounds = [](const treegen::LeafShapeMesh& m) {
        for (auto v : m.verts) {
            REQUIRE(v.x >= -1.0f - 1e-4f);
            REQUIRE(v.x <= +1.0f + 1e-4f);
            REQUIRE(v.y >= -1.0f - 1e-4f);
            REQUIRE(v.y <= +1.0f + 1e-4f);
        }
    };
    check_bounds(oak);
    check_bounds(pine);
    check_bounds(bir);
    check_bounds(map);

    // Indices all in range.
    auto check_indices = [](const treegen::LeafShapeMesh& m) {
        for (auto i : m.tris) {
            REQUIRE(i < m.verts.size());
        }
    };
    check_indices(oak);
    check_indices(pine);
    check_indices(bir);
    check_indices(map);
}

TEST_CASE("treegen_leaf_geometry: SingleCard emits 4 verts / 2 tris per site",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_single_card]") {
    const auto sites = make_synthetic_sites();
    treegen::LeafMeshOptions opts;
    opts.geometry_type = treegen::LeafGeometryType::SingleCard;
    opts.shape         = treegen::LeafShape::OakLobed;
    opts.leaf_size_m   = 0.12f;

    auto out = treegen::build_leaf_mesh(sites, opts);

    REQUIRE(out.positions.size() / 3 == sites.size() * 4);
    REQUIRE(out.normals.size()   / 3 == sites.size() * 4);
    REQUIRE(out.uvs.size()       / 2 == sites.size() * 4);
    REQUIRE(out.indices.size()       == sites.size() * 6);
    REQUIRE(out.wind_weights_packed.size() == sites.size() * 4 * 4);
    REQUIRE(out.material_slots.size() == sites.size() * 4);
}

TEST_CASE("treegen_leaf_geometry: ProceduralVeined hits per-shape vert/tri counts",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_procedural]") {
    const auto sites = make_synthetic_sites();

    struct Case {
        treegen::LeafShape shape;
        size_t expected_verts_per_leaf;
        size_t expected_tris_per_leaf;
    };
    const Case cases[] = {
        { treegen::LeafShape::OakLobed,      15, 14 },
        { treegen::LeafShape::PineNeedle,     6,  4 },
        { treegen::LeafShape::BirchSerrated, 16, 15 },
        { treegen::LeafShape::MapleStar,     16, 20 },
    };

    for (const auto& c : cases) {
        treegen::LeafMeshOptions opts;
        opts.geometry_type = treegen::LeafGeometryType::ProceduralVeined;
        opts.shape         = c.shape;
        opts.leaf_size_m   = 0.12f;
        auto out = treegen::build_leaf_mesh(sites, opts);
        INFO("shape index=" << static_cast<int>(c.shape)
             << " verts=" << out.positions.size() / 3
             << " tris=" << out.indices.size() / 3);
        REQUIRE(out.positions.size() / 3 == sites.size() * c.expected_verts_per_leaf);
        REQUIRE(out.indices.size()       == sites.size() * c.expected_tris_per_leaf * 3);
        REQUIRE(out.material_slots.size() == sites.size() * c.expected_verts_per_leaf);
    }
}

TEST_CASE("treegen_leaf_geometry: per-vertex normals are unit-length and equal site.normal",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_normals]") {
    const auto sites = make_synthetic_sites();
    // Site normals in make_synthetic_sites() are unit-length by construction
    // (±X / ±Y); per-vertex normal must equal site.normal exactly (no per-vert
    // shading; backlit-translucency contract).
    // Verts-per-leaf depends on geometry_type:
    //   SingleCard        → 4
    //   BentCrossCluster  → 6 * cluster_count_per_tip (here N=2 → 12)
    //   ProceduralVeined  → shape verts (OakLobed → 15)
    struct Case { treegen::LeafGeometryType geom; size_t verts_per_leaf; };
    const Case k_cases[] = {
        { treegen::LeafGeometryType::SingleCard,        4u },
        { treegen::LeafGeometryType::BentCrossCluster, 12u },
        { treegen::LeafGeometryType::ProceduralVeined, 15u },
    };
    for (const auto& c : k_cases) {
        treegen::LeafMeshOptions opts;
        opts.geometry_type = c.geom;
        opts.shape         = treegen::LeafShape::OakLobed;
        opts.leaf_size_m   = 0.10f;
        opts.cluster_count_per_tip = 2;
        auto out = treegen::build_leaf_mesh(sites, opts);
        const size_t n = out.normals.size() / 3;
        REQUIRE(n == sites.size() * c.verts_per_leaf);
        for (size_t i = 0; i < n; ++i) {
            const treegen::vec3 nv{
                out.normals[i * 3 + 0],
                out.normals[i * 3 + 1],
                out.normals[i * 3 + 2],
            };
            const float L = vlen(nv);
            INFO("geom=" << static_cast<int>(c.geom) << " i=" << i << " |n|=" << L);
            REQUIRE(std::abs(L - 1.0f) < 1e-4f);

            // Direction must equal the originating site.normal (sites are
            // already unit; no per-vert shading variation).
            const treegen::vec3& sn = sites[i / c.verts_per_leaf].normal;
            const treegen::vec3 d{nv.x - sn.x, nv.y - sn.y, nv.z - sn.z};
            const float d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            INFO("dir-mismatch sq=" << d2);
            REQUIRE(d2 < 1e-8f);
        }
    }
}

TEST_CASE("treegen_leaf_geometry: 1mm normal-direction jitter is along site normal",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_jitter]") {
    // Build the same leaf set twice — once with jitter disabled, once with
    // 1mm. Per-leaf centroid (mean of verts) is invariant under the in-plane
    // rotation that the emitter applies, so its only difference between the
    // two runs is the normal-direction jitter offset.
    const auto sites = make_synthetic_sites();
    const size_t verts_per_leaf = 12u;   // BentCrossCluster with N=2

    treegen::LeafMeshOptions opts_un;
    opts_un.geometry_type           = treegen::LeafGeometryType::BentCrossCluster;
    opts_un.shape                   = treegen::LeafShape::OakLobed;
    opts_un.leaf_size_m             = 0.10f;
    opts_un.cluster_count_per_tip   = 2;
    opts_un.wind_jitter_norm_eps_m  = 0.0f;
    auto out_un = treegen::build_leaf_mesh(sites, opts_un);

    treegen::LeafMeshOptions opts_jt = opts_un;
    opts_jt.wind_jitter_norm_eps_m  = 0.001f;   // 1mm
    auto out_jt = treegen::build_leaf_mesh(sites, opts_jt);

    REQUIRE(out_un.positions.size() == out_jt.positions.size());
    REQUIRE(out_un.positions.size() / 3 == sites.size() * verts_per_leaf);

    auto centroid = [&](const std::vector<float>& pos, size_t leaf_i) -> treegen::vec3 {
        treegen::vec3 c{0.0f, 0.0f, 0.0f};
        for (size_t v = 0; v < verts_per_leaf; ++v) {
            const size_t k = (leaf_i * verts_per_leaf + v) * 3;
            c.x += pos[k + 0];
            c.y += pos[k + 1];
            c.z += pos[k + 2];
        }
        const float inv = 1.0f / static_cast<float>(verts_per_leaf);
        return treegen::vec3{c.x * inv, c.y * inv, c.z * inv};
    };

    int nonzero_count = 0;
    for (size_t i = 0; i < sites.size(); ++i) {
        const treegen::vec3 cu = centroid(out_un.positions, i);
        const treegen::vec3 cj = centroid(out_jt.positions, i);
        const treegen::vec3 d{cj.x - cu.x, cj.y - cu.y, cj.z - cu.z};
        const float L = vlen(d);
        INFO("leaf=" << i << " |d|=" << L);
        // |jitter| <= eps/2 = 0.0005m; small float slack for centroid sum.
        REQUIRE(L <= 0.0005f + 1e-6f);

        if (L > 1e-8f) {
            ++nonzero_count;
            // Direction parallel to site.normal (already unit in fixture).
            const treegen::vec3& n = sites[i].normal;
            const float dn = d.x * n.x + d.y * n.y + d.z * n.z;
            const treegen::vec3 perp{
                d.x - dn * n.x,
                d.y - dn * n.y,
                d.z - dn * n.z,
            };
            const float perp_len = vlen(perp);
            INFO("perp_len=" << perp_len);
            REQUIRE(perp_len < 1e-5f);
        }
    }
    // At least one leaf must have jittered. (P ~ 10^-29 for all-zero with
    // pcg32 → not a real risk.)
    REQUIRE(nonzero_count > 0);
}

TEST_CASE("treegen_leaf_geometry: wind weights at d=1 are (0,0,0,255) and material slot = 1",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_wind_mat]") {
    // Sites with branch_depth_fraction = 1.0 (terminal twig) → pure leaf tier.
    auto sites = make_synthetic_sites();
    for (auto& s : sites) s.branch_depth_fraction = 1.0f;

    treegen::LeafMeshOptions opts;
    opts.geometry_type = treegen::LeafGeometryType::ProceduralVeined;
    opts.shape         = treegen::LeafShape::MapleStar;
    auto out = treegen::build_leaf_mesh(sites, opts);

    const size_t n_verts = out.positions.size() / 3;
    REQUIRE(out.wind_weights_packed.size() == n_verts * 4);
    REQUIRE(out.material_slots.size()      == n_verts);
    for (size_t i = 0; i < n_verts; ++i) {
        REQUIRE(out.wind_weights_packed[i * 4 + 0] == 0u);
        REQUIRE(out.wind_weights_packed[i * 4 + 1] == 0u);
        REQUIRE(out.wind_weights_packed[i * 4 + 2] == 0u);
        REQUIRE(out.wind_weights_packed[i * 4 + 3] == 255u);
        REQUIRE(out.material_slots[i] == 1);
    }
}

TEST_CASE("treegen_leaf_geometry: wind weights vary with branch_depth_fraction",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_wind_depth]") {
    // Build two leaf sets: one on thick branches (d=0.25), one on terminal
    // twigs (d=1.0). Wind weights must differ — thick-branch leaves get
    // trunk/branch influence while twig leaves are pure leaf tier.
    auto sites_thick = make_synthetic_sites();
    auto sites_thin  = make_synthetic_sites();
    for (auto& s : sites_thick) s.branch_depth_fraction = 0.25f;
    for (auto& s : sites_thin)  s.branch_depth_fraction = 1.0f;

    treegen::LeafMeshOptions opts;
    opts.geometry_type = treegen::LeafGeometryType::SingleCard;
    opts.shape         = treegen::LeafShape::OakLobed;

    auto out_thick = treegen::build_leaf_mesh(sites_thick, opts);
    auto out_thin  = treegen::build_leaf_mesh(sites_thin,  opts);

    REQUIRE(out_thick.wind_weights_packed.size() == out_thin.wind_weights_packed.size());
    REQUIRE(out_thick.wind_weights_packed.size() > 0u);

    // Thick-branch leaf (d=0.25): trunk=(0.75)^2=0.5625→143, branch=4*0.25*0.75=0.75→191,
    // twig=4*0.0625*0.75=0.1875→48, leaf=0.015625→4.
    REQUIRE(out_thick.wind_weights_packed[0] == 143u);  // trunk
    REQUIRE(out_thick.wind_weights_packed[1] == 191u);  // branch
    REQUIRE(out_thick.wind_weights_packed[2] == 48u);   // twig
    REQUIRE(out_thick.wind_weights_packed[3] == 4u);    // leaf

    // Terminal twig (d=1.0): (0,0,0,255).
    REQUIRE(out_thin.wind_weights_packed[0] == 0u);
    REQUIRE(out_thin.wind_weights_packed[1] == 0u);
    REQUIRE(out_thin.wind_weights_packed[2] == 0u);
    REQUIRE(out_thin.wind_weights_packed[3] == 255u);

    // Mid-depth (d=0.5) on a different geometry type.
    auto sites_mid = make_synthetic_sites();
    for (auto& s : sites_mid) s.branch_depth_fraction = 0.5f;
    opts.geometry_type = treegen::LeafGeometryType::BentCrossCluster;
    opts.cluster_count_per_tip = 2;
    opts.bend_half_angle = 0.392699081698724f;
    auto out_mid = treegen::build_leaf_mesh(sites_mid, opts);
    // d=0.5: trunk=0.25->64, branch=4*0.25=1.0->255, twig=4*0.25*0.5=0.5->128, leaf=0.125->32.
    REQUIRE(out_mid.wind_weights_packed[0] == 64u);
    REQUIRE(out_mid.wind_weights_packed[1] == 255u);
    REQUIRE(out_mid.wind_weights_packed[2] == 128u);
    REQUIRE(out_mid.wind_weights_packed[3] == 32u);
}

TEST_CASE("treegen_leaf_geometry: wind weights from generate_leaf_sites populate depth fraction",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_wind_pipeline]") {
    // Full pipeline: generate_leaf_sites → build_leaf_mesh. Verify that leaves
    // on deeper branches get different wind weights than leaves on shallower ones.
    namespace tsp = rynx::test_support;
    const auto fixture = tsp::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);

    auto leaf_opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
    auto sites = treegen::leaf_placement::generate_leaf_sites(skel, leaf_opts, seed);
    REQUIRE(sites.size() > 10u);

    // Verify branch_depth_fraction is populated and varies.
    float min_frac = 1.0f, max_frac = 0.0f;
    for (const auto& site : sites) {
        if (site.branch_depth_fraction < min_frac) min_frac = site.branch_depth_fraction;
        if (site.branch_depth_fraction > max_frac) max_frac = site.branch_depth_fraction;
    }
    REQUIRE(max_frac > min_frac);  // at least two different depth levels

    treegen::LeafMeshOptions mopts;
    mopts.geometry_type = treegen::LeafGeometryType::SingleCard;
    mopts.shape         = treegen::LeafShape::OakLobed;
    mopts.leaf_size_m   = s.tree.leaves.leaf_size_m;
    auto out = treegen::build_leaf_mesh(sites, mopts);

    // Wind weights must not all be identical (they used to be hardcoded (0,0,0,255)).
    const size_t n_verts = out.positions.size() / 3;
    REQUIRE(out.wind_weights_packed.size() == n_verts * 4);
    bool found_nonzero_trunk = false;
    bool found_nonzero_branch = false;
    for (size_t i = 0; i < n_verts; ++i) {
        if (out.wind_weights_packed[i * 4 + 0] > 0u) found_nonzero_trunk = true;
        if (out.wind_weights_packed[i * 4 + 1] > 0u) found_nonzero_branch = true;
    }
    // With depth-interpolated weights, some leaves must have nonzero trunk/branch.
    REQUIRE(found_nonzero_trunk);
    REQUIRE(found_nonzero_branch);
}

TEST_CASE("treegen_leaf_geometry: determinism — two runs byte-equal",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_determinism]") {
    const auto sites = make_synthetic_sites();
    treegen::LeafMeshOptions opts;
    opts.geometry_type = treegen::LeafGeometryType::BentCrossCluster;
    opts.shape         = treegen::LeafShape::OakLobed;
    opts.cluster_count_per_tip = 2;
    opts.bend_half_angle = 0.392699081698724f;

    auto a = treegen::build_leaf_mesh(sites, opts);
    auto b = treegen::build_leaf_mesh(sites, opts);

    REQUIRE(a.positions == b.positions);
    REQUIRE(a.normals   == b.normals);
    REQUIRE(a.uvs       == b.uvs);
    REQUIRE(a.indices   == b.indices);
    REQUIRE(a.wind_weights_packed == b.wind_weights_packed);
    REQUIRE(a.material_slots == b.material_slots);
    REQUIRE(a.tangents == b.tangents);
}

TEST_CASE("treegen_leaf_geometry: no trunk intersection — leaves outside trunk radius",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_no_trunk_intersect]") {
    // Use the real c3_oak fixture: leaf sites come from the shell-density
    // strategy on a grown oak. The trunk is centred at x=y=0 along +Z; we
    // assert each leaf's XY-distance from the trunk axis exceeds the trunk
    // radius at its z. (Leaf canopies live well outside the trunk cylinder.)
    namespace ts = rynx::test_support;
    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());
    treegen::Scenario s = treegen::load_scenario(fixture);
    const uint64_t seed = 42ull ^ s.scenario_fnv;
    treegen::TreeSkeleton skel = treegen::grow_skeleton(s.tree, seed);

    auto leaf_opts = treegen::leaf_placement::options_from_descriptor(s.tree.leaves);
    auto sites = treegen::leaf_placement::generate_leaf_sites(skel, leaf_opts, seed);
    REQUIRE(sites.size() > 100u);

    treegen::LeafMeshOptions opts;
    opts.geometry_type = treegen::LeafGeometryType::ProceduralVeined;
    opts.shape         = treegen::LeafShape::OakLobed;
    opts.leaf_size_m   = s.tree.leaves.leaf_size_m;
    auto out = treegen::build_leaf_mesh(sites, opts);

    // Tapered trunk radius: r(z) = r_base * (1 - z/H)^taper for z ∈ [0, H];
    // 0 above the crown. Allow a small slack since the actual trunk taper
    // ends well before the canopy starts.
    const float r_base = s.tree.trunk_base_radius_m;
    const float taper  = s.tree.taper_exponent;
    const float H      = s.tree.height_m;
    int intersect_count = 0;
    const size_t n_verts = out.positions.size() / 3;
    for (size_t i = 0; i < n_verts; ++i) {
        const float x = out.positions[i * 3 + 0];
        const float y = out.positions[i * 3 + 1];
        const float z = out.positions[i * 3 + 2];
        const float t = (z >= 0.0f && z <= H) ? (1.0f - z / H) : 0.0f;
        const float r_trunk = (t > 0.0f) ? r_base * std::pow(t, taper) : 0.0f;
        const float r_xy    = std::sqrt(x * x + y * y);
        if (r_xy < r_trunk - 0.05f) ++intersect_count;  // 5cm slack
    }
    INFO("intersect_count=" << intersect_count << " / " << n_verts);
    // The canopy starts ~halfway up the tree; trunk radius is small (<25cm)
    // by then. We tolerate a tiny number of clipped verts but not many.
    REQUIRE(intersect_count < static_cast<int>(n_verts / 100));
}

// ---- C2: BentCard + BentCrossCluster tests --------------------------------

TEST_CASE("treegen_leaf_geometry: BentCard emits 6 verts / 4 tris per site",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_bent_card]") {
    const auto sites = make_synthetic_sites();
    treegen::LeafMeshOptions opts;
    opts.geometry_type    = treegen::LeafGeometryType::BentCard;
    opts.shape            = treegen::LeafShape::OakLobed;
    opts.leaf_size_m      = 0.12f;
    opts.bend_half_angle  = 0.392699081698724f;  // pi/8

    auto out = treegen::build_leaf_mesh(sites, opts);

    REQUIRE(out.positions.size() / 3 == sites.size() * 6);
    REQUIRE(out.normals.size()   / 3 == sites.size() * 6);
    REQUIRE(out.uvs.size()       / 2 == sites.size() * 6);
    REQUIRE(out.indices.size()       == sites.size() * 12);
    REQUIRE(out.wind_weights_packed.size() == sites.size() * 6 * 4);
    REQUIRE(out.material_slots.size() == sites.size() * 6);
    REQUIRE(out.tangents.size() / 4 == sites.size() * 6);
}

TEST_CASE("treegen_leaf_geometry: BentCard dihedral fold — center verts lifted along normal",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_bent_card_fold]") {
    // Single site with known normal (+Z). Verify that center-crease verts
    // are displaced upward relative to the corner verts.
    std::vector<treegen::LeafSite> sites;
    treegen::LeafSite s;
    s.position = treegen::vec3{0.0f, 0.0f, 5.0f};
    s.normal   = treegen::vec3{0.0f, 0.0f, 1.0f};
    s.branch_id = 0;
    s.type = 0;
    sites.push_back(s);

    treegen::LeafMeshOptions opts;
    opts.geometry_type         = treegen::LeafGeometryType::BentCard;
    opts.shape                 = treegen::LeafShape::OakLobed;
    opts.leaf_size_m           = 1.0f;  // large for easy measurement
    opts.bend_half_angle       = 0.392699081698724f;  // pi/8
    opts.wind_jitter_norm_eps_m = 0.0f;

    auto out = treegen::build_leaf_mesh(sites, opts);
    REQUIRE(out.positions.size() / 3 == 6u);

    // Extract z-values. Corners (idx 0,2,3,5) should be at ~5.0 (site.z +
    // jitter=0). Center-crease verts (idx 1,4) should be lifted by
    // leaf_size * 0.5 * sin(pi/8) = 0.5 * sin(pi/8) ≈ 0.1913.
    const float expected_lift = 1.0f * 0.5f * std::sin(0.392699081698724f);
    const float corner_z = out.positions[0 * 3 + 2];  // vert 0.z
    const float mid_z_0  = out.positions[1 * 3 + 2];  // vert 1.z (center crease)
    const float mid_z_1  = out.positions[4 * 3 + 2];  // vert 4.z (center crease)

    INFO("corner_z=" << corner_z << " mid_z_0=" << mid_z_0 << " mid_z_1=" << mid_z_1
         << " expected_lift=" << expected_lift);
    // Center crease verts are lifted above corners
    REQUIRE(mid_z_0 > corner_z + 0.01f);
    REQUIRE(mid_z_1 > corner_z + 0.01f);
    // The lift is approximately expected_lift above the center
    REQUIRE(std::abs((mid_z_0 - 5.0f) - expected_lift) < 0.01f);
    REQUIRE(std::abs((mid_z_1 - 5.0f) - expected_lift) < 0.01f);
}

TEST_CASE("treegen_leaf_geometry: BentCrossCluster scales with cluster_count_per_tip",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_bent_cross]") {
    const auto sites = make_synthetic_sites();

    treegen::LeafMeshOptions opts;
    opts.geometry_type    = treegen::LeafGeometryType::BentCrossCluster;
    opts.shape            = treegen::LeafShape::OakLobed;
    opts.leaf_size_m      = 0.12f;
    opts.bend_half_angle  = 0.392699081698724f;

    // N=1: 6 verts / 4 tris per site.
    opts.cluster_count_per_tip = 1;
    auto out1 = treegen::build_leaf_mesh(sites, opts);
    REQUIRE(out1.positions.size() / 3 == sites.size() * 6);
    REQUIRE(out1.indices.size()       == sites.size() * 12);

    // N=2 (90 deg cross): 12 verts / 8 tris per site.
    opts.cluster_count_per_tip = 2;
    auto out2 = treegen::build_leaf_mesh(sites, opts);
    REQUIRE(out2.positions.size() / 3 == sites.size() * 12);
    REQUIRE(out2.indices.size()       == sites.size() * 24);

    // N=3: 18 verts / 12 tris per site.
    opts.cluster_count_per_tip = 3;
    auto out3 = treegen::build_leaf_mesh(sites, opts);
    REQUIRE(out3.positions.size() / 3 == sites.size() * 18);
    REQUIRE(out3.indices.size()       == sites.size() * 36);
}

TEST_CASE("treegen_leaf_geometry: BentCard/BentCrossCluster minimum solid angle guarantee",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_bent_min_area]") {
    // From any viewing direction, a BentCard with bend_half_angle alpha
    // guarantees that at least one half-plane has cos(theta) >= sin(alpha).
    // Test by sweeping 100 random view directions and computing the visible
    // projected area (sum of triangle areas projected onto view plane).
    std::vector<treegen::LeafSite> sites;
    treegen::LeafSite s;
    s.position = treegen::vec3{0.0f, 0.0f, 0.0f};
    s.normal   = treegen::vec3{0.0f, 0.0f, 1.0f};
    s.branch_id = 0;
    s.type = 0;
    sites.push_back(s);

    const float alpha = 0.392699081698724f; // pi/8 = 22.5 deg
    treegen::LeafMeshOptions opts;
    opts.geometry_type          = treegen::LeafGeometryType::BentCard;
    opts.shape                  = treegen::LeafShape::OakLobed;
    opts.leaf_size_m            = 1.0f;
    opts.bend_half_angle        = alpha;
    opts.wind_jitter_norm_eps_m = 0.0f;

    auto out = treegen::build_leaf_mesh(sites, opts);
    REQUIRE(out.positions.size() / 3 == 6u);
    REQUIRE(out.indices.size()       == 12u);

    // Extract triangle vertices
    auto get_v = [&](uint32_t i) -> treegen::vec3 {
        return { out.positions[i*3+0], out.positions[i*3+1], out.positions[i*3+2] };
    };

    // Sweep view directions on a fibonacci hemisphere + full sphere
    const int N = 200;
    const float golden_ratio = 1.6180339887498949f;
    const float k_pi = 3.14159265358979323846f;
    float min_projected_area = 1e9f;

    for (int i = 0; i < N; ++i) {
        const float theta = std::acos(1.0f - 2.0f * (static_cast<float>(i) + 0.5f) / static_cast<float>(N));
        const float phi   = 2.0f * k_pi * static_cast<float>(i) / golden_ratio;
        const treegen::vec3 view_dir = {
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            std::cos(theta),
        };

        // Sum projected area of all 4 triangles
        float total_area = 0.0f;
        for (size_t t = 0; t < out.indices.size(); t += 3) {
            const treegen::vec3 a = get_v(out.indices[t+0]);
            const treegen::vec3 b = get_v(out.indices[t+1]);
            const treegen::vec3 c = get_v(out.indices[t+2]);
            const treegen::vec3 ab = { b.x-a.x, b.y-a.y, b.z-a.z };
            const treegen::vec3 ac = { c.x-a.x, c.y-a.y, c.z-a.z };
            const treegen::vec3 cr = treegen::cross(ab, ac);
            // Projected area = |dot(cross, view_dir)| / 2
            const float proj = std::abs(treegen::dot(cr, view_dir)) * 0.5f;
            total_area += proj;
        }
        if (total_area < min_projected_area) min_projected_area = total_area;
    }

    // The dihedral fold guarantees nonzero projected area from every direction.
    // A flat card degenerates to zero when viewed exactly edge-on; the bent
    // card never does. The measured minimum is ~0.16 for alpha=pi/8, s=1.0.
    INFO("min_projected_area=" << min_projected_area);
    REQUIRE(min_projected_area > 0.05f);
}

TEST_CASE("treegen_leaf_geometry: BentCard determinism — two runs byte-equal",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_bent_determinism]") {
    const auto sites = make_synthetic_sites();
    treegen::LeafMeshOptions opts;
    opts.geometry_type         = treegen::LeafGeometryType::BentCrossCluster;
    opts.shape                 = treegen::LeafShape::OakLobed;
    opts.cluster_count_per_tip = 2;
    opts.bend_half_angle       = 0.392699081698724f;

    auto a = treegen::build_leaf_mesh(sites, opts);
    auto b = treegen::build_leaf_mesh(sites, opts);

    REQUIRE(a.positions == b.positions);
    REQUIRE(a.normals   == b.normals);
    REQUIRE(a.uvs       == b.uvs);
    REQUIRE(a.indices   == b.indices);
    REQUIRE(a.wind_weights_packed == b.wind_weights_packed);
    REQUIRE(a.material_slots == b.material_slots);
    REQUIRE(a.tangents == b.tangents);
}

TEST_CASE("treegen_leaf_geometry: BentCard normals equal site.normal",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_bent_normals]") {
    const auto sites = make_synthetic_sites();
    treegen::LeafMeshOptions opts;
    opts.geometry_type    = treegen::LeafGeometryType::BentCard;
    opts.shape            = treegen::LeafShape::OakLobed;
    opts.leaf_size_m      = 0.10f;
    opts.bend_half_angle  = 0.392699081698724f;

    auto out = treegen::build_leaf_mesh(sites, opts);
    const size_t n = out.normals.size() / 3;
    REQUIRE(n == sites.size() * 6);
    for (size_t i = 0; i < n; ++i) {
        const treegen::vec3 nv{
            out.normals[i*3+0], out.normals[i*3+1], out.normals[i*3+2],
        };
        const float L = vlen(nv);
        REQUIRE(std::abs(L - 1.0f) < 1e-4f);
        const treegen::vec3& sn = sites[i / 6].normal;
        const treegen::vec3 d{nv.x - sn.x, nv.y - sn.y, nv.z - sn.z};
        REQUIRE(d.x*d.x + d.y*d.y + d.z*d.z < 1e-8f);
    }
}

TEST_CASE("treegen_leaf_geometry: branch-strip TBN is orthonormal and non-degenerate",
          "[treegen][treegen_leaf_geometry][treegen_branch_strip_tbn]") {
    // Synthetic 2-node skeleton: root at origin, child at a non-axis-aligned position.
    treegen::TreeSkeleton skel;
    {
        treegen::BranchNode root;
        root.position = treegen::vec3{0.0f, 0.0f, 0.0f};
        root.radius = 0.05f;
        root.depth = 0;
        root.parent_index = -1;
        skel.nodes.push_back(root);

        treegen::BranchNode child;
        child.position = treegen::vec3{1.0f, 0.0f, 2.0f};
        child.radius = 0.04f;
        child.depth = 4;
        child.parent_index = 0;
        skel.nodes.push_back(child);
    }

    treegen::BranchStripOptions opts;
    opts.strip_width_m          = 0.4f;
    opts.strip_droop_angle      = 0.15f;
    opts.strip_angular_offset   = 0.0f;
    opts.strip_radius_threshold = 0.02f;
    opts.max_strips_per_segment = 3;

    const int min_branch_depth = 3;
    const float threshold = opts.strip_radius_threshold;
    auto out = treegen::build_branch_strip_mesh(skel, opts, min_branch_depth, threshold);

    // Must have emitted at least one strip quad (4 verts).
    const size_t n_verts = out.positions.size() / 3;
    REQUIRE(n_verts >= 4u);
    REQUIRE(out.tangents.size() / 4 == n_verts);
    REQUIRE(out.normals.size()  / 3 == n_verts);

    for (size_t i = 0; i < n_verts; ++i) {
        const treegen::vec3 T{
            out.tangents[i * 4 + 0],
            out.tangents[i * 4 + 1],
            out.tangents[i * 4 + 2],
        };
        const treegen::vec3 N{
            out.normals[i * 3 + 0],
            out.normals[i * 3 + 1],
            out.normals[i * 3 + 2],
        };
        const float T_len = vlen(T);
        const float N_len = vlen(N);
        const float TdotN = treegen::dot(T, N);
        const treegen::vec3 B = treegen::cross(N, T);
        const float B_len = vlen(B);

        INFO("vert=" << i << " |T|=" << T_len << " |N|=" << N_len
             << " T.N=" << TdotN << " |B|=" << B_len);
        REQUIRE(std::abs(T_len - 1.0f) < 1e-5f);
        REQUIRE(std::abs(N_len - 1.0f) < 1e-5f);
        REQUIRE(std::abs(TdotN) < 1e-5f);
        REQUIRE(B_len > 0.99f);
    }
}

TEST_CASE("treegen_leaf_geometry: tris_per_leaf SoT matches emitter output",
          "[treegen][treegen_leaf_geometry][treegen_leaf_geom_sot]") {
    const auto sites = make_synthetic_sites();
    struct Case {
        treegen::LeafGeometryType g;
        int cluster;
        int expected_tris_per_leaf;
    };
    const Case cases[] = {
        { treegen::LeafGeometryType::SingleCard,        1, 2 },
        { treegen::LeafGeometryType::BentCard,          1, 4 },
        { treegen::LeafGeometryType::BentCrossCluster,  2, 8 },
        { treegen::LeafGeometryType::BentCrossCluster,  1, 4 },
    };
    for (const auto& c : cases) {
        const int sot = treegen::tris_per_leaf(c.g, treegen::LeafShape::OakLobed, c.cluster);
        REQUIRE(sot == c.expected_tris_per_leaf);

        treegen::LeafMeshOptions opts;
        opts.geometry_type         = c.g;
        opts.shape                 = treegen::LeafShape::OakLobed;
        opts.cluster_count_per_tip = c.cluster;
        opts.bend_half_angle       = 0.392699081698724f;
        auto out = treegen::build_leaf_mesh(sites, opts);
        const int actual_tris = static_cast<int>(out.indices.size() / 3);
        const int actual_per_leaf = actual_tris / static_cast<int>(sites.size());
        INFO("g=" << static_cast<int>(c.g) << " cluster=" << c.cluster
             << " sot=" << sot << " actual=" << actual_per_leaf);
        REQUIRE(actual_per_leaf == sot);
    }
}

TEST_CASE("treegen_leaf_geometry: strip angular separation is pi/N",
          "[treegen][treegen_leaf_geometry][treegen_strip_angular]") {
    // For N strips, consecutive normals should be separated by pi/N radians.
    // Strips are bilateral (visible from both sides), so pi/N gives maximum
    // angular coverage: N=2 -> 90 deg, N=3 -> 60 deg.

    auto make_skel = []() {
        treegen::TreeSkeleton skel;
        treegen::BranchNode root;
        root.position = treegen::vec3{0.0f, 0.0f, 0.0f};
        root.radius = 0.06f;
        root.depth = 0;
        root.parent_index = -1;
        skel.nodes.push_back(root);

        treegen::BranchNode child;
        child.position = treegen::vec3{0.0f, 0.0f, 1.0f};
        child.radius = 0.05f;
        child.depth = 3;
        child.parent_index = 0;
        skel.nodes.push_back(child);
        return skel;
    };

    constexpr float k_pi = 3.14159265358979323846f;

    for (int N = 2; N <= 3; ++N) {
        auto skel = make_skel();

        treegen::BranchStripOptions opts;
        opts.strip_width_m          = 0.4f;
        opts.strip_droop_angle      = 0.0f;   // no droop — keeps normals in XY plane
        opts.strip_angular_offset   = 0.0f;
        opts.strip_radius_threshold = 0.01f;
        opts.min_strips_per_segment = N;
        opts.max_strips_per_segment = N;

        auto out = treegen::build_branch_strip_mesh(skel, opts, /*min_branch_depth=*/1, 0.01f);

        // N strips * 4 verts each.
        const size_t n_verts = out.positions.size() / 3;
        REQUIRE(n_verts == static_cast<size_t>(N) * 4u);

        // Extract per-strip normal from the first vertex of each quad.
        std::vector<treegen::vec3> strip_normals;
        for (int s = 0; s < N; ++s) {
            const size_t vi = static_cast<size_t>(s) * 4;
            strip_normals.push_back(treegen::vec3{
                out.normals[vi * 3 + 0],
                out.normals[vi * 3 + 1],
                out.normals[vi * 3 + 2],
            });
        }

        // Angular separation between consecutive strip normals should be pi/N.
        const float expected_sep = k_pi / static_cast<float>(N);
        for (int s = 0; s < N - 1; ++s) {
            const float d = treegen::dot(strip_normals[static_cast<size_t>(s)],
                                         strip_normals[static_cast<size_t>(s + 1)]);
            const float actual_angle = std::acos(std::min(1.0f, std::max(-1.0f, d)));
            INFO("N=" << N << " s=" << s << " expected=" << expected_sep
                 << " actual=" << actual_angle);
            REQUIRE(std::abs(actual_angle - expected_sep) < 0.01f);
        }
    }
}

TEST_CASE("treegen_leaf_geometry: min_strips_per_segment enforced on thin branches",
          "[treegen][treegen_leaf_geometry][treegen_strip_min_count]") {
    // Thin branches (radius=0.01m, well below the 0.04m 2-strip threshold)
    // should still emit min_strips_per_segment=2 strips per segment.
    treegen::TreeSkeleton skel;
    {
        treegen::BranchNode root;
        root.position = treegen::vec3{0.0f, 0.0f, 0.0f};
        root.radius = 0.01f;  // thin — below any radius threshold
        root.depth = 0;
        root.parent_index = -1;
        skel.nodes.push_back(root);

        treegen::BranchNode child;
        child.position = treegen::vec3{0.0f, 0.0f, 1.0f};
        child.radius = 0.01f;
        child.depth = 3;
        child.parent_index = 0;
        skel.nodes.push_back(child);
    }

    treegen::BranchStripOptions opts;
    opts.strip_width_m          = 0.4f;
    opts.strip_droop_angle      = 0.0f;
    opts.strip_angular_offset   = 0.0f;
    opts.strip_radius_threshold = 0.02f;  // avg_radius=0.01 < threshold*2=0.04
    opts.min_strips_per_segment = 2;
    opts.max_strips_per_segment = 3;

    auto out = treegen::build_branch_strip_mesh(skel, opts, /*min_branch_depth=*/1, 0.02f);

    // 1 qualifying segment * 2 strips * 4 verts = 8 verts.
    const size_t n_verts = out.positions.size() / 3;
    REQUIRE(n_verts == 8u);
    // 1 qualifying segment * 2 strips * 2 tris * 3 indices = 12 indices.
    REQUIRE(out.indices.size() == 12u);
}

TEST_CASE("treegen_leaf_geometry: per-seed angular offset changes strip orientation",
          "[treegen][treegen_leaf_geometry][needle_strip]") {
    // Two calls to build_branch_strip_mesh with different strip_angular_offset
    // values must produce different vertex normals — proving the offset actually
    // rotates the strip fan.
    treegen::TreeSkeleton skel;
    {
        treegen::BranchNode root;
        root.position = treegen::vec3{0.0f, 0.0f, 0.0f};
        root.radius = 0.05f;
        root.depth = 0;
        root.parent_index = -1;
        skel.nodes.push_back(root);

        treegen::BranchNode child;
        child.position = treegen::vec3{0.0f, 0.0f, 2.0f};
        child.radius = 0.04f;
        child.depth = 3;
        child.parent_index = 0;
        skel.nodes.push_back(child);
    }

    auto make_opts = [](float angular_offset) {
        treegen::BranchStripOptions opts;
        opts.strip_width_m          = 0.4f;
        opts.strip_droop_angle      = 0.0f;
        opts.strip_angular_offset   = angular_offset;
        opts.strip_radius_threshold = 0.01f;
        opts.min_strips_per_segment = 2;
        opts.max_strips_per_segment = 2;
        return opts;
    };

    auto opts_a = make_opts(0.0f);
    auto opts_b = make_opts(1.0f);
    auto out_a = treegen::build_branch_strip_mesh(skel, opts_a, /*min_branch_depth=*/1, 0.01f);
    auto out_b = treegen::build_branch_strip_mesh(skel, opts_b, /*min_branch_depth=*/1, 0.01f);

    // Both must emit the same number of verts (same skeleton + same strip count).
    const size_t n_a = out_a.normals.size() / 3;
    const size_t n_b = out_b.normals.size() / 3;
    REQUIRE(n_a >= 4u);
    REQUIRE(n_a == n_b);

    // At least one vertex normal must differ between the two runs.
    bool any_differ = false;
    for (size_t i = 0; i < n_a && !any_differ; ++i) {
        const treegen::vec3 na{out_a.normals[i*3+0], out_a.normals[i*3+1], out_a.normals[i*3+2]};
        const treegen::vec3 nb{out_b.normals[i*3+0], out_b.normals[i*3+1], out_b.normals[i*3+2]};
        const treegen::vec3 d{na.x - nb.x, na.y - nb.y, na.z - nb.z};
        if (d.x*d.x + d.y*d.y + d.z*d.z > 1e-8f) any_differ = true;
    }
    REQUIRE(any_differ);
}
