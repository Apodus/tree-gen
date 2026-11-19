// [crown_oracle] -- the crown probe (crown-shadows S-P0a, K17): the core's ray
// queries, the oracle's footprint and region logic, the format rule, the
// low-pass, the site rule, geometric normals only, and the report's schema.
// Every expected value below comes from the construction, never from the
// probe's own output.
#include "../external/catch2/catch.hpp"
#include "generated_trees.hpp"
#include "test_env.hpp"
#include "test_support_paths.hpp"

#include "../crown_oracle.hpp"
#include "../crown_probe.hpp"
#include "../crown_volume.hpp"
#include "../det_rng.hpp"
#include "../external/gltf_view.hpp"
#include "../json_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace cr = treegen::crown;
using treegen::vec3;

namespace {

constexpr double k_pi = 3.14159265358979323846;

// A leaf primitive under construction (positions xyz, uvs, triangle list).
struct Prim {
    std::vector<float>    pos;
    std::vector<float>    uv;
    std::vector<uint32_t> idx;

    uint32_t vert(vec3 p, float u, float v) {
        pos.insert(pos.end(), {p.x, p.y, p.z});
        uv.insert(uv.end(), {u, v});
        return uint32_t(pos.size() / 3 - 1);
    }
    // A square card of side `s` centred at c spanned by unit axes a, b; UV
    // [0,1]^2 over the card. Its own 4-vertex block, 2 triangles.
    void card(vec3 c, vec3 a, vec3 b, float s) {
        const float h = 0.5f * s;
        const uint32_t v0 = vert(c - a * h - b * h, 0, 0), v1 = vert(c + a * h - b * h, 1, 0);
        const uint32_t v2 = vert(c + a * h + b * h, 1, 1), v3 = vert(c - a * h + b * h, 0, 1);
        idx.insert(idx.end(), {v0, v1, v2, v0, v2, v3});
    }
    // A bent card as the generator writes one: 6 vertices in two rows of 3,
    // 4 triangles, one connected component; centroid at c.
    void bent_card(vec3 c, vec3 a, vec3 b, float s) {
        const float h = 0.5f * s;
        uint32_t v[6];
        for (int r = 0; r < 2; ++r)
            for (int k = 0; k < 3; ++k)
                v[r * 3 + k] = vert(c + a * (h * float(k - 1)) + b * (h * float(2 * r - 1)) + vec3{0, 0, k == 1 ? 0.02f : -0.01f},
                                    0.5f * k, float(r));
        idx.insert(idx.end(), {v[0], v[1], v[4], v[0], v[4], v[3], v[1], v[2], v[5], v[1], v[5], v[4]});
    }
    void strip(vec3 c, vec3 a, vec3 b, float s) { card(c, a, b, s); }
};

vec3 unit_from(float z, float phi) {
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return vec3{r * std::cos(phi), r * std::sin(phi), z};
}
void basis(vec3 n, vec3& a, vec3& b) {
    const vec3 ref = std::abs(n.z) < 0.9f ? vec3{0, 0, 1} : vec3{1, 0, 0};
    a = treegen::normalized(treegen::cross(n, ref));
    b = treegen::cross(n, a);
}

// Square cards in a slab [0, w]^2 x [0, d]: random centres; normals uniform on
// the sphere (`horizontal`: all +z); random in-plane rotation; random winding.
cr::CrownMesh card_slab(uint64_t seed, int cards, float side, float w, float d, bool horizontal) {
    treegen::pcg32 rng;
    rng.seed(seed, 0x5AB5EEDull);
    Prim p;
    for (int i = 0; i < cards; ++i) {
        const vec3 c{rng.next_float_01() * w, rng.next_float_01() * w, rng.next_float_01() * d};
        vec3 n = horizontal ? vec3{0, 0, 1} : unit_from(2.0f * rng.next_float_01() - 1.0f, float(2 * k_pi) * rng.next_float_01());
        vec3 a, b;
        basis(n, a, b);
        const float rot = float(2 * k_pi) * rng.next_float_01();
        const vec3 ra = a * std::cos(rot) + b * std::sin(rot), rb = treegen::cross(n, ra);
        if (rng.next_float_01() < 0.5f) p.card(c, ra, rb, side); else p.card(c, rb, ra, side);
    }
    cr::CrownMesh m;
    m.add_leaves(p.pos, p.uv, p.idx);
    return m;
}

// Unoccluded share of `n_side`^2 jittered-stratified parallel rays whose lines
// cross the plane z = z0 inside [x0, x0 + span]^2.
double ray_share(const cr::Probe& probe, vec3 dir, float x0, float span, float z0, int n_side, uint64_t seed) {
    treegen::pcg32 rng;
    rng.seed(seed, 0xA11CEull);
    int open = 0;
    const float step = span / float(n_side);
    for (int i = 0; i < n_side; ++i)
        for (int j = 0; j < n_side; ++j) {
            const vec3 hit{x0 + (float(i) + rng.next_float_01()) * step, x0 + (float(j) + rng.next_float_01()) * step, z0};
            const vec3 origin = hit - dir * 100.0f;
            open += probe.occluded(origin, dir, -1) ? 0 : 1;
        }
    return double(open) / double(n_side * n_side);
}

struct SeedStats { double mean, se; };
SeedStats four_seed(const std::vector<double>& v) {
    double m = 0;
    for (double x : v) m += x;
    m /= double(v.size());
    double var = 0;
    for (double x : v) var += (x - m) * (x - m);
    var /= double(v.size() - 1);
    return {m, std::sqrt(var / double(v.size()))};
}

}  // namespace

// ---- Oracle arm (a): one layer of alpha-textured cards ----------------------

TEST_CASE("[crown_oracle] (a) one layer of alpha-textured cards: T = 1 - f c", "[treegen][crown_oracle]") {
    // Lattice period 1 m, card side sqrt(0.8) m (fill f = 0.8), each card's UV
    // [0,1]^2 over an 8 x 8 atlas whose alpha is a one-texel checker: cut-out
    // fraction c = 0.5 exactly at nearest sampling. No two cards overlap in
    // projection along any direction (one plane).
    constexpr double f = 0.8, c = 0.5;
    const double expected = 1.0 - f * c;
    REQUIRE(expected >= 0.2);
    REQUIRE(expected <= 0.8);

    Prim p;
    const float side = float(std::sqrt(f));
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) p.card(vec3{i + 0.5f, j + 0.5f, 0.0f}, vec3{1, 0, 0}, vec3{0, 1, 0}, side);
    cr::CrownMesh m;
    m.add_leaves(p.pos, p.uv, p.idx);
    m.atlas.width = m.atlas.height = 8;
    m.atlas.cutoff = 0.5f;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) m.atlas.alpha.push_back(((x + y) % 2) ? 0 : 255);
    const cr::Probe probe(std::move(m));

    // 65,536 jittered-stratified rays over the 16 x 16 periods, tilted 30
    // degrees off the card normal.
    const vec3 dir = cr::sun_direction(60.0f, 30.0f).dir;
    const double t = ray_share(probe, dir, 0.0f, 16.0f, 0.0f, 256, 7);
    INFO("T = " << t << ", expected " << expected);
    CHECK(std::abs(t - expected) <= 0.005);
}

// ---- Oracle arm (b): randomly oriented opaque cards --------------------------

TEST_CASE("[crown_oracle] (b) random-orientation slab: T = exp(-0.5 L / sin e)", "[treegen][crown_oracle]") {
    // 36,864 cards of side 0.125 m in 24 x 24 x 2 m: L = 36864 * 0.015625 / 576 = 1.
    constexpr int cards = 36864;
    constexpr float side = 0.125f;
    const double L = cards * double(side) * side / (24.0 * 24.0);
    REQUIRE(L == 1.0);
    std::vector<double> t30, t70;
    for (uint64_t seed = 1; seed <= 4; ++seed) {
        const cr::Probe probe(card_slab(seed, cards, side, 24.0f, 2.0f, false));
        // Central 12 x 12 m at the mid-plane: a ray at 30 degrees crosses 3.5 m
        // horizontally inside the slab, so it stays inside it.
        t30.push_back(ray_share(probe, cr::sun_direction(30.0f, 0.0f).dir, 6.0f, 12.0f, 1.0f, 128, 100 + seed));
        t70.push_back(ray_share(probe, cr::sun_direction(70.0f, 0.0f).dir, 6.0f, 12.0f, 1.0f, 128, 200 + seed));
    }
    for (const auto& [e, v] : std::vector<std::pair<double, std::vector<double>*>>{{30.0, &t30}, {70.0, &t70}}) {
        const double expected = std::exp(-0.5 * L / std::sin(e * k_pi / 180.0));
        REQUIRE(expected >= 0.2);
        REQUIRE(expected <= 0.8);
        const SeedStats s = four_seed(*v);
        INFO("e = " << e << ": four-seed mean " << s.mean << " (se " << s.se << "), expected " << expected);
        CHECK(std::abs(s.mean - expected) <= 0.05 * expected);
        CHECK(0.05 * expected >= 3.0 * s.se);   // the band is at least three standard errors
    }
    // (b) varies with elevation.
    CHECK(four_seed(t70).mean - four_seed(t30).mean > 0.1);
}

// ---- Oracle arm (c): horizontal opaque cards ----------------------------------

TEST_CASE("[crown_oracle] (c) horizontal slab: T = exp(-L) at every elevation", "[treegen][crown_oracle]") {
    constexpr int cards = 36864;
    constexpr float side = 0.125f;
    const double L = cards * double(side) * side / (24.0 * 24.0);
    const double expected = std::exp(-L);
    REQUIRE(expected >= 0.2);
    REQUIRE(expected <= 0.8);
    const float elevations[3] = {30.0f, 45.0f, 70.0f};
    std::vector<double> t[3];
    for (uint64_t seed = 1; seed <= 4; ++seed) {
        const cr::Probe probe(card_slab(seed, cards, side, 24.0f, 2.0f, true));
        for (int k = 0; k < 3; ++k)
            t[k].push_back(ray_share(probe, cr::sun_direction(elevations[k], 0.0f).dir, 6.0f, 12.0f, 1.0f, 128, 300 + seed));
    }
    std::vector<double> means;
    for (int k = 0; k < 3; ++k) {
        const float e = elevations[k];
        const SeedStats s = four_seed(t[k]);
        INFO("e = " << e << ": four-seed mean " << s.mean << " (se " << s.se << "), expected " << expected);
        CHECK(std::abs(s.mean - expected) <= 0.05 * expected);
        CHECK(0.05 * expected >= 3.0 * s.se);
        means.push_back(s.mean);
    }
    // (c) does not vary with elevation.
    CHECK(*std::max_element(means.begin(), means.end()) - *std::min_element(means.begin(), means.end()) <= 0.05 * expected);
}

// ---- Saturation honesty ---------------------------------------------------------

TEST_CASE("[crown_oracle] saturated slab: not measurable, body opaque, no slope", "[treegen][crown_oracle]") {
    // Horizontal 0.5 m cards, leaf area index 6 in 12 x 12 x 1 m: T = exp(-6)
    // = 0.0025 through the slab at every elevation (under one ray in 64).
    constexpr float side = 0.5f, w = 12.0f;
    const int cards = int(6.0 * w * w / (side * side));
    const cr::Probe probe(card_slab(11, cards, side, w, 1.0f, true));
    const cr::CrownVolume volume = cr::CrownVolume::build(probe, cr::k_format_cells);
    std::vector<cr::FormatElevation> fmt;
    for (const float e : {30.0f, 45.0f, 60.0f, 90.0f}) {
        const cr::FootprintSet set = cr::measure_footprints(probe, cr::sun_direction(e, 0.0f), cr::PlaneKind::beam);
        // The slab's interior: lines that cross the mid-plane at least 3 m from
        // its sides (the sides of a finite slab transmit through its corners).
        const vec3 s = set.sun.dir;
        cr::FormatElevation fe;
        cr::add_format_footprints(probe, set, volume, fe, [&](const cr::Footprint& f) {
            const vec3 q = f.centre + s * ((0.5f - f.centre.z) / s.z);
            return q.x >= 3.0f && q.x <= w - 3.0f && q.y >= 3.0f && q.y <= w - 3.0f;
        });
        INFO("e = " << e << ": unsaturated interior footprints " << fe.unsaturated.size());
        CHECK(int(fe.unsaturated.size()) < cr::k_min_unsaturated);
        CHECK(fe.silhouette >= 40);   // the region is populated
        fmt.push_back(std::move(fe));
        const auto body = cr::body_stats(set, cr::k_body_edge_m);
        REQUIRE(body.has_value());
        CHECK(body->share_opaque >= 0.5f);   // body opaque at this elevation
    }
    const cr::FormatDecision d = cr::decide_format(fmt);
    CHECK(d.decision == "undecidable");
    for (const auto& r : d.rows) {
        CHECK_FALSE(r.measurable);
        CHECK(r.g_eff == 0.0f);   // no slope is fitted (the report writes null)
    }
    CHECK(d.g == 0.0f);
}

// ---- Format rule on synthetic volumes ----------------------------------------------

namespace {
cr::FormatDecision format_on_slab(bool horizontal) {
    // The format rule's subject is the two candidate grids and the decision,
    // not the oracle (arms (a)-(c) gate that). So the "oracle" transmittance is
    // the medium's own: leaf area index L in a slab of thickness d crossed at
    // elevation e has T = exp(-L G / sin e), G = 0.5 for random orientation and
    // sin e for horizontal cards. The grids are splatted from real cards: one
    // card per cell of a 128 x 128 lattice over 16 x 16 x 1 m (L = 1), jittered
    // inside its cell, so every 1 m grid column (N = 16) holds ~64 cards and a
    // random medium's count noise does not reach the comparison. (Cells must
    // stay thin against the slab: at N = 8 the 2 m z-cells smear the 1 m slab
    // and a 10-degree line leaves the grid's x range before it leaves them.)
    constexpr float side = 0.125f, w = 16.0f;
    constexpr int n = 128;
    const double L = double(n) * n * side * side / (double(w) * w);
    treegen::pcg32 rng;
    rng.seed(horizontal ? 21 : 22, 0x5AB5EEDull);
    Prim p;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const vec3 c{(i + rng.next_float_01()) * w / n, (j + rng.next_float_01()) * w / n, rng.next_float_01()};
            const vec3 nrm = horizontal ? vec3{0, 0, 1}
                                        : unit_from(2.0f * rng.next_float_01() - 1.0f, float(2 * k_pi) * rng.next_float_01());
            vec3 a, b;
            basis(nrm, a, b);
            const float rot = float(2 * k_pi) * rng.next_float_01();
            const vec3 ra = a * std::cos(rot) + b * std::sin(rot);
            p.card(c, ra, treegen::cross(nrm, ra), side);
        }
    cr::CrownMesh m;
    m.add_leaves(p.pos, p.uv, p.idx);
    const cr::Probe probe(std::move(m));
    const cr::CrownVolume volume = cr::CrownVolume::build(probe, 16);
    // The splat conserves area: sum of rho * cell volume = the cards' area.
    CHECK(std::abs(volume.total_area_m2() - L * w * w) <= 1e-3 * L * w * w);

    std::vector<cr::FormatElevation> fmt;
    for (const float e : {10.0f, 20.0f, 30.0f, 45.0f, 60.0f, 70.0f, 75.0f, 90.0f}) {
        const vec3 s = cr::sun_direction(e, 0.0f).dir;
        const double g = horizontal ? std::sin(e * k_pi / 180.0) : 0.5;
        cr::FormatElevation fe;
        fe.elevation_deg = e;
        // 8 x 8 footprints of 0.5 m on the mid-plane over [6, 10]^2: at 10
        // degrees a line spans 2.84 m either side, inside the slab.
        for (int fi = 0; fi < 8; ++fi)
            for (int fj = 0; fj < 8; ++fj) {
                cr::FormatFootprint f;
                f.t_oracle = float(std::exp(-L * g / std::sin(e * k_pi / 180.0)));
                for (int a = 0; a < 8; ++a)
                    for (int b = 0; b < 8; ++b) {
                        const vec3 q{6.0f + 0.5f * fi + (a + 0.5f) / 16.0f, 6.0f + 0.5f * fj + (b + 0.5f) / 16.0f, 0.5f};
                        f.integrals.push_back(volume.integrals(q - s * 50.0f, s));
                    }
                fe.unsaturated.push_back(std::move(f));
            }
        fe.silhouette = int(fe.unsaturated.size());
        fmt.push_back(std::move(fe));
    }
    return cr::decide_format(fmt);
}
}  // namespace

TEST_CASE("[crown_oracle] format rule: random orientation decides scalar, G_eff 0.5", "[treegen][crown_oracle]") {
    const cr::FormatDecision d = format_on_slab(false);
    for (const auto& r : d.rows) {
        INFO("e " << r.elevation_deg << " n " << r.unsaturated << " scalar " << r.scalar_mae << " axis " << r.axis_mae
                  << " g_eff " << r.g_eff);
        CHECK(r.measurable);
        CHECK(std::abs(r.g_eff - 0.5f) <= 0.03f);
    }
    CHECK(d.decision == "scalar");
}

TEST_CASE("[crown_oracle] format rule: horizontal cards decide axis-projected, G_eff sin(e)", "[treegen][crown_oracle]") {
    const cr::FormatDecision d = format_on_slab(true);
    for (const auto& r : d.rows) {
        INFO("e " << r.elevation_deg << " n " << r.unsaturated << " scalar " << r.scalar_mae << " axis " << r.axis_mae
                  << " g_eff " << r.g_eff);
        CHECK(r.measurable);
        CHECK(std::abs(r.g_eff - float(std::sin(r.elevation_deg * k_pi / 180.0))) <= 0.03f);
        CHECK(r.axis_mae <= cr::k_format_max_mae);
    }
    CHECK(d.decision == "axis_projected");
    CHECK(d.clumping == 1.0f);
}

// ---- Region logic ---------------------------------------------------------------------

namespace {
// An opaque disc (bark: a 96-segment fan) of radius r centred at c, normal n.
cr::Probe disc(vec3 c, vec3 n, float r) {
    vec3 a, b;
    basis(n, a, b);
    std::vector<float> pos{c.x, c.y, c.z};
    std::vector<uint32_t> idx;
    constexpr int seg = 96;
    for (int k = 0; k < seg; ++k) {
        const float ph = float(2 * k_pi) * float(k) / seg;
        const vec3 p = c + a * (r * std::cos(ph)) + b * (r * std::sin(ph));
        pos.insert(pos.end(), {p.x, p.y, p.z});
        idx.insert(idx.end(), {0u, uint32_t(1 + k), uint32_t(1 + (k + 1) % seg)});
    }
    cr::CrownMesh m;
    m.add_bark(pos, idx);
    return cr::Probe(std::move(m));
}

void check_disc_regions(float elevation, float radius) {
    const cr::SunDirection sun = cr::sun_direction(elevation, 0.0f);
    const vec3 c{0.3f, -0.2f, 6.0f};
    const cr::Probe probe = disc(c, sun.dir, radius);
    const cr::FootprintSet set = cr::measure_footprints(probe, sun, cr::PlaneKind::beam);
    int checked = 0, body = 0;
    for (const cr::Footprint& f : set.footprints) {
        // r: the footprint's line's distance from the disc centre, in metres
        // perpendicular to the sun (the beam metric, whatever plane was used).
        const vec3 q = f.centre - c;
        const float r = treegen::length(q - sun.dir * treegen::dot(q, sun.dir));
        if (r > radius - 0.36f) continue;   // fully inside the disc: T = 0
        REQUIRE(f.silhouette());
        // Outside centres (T >= 0.98) lie 0.2-0.36 m beyond the edge, and the
        // nearest lattice point to a radial line is within half a diagonal
        // (0.354 m): R - r + 0.2 <= d <= R - r + 0.714.
        INFO("e " << elevation << " r " << r << " d " << f.edge_distance);
        CHECK(f.edge_distance - (radius - r) >= -0.01f);
        CHECK(f.edge_distance - (radius - r) <= 0.75f);
        if (radius - r < 0.25f) CHECK_FALSE(f.body());
        if (radius - r >= 1.0f) CHECK(f.body());
        ++checked;
        body += f.body() ? 1 : 0;
    }
    CHECK(checked >= 70);
    CHECK(body >= 25);
}
}  // namespace

TEST_CASE("[crown_oracle] region logic: rim and body by edge distance in the beam plane", "[treegen][crown_oracle]") {
    check_disc_regions(90.0f, 3.0f);   // perpendicular to a zenith sun
    check_disc_regions(45.0f, 3.0f);   // tilted 45 degrees, perpendicular to a 45-degree sun

    // Degenerate: a crown with no footprint 1 m from an outside centre has no
    // body; the report says null. (Outside centres lie up to 0.714 m beyond the
    // edge, so that needs a radius under ~0.29 m, not "under 2 m across": a
    // 1.8 m disc has body.)
    const cr::Probe small = disc(vec3{0, 0, 6}, vec3{0, 0, 1}, 0.25f);
    const cr::FootprintSet set = cr::measure_footprints(small, cr::sun_direction(90.0f, 0.0f), cr::PlaneKind::beam);
    CHECK_FALSE(cr::body_stats(set, cr::k_body_edge_m).has_value());
    cr::ProbeOptions o;
    o.elevations_deg = {90.0f};
    o.azimuths = 1;
    const std::string doc = cr::probe_report_json(small, o, "{}");
    CHECK(doc.find("\"body\": null") != std::string::npos);
    CHECK(doc.find("\"body\": {") == std::string::npos);
}

// ---- Low-pass ----------------------------------------------------------------------------

TEST_CASE("[crown_oracle] low-pass: identity at 0, Gaussian second moment, conserved", "[treegen][crown_oracle]") {
    // One filled cell (all four channels 1 m2/m3) at the centre of 9^3 cells of 0.2 m.
    constexpr int n = 9;
    constexpr float cell = 0.2f;
    std::vector<float> cells(size_t(n) * n * n * 4, 0.0f);
    const size_t centre = ((size_t(4) * n + 4) * n + 4) * 4;
    for (int ch = 0; ch < 4; ++ch) cells[centre + size_t(ch)] = 1.0f;
    const vec3 origin{-0.9f, -0.9f, -0.9f};   // the filled cell spans [-0.1, 0.1]^3
    const cr::CrownVolume v = cr::CrownVolume::from_cells(origin, cell, n, n, n, cells);
    const cr::ExtinctionModel axis{};
    const vec3 dir{1, 0, 0};

    // sigma 0: the unfiltered march, bit for bit.
    for (const float y : {-0.05f, 0.0f, 0.07f})
        CHECK(v.optical_depth(vec3{-5.0f, y, 0.01f}, dir, 0.0f, axis) == v.integrals(vec3{-5.0f, y, 0.01f}, dir)[1]);

    // sigma 0.6 m: the optical-depth profile over the ray offsets (y, z).
    constexpr float sigma = 0.6f;
    const double step = 0.05;
    double m0 = 0, my2 = 0, mz2 = 0;
    for (int iy = -60; iy <= 60; ++iy)
        for (int iz = -60; iz <= 60; ++iz) {
            const double y = iy * step, z = iz * step;
            const double tau = v.optical_depth(vec3{-10.0f, float(y), float(z)}, dir, sigma, axis);
            m0 += tau; my2 += tau * y * y; mz2 += tau * z * z;
        }
    const double expected_var = double(sigma) * sigma + double(cell) * cell / 12.0;
    const double integral = m0 * step * step;
    INFO("var y " << my2 / m0 << " z " << mz2 / m0 << " expected " << expected_var << "; integral " << integral);
    CHECK(std::abs(my2 / m0 - expected_var) <= 0.05 * expected_var);
    CHECK(std::abs(mz2 / m0 - expected_var) <= 0.05 * expected_var);
    // Unfiltered, the integral of optical depth over the offsets is rho * cell^3.
    CHECK(std::abs(integral - double(cell) * cell * cell) <= 0.01 * double(cell) * cell * cell);
}

// ---- Core self-test --------------------------------------------------------------------------

TEST_CASE("[crown_oracle] core: skip_site skips exactly that site; segments are finite", "[treegen][crown_oracle]") {
    // A bark triangle first (off the ray), then site 0 at z = 1 and site 1 at
    // z = 2 above the origin, and site 2 at z = 1 beside it.
    cr::CrownMesh m;
    const std::vector<float> bark{10, 10, 0, 11, 10, 0, 10, 11, 0};
    const std::vector<uint32_t> bark_idx{0, 1, 2};
    m.add_bark(bark, bark_idx);
    Prim p;
    p.card(vec3{0, 0, 1}, vec3{1, 0, 0}, vec3{0, 1, 0}, 1.0f);
    p.card(vec3{0, 0, 2}, vec3{1, 0, 0}, vec3{0, 1, 0}, 1.0f);
    p.card(vec3{5, 0, 1}, vec3{1, 0, 0}, vec3{0, 1, 0}, 1.0f);
    m.add_leaves(p.pos, p.uv, p.idx);
    REQUIRE(m.sites.size() == 3u);
    REQUIRE(m.triangles[m.sites[0].triangles[0]].site == 0);
    REQUIRE(m.sites[0].triangles[0] != 0u);   // site ids and triangle indices differ
    const cr::Probe probe(std::move(m));
    const vec3 up{0, 0, 1};
    CHECK(probe.occluded(vec3{0, 0, 0}, up, -1));
    CHECK(probe.occluded(vec3{0, 0, 0}, up, 0));       // site 1 still blocks
    CHECK(probe.occluded(vec3{0, 0, 1.5f}, up, -1));   // site 1 above
    CHECK_FALSE(probe.occluded(vec3{0, 0, 1.5f}, up, 1));
    CHECK_FALSE(probe.occluded(vec3{0, 0, 0}, vec3{-1, 0, 0}, -1));
    CHECK_FALSE(probe.occluded(vec3{5, 0, 0}, up, 2));  // only site 2 is on this ray
    CHECK(probe.occluded(vec3{5, 0, 0}, up, 0));
    // Segments: a point in front of the card is visible, behind it is not.
    CHECK_FALSE(probe.occluded_segment(vec3{0, 0, 0}, vec3{0, 0, 0.5f}, -1));
    CHECK(probe.occluded_segment(vec3{0, 0, 0}, vec3{0, 0, 1.5f}, -1));
    CHECK_FALSE(probe.occluded_segment(vec3{0, 0, 0}, vec3{0, 0, 1.5f}, 0));
}

// ---- Site rule (K17) -------------------------------------------------------------------------

TEST_CASE("[crown_oracle] site rule: crossed bent cards and strips", "[treegen][crown_oracle]") {
    Prim p;
    const vec3 x{1, 0, 0}, y{0, 1, 0}, z{0, 0, 1};
    for (int s = 0; s < 3; ++s) {
        const vec3 c{float(s), 0.5f * s, 3.0f};
        p.bent_card(c, x, z, 0.6f);   // two crossed cards, back to back
        p.bent_card(c, y, z, 0.6f);
    }
    p.strip(vec3{5, 0, 3}, x, y, 0.4f);
    p.strip(vec3{6, 0, 3}, x, y, 0.4f);
    const cr::SiteRule r = cr::derive_sites(p.pos, p.idx);
    std::map<int, int> hist;
    for (const auto& s : r.sites) ++hist[s.cards];
    CHECK(r.sites.size() == 5u);
    CHECK(hist == std::map<int, int>{{1, 2}, {2, 3}});

    // Two non-adjacent sites at one centroid stay two sites.
    Prim q;
    q.bent_card(vec3{0, 0, 3}, x, z, 0.6f);
    q.bent_card(vec3{0, 0, 3}, y, z, 0.6f);
    q.bent_card(vec3{2, 0, 3}, x, z, 0.6f);
    q.bent_card(vec3{0, 0, 3}, x, z, 0.6f);
    const cr::SiteRule r2 = cr::derive_sites(q.pos, q.idx);
    REQUIRE(r2.sites.size() == 3u);
    CHECK(r2.sites[0].cards == 2);
    CHECK(r2.sites[2].cards == 1);
}

TEST_CASE("[crown_oracle] site rule: c3_oak 42 L0 sites equal the emitter's kept sites", "[treegen][crown_oracle]") {
    const treegen::GeneratedTree& g = rynx::test_support::generated_tree("oak", 42);
    const cr::LoadedCrown c = cr::load_crown_from_glb(g.glb, 0);
    const size_t kept = g.lods[0].leaf_kept_sites.size();
    INFO("sites " << c.mesh.sites.size() << " kept " << kept);
    REQUIRE(kept > 100u);
    CHECK(c.mesh.sites.size() == kept);
    for (const auto& s : c.mesh.sites) CHECK(s.cards == 2);   // crossed pair per site (the oak's cluster count)
}

// ---- Geometric normals only (K16) -------------------------------------------------------------

namespace {
cr::ProbeOptions one_direction() {
    cr::ProbeOptions o;
    o.elevations_deg = {45.0f};
    o.azimuths = 1;
    return o;
}
std::string strip_input(const std::string& doc) {
    const size_t a = doc.find("\"input\":"), b = doc.find("\"tree\":");
    return doc.substr(0, a) + doc.substr(b);
}
}  // namespace

TEST_CASE("[crown_oracle] the vertex NORMAL never reaches the report (K16)", "[treegen][crown_oracle]") {
    namespace gv = rynx::graphics::gltf_view;
    const treegen::GeneratedTree& g = rynx::test_support::generated_tree("oak", 42);
    std::vector<uint8_t> bytes = g.glb;

    // Overwrite every leaf primitive's NORMAL accessor, in place, with
    // arbitrary unit vectors.
    const std::span<const char> view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    const gv::glb_view glb = gv::parse_glb_header(view);
    REQUIRE(glb.valid);
    gv::jparser parser(glb.json_begin, glb.json_len);
    const gv::jval root = parser.parse_value();
    REQUIRE(parser.good());
    const size_t bin_offset = size_t(glb.bin_begin - view.data());
    int overwritten = 0;
    treegen::pcg32 rng;
    rng.seed(5, 9);
    for (const gv::jval& mesh : root.find("meshes")->as_arr())
        for (const gv::jval& prim : mesh.find("primitives")->as_arr()) {
            gv::pbr_material_desc mat;
            REQUIRE(gv::extract_pbr_material(root, prim.find("material")->as_int(), mat, nullptr));
            if (mat.alpha_mode != 1) continue;
            gv::accessor acc;
            REQUIRE(gv::resolve_accessor(root, prim.find("attributes")->find("NORMAL")->as_int(), acc));
            REQUIRE(acc.components == 3);
            for (uint32_t i = 0; i < acc.count; ++i) {
                const vec3 n = unit_from(2.0f * rng.next_float_01() - 1.0f, 6.2831853f * rng.next_float_01());
                const float f3[3] = {n.x, n.y, n.z};
                std::memcpy(bytes.data() + bin_offset + acc.first_byte() + size_t(i) * acc.stride(), f3, 12);
            }
            ++overwritten;
        }
    REQUIRE(overwritten == 3);
    REQUIRE(bytes != g.glb);

    const cr::ProbeOptions o = one_direction();
    const std::string a = cr::probe_glb_report(g.glb, 0, o, "{}");
    const std::string b = cr::probe_glb_report(bytes, 0, o, "{}");
    CHECK(a.find("\"format\": {") != std::string::npos);   // the axis grid (normals' consumer) is in the report
    CHECK(a == b);
}

// ---- What is measured is what ships ----------------------------------------------------------

TEST_CASE("[crown_oracle] scenario path and --glb of the written file agree", "[crown_oracle]") {   // spawns the tool
    namespace fs = std::filesystem;
    const fs::path exe = treegen::test_env::treegen_exe();
    for (const char* species : {"oak", "pine"}) {
        INFO("species " << species);
        const fs::path glb = treegen::test_env::scratch(std::string("probe_") + species + ".glb");
        const fs::path a = treegen::test_env::scratch(std::string("probe_scn_") + species + ".json");
        const fs::path b = treegen::test_env::scratch(std::string("probe_glb_") + species + ".json");
        const std::string flags = " --sun-elevations=45 --sun-azimuths=1 --eyes=0,-12,1.7";
        std::ostringstream s1, s2;
        s1 << "--scenario=\"" << rynx::test_support::treegen_file(std::string("scenarios/c3_") + species + ".json")
           << "\" --seed=42 --out=\"" << glb.string() << "\" --probe-crown=\"" << a.string() << "\"" << flags;
        s2 << "--glb=\"" << glb.string() << "\" --probe-crown=\"" << b.string() << "\"" << flags;
        REQUIRE(rynx::test_support::run_exe_in_dir(exe.string(), glb.parent_path().string(), s1.str()) == 0);
        REQUIRE(rynx::test_support::run_exe_in_dir(exe.string(), glb.parent_path().string(), s2.str()) == 0);
        auto slurp = [](const fs::path& p) {
            std::ifstream f(p, std::ios::binary);
            return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        };
        const std::string da = slurp(a), db = slurp(b);
        REQUIRE(da.find("\"sunlit\": {") != std::string::npos);
        REQUIRE(da.find("\"oracle\": {") != std::string::npos);
        CHECK(da.find("\"source\": \"scenario\"") != std::string::npos);
        CHECK(db.find("\"source\": \"glb\"") != std::string::npos);
        CHECK(strip_input(da) == strip_input(db));
    }
}

// ---- Schema (K17 Output) -------------------------------------------------------------------------

namespace {
void key_paths(const treegen::json::value& v, const std::string& prefix, std::set<std::string>& out) {
    using K = treegen::json::value::kind_t;
    if (v.kind == K::obj_t) {
        for (const auto& [k, child] : v.o) {
            const std::string p = prefix.empty() ? k : prefix + "." + k;
            // Numeric-keyed maps (cards_per_site) are data, not fields.
            if (!k.empty() && std::isdigit(static_cast<unsigned char>(k[0]))) { out.insert(prefix + ".<n>"); continue; }
            out.insert(p);
            key_paths(child, p, out);
        }
    } else if (v.kind == K::arr_t) {
        for (const auto& e : v.a) key_paths(e, prefix + "[]", out);
    }
}

// The schema-1 field list (names; units are in the names). Adding a field
// adds a line here; renaming or repurposing one bumps k_report_schema.
const std::set<std::string> k_schema_1 = {
    "schema", "input", "input.source", "input.scenario", "input.seed", "input.probe_lod", "tree",
    "tree.bark_triangles", "tree.leaf_triangles", "tree.sites", "tree.cards_per_site", "tree.cards_per_site.<n>",
    "tree.alpha_coverage", "tree.card_area_m2", "tree.opaque_leaf_area_m2", "tree.crown_ground_projected_area_m2",
    "tree.leaf_area_index", "tree.leaf_aabb_min_m", "tree.leaf_aabb_max_m", "tree.leaf_area_density_m2_per_m3",
    "tree.trunk_base_z_m", "tree.lowest_leaf_height_m",
    "modes", "modes.oracle", "modes.oracle.footprint_m", "modes.oracle.rays_per_footprint", "modes.oracle.trunk_axis",
    "modes.oracle.directions",
    "modes.oracle.directions[].elevation_deg", "modes.oracle.directions[].azimuth_deg",
    "modes.oracle.directions[].footprints", "modes.oracle.directions[].silhouette", "modes.oracle.directions[].rim",
    "modes.oracle.directions[].body", "modes.oracle.directions[].body.footprints", "modes.oracle.directions[].body.mean",
    "modes.oracle.directions[].body.p50", "modes.oracle.directions[].body.p95",
    "modes.oracle.directions[].body.share_below_0_05", "modes.oracle.directions[].body_opaque",
    "modes.oracle.directions[].body_mean_by_min_edge_m", "modes.oracle.directions[].body_mean_by_min_edge_m.<n>",
    "modes.oracle.directions[].t_by_edge_distance", "modes.oracle.directions[].t_by_edge_distance[].d_min_m",
    "modes.oracle.directions[].t_by_edge_distance[].d_max_m", "modes.oracle.directions[].t_by_edge_distance[].footprints",
    "modes.oracle.directions[].t_by_edge_distance[].mean",
    "modes.oracle.directions[].crossings", "modes.oracle.directions[].crossings.silhouette",
    "modes.oracle.directions[].crossings.silhouette.geometric", "modes.oracle.directions[].crossings.silhouette.opaque",
    "modes.oracle.directions[].crossings.body", "modes.oracle.directions[].crossings.body.geometric",
    "modes.oracle.directions[].crossings.body.opaque",
    "modes.oracle.directions[].ground_plane", "modes.oracle.directions[].ground_plane.footprints",
    "modes.oracle.directions[].ground_plane.silhouette", "modes.oracle.directions[].ground_plane.body",
    "modes.oracle.directions[].ground_plane.body.footprints", "modes.oracle.directions[].ground_plane.body.mean",
    "modes.oracle.directions[].ground_plane.body.p50", "modes.oracle.directions[].ground_plane.body.p95",
    "modes.oracle.directions[].ground_plane.body.share_below_0_05",
    "modes.oracle.directions[].ground_plane.body_mean_by_min_edge_m",
    "modes.oracle.directions[].ground_plane.body_mean_by_min_edge_m.<n>",
    "modes.oracle.directions[].t_by_trunk_distance", "modes.oracle.directions[].t_by_trunk_distance[].r_min_m",
    "modes.oracle.directions[].t_by_trunk_distance[].r_max_m",
    "modes.oracle.directions[].t_by_trunk_distance[].footprints", "modes.oracle.directions[].t_by_trunk_distance[].mean",
    "modes.oracle.directions[].sunlit_by_depth", "modes.oracle.directions[].sunlit_by_depth[].depth_min_m",
    "modes.oracle.directions[].sunlit_by_depth[].depth_max_m", "modes.oracle.directions[].sunlit_by_depth[].sites",
    "modes.oracle.directions[].sunlit_by_depth[].sunlit_share",
    "modes.oracle.stop_rule", "modes.oracle.stop_rule.body_opaque_by_elevation",
    "modes.oracle.stop_rule.body_opaque_by_elevation[].elevation_deg",
    "modes.oracle.stop_rule.body_opaque_by_elevation[].body_opaque_every_azimuth",
    "modes.oracle.stop_rule.crown_opaque_to_direct_sun",
    "modes.oracle.format", "modes.oracle.format.cells_longest_axis", "modes.oracle.format.rays_per_footprint",
    "modes.oracle.format.cell_m",
    "modes.oracle.format.decision", "modes.oracle.format.g", "modes.oracle.format.clumping", "modes.oracle.format.rows",
    "modes.oracle.format.rows[].elevation_deg", "modes.oracle.format.rows[].unsaturated_footprints",
    "modes.oracle.format.rows[].measurable", "modes.oracle.format.rows[].saturated_share",
    "modes.oracle.format.rows[].scalar_mae", "modes.oracle.format.rows[].axis_mae", "modes.oracle.format.rows[].g_eff",
    "modes.sunlit", "modes.sunlit.sites", "modes.sunlit.open_sky", "modes.sunlit.open_sky.directions",
    "modes.sunlit.open_sky.mean", "modes.sunlit.open_sky.p10", "modes.sunlit.open_sky.p50", "modes.sunlit.open_sky.p90",
    "modes.sunlit.directions", "modes.sunlit.directions[].elevation_deg", "modes.sunlit.directions[].azimuth_deg",
    "modes.sunlit.directions[].share", "modes.sunlit.directions[].quartiles", "modes.sunlit.directions[].deciles",
    "modes.sunlit.directions[].eyes", "modes.sunlit.directions[].eyes[].eye_m",
    "modes.sunlit.directions[].eyes[].visible_sites", "modes.sunlit.directions[].eyes[].share",
};

std::set<std::string> schema_mismatch(const std::string& doc) {
    const auto parsed = treegen::json::parse(doc);
    if (!parsed.ok) return {"<unparseable>"};
    std::set<std::string> got;
    key_paths(parsed.root, "", got);
    std::set<std::string> diff;
    for (const auto& k : got) if (!k_schema_1.count(k)) diff.insert("+" + k);
    for (const auto& k : k_schema_1) if (!got.count(k)) diff.insert("-" + k);
    return diff;
}
}  // namespace

TEST_CASE("[crown_oracle] report schema 1: the exact field list", "[treegen][crown_oracle]") {
    const treegen::GeneratedTree& g = rynx::test_support::generated_tree("oak", 42);
    cr::ProbeOptions o;
    o.elevations_deg = {90.0f};
    o.azimuths = 1;
    o.eyes = {vec3{0, -12, 1.7f}};
    const std::string doc = cr::probe_glb_report(g.glb, 0, o, cr::probe_input_scenario("c3_oak.json", 42, 0));
    REQUIRE(doc.find("\"schema\": 1,") != std::string::npos);
    REQUIRE(cr::k_report_schema == 1);
    const auto diff = schema_mismatch(doc);
    for (const auto& d : diff) UNSCOPED_INFO(d);
    CHECK(diff.empty());

    // Positive control: one field removed is caught.
    std::string cut = doc;
    const size_t at = cut.find("\"leaf_area_index\": ");
    REQUIRE(at != std::string::npos);
    cut.erase(at, cut.find('\n', at) - at + 1);
    CHECK(schema_mismatch(cut) == std::set<std::string>{"-tree.leaf_area_index"});
}
