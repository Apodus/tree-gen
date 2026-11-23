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
#include "../generate_tree.hpp"
#include "../glb_writer.hpp"
#include "../json_reader.hpp"
#include "../png_encoder.hpp"
#include "../scenario.hpp"

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
constexpr cr::Occluders kAll    = cr::Occluders::leaves_and_bark;
constexpr cr::Occluders kLeaves = cr::Occluders::leaves_only;

// The report's GLB digest computed here, independently of crown::glb_digest:
// fnv1a64 (det_rng.hpp) over every byte, "0x" and 16 hex digits.
std::string expected_digest(std::string_view bytes) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "0x%016llx", static_cast<unsigned long long>(treegen::fnv1a64(bytes)));
    return buf;
}
std::string expected_digest(const std::vector<uint8_t>& b) {
    return expected_digest(std::string_view(reinterpret_cast<const char*>(b.data()), b.size()));
}

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
            open += probe.occluded(origin, dir, -1, kAll) ? 0 : 1;
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

TEST_CASE("[crown_oracle] (a') an asymmetric alpha band: T, by half-card, and per-triangle coverage", "[treegen][crown_oracle]") {
    // One 1 m card at z = 0, UV [0,1]^2 (v along +y). An 8 x 8 atlas whose rows
    // 0-2 (v < 0.375) hold alpha 128 and the rest 127, cutoff 0.5 (passes at
    // >= 127.5): the opaque band is 3/8 of the card, on ONE side of it, and the
    // texel values sit on either side of the cutoff. Expected from the
    // construction: T over the card 1 - 3/8; over its v < 0.5 half 1 - 3/4; the
    // triangle (0,0)(1,0)(1,1) covers 21 of its 36 texel centres opaque, the
    // triangle (0,0)(1,1)(0,1) 6 of 36. An inverted test, a changed cutoff and a
    // v-flip each change at least one of them.
    Prim p;
    p.card(vec3{0, 0, 0}, vec3{1, 0, 0}, vec3{0, 1, 0}, 1.0f);
    cr::CrownMesh m;
    m.add_leaves(p.pos, p.uv, p.idx);
    m.atlas.width = m.atlas.height = 8;
    m.atlas.cutoff = 0.5f;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) m.atlas.alpha.push_back(y < 3 ? 128 : 127);
    const cr::Probe probe(std::move(m));
    REQUIRE(probe.mesh().triangles.size() == 2u);
    CHECK(std::abs(probe.triangle_coverage(0) - 21.0f / 36.0f) <= 1e-6f);
    CHECK(std::abs(probe.triangle_coverage(1) - 6.0f / 36.0f) <= 1e-6f);

    // Zenith rays, 64 x 64 strata over [x0, x0 + 1] x [y0, y0 + h]; strata edges
    // fall on the band edge (y = -0.125 = 24/64 - 0.5), so the shares are exact.
    auto share = [&](float y0, float h) {
        treegen::pcg32 rng;
        rng.seed(3, 0xA11CEull);
        int open = 0, n = 0;
        for (int i = 0; i < 64; ++i)
            for (int j = 0; j < 64; ++j) {
                const vec3 o{-0.5f + (i + rng.next_float_01()) / 64.0f, y0 + (j + rng.next_float_01()) * h / 64.0f, -1.0f};
                open += probe.occluded(o, vec3{0, 0, 1}, -1, kAll) ? 0 : 1;
                ++n;
            }
        return double(open) / n;
    };
    // One ray of 4096 of slack for barycentric rounding at the band edge.
    CHECK(std::abs(share(-0.5f, 1.0f) - (1.0 - 3.0 / 8.0)) <= 1.0 / 4096);
    CHECK(std::abs(share(-0.5f, 0.5f) - (1.0 - 3.0 / 4.0)) <= 1.0 / 4096);
    CHECK(share(0.0f, 0.5f) == 1.0);
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
        const cr::FootprintSet set = cr::measure_footprints(probe, cr::sun_direction(e, 30.0f), cr::PlaneKind::beam, kLeaves);
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
// Card normals: uniform on the sphere; all +z; all +x; all +y (vertical cards:
// only an axis grid built from the cards' own normals fits them). The sun
// stands at azimuth 30 degrees, off both axes, so every |dir| component is
// non-zero and each axis channel carries a share of every arm's projection.
enum class Orient { random, horizontal, facing_x, facing_y };
constexpr float k_format_azimuth_deg = 30.0f;
// error_amp: the oracle is offset by +amp at even and -amp at odd elevation
// indices, an error of known size that no clumping factor can absorb.
cr::FormatDecision format_on_slab(Orient orient, double error_amp = 0.0) {
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
    rng.seed(orient == Orient::horizontal ? 21 : orient == Orient::random ? 22 : orient == Orient::facing_x ? 23 : 24, 0x5AB5EEDull);
    Prim p;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const vec3 c{(i + rng.next_float_01()) * w / n, (j + rng.next_float_01()) * w / n, rng.next_float_01()};
            const vec3 nrm = orient == Orient::horizontal ? vec3{0, 0, 1}
                           : orient == Orient::facing_x   ? vec3{1, 0, 0}
                           : orient == Orient::facing_y   ? vec3{0, 1, 0}
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
    int ei = 0;
    for (const float e : {10.0f, 20.0f, 30.0f, 45.0f, 60.0f, 70.0f, 75.0f, 90.0f}) {
        const vec3 s = cr::sun_direction(e, k_format_azimuth_deg).dir;
        // G = |n . s| for the oriented arms (the cards are one-sided in
        // projection), 0.5 for random orientation.
        const double g = orient == Orient::horizontal ? std::abs(double(s.z))
                       : orient == Orient::facing_x   ? std::abs(double(s.x))
                       : orient == Orient::facing_y   ? std::abs(double(s.y))
                                                      : 0.5;
        const double offset = (ei++ % 2 == 0) ? error_amp : -error_amp;
        cr::FormatElevation fe;
        fe.elevation_deg = e;
        // 8 x 8 footprints of 0.5 m on the mid-plane over [6, 10]^2: at 10
        // degrees a line spans 2.84 m either side, inside the slab.
        for (int fi = 0; fi < 8; ++fi)
            for (int fj = 0; fj < 8; ++fj) {
                cr::FormatFootprint f;
                f.t_oracle = float(std::exp(-L * g / std::sin(e * k_pi / 180.0)) + offset);
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
    const cr::FormatDecision d = format_on_slab(Orient::random);
    for (const auto& r : d.rows) {
        INFO("e " << r.elevation_deg << " n " << r.unsaturated << " scalar " << r.scalar_mae << " axis " << r.axis_mae
                  << " g_eff " << r.g_eff);
        CHECK(r.measurable);
        CHECK(std::abs(r.g_eff - 0.5f) <= 0.03f);
    }
    CHECK(d.decision == "scalar");
}

TEST_CASE("[crown_oracle] format rule: horizontal cards decide axis-projected, G_eff sin(e)", "[treegen][crown_oracle]") {
    const cr::FormatDecision d = format_on_slab(Orient::horizontal);
    for (const auto& r : d.rows) {
        INFO("e " << r.elevation_deg << " n " << r.unsaturated << " scalar " << r.scalar_mae << " axis " << r.axis_mae
                  << " g_eff " << r.g_eff);
        CHECK(r.measurable);
        CHECK(std::abs(r.g_eff - float(std::sin(r.elevation_deg * k_pi / 180.0))) <= 0.03f);
    }
    CHECK(d.decision == "axis_projected");
    // Planar rank-1 F2_tensor has unclumped factor sqrt(3/4); Step 5 fits c = sqrt(4/3) = 1.1547.
    CHECK(std::abs(d.clumping - float(std::sqrt(4.0 / 3.0))) <= 0.03f);
}

TEST_CASE("[crown_oracle] format rule: vertical cards facing the sun decide axis-projected from their own normals", "[treegen][crown_oracle]") {
    // G = |s.x| (+x cards) or |s.y| (+y cards) under the azimuth-30 sun: no
    // single scalar G fits, and a directional tensor built from any normal but
    // the cards' own is off by far more than the threshold.
    for (const Orient o : {Orient::facing_x, Orient::facing_y}) {
        INFO((o == Orient::facing_x ? "cards facing +x" : "cards facing +y"));
        const cr::FormatDecision d = format_on_slab(o);
        for (const auto& r : d.rows) {
            INFO("e " << r.elevation_deg << " scalar " << r.scalar_mae << " axis " << r.axis_mae);
            CHECK(r.measurable);
        }
        CHECK(d.decision == "axis_projected");
        CHECK(d.clumping >= 0.99f);
        CHECK(d.clumping <= 1.20f);
    }
}

TEST_CASE("[crown_oracle] format rule: an error of known size above the threshold escalates", "[treegen][crown_oracle]") {
    // The same horizontal slab with the oracle moved by +-0.12 (alternating by
    // elevation): every elevation exceeds the 0.05 threshold even after the
    // planar sqrt(3/4) offset, and one clumping factor cannot follow a sign
    // that alternates, so the rule must escalate.
    const cr::FormatDecision d = format_on_slab(Orient::horizontal, 0.12);
    for (const auto& r : d.rows) {
        INFO("e " << r.elevation_deg << " scalar " << r.scalar_mae << " axis " << r.axis_mae);
        CHECK(r.measurable);
        CHECK(r.axis_mae > cr::k_format_max_mae);
    }
    CHECK(d.decision == "escalate");
}

TEST_CASE("[crown_oracle] format rule: the oracle's sampling noise sits well under the threshold", "[treegen][crown_oracle]") {
    // Expected MAE of a footprint mean of n binary rays against an exact model,
    // at the worst T (0.5): sqrt(T(1-T)/n) * sqrt(2/pi). At the 64 rays that
    // select footprints it equals the 0.05 threshold (the reason for the
    // re-measure); at the format rule's 1024 it is a quarter of it.
    auto floor_mae = [](int n) { return std::sqrt(0.25 / n) * std::sqrt(2.0 / k_pi); };
    const int n64 = cr::k_rays_per_side * cr::k_rays_per_side;
    const int n_fmt = cr::k_format_rays_per_side * cr::k_format_rays_per_side;
    INFO("noise floor MAE: " << floor_mae(n64) << " at " << n64 << " rays, " << floor_mae(n_fmt) << " at " << n_fmt
                             << " rays; threshold " << cr::k_format_max_mae);
    CHECK(floor_mae(n64) >= 0.9 * cr::k_format_max_mae);
    CHECK(floor_mae(n_fmt) <= 0.25 * cr::k_format_max_mae);
    CHECK(cr::k_format_ray_stream != cr::k_footprint_ray_stream);
}

TEST_CASE("[crown_oracle] format rule: bark is outside its population; everything else sees it", "[treegen][crown_oracle]") {
    // The same random-orientation slab (L = 1, 6 x 6 x 1 m) with and without an
    // opaque bark plank across it, inside the leaf box so the footprint grid and
    // rays are identical. The format section (leaves-only oracle, R-CR1) must be
    // byte-identical; the leaves + bark sections must see the plank.
    cr::CrownMesh with_plank = card_slab(5, 2304, 0.125f, 6.0f, 1.0f, false);
    const std::vector<float> plank{2.0f, 0.5f, 0.5f, 3.5f, 0.5f, 0.5f, 3.5f, 5.5f, 0.5f, 2.0f, 5.5f, 0.5f};
    with_plank.add_bark(plank, std::vector<uint32_t>{0, 1, 2, 0, 2, 3});
    const cr::Probe a(card_slab(5, 2304, 0.125f, 6.0f, 1.0f, false)), b(std::move(with_plank));
    for (const auto& [p, q] : {std::pair{a.bounds_min(), b.bounds_min()}, std::pair{a.bounds_max(), b.bounds_max()}}) {
        REQUIRE(p.x == q.x);
        REQUIRE(p.y == q.y);
        REQUIRE(p.z == q.z);
    }
    cr::ProbeOptions o;
    o.elevations_deg = {30.0f, 45.0f, 60.0f, 90.0f};
    o.azimuths = 1;
    o.sunlit = false;
    const std::string da = cr::probe_report_json(a, o, "{}"), db = cr::probe_report_json(b, o, "{}");
    const size_t fa = da.find("\"format\": {"), fb = db.find("\"format\": {");
    REQUIRE(fa != std::string::npos);
    REQUIRE(fb != std::string::npos);
    CHECK(da.find("\"decision\": \"undecidable\"") == std::string::npos);   // the rows carry numbers
    CHECK(da.substr(fa) == db.substr(fb));
    // Positive control: the whole-crown directions (T, regions, stop rule) see the plank.
    const size_t ma = da.find("\"directions\":"), mb = db.find("\"directions\":");
    REQUIRE(ma < fa);
    REQUIRE(mb < fb);
    CHECK(da.substr(ma, fa - ma) != db.substr(mb, fb - mb));
}

TEST_CASE("[crown_oracle] each sampler call site traces its own stream", "[treegen][crown_oracle]") {
    // Random-orientation slab, L = 1 in 6 x 6 x 1 m, sun at elevation 60 and
    // azimuth 30 (off the axes). Both call sites are re-traced here:
    // every footprint's T must be its selection-stream share, and every kept
    // footprint's t_oracle its format-stream share, exactly; the fixture must
    // tell the two apart on most footprints (positive control), so either call
    // site passing the other stream is red.
    const cr::Probe probe(card_slab(5, 2304, 0.125f, 6.0f, 1.0f, false));
    const cr::CrownVolume volume = cr::CrownVolume::build(probe, cr::k_format_cells);
    const cr::FootprintSet set = cr::measure_footprints(probe, cr::sun_direction(60.0f, 30.0f), cr::PlaneKind::beam, kLeaves);
    cr::FormatElevation fe;
    cr::add_format_footprints(probe, set, volume, fe);

    auto open_share = [&](size_t k, cr::RayStream s) {
        std::vector<vec3> rays;
        set.rays(k, rays, s);
        int open = 0;
        for (const vec3& o : rays) open += probe.occluded(o, set.sun.dir, -1, set.occluders) ? 0 : 1;
        return float(open) / float(rays.size());
    };
    size_t c = 0, partial = 0;
    int differ_sel = 0, differ_fmt = 0;
    for (size_t k = 0; k < set.footprints.size(); ++k) {
        const float t = set.footprints[k].t;
        const float sel = open_share(k, cr::RayStream::selection), fmt = open_share(k, cr::RayStream::format);
        INFO("footprint " << k << ": T " << t << ", selection stream " << sel << ", format stream " << fmt);
        CHECK(t == sel);
        if (t > 0.0f && t < 1.0f) { ++partial; differ_sel += t != fmt ? 1 : 0; }
        if (t < cr::k_opaque_t || t > cr::k_unsaturated_hi) continue;   // the rule's selection
        REQUIRE(c < fe.unsaturated.size());
        CHECK(fe.unsaturated[c].t_oracle == fmt);
        differ_fmt += fe.unsaturated[c].t_oracle != t ? 1 : 0;
        ++c;
    }
    INFO("kept " << c << ", format disagrees with selection on " << differ_fmt << "; partial " << partial << ", " << differ_sel);
    CHECK(c == fe.unsaturated.size());
    CHECK(c >= size_t(cr::k_min_unsaturated));
    CHECK(differ_fmt * 2 >= int(c));
    CHECK(differ_sel * 2 >= int(partial));
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
    const cr::FootprintSet set = cr::measure_footprints(probe, sun, cr::PlaneKind::beam, kAll);
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
        if (radius - r >= 1.0f) CHECK(f.body());   // the rim side is the lattice plate's (exact) arm
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
    const cr::FootprintSet set = cr::measure_footprints(small, cr::sun_direction(90.0f, 0.0f), cr::PlaneKind::beam, kAll);
    CHECK_FALSE(cr::body_stats(set, cr::k_body_edge_m).has_value());
    cr::ProbeOptions o;
    o.elevations_deg = {90.0f};
    o.azimuths = 1;
    const std::string doc = cr::probe_report_json(small, o, "{}");
    CHECK(doc.find("\"body\": null") != std::string::npos);
    CHECK(doc.find("\"body\": {") == std::string::npos);

    // Recorded, not a statement of intent: the plan's "a crown under 2 m across
    // has no body" disagrees with 5.1's d (to the nearest OUTSIDE centre, which
    // lies 0.2-0.36 m beyond the edge). Under the definition as written a 1.8 m
    // disc HAS body footprints; this row pins what the definition gives today
    // until the plan is corrected on one side or the other.
    const cr::Probe mid = disc(vec3{0, 0, 6}, vec3{0, 0, 1}, 0.9f);
    const cr::FootprintSet mid_set = cr::measure_footprints(mid, cr::sun_direction(90.0f, 0.0f), cr::PlaneKind::beam, kAll);
    const auto mid_body = cr::body_stats(mid_set, cr::k_body_edge_m);
    float max_d = 0.0f;
    for (const auto& f : mid_set.footprints) max_d = std::max(max_d, f.edge_distance);
    INFO("1.8 m disc at zenith: body footprints " << (mid_body ? mid_body->footprints : 0) << ", max edge distance " << max_d);
    CHECK(mid_body.has_value());
}

namespace {
// Opaque axis-aligned rectangles (bark) in the plane z, each [x0, x1] x [y0, y1].
cr::Probe plate(float z, const std::vector<std::array<float, 4>>& rects) {
    std::vector<float> pos;
    std::vector<uint32_t> idx;
    for (const auto& r : rects) {
        const uint32_t b = uint32_t(pos.size() / 3);
        pos.insert(pos.end(), {r[0], r[2], z, r[1], r[2], z, r[1], r[3], z, r[0], r[3], z});
        idx.insert(idx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    cr::CrownMesh m;
    m.add_bark(pos, idx);
    return cr::Probe(std::move(m));
}
}  // namespace

TEST_CASE("[crown_oracle] region logic: an exact lattice plate's rim and body", "[treegen][crown_oracle]") {
    // A 3 x 3 m plate with its four 0.5 m corners notched, at a zenith sun. Its
    // edges lie on footprint boundaries (the grid is centred on the symmetric
    // box), so every T is exactly 0 or 1 and every edge distance is a lattice
    // distance: 32 silhouette footprints; d = 0.5 on the 16 edge footprints and
    // 0.707 on the 4 inner corners (rim, 20); d = 1.0 on 8 and 1.414 on 4 (body, 12).
    // Edge distances are centre to centre, so d is 0.5 sqrt(n): a threshold of
    // 0.70 makes the 0.707 ring body, ">" for ">=" makes the 1.0 ring rim, and any
    // threshold in (0.707, 1.0] is the same rule.
    const cr::Probe probe = plate(6.0f, {{-1.5f, 1.5f, -1.0f, 1.0f}, {-1.0f, 1.0f, -1.5f, 1.5f}});
    const cr::FootprintSet set = cr::measure_footprints(probe, cr::sun_direction(90.0f, 0.0f), cr::PlaneKind::beam, kAll);
    std::map<int, int> d_hist;   // round(100 d) -> count, silhouette footprints
    int sil = 0, body = 0;
    for (const cr::Footprint& f : set.footprints) {
        CHECK((f.t == 0.0f || f.t == 1.0f));
        if (!f.silhouette()) continue;
        ++sil;
        ++d_hist[int(std::lround(100.0f * f.edge_distance))];
        body += f.body() ? 1 : 0;
        const float r = std::max(std::abs(f.u), std::abs(f.v));
        INFO("footprint (" << f.u << ", " << f.v << ") d " << f.edge_distance);
        CHECK(f.body() == (r < 1.0f && !(std::abs(f.u) > 0.5f && std::abs(f.v) > 0.5f)));
    }
    CHECK(sil == 32);
    CHECK(d_hist == std::map<int, int>{{50, 16}, {71, 4}, {100, 8}, {141, 4}});
    CHECK(body == 12);
    const auto stats = cr::body_stats(set, cr::k_body_edge_m);
    REQUIRE(stats.has_value());
    CHECK(stats->footprints == 12);
    CHECK(stats->mean == 0.0f);

    cr::ProbeOptions o;
    o.elevations_deg = {90.0f};
    o.azimuths = 1;
    o.sunlit = false;
    const std::string doc = cr::probe_report_json(probe, o, "{}");
    const size_t sil_at = doc.find("\"silhouette\": 32,"), rim_at = doc.find("\"rim\": 20,");
    REQUIRE(sil_at != std::string::npos);
    REQUIRE(rim_at != std::string::npos);
    const size_t body_at = doc.find("\"body\": {", rim_at);
    REQUIRE(body_at != std::string::npos);
    CHECK(doc.find("\"footprints\": 12,", body_at) == doc.find("\"footprints\":", body_at));
}

// ---- Stop rule ---------------------------------------------------------------------------------

namespace {
// A synthetic beam set: body footprints (d = 2 m) with the given T, plus rim ones (d = 0.5 m).
cr::FootprintSet synthetic_set(float elevation, const std::vector<float>& body_t, int rim = 3) {
    cr::FootprintSet s;
    s.sun = cr::sun_direction(elevation, 0.0f);
    for (float t : body_t) { cr::Footprint f; f.t = t; f.edge_distance = 2.0f; s.footprints.push_back(f); }
    for (int k = 0; k < rim; ++k) { cr::Footprint f; f.t = 0.0f; f.edge_distance = 0.5f; s.footprints.push_back(f); }
    return s;
}
}  // namespace

TEST_CASE("[crown_oracle] stop rule: one direction's body opacity at each threshold", "[treegen][crown_oracle]") {
    // T < 0.05 is opaque (strict); a share of at least 0.5 makes the body opaque (inclusive).
    CHECK(cr::body_opaque(synthetic_set(45, {0.0499f, 0.9f})));          // share 0.5 exactly
    CHECK_FALSE(cr::body_opaque(synthetic_set(45, {0.05f, 0.9f})));      // T at the threshold is not opaque
    CHECK_FALSE(cr::body_opaque(synthetic_set(45, {0.0f, 0.9f, 0.9f})));  // share 1/3
    CHECK(cr::body_opaque(synthetic_set(45, {0.0f, 0.0f, 0.9f})));       // share 2/3
    CHECK_FALSE(cr::body_opaque(synthetic_set(45, {}, 5)));              // no body: not opaque (rim alone is opaque)
}

TEST_CASE("[crown_oracle] SE: a body's statistics carry the stated standard errors", "[treegen][crown_oracle]") {
    // Four body footprints T = {0.5, 0.5, 0, 0.9} of 64 rays: mean 0.475, ray-sampling
    // SE sqrt((0.25 + 0.25 + 0 + 0.09) / 64) / 4; share below 0.05 is 1/4, SE sqrt(3/16 / 4).
    const auto b = cr::body_stats(synthetic_set(45, {0.5f, 0.5f, 0.0f, 0.9f}, 0), cr::k_body_edge_m);
    REQUIRE(b.has_value());
    CHECK(std::abs(b->mean - 0.475f) <= 1e-6f);
    CHECK(std::abs(b->mean_se - std::sqrt(0.59 / 64.0) / 4.0) <= 1e-7);
    CHECK(b->share_opaque == 0.25f);
    CHECK(std::abs(b->share_opaque_se - std::sqrt(3.0 / 16.0 / 4.0)) <= 1e-7);
}

TEST_CASE("[crown_oracle] stop rule: the crown verdict over elevations and azimuths", "[treegen][crown_oracle]") {
    using V = std::vector<std::pair<float, bool>>;
    auto verdict = [](const V& v) { return cr::stop_rule(v).crown_opaque_to_direct_sun; };
    CHECK(verdict({{30.0f, true}}));                       // 30 degrees counts (inclusive)
    CHECK_FALSE(verdict({{29.99f, true}}));                // below 30 never decides
    CHECK_FALSE(verdict({}));                              // nothing sampled at 30 or above
    CHECK(verdict({{20.0f, false}, {30.0f, true}}));       // an open low sun is ignored
    CHECK_FALSE(verdict({{30.0f, true}, {60.0f, false}})); // every elevation >= 30 must be opaque
    CHECK_FALSE(verdict({{45.0f, true}, {45.0f, false}})); // ... at every azimuth
    CHECK(verdict({{45.0f, true}, {45.0f, true}, {90.0f, true}}));
    const cr::StopRule r = cr::stop_rule({{60.0f, true}, {20.0f, false}, {60.0f, false}});
    CHECK(r.opaque_by_elevation == V{{20.0f, false}, {60.0f, false}});   // ascending, per-elevation AND
}

TEST_CASE("[crown_oracle] stop rule: the report states the verdict it computes", "[treegen][crown_oracle]") {
    // Horizontal 0.5 m cards at leaf area index 6: opaque at every elevation; a
    // small disc has no body: not opaque. Read from the report, not the struct.
    constexpr float side = 0.5f, w = 12.0f;
    const cr::Probe slab(card_slab(11, int(6.0 * w * w / (side * side)), side, w, 1.0f, true));
    cr::ProbeOptions o;
    o.elevations_deg = {20.0f, 30.0f, 45.0f};
    o.azimuths = 2;
    o.sunlit = false;
    const std::string a = cr::probe_report_json(slab, o, "{}");
    CHECK(a.find("\"crown_opaque_to_direct_sun\": true") != std::string::npos);
    const cr::Probe small = disc(vec3{0, 0, 6}, vec3{0, 0, 1}, 0.25f);
    const std::string b = cr::probe_report_json(small, o, "{}");
    CHECK(b.find("\"crown_opaque_to_direct_sun\": false") != std::string::npos);
}

// ---- Low-pass ----------------------------------------------------------------------------

TEST_CASE("[crown_oracle] low-pass: identity at 0, Gaussian second moment, conserved", "[treegen][crown_oracle]") {
    // One filled cell (rank-1 M_xx = 1 m2/m3, others 0) at the centre of 9^3 cells of 0.2 m.
    constexpr int n = 9;
    constexpr float cell = 0.2f;
    std::vector<float> cells(size_t(n) * n * n * 6, 0.0f);
    const size_t centre = ((size_t(4) * n + 4) * n + 4) * 6;
    cells[centre + 0] = 1.0f;  // M_xx = 1 -> rho = 1, w=(1,0,0) -> sigma = sqrt(3/4) * 1
    const vec3 origin{-0.9f, -0.9f, -0.9f};   // the filled cell spans [-0.1, 0.1]^3
    const cr::CrownVolume v = cr::CrownVolume::from_cells(origin, cell, n, n, n, cells);
    const cr::ExtinctionModel tensor{cr::ExtinctionKind::tensor, 0.5f, 1.0f};
    const vec3 dir{1, 0, 0};

    // sigma 0: the unfiltered march, bit for bit.
    for (const float y : {-0.05f, 0.0f, 0.07f})
        CHECK(v.optical_depth(vec3{-5.0f, y, 0.01f}, dir, 0.0f, tensor) == v.integrals(vec3{-5.0f, y, 0.01f}, dir)[1]);

    // sigma 0.6 m: the optical-depth profile over the ray offsets (y, z).
    constexpr float sigma = 0.6f;
    const double step = 0.05;
    double m0 = 0, my2 = 0, mz2 = 0;
    for (int iy = -60; iy <= 60; ++iy)
        for (int iz = -60; iz <= 60; ++iz) {
            const double y = iy * step, z = iz * step;
            const double tau = v.optical_depth(vec3{-10.0f, float(y), float(z)}, dir, sigma, tensor);
            m0 += tau; my2 += tau * y * y; mz2 += tau * z * z;
        }
    const double expected_var = double(sigma) * sigma + double(cell) * cell / 12.0;
    const double expected_integral = std::sqrt(0.75) * double(cell) * cell * cell;
    const double integral = m0 * step * step;
    INFO("var y " << my2 / m0 << " z " << mz2 / m0 << " expected " << expected_var << "; integral " << integral
                  << " expected " << expected_integral);
    CHECK(std::abs(my2 / m0 - expected_var) <= 0.05 * expected_var);
    CHECK(std::abs(mz2 / m0 - expected_var) <= 0.05 * expected_var);
    CHECK(std::abs(integral - expected_integral) <= 0.01 * expected_integral);
}

TEST_CASE("[crown_oracle] low-pass: a diagonal march sees every blur pass and the channel it reads", "[treegen][crown_oracle]") {
    // One filled cell with distinct PSD tensor channels (including non-zero off-diagonal M_xy),
    // marched along (1, 1, 0)/sqrt 2. All 6 channels are blurred independently in FP32; because
    // sigma_tensor is homogeneous of degree 1 in M, the offset integral and Gaussian second
    // moment are conserved identically across all 3 separable blur passes.
    constexpr int n = 9;
    constexpr float cell = 0.2f;
    std::vector<float> cells(size_t(n) * n * n * 6, 0.0f);
    const size_t centre = ((size_t(4) * n + 4) * n + 4) * 6;
    // M = diag(0.5, 2.0, 7.0) + off-diag M_xy = 0.4 (PSD: 0.5*2.0 > 0.16)
    cells[centre + 0] = 0.5f;   // M_xx
    cells[centre + 1] = 2.0f;   // M_yy
    cells[centre + 2] = 7.0f;   // M_zz
    cells[centre + 3] = 0.4f;   // M_xy
    cells[centre + 4] = -0.2f;  // M_xz
    cells[centre + 5] = 0.3f;   // M_yz
    const cr::CrownVolume v = cr::CrownVolume::from_cells(vec3{-0.9f, -0.9f, -0.9f}, cell, n, n, n, cells);
    const cr::ExtinctionModel tensor{cr::ExtinctionKind::tensor, 0.5f, 1.0f};
    const double s2 = std::sqrt(0.5);
    const vec3 dir{float(s2), float(s2), 0.0f}, wv{float(-s2), float(s2), 0.0f}, zv{0, 0, 1};
    constexpr float sigma = 0.6f;
    const double step = 0.05;
    double m0 = 0, mw2 = 0, mz2 = 0;
    for (int iw = -60; iw <= 60; ++iw)
        for (int iz = -60; iz <= 60; ++iz) {
            const double w = iw * step, z = iz * step;
            const vec3 o = wv * float(w) + zv * float(z) - dir * 10.0f;
            const double tau = v.optical_depth(o, dir, sigma, tensor);
            m0 += tau; mw2 += tau * w * w; mz2 += tau * z * z;
        }
    const double expected_var = double(sigma) * sigma + double(cell) * cell / 12.0;
    const double rho = 0.5 + 2.0 + 7.0;
    const double quad = 0.5 * 0.5 + 0.5 * 2.0 + 2.0 * 0.5 * 0.4;
    const double expected_integral = std::sqrt(0.75 * rho * quad) * double(cell) * cell * cell;
    const double integral = m0 * step * step;
    INFO("var w " << mw2 / m0 << " z " << mz2 / m0 << " expected " << expected_var << "; integral " << integral
                  << " expected " << expected_integral);
    CHECK(std::abs(mw2 / m0 - expected_var) <= 0.05 * expected_var);
    CHECK(std::abs(mz2 / m0 - expected_var) <= 0.05 * expected_var);
    CHECK(std::abs(integral - expected_integral) <= 0.01 * expected_integral);
}

// ---- Sunlit mode -------------------------------------------------------------------------------

TEST_CASE("[crown_oracle] sunlit mode: a two-layer crown at a zenith sun", "[treegen][crown_oracle]") {
    // 4 x 4 bent cards (generator shape: the ridge sits 0.02 m above the site
    // centroid, so a site's own card is ON its ray to the sun) at z = 2, and the
    // same 16 directly below at z = 1. Opaque (no atlas). At zenith every top
    // site sees the sun only because its own site is excluded, and every bottom
    // site is shadowed by the card above it: share 0.5; ranked nearest the sun
    // first (the top layer), quartiles {1, 1, 0, 0} and deciles as derived below.
    Prim p;
    for (const float h : {2.0f, 1.0f})
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) p.bent_card(vec3{float(i), float(j), h}, vec3{1, 0, 0}, vec3{0, 1, 0}, 0.6f);
    cr::CrownMesh m;
    m.add_leaves(p.pos, p.uv, p.idx);
    REQUIRE(m.sites.size() == 32u);
    const cr::Probe probe(std::move(m));
    cr::ProbeOptions o;
    o.oracle = false;
    o.elevations_deg = {90.0f};
    o.azimuths = 1;
    const auto parsed = treegen::json::parse(cr::probe_report_json(probe, o, "{}"));
    REQUIRE(parsed.ok);
    auto field = [](const treegen::json::value& v, const char* k) -> const treegen::json::value& {
        const treegen::json::value* f = v.find(k);
        REQUIRE(f);
        return *f;
    };
    const auto& dirs = field(field(field(parsed.root, "modes"), "sunlit"), "directions");
    REQUIRE(dirs.a.size() == 1u);
    const auto& d0 = dirs.a[0];
    auto nums = [](const treegen::json::value& arr) {
        std::vector<double> v;
        for (const auto& e : arr.a) v.push_back(e.n);
        return v;
    };
    CHECK(field(d0, "share").n == 0.5);
    CHECK(nums(field(d0, "quartiles")) == std::vector<double>{1, 1, 0, 0});
    // 32 ranked sites into 10 groups by floor(10 r / 32): groups 0-4 hold ranks
    // 0-15 (the lit top layer), groups 5-9 ranks 16-31.
    CHECK(nums(field(d0, "deciles")) == std::vector<double>{1, 1, 1, 1, 1, 0, 0, 0, 0, 0});
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
    for (const cr::Occluders occ : {kAll, kLeaves}) {   // leaves answer the same in both populations
        INFO("occluders " << cr::occluders_name(occ));
        CHECK(probe.occluded(vec3{0, 0, 0}, up, -1, occ));
        CHECK(probe.occluded(vec3{0, 0, 0}, up, 0, occ));       // site 1 still blocks
        CHECK(probe.occluded(vec3{0, 0, 1.5f}, up, -1, occ));   // site 1 above
        CHECK_FALSE(probe.occluded(vec3{0, 0, 1.5f}, up, 1, occ));
        CHECK_FALSE(probe.occluded(vec3{0, 0, 0}, vec3{-1, 0, 0}, -1, occ));
        CHECK_FALSE(probe.occluded(vec3{5, 0, 0}, up, 2, occ));  // only site 2 is on this ray
        CHECK(probe.occluded(vec3{5, 0, 0}, up, 0, occ));
        // Segments: a point in front of the card is visible, behind it is not.
        CHECK_FALSE(probe.occluded_segment(vec3{0, 0, 0}, vec3{0, 0, 0.5f}, -1, occ));
        CHECK(probe.occluded_segment(vec3{0, 0, 0}, vec3{0, 0, 1.5f}, -1, occ));
        CHECK_FALSE(probe.occluded_segment(vec3{0, 0, 0}, vec3{0, 0, 1.5f}, 0, occ));
    }
    // The bark triangle blocks the whole crown and is absent from the leaves-only population.
    const vec3 under_bark{10.2f, 10.2f, -1.0f};
    CHECK(probe.occluded(under_bark, up, -1, kAll));
    CHECK_FALSE(probe.occluded(under_bark, up, -1, kLeaves));
    CHECK(probe.occluded_segment(under_bark, vec3{10.2f, 10.2f, 1.0f}, -1, kAll));
    CHECK_FALSE(probe.occluded_segment(under_bark, vec3{10.2f, 10.2f, 1.0f}, -1, kLeaves));
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

TEST_CASE("[crown_oracle] the loader reads every triangle the emitter wrote", "[treegen][crown_oracle]") {
    // Both probe paths share the loader, so their agreement cannot see a
    // primitive it drops; the emitter's own index counts can.
    for (const char* species : {"oak", "pine"}) {
        const treegen::GeneratedTree& g = rynx::test_support::generated_tree(species, 42);
        for (int lod = 0; lod < int(g.lods.size()); ++lod) {
            const cr::LoadedCrown c = cr::load_crown_from_glb(g.glb, lod);
            const auto& l = g.lods[size_t(lod)];
            INFO(species << " L" << lod << ": bark " << c.mesh.bark_triangles << " / " << l.indices_u32.size() / 3
                         << ", leaf " << c.mesh.leaf_triangles << " / " << l.leaf_indices_u32.size() / 3);
            REQUIRE(l.has_leaves);
            CHECK(c.mesh.bark_triangles == l.indices_u32.size() / 3);
            CHECK(c.mesh.leaf_triangles == l.leaf_indices_u32.size() / 3);
            CHECK(c.mesh.bark_triangles > 0u);
            CHECK(c.mesh.leaf_triangles > 0u);
        }
    }
}

// ---- Geometric normals only (K16) -------------------------------------------------------------

namespace {
// Four elevations spanning 70 degrees: the format rule decides, so the
// table (the axis grid, the normals' only possible consumer) carries numbers.
cr::ProbeOptions four_directions() {
    cr::ProbeOptions o;
    o.elevations_deg = {20.0f, 45.0f, 70.0f, 90.0f};
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

    const cr::ProbeOptions o = four_directions();
    const std::string a = cr::probe_glb_report(g.glb, 0, o, cr::ProbeInput::glb("oak_orig.glb"));
    const std::string b = cr::probe_glb_report(bytes, 0, o, cr::ProbeInput::glb("oak_mut.glb"));
    CHECK(a.find("\"format\": {") != std::string::npos);   // the axis grid (normals' consumer) is in the report
    CHECK(a.find("\"decision\": \"undecidable\"") == std::string::npos);   // ... with numbers, not nulls
    CHECK(a.find("\"axis_mae\": 0.") != std::string::npos);
    CHECK(a.find("\"glb_fnv1a64\": \"" + expected_digest(g.glb) + "\"") != std::string::npos);
    CHECK(b.find("\"glb_fnv1a64\": \"" + expected_digest(bytes) + "\"") != std::string::npos);
    CHECK(strip_input(a) == strip_input(b));
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
        // Both reports name the bytes of the file the tool wrote.
        const std::string file = slurp(glb);
        const std::string digest = "\"glb_fnv1a64\": \"" + expected_digest(file) + "\"";
        CHECK(da.find(digest) != std::string::npos);
        CHECK(db.find(digest) != std::string::npos);
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
    "schema", "input", "input.source", "input.glb_fnv1a64", "input.probe_lod", "tree",
    "tree.bark_triangles", "tree.leaf_triangles", "tree.sites", "tree.cards_per_site", "tree.cards_per_site.<n>",
    "tree.alpha_coverage", "tree.card_area_m2", "tree.opaque_leaf_area_m2", "tree.crown_ground_projected_area_m2",
    "tree.leaf_area_index", "tree.leaf_aabb_min_m", "tree.leaf_aabb_max_m", "tree.leaf_area_density_m2_per_m3",
    "tree.trunk_base_z_m", "tree.lowest_leaf_height_m",
    "modes", "modes.oracle", "modes.oracle.footprint_m", "modes.oracle.rays_per_footprint", "modes.oracle.trunk_axis",
    "modes.oracle.occluders", "modes.oracle.format.occluders",
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
    // R-CR6 / P2: the standard error beside every sampled statistic and jitter-dependent count.
    "tree.crown_ground_projected_area_m2_se", "tree.leaf_area_index_se",
    "modes.oracle.directions[].silhouette_se", "modes.oracle.directions[].rim_se",
    "modes.oracle.directions[].body.footprints_se",
    "modes.oracle.directions[].body.mean_se", "modes.oracle.directions[].body.p50_se",
    "modes.oracle.directions[].body.p95_se", "modes.oracle.directions[].body.share_below_0_05_se",
    "modes.oracle.directions[].body_mean_by_min_edge_m_se", "modes.oracle.directions[].body_mean_by_min_edge_m_se.<n>",
    "modes.oracle.directions[].t_by_edge_distance[].mean_se",
    "modes.oracle.directions[].crossings.silhouette.geometric_se", "modes.oracle.directions[].crossings.silhouette.opaque_se",
    "modes.oracle.directions[].crossings.body.geometric_se", "modes.oracle.directions[].crossings.body.opaque_se",
    "modes.oracle.directions[].ground_plane.silhouette_se",
    "modes.oracle.directions[].ground_plane.body.footprints_se",
    "modes.oracle.directions[].ground_plane.body.mean_se", "modes.oracle.directions[].ground_plane.body.p50_se",
    "modes.oracle.directions[].ground_plane.body.p95_se", "modes.oracle.directions[].ground_plane.body.share_below_0_05_se",
    "modes.oracle.directions[].ground_plane.body_mean_by_min_edge_m_se",
    "modes.oracle.directions[].ground_plane.body_mean_by_min_edge_m_se.<n>",
    "modes.oracle.directions[].t_by_trunk_distance[].mean_se",
    "modes.oracle.directions[].sunlit_by_depth[].sunlit_share_se",
    "modes.oracle.format.g_se", "modes.oracle.format.clumping_se",
    "modes.oracle.format.rows[].unsaturated_footprints_se",
    "modes.oracle.format.rows[].saturated_share_se", "modes.oracle.format.rows[].scalar_mae_se",
    "modes.oracle.format.rows[].axis_mae_se", "modes.oracle.format.rows[].g_eff_se",
    "modes.sunlit.open_sky.mean_se", "modes.sunlit.open_sky.p10_se", "modes.sunlit.open_sky.p50_se",
    "modes.sunlit.open_sky.p90_se", "modes.sunlit.directions[].share_se", "modes.sunlit.directions[].quartiles_se",
    "modes.sunlit.directions[].deciles_se", "modes.sunlit.directions[].eyes[].share_se",
};

// The input object's source-specific fields.
const std::set<std::string> k_schema_1_scenario_input = {"input.scenario", "input.seed"};
const std::set<std::string> k_schema_1_glb_input      = {"input.glb"};

std::set<std::string> schema_mismatch(const std::string& doc, const std::set<std::string>& input_fields) {
    const auto parsed = treegen::json::parse(doc);
    if (!parsed.ok) return {"<unparseable>"};
    std::set<std::string> got, want = k_schema_1;
    want.insert(input_fields.begin(), input_fields.end());
    key_paths(parsed.root, "", got);
    std::set<std::string> diff;
    for (const auto& k : got) if (!want.count(k)) diff.insert("+" + k);
    for (const auto& k : want) if (!got.count(k)) diff.insert("-" + k);
    return diff;
}
}  // namespace

TEST_CASE("[crown_oracle] report schema 1: the exact field list", "[treegen][crown_oracle]") {
    const treegen::GeneratedTree& g = rynx::test_support::generated_tree("oak", 42);
    cr::ProbeOptions o;
    o.elevations_deg = {90.0f};
    o.azimuths = 1;
    o.eyes = {vec3{0, -12, 1.7f}};
    const std::string doc = cr::probe_glb_report(g.glb, 0, o, cr::ProbeInput::scenario("c3_oak.json", 42));
    REQUIRE(doc.find("\"schema\": 1,") != std::string::npos);
    REQUIRE(cr::k_report_schema == 1);
    const auto diff = schema_mismatch(doc, k_schema_1_scenario_input);
    for (const auto& d : diff) UNSCOPED_INFO(d);
    CHECK(diff.empty());
    // The same document from --glb: only the input object's source fields differ.
    const std::string glb_doc = cr::probe_glb_report(g.glb, 0, o, cr::ProbeInput::glb("oak_42.glb"));
    const auto glb_diff = schema_mismatch(glb_doc, k_schema_1_glb_input);
    for (const auto& d : glb_diff) UNSCOPED_INFO("glb input: " << d);
    CHECK(glb_diff.empty());
    // The digest names these bytes.
    CHECK(doc.find("\"glb_fnv1a64\": \"" + expected_digest(g.glb) + "\"") != std::string::npos);
    CHECK(glb_doc.find("\"glb_fnv1a64\": \"" + expected_digest(g.glb) + "\"") != std::string::npos);

    // Positive control: one field removed is caught.
    std::string cut = doc;
    const size_t at = cut.find("\"leaf_area_index\": ");
    REQUIRE(at != std::string::npos);
    cut.erase(at, cut.find('\n', at) - at + 1);
    CHECK(schema_mismatch(cut, k_schema_1_scenario_input) == std::set<std::string>{"-tree.leaf_area_index"});
}

// ---- S-P0b gates (P1-P5, D1-D6) -----------------------------------------------

TEST_CASE("[crown_oracle] G-P1 pin_band floors over zero-SE degenerate ends", "[treegen][crown_oracle]") {
    SECTION("share at 0 and 1 keeps 1/n floor") {
        constexpr size_t n = 20;
        CHECK(cr::detail::se_share(0.0, n) == 0.0);
        CHECK(cr::detail::se_share(1.0, n) == 0.0);
        CHECK(cr::pin_band_share(cr::detail::se_share(0.0, n), n) == Approx(1.0 / double(n)));
        CHECK(cr::pin_band_share(cr::detail::se_share(1.0, n), n) == Approx(1.0 / double(n)));
    }
    SECTION("all-zero T mean keeps 1/(rays_per_fp * n) floor") {
        const std::vector<float> zeros(10, 0.0f);
        CHECK(cr::detail::se_mean_t(zeros, 64) == 0.0);
        CHECK(cr::pin_band_mean_t(cr::detail::se_mean_t(zeros, 64), zeros.size(), 64) == Approx(1.0 / 640.0));
    }
    SECTION("tied order stats quantile keeps parameterized 1/step_count floor") {
        const std::vector<float> tied(12, 0.5f);
        CHECK(cr::detail::se_quantile(tied, 0.5) == 0.0);
        CHECK(cr::pin_band_quantile(cr::detail::se_quantile(tied, 0.5), 64) == Approx(1.0 / 64.0));
        CHECK(cr::pin_band_quantile(cr::detail::se_quantile(tied, 0.5), 1024) == Approx(1.0 / 1024.0));
    }
    SECTION("count 0 keeps 1.0 floor and non-zero SE scales by 3") {
        CHECK(cr::pin_band_count(0.0) == Approx(1.0));
        CHECK(cr::pin_band(0.1, 0.01) == Approx(0.3));
    }
}

TEST_CASE("[crown_oracle] G-P2 rim straddling T=0.98 produces non-zero count and area SEs", "[treegen][crown_oracle]") {
    auto field = [](const treegen::json::value& v, const char* k) -> const treegen::json::value& {
        const treegen::json::value* f = v.find(k);
        REQUIRE(f);
        return *f;
    };

    // 3.04 x 3.04 m leaf slab (two stacked cards at z = 2.0 and z = 2.5 so the
    // leaf AABB has non-zero volume): centred at (0, 0), so its edges extend
    // 0.02 m into the outer 0.5 m footprint ring (4% covered -> T ~ 0.96,
    // straddling k_silhouette_t = 0.98). Inner [-1.5, 1.5]^2 is fully opaque.
    Prim p;
    p.card(vec3{0, 0, 2.0f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 3.04f);
    p.card(vec3{0, 0, 2.5f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 3.04f);
    cr::CrownMesh m;
    m.add_leaves(p.pos, p.uv, p.idx);
    const cr::Probe probe(std::move(m));
    cr::ProbeOptions o;
    o.elevations_deg = {90.0f};
    o.azimuths = 1;
    o.sunlit = false;
    const auto parsed = treegen::json::parse(cr::probe_report_json(probe, o, "{}"));
    REQUIRE(parsed.ok);
    const auto& tree = field(parsed.root, "tree");
    const auto& d0   = field(field(field(parsed.root, "modes"), "oracle"), "directions").a.at(0);
    const auto& body = field(d0, "body");
    const auto& gp   = field(d0, "ground_plane");
    const auto& gpb  = field(gp, "body");

    const double sil_se    = field(d0, "silhouette_se").n;
    const double rim_se    = field(d0, "rim_se").n;
    const double bfp_se    = field(body, "footprints_se").n;
    const double gsil_se   = field(gp, "silhouette_se").n;
    const double gbfp_se   = field(gpb, "footprints_se").n;
    const double area      = field(tree, "crown_ground_projected_area_m2").n;
    const double area_se   = field(tree, "crown_ground_projected_area_m2_se").n;
    const double opaque_a  = field(tree, "opaque_leaf_area_m2").n;
    const double lai_se    = field(tree, "leaf_area_index_se").n;

    const auto& fmt_rows = field(field(field(field(parsed.root, "modes"), "oracle"), "format"), "rows").a;
    REQUIRE(fmt_rows.size() == 1u);
    const double unsat_se  = field(fmt_rows[0], "unsaturated_footprints_se").n;

    INFO("sil_se=" << sil_se << " rim_se=" << rim_se << " body_fp_se=" << bfp_se
                   << " area_se=" << area_se << " lai_se=" << lai_se << " unsat_se=" << unsat_se);
    CHECK(sil_se > 0.0);
    CHECK(rim_se > 0.0);
    CHECK(bfp_se > 0.0);
    CHECK(gsil_se > 0.0);
    CHECK(gbfp_se > 0.0);
    CHECK(area_se > 0.0);
    CHECK(lai_se > 0.0);
    CHECK(unsat_se > 0.0);
    CHECK(area_se == Approx(0.25 * sil_se).margin(1e-6));
    CHECK(lai_se == Approx((opaque_a / (area * area)) * area_se).margin(1e-6));

    // Exact lattice-aligned 3.0 x 3.0 m leaf slab: every T is 0 or 1, so count SEs are 0.
    Prim exact;
    exact.card(vec3{0, 0, 2.0f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 3.0f);
    exact.card(vec3{0, 0, 2.5f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 3.0f);
    cr::CrownMesh em;
    em.add_leaves(exact.pos, exact.uv, exact.idx);
    const cr::Probe eprobe(std::move(em));
    const auto ep = treegen::json::parse(cr::probe_report_json(eprobe, o, "{}"));
    REQUIRE(ep.ok);
    CHECK(field(field(ep.root, "tree"), "crown_ground_projected_area_m2_se").n == 0.0);
    CHECK(field(field(field(field(ep.root, "modes"), "oracle"), "directions").a.at(0), "silhouette_se").n == 0.0);
    CHECK(field(field(field(field(field(ep.root, "modes"), "oracle"), "format"), "rows").a.at(0), "unsaturated_footprints_se").n == 0.0);

    // Direct multi-ray crossing check (Neon L4): a footprint at T = 60/64 (0.9375,
    // > 1/64 away from k_silhouette_t = 0.98) still carries non-zero binomial crossing SE.
    cr::FootprintSet multi;
    cr::Footprint fp;
    fp.t = 60.0f / 64.0f;
    multi.footprints.push_back(fp);
    const double p60 = 60.0 / 64.0;
    const double q_sil_60 = 1.0 - std::pow(p60, 64) - 64.0 * std::pow(p60, 63) * (1.0 - p60);
    const double multi_se = cr::footprint_count_se(multi, cr::k_body_edge_m, false).silhouette_se;
    CHECK(multi_se > 0.0);
    CHECK(multi_se == Approx(std::sqrt(q_sil_60 * (1.0 - q_sil_60))).margin(1e-6));
}

TEST_CASE("[crown_oracle] G-P3 exposed SE formulas against closed forms", "[treegen][crown_oracle]") {
    SECTION("se_mean, se_mean_t, and se_share on closed forms") {
        const std::vector<double> constant{5.0, 5.0, 5.0, 5.0};
        CHECK(cr::detail::se_mean(constant) == 0.0);
        // Two-valued {1, 3}: mean 2, sum sq dev 2, sample var = 2/(2-1) = 2, SE = sqrt(2/2) = 1.
        // Using n instead of n-1 gives sqrt(0.5) != 1.
        const std::vector<double> two{1.0, 3.0};
        CHECK(cr::detail::se_mean(two) == Approx(1.0).margin(1e-12));
        const std::vector<float> four{1.0f, 3.0f, 1.0f, 3.0f};
        CHECK(cr::detail::se_mean(four) == Approx(std::sqrt(1.0 / 3.0)).margin(1e-6));
        const std::vector<float> halves{0.5f, 0.5f};
        CHECK(cr::detail::se_mean_t(halves, 64) == Approx(std::sqrt(0.5 / 64.0) / 2.0).margin(1e-12));
        CHECK(cr::detail::se_share(0.25, 16) == Approx(std::sqrt(0.25 * 0.75 / 16.0)).margin(1e-12));
    }
    SECTION("se_quantile on known ranks and small-n one-sided clamp") {
        // Non-uniform sorted spacing so off-by-one rank shifts change the spread.
        const std::vector<float> v{0, 1, 3, 6, 10, 15, 21, 28, 36, 45};
        // p50, n=10: d = sqrt(2.5) = 1.58114 -> lo_pos = 3.41886 (idx 3 -> 6),
        // hi_pos = 6.58114 (idx 6 -> 21); interior -> 0.5 * (21 - 6) = 7.5.
        CHECK(cr::detail::se_quantile(v, 0.50) == Approx(7.5).margin(1e-12));
        // Small-n p95, n=10: q*n = 9.5, d = sqrt(0.475) = 0.6892 -> hi_pos = 10.1892 > 10
        // (upper clamp!), lo_pos = 8.8108 (idx 8 -> 36), hi clamps to idx 9 -> 45.
        // Full one-sided spread = 45 - 36 = 9.0 (halving bug would give 4.5).
        CHECK(cr::detail::se_quantile(v, 0.95) == Approx(9.0).margin(1e-12));
        // Small-n p05, n=10: lo_pos = 0.5 - 0.6892 <= 0 (lower clamp!), hi_pos = 1.1892 (idx 1 -> 1).
        // Full one-sided spread = 1 - 0 = 1.0.
        CHECK(cr::detail::se_quantile(v, 0.05) == Approx(1.0).margin(1e-12));
        // Exact discrete handoff (Carbon Low #3): n=19, q=0.95 -> q*n = 18.05, d = 0.95,
        // hi_pos = 19.0 (not > 19), k_hi == k_q == 18, k_lo == 17 -> full one-sided step.
        std::vector<float> v19(19);
        for (int i = 0; i < 19; ++i) v19[size_t(i)] = float(i * i);
        CHECK(cr::detail::se_quantile(v19, 0.95) == Approx(double(18 * 18 - 17 * 17)).margin(1e-12));
        // Exact discrete handoff at lower end: n=10, q=0.10 -> k_lo == k_q == 0, k_hi == 1.
        CHECK(cr::detail::se_quantile(v, 0.10) == Approx(1.0).margin(1e-12));
    }
    SECTION("mae_se on known |err| list") {
        // {0.1, 0.3, 0.1, 0.3}: mean 0.2, SSE 0.04, var/n = 0.04 / 3 / 4 = 1/300.
        const std::vector<double> errs{0.1, 0.3, 0.1, 0.3};
        CHECK(cr::detail::mae_se(errs) == Approx(std::sqrt(1.0 / 300.0)).margin(1e-12));
    }
    SECTION("fit_se and analytic derivatives") {
        cr::FormatFootprint fp;
        fp.integrals = {{2.0f, 1.5f}};
        CHECK(cr::detail::dt_scalar(fp, 0.5) == Approx(-2.0 * std::exp(-1.0)).margin(1e-6));
        CHECK(cr::detail::dt_axis(fp, 0.8)   == Approx(-1.5 * std::exp(-1.2)).margin(1e-6));
        // SSE = 0.01 + 0.04 + 0.09 + 0.16 = 0.30, n - 1 = 3, J2 = 1 + 4 + 1 + 4 = 10 -> sqrt(0.01) = 0.1.
        const std::vector<double> res{0.1, -0.2, 0.3, -0.4};
        const std::vector<double> der{-1.0, -2.0, -1.0, -2.0};
        CHECK(cr::detail::fit_se(res, der) == Approx(0.1).margin(1e-12));
    }
}

TEST_CASE("[crown_oracle] G-P4 re-jitter stability across pinned elevations on fixture tree", "[treegen][crown_oracle]") {
    const treegen::GeneratedTree& g = rynx::test_support::generated_tree("oak", 42);
    const cr::Probe probe(cr::load_crown_from_glb(g.glb, 0).mesh);
    const std::vector<float> elevations = {10.0f, 20.0f, 30.0f, 45.0f, 60.0f, 70.0f, 75.0f, 90.0f};
    const int rays_fp = cr::rays_per_side(cr::RayStream::selection) * cr::rays_per_side(cr::RayStream::selection);
    int moved_any = 0;

    // Zenith ground-projected area & LAI via ProbeOptions::selection_stream plumbing.
    {
        cr::ProbeOptions oa, ob;
        oa.oracle = false; oa.sunlit = false; oa.selection_stream = cr::RayStream::selection;
        ob.oracle = false; ob.sunlit = false; ob.selection_stream = cr::RayStream::selection_alt;
        const auto pa = treegen::json::parse(cr::probe_report_json(probe, oa, "{}"));
        const auto pb = treegen::json::parse(cr::probe_report_json(probe, ob, "{}"));
        REQUIRE(pa.ok);
        REQUIRE(pb.ok);
        const auto& ta = *pa.root.find("tree");
        const auto& tb = *pb.root.find("tree");
        const double area_a = ta.find("crown_ground_projected_area_m2")->n;
        const double area_b = tb.find("crown_ground_projected_area_m2")->n;
        const double area_se = std::hypot(ta.find("crown_ground_projected_area_m2_se")->n,
                                          tb.find("crown_ground_projected_area_m2_se")->n);
        const double lai_a = ta.find("leaf_area_index")->n;
        const double lai_b = tb.find("leaf_area_index")->n;
        const double lai_se = std::hypot(ta.find("leaf_area_index_se")->n,
                                         tb.find("leaf_area_index_se")->n);
        const double fp_area = double(cr::k_footprint_m) * cr::k_footprint_m;
        const double sil_n = std::max(1.0, area_a / fp_area);
        INFO("zenith area: " << area_a << " vs " << area_b << " combined se " << area_se);
        CHECK(std::abs(area_a - area_b) <= cr::pin_band(area_se, fp_area));
        INFO("zenith lai: " << lai_a << " vs " << lai_b << " combined se " << lai_se);
        CHECK(std::abs(lai_a - lai_b) <= cr::pin_band(lai_se, lai_a / sil_n));
        if (area_a != area_b) ++moved_any;
    }

    // Per pinned elevation (10..90 deg): re-jitter selection vs selection_alt.
    // Two independent draws X_a, X_b have difference SE = hypot(se_a, se_b).
    for (float e : elevations) {
        const cr::SunDirection sun = cr::sun_direction(e, 0.0f);
        const cr::FootprintSet sa = cr::measure_footprints(probe, sun, cr::PlaneKind::beam, kAll, cr::RayStream::selection);
        const cr::FootprintSet sb = cr::measure_footprints(probe, sun, cr::PlaneKind::beam, kAll, cr::RayStream::selection_alt);
        int sil_a = 0, sil_b = 0, rim_a = 0, rim_b = 0;
        for (const auto& f : sa.footprints) if (f.silhouette()) { ++sil_a; if (!f.body()) ++rim_a; }
        for (const auto& f : sb.footprints) if (f.silhouette()) { ++sil_b; if (!f.body()) ++rim_b; }
        const cr::CountSEs csea = cr::footprint_count_se(sa, cr::k_body_edge_m);
        const cr::CountSEs cseb = cr::footprint_count_se(sb, cr::k_body_edge_m);
        const auto ba = cr::body_stats(sa, cr::k_body_edge_m);
        const auto bb = cr::body_stats(sb, cr::k_body_edge_m);
        INFO("e=" << e << " sil " << sil_a << " vs " << sil_b << " (se " << csea.silhouette_se << "/" << cseb.silhouette_se << ")"
                  << " rim " << rim_a << " vs " << rim_b << " (se " << csea.rim_se << "/" << cseb.rim_se << ")");
        CHECK(std::abs(sil_a - sil_b) <= cr::pin_band_count(std::hypot(csea.silhouette_se, cseb.silhouette_se)));
        CHECK(std::abs(rim_a - rim_b) <= cr::pin_band_count(std::hypot(csea.rim_se, cseb.rim_se)));
        if (sil_a != sil_b || rim_a != rim_b) ++moved_any;
        REQUIRE(ba.has_value());
        REQUIRE(bb.has_value());
        INFO("e=" << e << " body fp " << ba->footprints << " vs " << bb->footprints << " (se " << ba->footprints_se << "/" << bb->footprints_se << ")"
                  << " mean " << ba->mean << " vs " << bb->mean << " (se " << ba->mean_se << "/" << bb->mean_se << ")"
                  << " p50 " << ba->p50 << " vs " << bb->p50 << " (se " << ba->p50_se << "/" << bb->p50_se << ")"
                  << " p95 " << ba->p95 << " vs " << bb->p95 << " (se " << ba->p95_se << "/" << bb->p95_se << ")"
                  << " opaque " << ba->share_opaque << " vs " << bb->share_opaque << " (se " << ba->share_opaque_se << "/" << bb->share_opaque_se << ")");
        CHECK(std::abs(ba->footprints - bb->footprints) <= cr::pin_band_count(std::hypot(ba->footprints_se, bb->footprints_se)));
        CHECK(std::abs(ba->mean - bb->mean) <= cr::pin_band_mean_t(std::hypot(ba->mean_se, bb->mean_se), size_t(ba->footprints), rays_fp));
        CHECK(std::abs(ba->p50 - bb->p50) <= cr::pin_band_quantile(std::hypot(ba->p50_se, bb->p50_se), rays_fp));
        CHECK(std::abs(ba->p95 - bb->p95) <= cr::pin_band_quantile(std::hypot(ba->p95_se, bb->p95_se), rays_fp));
        CHECK(std::abs(ba->share_opaque - bb->share_opaque) <= cr::pin_band_share(std::hypot(ba->share_opaque_se, bb->share_opaque_se), size_t(ba->footprints)));
        if (ba->mean != bb->mean) ++moved_any;
    }
    CHECK(moved_any > 0);   // positive control: the two streams are genuinely distinct
}

TEST_CASE("[crown_oracle] G-P5a low-sun slab at 10 and 15 deg", "[treegen][crown_oracle]") {
    // 40 m x 40 m x 1.0 m slab so at 10 deg a ray spans 1.0 / tan(10 deg) = 5.67 m
    // horizontally and stays strictly inside slab edges for the central [14, 26]^2
    // window (14 m margin on every side).
    SECTION("horizontal slab: T = exp(-L) at 10 and 15 deg") {
        constexpr float w = 40.0f, d = 1.0f, side = 0.25f;
        constexpr int cards = 25600;   // 25600 * 0.0625 / 1600 = 1.0
        const double L = cards * double(side) * side / (double(w) * w);
        REQUIRE(L == Approx(1.0));
        const double expected = std::exp(-L);
        const float elevations[2] = {10.0f, 15.0f};
        std::vector<double> t[2];
        for (uint64_t seed = 1; seed <= 4; ++seed) {
            const cr::Probe probe(card_slab(seed, cards, side, w, d, true));
            for (int k = 0; k < 2; ++k)
                t[k].push_back(ray_share(probe, cr::sun_direction(elevations[k], 0.0f).dir, 14.0f, 12.0f, 0.5f, 128, 400 + seed));
        }
        for (int k = 0; k < 2; ++k) {
            const SeedStats s = four_seed(t[k]);
            INFO("horizontal e = " << elevations[k] << ": mean " << s.mean << " (se " << s.se << "), expected " << expected);
            CHECK(std::abs(s.mean - expected) <= 0.05 * expected);
            CHECK(0.05 * expected >= 3.0 * s.se);
        }
    }
    SECTION("random-orientation slab: T = exp(-0.5 L / sin e) at 10 and 15 deg") {
        // L = 0.25 puts T(10 deg) = 0.487 and T(15 deg) = 0.617 squarely in [0.2, 0.8],
        // gating oblique cos(e)/sin(e) path length directly at low sun.
        constexpr float w = 40.0f, d = 1.0f, side = 0.25f;
        constexpr int cards = 6400;    // 6400 * 0.0625 / 1600 = 0.25
        const double L = cards * double(side) * side / (double(w) * w);
        REQUIRE(L == Approx(0.25));
        const float elevations[2] = {10.0f, 15.0f};
        std::vector<double> t[2];
        for (uint64_t seed = 1; seed <= 4; ++seed) {
            const cr::Probe probe(card_slab(10 + seed, cards, side, w, d, false));
            for (int k = 0; k < 2; ++k)
                t[k].push_back(ray_share(probe, cr::sun_direction(elevations[k], 0.0f).dir, 14.0f, 12.0f, 0.5f, 128, 500 + seed));
        }
        for (int k = 0; k < 2; ++k) {
            const double expected = std::exp(-0.5 * L / std::sin(elevations[k] * k_pi / 180.0));
            REQUIRE(expected >= 0.2);
            REQUIRE(expected <= 0.8);
            const SeedStats s = four_seed(t[k]);
            INFO("random e = " << elevations[k] << ": mean " << s.mean << " (se " << s.se << "), expected " << expected);
            CHECK(std::abs(s.mean - expected) <= 0.05 * expected);
            CHECK(0.05 * expected >= 3.0 * s.se);
        }
    }
}

TEST_CASE("[crown_oracle] G-P5b/P5c optical_depth directional (3,2,1) and scalar channel gates", "[treegen][crown_oracle]") {
    constexpr int n = 9;
    constexpr float cell = 0.2f;
    std::vector<float> cells(size_t(n) * n * n * 6, 0.0f);
    const size_t centre = ((size_t(4) * n + 4) * n + 4) * 6;
    // Distinct 6-channel PSD tensor so any channel swap or off-diagonal sign error fails.
    cells[centre + 0] = 0.5f;   // M_xx
    cells[centre + 1] = 2.0f;   // M_yy
    cells[centre + 2] = 7.0f;   // M_zz
    cells[centre + 3] = 0.4f;   // M_xy
    cells[centre + 4] = -0.3f;  // M_xz
    cells[centre + 5] = 0.6f;   // M_yz
    const cr::CrownVolume v = cr::CrownVolume::from_cells(vec3{-0.9f, -0.9f, -0.9f}, cell, n, n, n, cells);
    // Ray through origin along normalized({3, 2, 1}) exercises all 6 tensor channels
    // asymmetrically; crosses [-0.1, 0.1]^3 between x = -0.1 and x = +0.1:
    // exact path length = 0.2 * sqrt(14) / 3.
    const vec3 dir = treegen::normalized(vec3{3.0f, 2.0f, 1.0f});
    const vec3 origin = -dir * 5.0f;
    const double len = 0.2 * std::sqrt(14.0) / 3.0;
    const double rho = 0.5 + 2.0 + 7.0;

    SECTION("G-P5b tensor along non-symmetric (3, 2, 1)") {
        const cr::ExtinctionModel tensor{cr::ExtinctionKind::tensor, 0.5f, 1.0f};
        const double wx = double(dir.x), wy = double(dir.y), wz = double(dir.z);
        const double quad = wx * wx * 0.5 + wy * wy * 2.0 + wz * wz * 7.0
                          + 2.0 * (wx * wy * 0.4 + wx * wz * (-0.3) + wy * wz * 0.6);
        const double expected = std::sqrt(0.75 * rho * quad) * len;
        const float tau = v.optical_depth(origin, dir, 0.0f, tensor);
        CHECK(tau == Approx(expected).margin(1e-5));
        CHECK(tau == v.integrals(origin, dir)[1]);
    }
    SECTION("G-P5c scalar extinction reads tr(M) * g * clumping") {
        const cr::ExtinctionModel scalar{cr::ExtinctionKind::scalar, 0.7f, 1.25f};
        const double expected = 0.7 * rho * len * 1.25;
        CHECK(v.optical_depth(origin, dir, 0.0f, scalar) == Approx(expected).margin(1e-5));
    }
}

TEST_CASE("[crown_oracle] G-D2 load_crown_from_glb decodes 8x8 asymmetric RGBA atlas and alphaCutoff", "[treegen][crown_oracle]") {
    // 8 x 8 RGBA PNG: rows 0-2 have A = 166 (>= 0.65 * 255 = 165.75, passes) and R = 0;
    // rows 3-7 have A = 165 (< 165.75, fails at 0.65, but >= 127.5 so passes if cutoff hardcoded to 0.5) and R = 255.
    std::vector<uint8_t> rgba(8 * 8 * 4, 0);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            const size_t i = (size_t(y) * 8 + size_t(x)) * 4;
            rgba[i + 0] = y < 3 ? 0 : 255;     // R inverted vs A (catches channel 3 -> 0)
            rgba[i + 1] = 10;
            rgba[i + 2] = 20;
            rgba[i + 3] = y < 3 ? 166 : 165;   // straddles 0.65 * 255 = 165.75; both >= 0.5 * 255
        }
    treegen::ImageData img;
    img.png_bytes = treegen::encode_png_rgba8(8, 8, 8 * 4, rgba.data());
    REQUIRE_FALSE(img.png_bytes.empty());
    const std::vector<treegen::ImageData> images = {img, img};

    // 2-triangle 1 m square card at z = 0, UV [0,1]^2 (same layout as (a')).
    const std::vector<float> pos = {-0.5f, -0.5f, 0.0f,  0.5f, -0.5f, 0.0f,  0.5f, 0.5f, 0.0f,  -0.5f, 0.5f, 0.0f};
    const std::vector<float> nrm = {0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1};
    const std::vector<float> uv  = {0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f};
    const std::vector<uint32_t> idx = {0, 1, 2,  0, 2, 3};

    treegen::PrimitiveData prim{};
    prim.positions   = pos;
    prim.normals     = nrm;
    prim.uvs         = uv;
    prim.indices_u32 = idx;
    prim.material.alpha_mode             = "MASK";
    prim.material.alpha_cutoff           = 0.65f;
    prim.material.base_color_tex_index   = 0;
    prim.material.translucency_tex_index = 1;
    prim.material.sampler_index          = 1;

    const treegen::PrimitiveData prims[1] = {prim};
    treegen::MeshData mesh{};
    mesh.primitives = prims;
    const treegen::MeshData meshes[1] = {mesh};

    std::vector<uint8_t> glb;
    std::string err;
    REQUIRE(treegen::build_glb_multi_mesh(meshes, images, glb, &err));

    cr::LoadedCrown loaded = cr::load_crown_from_glb(glb, 0);
    CHECK(loaded.alpha_cutoff == Approx(0.65f));
    CHECK(loaded.mesh.bark_triangles == 0u);
    REQUIRE(loaded.mesh.leaf_triangles == 2u);
    const cr::Probe probe(std::move(loaded.mesh));
    // Exact (a') coverage integers: tri 0 covers 21/36 opaque texels, tri 1 covers 6/36.
    CHECK(std::abs(probe.triangle_coverage(0) - 21.0f / 36.0f) <= 1e-6f);
    CHECK(std::abs(probe.triangle_coverage(1) - 6.0f / 36.0f) <= 1e-6f);
    // Exact hit pattern: v = 0.2 (y = -0.3, row 1) occluded; v = 0.8 (y = +0.3, row 6) open.
    CHECK(probe.occluded(vec3{0.0f, -0.3f, -1.0f}, vec3{0, 0, 1}, -1, kAll));
    CHECK_FALSE(probe.occluded(vec3{0.0f, 0.3f, -1.0f}, vec3{0, 0, 1}, -1, kAll));
}

TEST_CASE("[crown_oracle] G-D3 two-layer sunlit oblique sun, eyes, open_sky, sunlit_by_depth", "[treegen][crown_oracle]") {
    auto field = [](const treegen::json::value& v, const char* k) -> const treegen::json::value& {
        const treegen::json::value* f = v.find(k);
        REQUIRE(f);
        return *f;
    };

    // Asymmetric two-layer crown covering [-0.3, 2.1]^2:
    // top at z = 3.0: 4 x 4 = 16 cards of side 0.6 m (80% of sites, open_sky = 1.0);
    // bottom at z = 1.0: 2 x 2 = 4 cards of side 1.2 m (20% of sites, occluded above).
    // The 16-vs-4 layer asymmetry breaks z-reflection symmetry so flipping +d -> -d
    // in open_sky changes the distribution as well as ignoring the upper layer.
    Prim p;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            p.card(vec3{0.6f * float(i), 0.6f * float(j), 3.0f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 0.6f);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            p.card(vec3{0.3f + 1.2f * float(i), 0.3f + 1.2f * float(j), 1.0f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 1.2f);
    cr::CrownMesh m;
    m.add_leaves(p.pos, p.uv, p.idx);
    REQUIRE(m.sites.size() == 20u);
    const cr::Probe probe(std::move(m));

    // Analytic open_sky (64 Fibonacci upper-hemisphere directions, exact rect intersection with z = 3.0 sheet).
    std::vector<vec3> sky_dirs;
    const double golden = k_pi * (3.0 - std::sqrt(5.0));
    for (int k = 0; k < cr::k_open_sky_directions; ++k) {
        const double z = (k + 0.5) / cr::k_open_sky_directions, r = std::sqrt(1.0 - z * z), ph = golden * k;
        sky_dirs.push_back(vec3{float(r * std::cos(ph)), float(r * std::sin(ph)), float(z)});
    }
    std::vector<float> expected_sky;
    for (const auto& st : probe.mesh().sites) {
        int open = 0;
        for (const vec3& d : sky_dirs) {
            bool hit = false;
            if (st.centroid.z < 2.0f) {
                const float t = (3.0f - st.centroid.z) / d.z;
                const float x = st.centroid.x + t * d.x, y = st.centroid.y + t * d.y;
                hit = (x >= -0.3f && x <= 2.1f && y >= -0.3f && y <= 2.1f);
            }
            if (!hit) ++open;
        }
        expected_sky.push_back(float(open) / float(sky_dirs.size()));
    }
    std::sort(expected_sky.begin(), expected_sky.end());
    double exp_sky_mean = 0.0;
    for (float x : expected_sky) exp_sky_mean += x;
    exp_sky_mean /= double(expected_sky.size());
    auto pct = [&](double q) {
        const size_t k = static_cast<size_t>(std::ceil(q * double(expected_sky.size())));
        return double(expected_sky[std::min(expected_sky.size() - 1, k == 0 ? 0 : k - 1)]);
    };
    REQUIRE(pct(0.10) < 0.90);   // bottom layer is genuinely shadowed by the upper layer
    REQUIRE(pct(0.50) == 1.0);   // 16 of 20 sites sit on the unoccluded top layer

    // Oblique sun 45 deg, azimuth 30 deg (azimuths = 12 -> index 1 is az = 30 deg).
    cr::ProbeOptions o;
    o.oracle = true;
    o.sunlit = true;
    o.elevations_deg = {45.0f};
    o.azimuths = 12;
    o.eyes = {vec3{0.9f, 0.9f, -10.0f}, vec3{0.9f, 0.9f, 10.0f}};
    const auto parsed = treegen::json::parse(cr::probe_report_json(probe, o, "{}"));
    REQUIRE(parsed.ok);

    // open_sky checks (16 top sites = 1.0, 4 bottom sites occluded by top layer).
    const auto& sky = field(field(field(parsed.root, "modes"), "sunlit"), "open_sky");
    CHECK(field(sky, "mean").n == Approx(exp_sky_mean).margin(1e-5));
    CHECK(field(sky, "p10").n  == Approx(pct(0.10)).margin(1e-5));
    CHECK(field(sky, "p50").n  == Approx(1.0).margin(1e-5));
    CHECK(field(sky, "p90").n  == Approx(1.0).margin(1e-5));

    // Sunlit direction (45 deg, az 30 deg): ray from bottom (x, y, 1) hits z=3 at
    // (x + 2*cos(30), y + 2*sin(30)) = (x + 1.732, y + 1.0). Inside [-0.3, 2.1]^2
    // only for (0.3, 0.3, 1.0) -> (2.032, 1.3): 1 bottom site shadowed, 3 bottom + 16 top = 19/20 lit.
    const auto& sdirs = field(field(field(parsed.root, "modes"), "sunlit"), "directions").a;
    const treegen::json::value* sd30 = nullptr;
    for (const auto& d : sdirs)
        if (std::abs(field(d, "azimuth_deg").n - 30.0) < 1e-3) { sd30 = &d; break; }
    REQUIRE(sd30 != nullptr);
    CHECK(field(*sd30, "share").n == Approx(19.0 / 20.0).margin(1e-6));

    // Below-crown eye sees only the 4 bottom sites (3/4 lit); above-crown eye sees only 16 top (16/16 lit).
    const auto& eyes = field(*sd30, "eyes").a;
    REQUIRE(eyes.size() == 2u);
    CHECK(field(eyes[0], "visible_sites").n == 4.0);
    CHECK(field(eyes[0], "share").n == Approx(3.0 / 4.0).margin(1e-6));
    CHECK(field(eyes[1], "visible_sites").n == 16.0);
    CHECK(field(eyes[1], "share").n == Approx(1.0).margin(1e-6));

    // sunlit_by_depth at oblique 45 deg, az 30 deg: match analytic bins.
    const auto& odirs = field(field(field(parsed.root, "modes"), "oracle"), "directions").a;
    const treegen::json::value* od30 = nullptr;
    for (const auto& d : odirs)
        if (std::abs(field(d, "azimuth_deg").n - 30.0) < 1e-3) { od30 = &d; break; }
    REQUIRE(od30 != nullptr);
    const vec3 s30 = cr::sun_direction(45.0f, 30.0f).dir;
    const float top30 = treegen::dot(vec3{2.1f, 2.1f, 3.0f}, s30);
    std::map<int, std::pair<int, int>> exp_bins;
    for (const auto& st : probe.mesh().sites) {
        const int bin = int(std::floor((top30 - treegen::dot(st.centroid, s30)) / 0.5f));
        const bool is_bot = st.centroid.z < 2.0f;
        const bool shadowed = is_bot && st.centroid.x < 0.9f && st.centroid.y < 0.9f;
        ++exp_bins[bin].first;
        exp_bins[bin].second += shadowed ? 0 : 1;
    }
    const auto& got_bins = field(*od30, "sunlit_by_depth").a;
    REQUIRE(got_bins.size() == exp_bins.size());
    size_t bi = 0;
    for (const auto& [k, cnt] : exp_bins) {
        CHECK(field(got_bins[bi], "depth_min_m").n == Approx(k * 0.5).margin(1e-5));
        CHECK(field(got_bins[bi], "sites").n == double(cnt.first));
        CHECK(field(got_bins[bi], "sunlit_share").n == Approx(double(cnt.second) / double(cnt.first)).margin(1e-6));
        ++bi;
    }

    // Zenith sunlit_by_depth: exact 2 bins [0.0, 0.5) (16 sites, 1.0) and [2.0, 2.5) (4 sites, 0.0).
    cr::ProbeOptions oz;
    oz.oracle = true;
    oz.sunlit = false;
    oz.elevations_deg = {90.0f};
    oz.azimuths = 1;
    const auto pz = treegen::json::parse(cr::probe_report_json(probe, oz, "{}"));
    REQUIRE(pz.ok);
    const auto& zbins = field(field(field(field(pz.root, "modes"), "oracle"), "directions").a.at(0), "sunlit_by_depth").a;
    REQUIRE(zbins.size() == 2u);
    CHECK(field(zbins[0], "depth_min_m").n == Approx(0.0));
    CHECK(field(zbins[0], "sites").n == 16.0);
    CHECK(field(zbins[0], "sunlit_share").n == Approx(1.0));
    CHECK(field(zbins[1], "depth_min_m").n == Approx(2.0));
    CHECK(field(zbins[1], "sites").n == 4.0);
    CHECK(field(zbins[1], "sunlit_share").n == Approx(0.0));
}

TEST_CASE("[crown_oracle] G-D4 emit_levels rejects mismatched scenario_fnv and probe_glb_report hashes actual bytes", "[treegen][crown_oracle]") {
    SECTION("emit_levels throws on mismatched GrownTree::scenario_fnv") {
        const treegen::Scenario oak  = treegen::load_scenario(rynx::test_support::treegen_file("scenarios/c3_oak.json"));
        const treegen::Scenario pine = treegen::load_scenario(rynx::test_support::treegen_file("scenarios/c3_pine.json"));
        REQUIRE(oak.scenario_fnv != pine.scenario_fnv);
        const treegen::GrownTree grown = treegen::grow_tree(oak, 42);
        CHECK(grown.scenario_fnv == oak.scenario_fnv);
        CHECK_THROWS_AS(treegen::emit_levels(pine, grown), std::logic_error);
        treegen::GrownTree tampered = grown;
        tampered.scenario_fnv ^= 1ull;
        CHECK_THROWS_AS(treegen::emit_levels(oak, std::move(tampered)), std::logic_error);
        CHECK_NOTHROW(treegen::emit_levels(oak, grown));
    }
    SECTION("probe_glb_report computes glb_fnv1a64 and probe_lod from actual probed bytes and lod") {
        const treegen::GeneratedTree& g = rynx::test_support::generated_tree("oak", 42);
        cr::ProbeOptions o;
        o.oracle = false;
        o.sunlit = false;
        const std::string doc = cr::probe_glb_report(g.glb, 1, o, cr::ProbeInput::glb("oak_lod1.glb"));
        CHECK(doc.find("\"probe_lod\": 1") != std::string::npos);
        CHECK(doc.find("\"glb_fnv1a64\": \"" + expected_digest(g.glb) + "\"") != std::string::npos);
    }
}

// ---- S-P1 Crown Density Volume Gates (F2_tensor Option 2A) -------------------

TEST_CASE("[crown_volume_fidelity] F2_tensor Option 2A matches leaves-only oracle across all 4 species and grazing zero",
          "[treegen][crown_volume_fidelity]") {
    const std::vector<float> elevations = {10.0f, 20.0f, 30.0f, 45.0f, 60.0f, 70.0f, 75.0f, 90.0f};
    const auto suns = cr::sun_directions(elevations, 4);
    REQUIRE(suns.size() == 29u);  // 7 * 4 + 1 deduped zenith

    struct Spec { const char* name; double max_overall; double max_el_tri; double max_el_dda; };
    const Spec specs[4] = {
        {"oak",   0.050, 0.053, 0.050},
        {"maple", 0.050, 0.053, 0.050},
        {"birch", 0.050, 0.053, 0.050},
        {"pine",  0.060, 0.060, 0.060},
    };

    for (const Spec& sp : specs) {
        const treegen::GeneratedTree& g = rynx::test_support::generated_tree(sp.name, 42);
        REQUIRE(g.crown_volume.has_value());
        const cr::PackedCrownVolume& packed = *g.crown_volume;
        const cr::Probe probe(cr::load_crown_from_glb(g.glb, 0).mesh);

        struct Acc {
            double abs_tri = 0.0, abs_dda = 0.0;
            double so = 0.0, sp = 0.0, soo = 0.0, spp = 0.0, sop = 0.0;
            size_t n = 0;
            void add(double o, double t_tri, double t_dda) {
                abs_tri += std::abs(t_tri - o);
                abs_dda += std::abs(t_dda - o);
                so += o; sp += t_tri; soo += o * o; spp += t_tri * t_tri; sop += o * t_tri;
                ++n;
            }
        };
        std::map<float, Acc> by_el;
        Acc all;

        for (const cr::SunDirection& sun : suns) {
            const cr::FootprintSet set = cr::measure_footprints(probe, sun, cr::PlaneKind::beam, kLeaves);
            std::vector<size_t> chosen;
            for (size_t k = 0; k < set.footprints.size(); ++k) {
                const float t = set.footprints[k].t;
                if (t >= cr::k_opaque_t && t <= cr::k_unsaturated_hi) chosen.push_back(k);
            }
            struct Res { double o, tri, dda; };
            std::vector<Res> rows(chosen.size());
            cr::parallel_for(chosen.size(), [&](size_t ci) {
                std::vector<vec3> rays;
                set.rays(chosen[ci], rays, cr::RayStream::format);
                int open = 0;
                double s_tri = 0.0, s_dda = 0.0;
                for (const vec3& o : rays) {
                    open  += probe.occluded(o, sun.dir, -1, kLeaves) ? 0 : 1;
                    s_tri += packed.transmittance_trilinear(o, sun.dir, cr::k_default_trilinear_steps, false);
                    s_dda += packed.transmittance_dda(o, sun.dir, false);
                }
                const double inv = 1.0 / double(rays.size());
                rows[ci] = {double(open) * inv, s_tri * inv, s_dda * inv};
            });
            for (const auto& r : rows) {
                by_el[sun.elevation_deg].add(r.o, r.tri, r.dda);
                all.add(r.o, r.tri, r.dda);
            }
        }

        REQUIRE(all.n >= 300u);
        const double mae_tri = all.abs_tri / double(all.n);
        const double dn = double(all.n);
        const double cov = all.sop - (all.so * all.sp) / dn;
        const double vo  = all.soo - (all.so * all.so) / dn;
        const double vp  = all.spp - (all.sp * all.sp) / dn;
        const double r_corr = cov / std::sqrt(std::max(1e-30, vo * vp));

        INFO(sp.name << " overall: tri_MAE=" << mae_tri << " r=" << r_corr << " n=" << all.n);
        CHECK(mae_tri <= sp.max_overall);
        CHECK(r_corr >= 0.90);

        for (const auto& [el, a] : by_el) {
            const double el_tri = a.abs_tri / double(a.n);
            const double el_dda = a.abs_dda / double(a.n);
            INFO(sp.name << " el=" << el << " n=" << a.n << " tri_MAE=" << el_tri << " dda_MAE=" << el_dda);
            CHECK(a.n >= size_t(cr::k_min_unsaturated));
            CHECK(el_tri <= sp.max_el_tri);
            CHECK(el_dda <= sp.max_el_dda);
        }
    }

    // Grazing flat-card arm (catches M3 UNORM+0.5 grazing floor): horizontal card
    // with normal +z must produce bit-exact 0.0 extinction along both (1,0,0)
    // and 45-deg azimuth (1/sqrt(2), 1/sqrt(2), 0).
    {
        Prim p;
        p.card(vec3{0.0f, 0.0f, 0.0f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 1.0f);
        cr::CrownMesh m;
        m.add_leaves(p.pos, p.uv, p.idx);
        const cr::Probe probe(std::move(m));
        const cr::CrownVolume vol = cr::CrownVolume::build_shipped(probe);
        const cr::PackedCrownVolume packed = vol.pack_option2a();
        const float s2 = float(std::sqrt(0.5));
        const vec3 grazing_dirs[3] = {vec3{1, 0, 0}, vec3{0, 1, 0}, vec3{s2, s2, 0}};
        for (const vec3& wd : grazing_dirs) {
            CHECK(packed.sample_trilinear_sigma(vec3{0.5f, 0.5f, 0.25f}, wd) == 0.0f);
            CHECK(packed.sample_voxel_sigma(packed.dims[0] / 2, packed.dims[1] / 2, 0, wd) == 0.0f);
        }
        // Positive control: along +z normal, extinction is strongly non-zero.
        CHECK(packed.sample_voxel_sigma(packed.dims[0] / 2, packed.dims[1] / 2, 0, vec3{0, 0, 1}) > 1.0f);
    }
}

TEST_CASE("[crown_volume_self_shadow] K4 receiver exclusion depth-binned sunlit share on oak_42 at 45 and 70 deg",
          "[treegen][crown_volume_self_shadow]") {
    const treegen::GeneratedTree& g = rynx::test_support::generated_tree("oak", 42);
    REQUIRE(g.crown_volume.has_value());
    const cr::PackedCrownVolume& packed = *g.crown_volume;
    const cr::Probe probe(cr::load_crown_from_glb(g.glb, 0).mesh);

    for (float el : {45.0f, 70.0f}) {
        const auto suns = cr::sun_directions({el}, 4);
        struct Bin { int count = 0; double oracle_sum = 0.0, pred_sum = 0.0; };
        std::map<int, Bin> bins;
        for (const cr::SunDirection& sun : suns) {
            float top = -1e30f;
            for (const cr::Triangle& t : probe.mesh().triangles)
                if (t.site >= 0) for (const vec3& p : t.p) top = std::max(top, treegen::dot(p, sun.dir));
            for (const cr::Site& st : probe.mesh().sites) {
                const int b = int(std::floor((top - treegen::dot(st.centroid, sun.dir)) / 0.5f));
                const double o = probe.occluded(st.centroid, sun.dir, st.id, kLeaves) ? 0.0 : 1.0;
                const double p = packed.transmittance_trilinear(st.centroid, sun.dir, cr::k_default_trilinear_steps, true);
                auto& acc = bins[b];
                ++acc.count;
                acc.oracle_sum += o;
                acc.pred_sum   += p;
            }
        }
        int checked_bins = 0;
        for (const auto& [k, a] : bins) {
            if (a.count < 20) continue;
            ++checked_bins;
            const double o_share = a.oracle_sum / double(a.count);
            const double p_share = a.pred_sum / double(a.count);
            const double err = std::abs(p_share - o_share);
            INFO("oak_42 el=" << el << " bin [" << (k * 0.5) << "," << ((k + 1) * 0.5) << ") n=" << a.count
                              << " oracle=" << o_share << " pred=" << p_share << " err=" << err);
            CHECK(err <= 0.15);
        }
        CHECK(checked_bins >= 10);
    }
}

TEST_CASE("[crown_volume_invariants] trace conservation, PSD/NaN-free, K8 lowpass, K16 normals, L0-AABB/union-pad, 90deg dedup, GLB v2",
          "[treegen][crown_volume_invariants]") {
    namespace gv = rynx::graphics::gltf_view;

    SECTION("unclamped trace conservation (< 1e-4) on oak_42") {
        const treegen::GeneratedTree& g = rynx::test_support::generated_tree("oak", 42);
        const cr::Probe probe(cr::load_crown_from_glb(g.glb, 0).mesh);
        double exact_opaque_area = 0.0;
        for (uint32_t ti = 0; ti < probe.mesh().triangles.size(); ++ti) {
            const cr::Triangle& t = probe.mesh().triangles[ti];
            if (t.site < 0) continue;
            const double area = 0.5 * treegen::length(treegen::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
            exact_opaque_area += area * probe.triangle_coverage(ti);
        }
        const cr::CrownVolume vol = cr::CrownVolume::build_shipped(probe);
        const double rel_err = std::abs(vol.total_area_m2() - exact_opaque_area) / exact_opaque_area;
        INFO("vol area=" << vol.total_area_m2() << " exact=" << exact_opaque_area << " rel_err=" << rel_err);
        CHECK(rel_err < 1e-4);
    }

    SECTION("PSD / NaN-free Option 2A decode across 1000 Fibonacci directions on bimodal and 45-deg diagonal cards") {
        Prim p;
        const float s2 = float(std::sqrt(0.5));
        // Diagonal 45-deg cards (+n and -n off-diagonals) + orthogonal bimodal cards.
        p.card(vec3{0.0f, 0.0f, 0.0f}, vec3{s2, -s2, 0}, vec3{0, 0, 1}, 0.5f);  // normal (+s2, +s2, 0) -> M_xy > 0
        p.card(vec3{0.6f, 0.0f, 0.0f}, vec3{s2,  s2, 0}, vec3{0, 0, 1}, 0.5f);  // normal (-s2, +s2, 0) -> M_xy < 0
        p.card(vec3{0.0f, 0.6f, 0.0f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 0.5f);     // bimodal +z
        p.card(vec3{0.0f, 0.6f, 0.0f}, vec3{0, 1, 0}, vec3{0, 0, 1}, 0.5f);     // bimodal +x
        cr::CrownMesh m;
        m.add_leaves(p.pos, p.uv, p.idx);
        const cr::Probe probe(std::move(m));
        const cr::CrownVolume vol = cr::CrownVolume::build_shipped(probe);
        const cr::PackedCrownVolume packed = vol.pack_option2a();

        const double golden = k_pi * (3.0 - std::sqrt(5.0));
        bool saw_negative_offdiag = false;
        for (size_t i = 0; i < packed.tex1.size(); i += 4)
            if (packed.tex1[i] < 0) { saw_negative_offdiag = true; break; }
        CHECK(saw_negative_offdiag);

        for (int k = 0; k < 1000; ++k) {
            const double z = 1.0 - 2.0 * (double(k) + 0.5) / 1000.0;
            const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double ph = golden * double(k);
            const vec3 w{float(r * std::cos(ph)), float(r * std::sin(ph)), float(z)};
            for (float u : {0.1f, 0.5f, 0.9f}) {
                const float st = packed.sample_trilinear_sigma(vec3{u, 0.5f, 0.5f}, w);
                REQUIRE(std::isfinite(st));
                REQUIRE(st >= 0.0f);
            }
        }
    }

    SECTION("K8 0.6m FP32 ensure_lowpass keeps _RYNX_CROWN_VOLUME strictly unblurred") {
        constexpr int n = 9;
        constexpr float cell = 0.2f;
        std::vector<float> cells(size_t(n) * n * n * 6, 0.0f);
        const size_t c_idx = ((size_t(4) * n + 4) * n + 4) * 6;
        cells[c_idx + 2] = 2.0f;  // M_zz = 2.0 in center voxel only
        const cr::CrownVolume v = cr::CrownVolume::from_cells(vec3{-0.9f, -0.9f, -0.9f}, cell, n, n, n, cells);

        // Invoke K8 0.6 m low-pass first, verify conservation (<1%) & second moment (within 5%).
        constexpr float sigma = 0.6f;
        const cr::ExtinctionModel tensor{cr::ExtinctionKind::tensor, 0.5f, 1.0f};
        const vec3 dir{0, 0, 1};
        const double step = 0.05;
        double m0 = 0.0, mx2 = 0.0, my2 = 0.0;
        for (int ix = -60; ix <= 60; ++ix)
            for (int iy = -60; iy <= 60; ++iy) {
                const double x = ix * step, y = iy * step;
                const double tau = v.optical_depth(vec3{float(x), float(y), -10.0f}, dir, sigma, tensor);
                m0 += tau; mx2 += tau * x * x; my2 += tau * y * y;
            }
        const double exp_var = double(sigma) * sigma + double(cell) * cell / 12.0;
        const double exp_int = std::sqrt(0.75) * 2.0 * double(cell) * cell * cell;
        CHECK(std::abs(mx2 / m0 - exp_var) <= 0.05 * exp_var);
        CHECK(std::abs(my2 / m0 - exp_var) <= 0.05 * exp_var);
        CHECK(std::abs(m0 * step * step - exp_int) <= 0.01 * exp_int);

        // Packing after ensure_lowpass MUST remain strictly unblurred (sigma = 0):
        // only the single center voxel (4,4,4) may be non-zero.
        const cr::PackedCrownVolume packed = v.pack_option2a();
        int nonzero_voxels = 0;
        for (int z = 0; z < n; ++z)
            for (int y = 0; y < n; ++y)
                for (int x = 0; x < n; ++x) {
                    const size_t b = ((size_t(z) * n + y) * n + x) * 4u;
                    const bool nz = (packed.tex0[b] | packed.tex0[b + 1] | packed.tex0[b + 2]) != 0;
                    if (nz) {
                        ++nonzero_voxels;
                        CHECK(x == 4); CHECK(y == 4); CHECK(z == 4);
                    }
                }
        CHECK(nonzero_voxels == 1);
    }

    SECTION("K16 geometric-normal invariance (corrupted vertex NORMAL does not change packed volume)") {
        // Vertical card in YZ plane (geometric normal = +x) with bogus vertex NORMAL = +z.
        const std::vector<float> pos = {0.0f, -0.5f, -0.5f,  0.0f, 0.5f, -0.5f,  0.0f, 0.5f, 0.5f,  0.0f, -0.5f, 0.5f};
        const std::vector<float> bogus_nrm = {0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1};
        const std::vector<float> true_nrm  = {1, 0, 0,  1, 0, 0,  1, 0, 0,  1, 0, 0};
        const std::vector<float> uv  = {0, 0,  1, 0,  1, 1,  0, 1};
        const std::vector<uint32_t> idx = {0, 1, 2,  0, 2, 3};
        std::vector<uint8_t> rgba(4 * 4 * 4, 255);
        treegen::ImageData img;
        img.png_bytes = treegen::encode_png_rgba8(4, 4, 16, rgba.data());
        const std::vector<treegen::ImageData> images = {img, img};

        auto make_glb = [&](const std::vector<float>& nrm) {
            treegen::PrimitiveData pr{};
            pr.positions = pos; pr.normals = nrm; pr.uvs = uv; pr.indices_u32 = idx;
            pr.material.alpha_mode = "MASK"; pr.material.alpha_cutoff = 0.5f;
            pr.material.base_color_tex_index = 0; pr.material.translucency_tex_index = 1;
            treegen::MeshData md{};
            const treegen::PrimitiveData arr[1] = {pr};
            md.primitives = arr;
            const treegen::MeshData marr[1] = {md};
            std::vector<uint8_t> out;
            std::string err;
            REQUIRE(treegen::build_glb_multi_mesh(marr, images, out, &err));
            return out;
        };
        const auto glb_true  = make_glb(true_nrm);
        const auto glb_bogus = make_glb(bogus_nrm);
        const cr::PackedCrownVolume p_true  = cr::CrownVolume::build_shipped(cr::Probe(cr::load_crown_from_glb(glb_true, 0).mesh)).pack_option2a();
        const cr::PackedCrownVolume p_bogus = cr::CrownVolume::build_shipped(cr::Probe(cr::load_crown_from_glb(glb_bogus, 0).mesh)).pack_option2a();
        CHECK(p_true.tex0 == p_bogus.tex0);
        CHECK(p_true.tex1 == p_bogus.tex1);
        CHECK(p_bogus.sample_voxel_sigma(0, p_bogus.dims[1] / 2, p_bogus.dims[2] / 2, vec3{1, 0, 0}) > 1.0f);
        CHECK(p_bogus.sample_voxel_sigma(0, p_bogus.dims[1] / 2, p_bogus.dims[2] / 2, vec3{0, 0, 1}) == 0.0f);
    }

    SECTION("L0-AABB cell sizing (catches M6 union-ext sizing) and union-of-LODs zero-padded outer shell") {
        // Synthetic M6 tripwire: L0 extent = 10m (cell_m = max(0.20, 10/64) = 0.20m),
        // union extent = 16m (16/64 = 0.25m != 0.20m).
        Prim p;
        p.card(vec3{0.0f, 0.0f, 0.0f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 1.0f);
        p.card(vec3{9.0f, 0.0f, 0.0f}, vec3{1, 0, 0}, vec3{0, 1, 0}, 1.0f);
        cr::CrownMesh m;
        m.add_leaves(p.pos, p.uv, p.idx);
        const cr::Probe probe(std::move(m));
        const cr::CrownVolume vol = cr::CrownVolume::build_shipped(probe, vec3{-3.5f, -1.0f, -1.0f}, vec3{12.5f, 1.0f, 1.0f});
        CHECK(vol.cell_size() == Approx(0.20f).margin(1e-6f));
        CHECK(vol.box_min().x <= -3.5f);
        CHECK(vol.box_max().x >= 12.5f);

        // Real trees: verify L0..L2 leaf vertices inside [box_min, box_max] and outer padded shell all-zero.
        for (const char* sp : {"oak", "maple", "birch", "pine"}) {
            const treegen::GeneratedTree& g = rynx::test_support::generated_tree(sp, 42);
            REQUIRE(g.crown_volume.has_value());
            const cr::PackedCrownVolume& packed = *g.crown_volume;
            const cr::Probe l0_probe(cr::load_crown_from_glb(g.glb, 0).mesh);
            vec3 umin{1e30f, 1e30f, 1e30f}, umax{-1e30f, -1e30f, -1e30f};
            for (const auto& lod : g.lods) {
                if (!lod.has_leaves) continue;
                for (size_t i = 0; i + 2 < lod.leaf_positions.size(); i += 3) {
                    const float x = lod.leaf_positions[i + 0];
                    const float y = lod.leaf_positions[i + 1];
                    const float z = lod.leaf_positions[i + 2];
                    REQUIRE(x >= packed.box_min.x - 1e-5f);
                    REQUIRE(x <= packed.box_max.x + 1e-5f);
                    REQUIRE(y >= packed.box_min.y - 1e-5f);
                    REQUIRE(y <= packed.box_max.y + 1e-5f);
                    REQUIRE(z >= packed.box_min.z - 1e-5f);
                    REQUIRE(z <= packed.box_max.z + 1e-5f);
                    umin = {std::min(umin.x, x), std::min(umin.y, y), std::min(umin.z, z)};
                    umax = {std::max(umax.x, x), std::max(umax.y, y), std::max(umax.z, z)};
                }
            }
            const cr::CrownVolume vol_sp = cr::CrownVolume::build_shipped(l0_probe, umin, umax);
            const auto pl = vol_sp.pad_lo(), ld = vol_sp.l0_dims(), d = vol_sp.dims();
            size_t shell_voxels = 0;
            for (int z = 0; z < d[2]; ++z)
                for (int y = 0; y < d[1]; ++y)
                    for (int x = 0; x < d[0]; ++x) {
                        const bool in_l0 = (x >= pl[0] && x < pl[0] + ld[0] &&
                                            y >= pl[1] && y < pl[1] + ld[1] &&
                                            z >= pl[2] && z < pl[2] + ld[2]);
                        if (in_l0) continue;
                        ++shell_voxels;
                        const size_t b = ((size_t(z) * size_t(d[1]) + size_t(y)) * size_t(d[0]) + size_t(x)) * 4u;
                        REQUIRE((packed.tex0[b] | packed.tex0[b + 1] | packed.tex0[b + 2] | packed.tex0[b + 3]) == 0u);
                        REQUIRE((packed.tex1[b] | packed.tex1[b + 1] | packed.tex1[b + 2] | packed.tex1[b + 3]) == 0);
                    }
            if (std::string(sp) == "oak") CHECK(shell_voxels > 0u);  // positive control: oak_42 has L2 padding
        }
    }

    SECTION("zenith 90 deg single-azimuth deduplication") {
        const auto dirs = cr::sun_directions({45.0f, 90.0f}, 4);
        REQUIRE(dirs.size() == 5u);
        int z_cnt = 0;
        for (const auto& d : dirs) {
            if (d.elevation_deg >= 89.99f) {
                ++z_cnt;
                CHECK(d.azimuth_deg == 0.0f);
            }
        }
        CHECK(z_cnt == 1);
    }

    SECTION("GLB _RYNX_CROWN_VOLUME v2 JSON and binary bufferViews match PackedCrownVolume byte-for-byte") {
        const treegen::GeneratedTree& g = rynx::test_support::generated_tree("oak", 42);
        REQUIRE(g.crown_volume.has_value());
        const auto& cv = *g.crown_volume;
        const std::span<const char> view(reinterpret_cast<const char*>(g.glb.data()), g.glb.size());
        const gv::glb_view glb = gv::parse_glb_header(view);
        REQUIRE(glb.valid);
        gv::jparser parser(glb.json_begin, glb.json_len);
        const gv::jval root = parser.parse_value();
        REQUIRE(parser.good());

        const gv::jval* ext_used = root.find("extensionsUsed");
        REQUIRE(ext_used != nullptr);
        bool has_ext_used = false;
        for (const auto& e : ext_used->as_arr())
            if (e.as_str() == "_RYNX_CROWN_VOLUME") has_ext_used = true;
        CHECK(has_ext_used);

        const gv::jval* meshes = root.find("meshes");
        REQUIRE(meshes != nullptr);
        REQUIRE(!meshes->as_arr().empty());
        const gv::jval* m0_ext = meshes->as_arr()[0].find("extensions");
        REQUIRE(m0_ext != nullptr);
        const gv::jval* cvol = m0_ext->find("_RYNX_CROWN_VOLUME");
        REQUIRE(cvol != nullptr);
        CHECK(cvol->find("version")->as_int() == 2);
        CHECK(cvol->find("format")->as_str() == "f2_tensor_unorm_snorm");
        CHECK(cvol->find("dims")->as_arr()[0].as_int() == cv.dims[0]);
        CHECK(cvol->find("dims")->as_arr()[1].as_int() == cv.dims[1]);
        CHECK(cvol->find("dims")->as_arr()[2].as_int() == cv.dims[2]);
        CHECK(cvol->find("cell_m")->n == Approx(cv.cell_m).margin(1e-6));
        CHECK(cvol->find("rho_cap")->n == Approx(cv.rho_cap).margin(1e-5));
        CHECK(cvol->find("clumping")->n == Approx(cr::k_default_crown_clumping).margin(1e-6));

        const int bv0 = cvol->find("tex0_buffer_view")->as_int();
        const int bv1 = cvol->find("tex1_buffer_view")->as_int();
        const auto& bvs = root.find("bufferViews")->as_arr();
        const auto& jbv0 = bvs[size_t(bv0)];
        const auto& jbv1 = bvs[size_t(bv1)];
        const size_t off0 = size_t(jbv0.find("byteOffset")->n);
        const size_t len0 = size_t(jbv0.find("byteLength")->n);
        const size_t off1 = size_t(jbv1.find("byteOffset")->n);
        const size_t len1 = size_t(jbv1.find("byteLength")->n);
        REQUIRE(len0 == cv.tex0.size());
        REQUIRE(len1 == cv.tex1.size());
        CHECK(std::memcmp(glb.bin_begin + off0, cv.tex0.data(), len0) == 0);
        CHECK(std::memcmp(glb.bin_begin + off1, cv.tex1.data(), len1) == 0);
    }
}
