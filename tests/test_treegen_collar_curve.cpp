// [treegen_collar_curve] — pins Hermite collar curve evaluator.
// Chord-length tangent scaling + smoothstep radius blend + edge cases.

#include "../external/catch2/catch.hpp"

#include "../collar_curve.hpp"
#include "../vec3.hpp"

#include <cmath>

namespace ts = treegen;

namespace {

inline float approx_len(ts::vec3 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline bool is_finite_vec(ts::vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

} // anonymous namespace

// ---------- straight pipe ----------

TEST_CASE("[treegen_collar_curve] straight pipe endpoints",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {0, 0, 1};
    c.t0_dir = {0, 0, 1}; c.t1_dir = {0, 0, 1};
    c.r0 = 0.5f; c.r1 = 0.5f;

    auto s0 = ts::eval_collar_curve(c, 0.0f);
    REQUIRE(s0.position.x == Approx(0.0f).margin(1e-5f));
    REQUIRE(s0.position.y == Approx(0.0f).margin(1e-5f));
    REQUIRE(s0.position.z == Approx(0.0f).margin(1e-5f));
    REQUIRE(s0.tangent.z  == Approx(1.0f).margin(1e-5f));
    REQUIRE(s0.radius     == Approx(0.5f).margin(1e-5f));

    auto s1 = ts::eval_collar_curve(c, 1.0f);
    REQUIRE(s1.position.x == Approx(0.0f).margin(1e-5f));
    REQUIRE(s1.position.y == Approx(0.0f).margin(1e-5f));
    REQUIRE(s1.position.z == Approx(1.0f).margin(1e-5f));
    REQUIRE(s1.tangent.z  == Approx(1.0f).margin(1e-5f));
    REQUIRE(s1.radius     == Approx(0.5f).margin(1e-5f));
}

TEST_CASE("[treegen_collar_curve] straight pipe midpoint",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {0, 0, 1};
    c.t0_dir = {0, 0, 1}; c.t1_dir = {0, 0, 1};
    c.r0 = 0.5f; c.r1 = 0.5f;

    auto s = ts::eval_collar_curve(c, 0.5f);
    REQUIRE(s.position.x == Approx(0.0f).margin(1e-5f));
    REQUIRE(s.position.y == Approx(0.0f).margin(1e-5f));
    REQUIRE(s.position.z == Approx(0.5f).margin(1e-5f));
    REQUIRE(s.tangent.z  == Approx(1.0f).margin(1e-5f));
    REQUIRE(s.radius     == Approx(0.5f).margin(1e-5f));
}

TEST_CASE("[treegen_collar_curve] straight pipe stays on Z axis",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {0, 0, 1};
    c.t0_dir = {0, 0, 1}; c.t1_dir = {0, 0, 1};
    c.r0 = 0.5f; c.r1 = 0.5f;

    for (int i = 0; i <= 10; ++i) {
        const float t = static_cast<float>(i) / 10.0f;
        auto s = ts::eval_collar_curve(c, t);
        REQUIRE(std::abs(s.position.x) < 1e-5f);
        REQUIRE(std::abs(s.position.y) < 1e-5f);
    }
}

// ---------- 90-degree bend ----------

TEST_CASE("[treegen_collar_curve] 90-degree bend endpoints",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {1, 0, 1};
    c.t0_dir = {0, 0, 1}; c.t1_dir = {1, 0, 0};
    c.r0 = 0.3f; c.r1 = 0.1f;

    auto s0 = ts::eval_collar_curve(c, 0.0f);
    REQUIRE(s0.position.x == Approx(0.0f).margin(1e-5f));
    REQUIRE(s0.position.y == Approx(0.0f).margin(1e-5f));
    REQUIRE(s0.position.z == Approx(0.0f).margin(1e-5f));
    REQUIRE(s0.tangent.z  == Approx(1.0f).margin(1e-5f));
    REQUIRE(s0.radius     == Approx(0.3f).margin(1e-5f));

    auto s1 = ts::eval_collar_curve(c, 1.0f);
    REQUIRE(s1.position.x == Approx(1.0f).margin(1e-5f));
    REQUIRE(s1.position.y == Approx(0.0f).margin(1e-5f));
    REQUIRE(s1.position.z == Approx(1.0f).margin(1e-5f));
    REQUIRE(s1.tangent.x  == Approx(1.0f).margin(1e-5f));
    REQUIRE(s1.radius     == Approx(0.1f).margin(1e-5f));
}

TEST_CASE("[treegen_collar_curve] 90-degree bend radius at midpoint",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {1, 0, 1};
    c.t0_dir = {0, 0, 1}; c.t1_dir = {1, 0, 0};
    c.r0 = 0.3f; c.r1 = 0.1f;

    auto s = ts::eval_collar_curve(c, 0.5f);
    // smoothstep(0.5) = 0.5*0.5*(3-2*0.5) = 0.25*2 = 0.5
    // r = 0.3 + (0.1-0.3)*0.5 = 0.2
    REQUIRE(s.radius == Approx(0.2f).margin(1e-5f));
}

TEST_CASE("[treegen_collar_curve] 90-degree bend stays in XZ plane",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {1, 0, 1};
    c.t0_dir = {0, 0, 1}; c.t1_dir = {1, 0, 0};
    c.r0 = 0.3f; c.r1 = 0.1f;

    for (int i = 0; i <= 10; ++i) {
        const float t = static_cast<float>(i) / 10.0f;
        auto s = ts::eval_collar_curve(c, t);
        REQUIRE(std::abs(s.position.y) < 1e-5f);
    }
}

TEST_CASE("[treegen_collar_curve] 90-degree bend midpoint off straight line",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {1, 0, 1};
    c.t0_dir = {0, 0, 1}; c.t1_dir = {1, 0, 0};
    c.r0 = 0.3f; c.r1 = 0.1f;

    auto s = ts::eval_collar_curve(c, 0.5f);
    // Straight midpoint would be (0.5, 0, 0.5).
    const float dx = s.position.x - 0.5f;
    const float dz = s.position.z - 0.5f;
    const float off = std::sqrt(dx * dx + dz * dz);
    REQUIRE(off > 0.01f);
}

// ---------- S-curve ----------

TEST_CASE("[treegen_collar_curve] S-curve endpoints",
          "[treegen][treegen_collar_curve]") {
    // Same-direction perpendicular tangents produce an S-curve:
    // x(t) = 2t(2t-1)(t-1), roots at 0, 0.5, 1.
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {0, 0, 2};
    c.t0_dir = {1, 0, 0}; c.t1_dir = {1, 0, 0};
    c.r0 = 0.5f; c.r1 = 0.2f;

    auto s0 = ts::eval_collar_curve(c, 0.0f);
    REQUIRE(s0.position.x == Approx(0.0f).margin(1e-5f));
    REQUIRE(s0.position.z == Approx(0.0f).margin(1e-5f));

    auto s1 = ts::eval_collar_curve(c, 1.0f);
    REQUIRE(s1.position.x == Approx(0.0f).margin(1e-5f));
    REQUIRE(s1.position.z == Approx(2.0f).margin(1e-5f));
}

TEST_CASE("[treegen_collar_curve] S-curve symmetry at midpoint",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {0, 0, 2};
    c.t0_dir = {1, 0, 0}; c.t1_dir = {1, 0, 0};
    c.r0 = 0.5f; c.r1 = 0.2f;

    auto s = ts::eval_collar_curve(c, 0.5f);
    REQUIRE(std::abs(s.position.x) < 1e-5f);
}

TEST_CASE("[treegen_collar_curve] S-curve lateral deflection",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {0, 0, 2};
    c.t0_dir = {1, 0, 0}; c.t1_dir = {1, 0, 0};
    c.r0 = 0.5f; c.r1 = 0.2f;

    auto sq = ts::eval_collar_curve(c, 0.25f);
    REQUIRE(sq.position.x > 0.0f);

    auto s3q = ts::eval_collar_curve(c, 0.75f);
    REQUIRE(s3q.position.x < 0.0f);
}

// ---------- edge cases ----------

TEST_CASE("[treegen_collar_curve] degenerate chord produces no NaN",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {1, 2, 3}; c.p1 = {1, 2, 3};
    c.t0_dir = {0, 1, 0}; c.t1_dir = {1, 0, 0};
    c.r0 = 0.4f; c.r1 = 0.1f;

    auto s = ts::eval_collar_curve(c, 0.5f);
    REQUIRE(is_finite_vec(s.position));
    REQUIRE(is_finite_vec(s.tangent));
    REQUIRE(std::isfinite(s.radius));
    // Position matches p0 for degenerate chord.
    REQUIRE(s.position.x == Approx(1.0f).margin(1e-5f));
    REQUIRE(s.position.y == Approx(2.0f).margin(1e-5f));
    REQUIRE(s.position.z == Approx(3.0f).margin(1e-5f));
}

TEST_CASE("[treegen_collar_curve] zero tangent direction falls back",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0, 0, 0}; c.p1 = {0, 0, 1};
    c.t0_dir = {0, 0, 0}; c.t1_dir = {0, 0, 0};
    c.r0 = 0.5f; c.r1 = 0.5f;

    for (int i = 0; i <= 10; ++i) {
        const float t = static_cast<float>(i) / 10.0f;
        auto s = ts::eval_collar_curve(c, t);
        REQUIRE(is_finite_vec(s.position));
        REQUIRE(is_finite_vec(s.tangent));
        REQUIRE(std::isfinite(s.radius));
    }
}

TEST_CASE("[treegen_collar_curve] degenerate chord + zero tangent",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {5, 5, 5}; c.p1 = {5, 5, 5};
    c.t0_dir = {0, 0, 0}; c.t1_dir = {0, 0, 0};
    c.r0 = 0.3f; c.r1 = 0.3f;

    auto s = ts::eval_collar_curve(c, 0.5f);
    REQUIRE(is_finite_vec(s.position));
    REQUIRE(is_finite_vec(s.tangent));
    // Tangent fallback to {0,0,1}.
    REQUIRE(s.tangent.z == Approx(1.0f).margin(1e-5f));
}

TEST_CASE("[treegen_collar_curve] determinism",
          "[treegen][treegen_collar_curve]") {
    ts::CollarCurve c;
    c.p0 = {0.1f, 0.2f, 0.3f}; c.p1 = {1.0f, 2.0f, 3.0f};
    c.t0_dir = {0.5f, 0.0f, 1.0f}; c.t1_dir = {-0.3f, 0.7f, 0.1f};
    c.r0 = 0.4f; c.r1 = 0.15f;

    for (int i = 0; i <= 20; ++i) {
        const float t = static_cast<float>(i) / 20.0f;
        auto a = ts::eval_collar_curve(c, t);
        auto b = ts::eval_collar_curve(c, t);
        REQUIRE(a.position.x == b.position.x);
        REQUIRE(a.position.y == b.position.y);
        REQUIRE(a.position.z == b.position.z);
        REQUIRE(a.tangent.x  == b.tangent.x);
        REQUIRE(a.tangent.y  == b.tangent.y);
        REQUIRE(a.tangent.z  == b.tangent.z);
        REQUIRE(a.radius     == b.radius);
    }
}
