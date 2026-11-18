// [cpu_raster] — the offline triangle rasteriser: exact coverage, depth order
// independent of draw order, the alpha cut, degenerate input.

#include "../external/catch2/catch.hpp"

#include "../cpu_raster.hpp"

#include <array>
#include <cstdint>

namespace cr = treegen::cpu_raster;

namespace {

int covered_count(const cr::target& t) {
    int n = 0;
    for (int y = 0; y < t.h; ++y)
        for (int x = 0; x < t.w; ++x)
            n += t.covered(x, y) ? 1 : 0;
    return n;
}

// Right triangle (0,0) (8,0) (0,8): centres (x+.5, y+.5) with x+y <= 7, edges
// inclusive -> 8+7+...+1 = 36 pixels. Exact in float (small integers + .5).
constexpr int k_right_tri_px = 36;

cr::vertex v(float x, float y, float d, float u = 0.0f, float vv = 0.0f) { return { x, y, d, u, vv }; }

}  // namespace

TEST_CASE("cpu_raster: exact coverage of a right triangle, either winding", "[cpu_raster]") {
    const cr::texture_view none{};
    for (int winding = 0; winding < 2; ++winding) {
        INFO("winding " << winding);
        cr::target t(16, 16);
        const int n = winding == 0
            ? cr::draw_triangle(t, v(0, 0, 1), v(8, 0, 1), v(0, 8, 1), none, 1)
            : cr::draw_triangle(t, v(0, 0, 1), v(0, 8, 1), v(8, 0, 1), none, 1);
        REQUIRE(n == k_right_tri_px);
        REQUIRE(covered_count(t) == k_right_tri_px);
        REQUIRE(t.covered(0, 0));
        REQUIRE(t.covered(0, 7));
        REQUIRE_FALSE(t.covered(4, 4));   // centre (4.5, 4.5): x+y = 9 > 8
        REQUIRE_FALSE(t.covered(8, 0));
    }
}

TEST_CASE("cpu_raster: the nearer triangle wins regardless of draw order", "[cpu_raster]") {
    // Two full-target quads (two triangles each), near = red at depth 2, far = blue at 1.
    const std::array<uint8_t, 4> red{ 255, 0, 0, 255 }, blue{ 0, 0, 255, 255 };
    const cr::texture_view near_tex{ red.data(), 1, 1 }, far_tex{ blue.data(), 1, 1 };
    const auto quad = [](cr::target& t, float d, const cr::texture_view& tex) {
        const float s = float(t.w);
        return cr::draw_triangle(t, v(0, 0, d), v(s, 0, d), v(s, s, d), tex, 1)
             + cr::draw_triangle(t, v(0, 0, d), v(s, s, d), v(0, s, d), tex, 1);
    };
    for (int order = 0; order < 2; ++order) {
        INFO("order " << order);
        cr::target t(8, 8);
        if (order == 0) { quad(t, 2.0f, near_tex); REQUIRE(quad(t, 1.0f, far_tex) == 0); }
        else            { quad(t, 1.0f, far_tex);  REQUIRE(quad(t, 2.0f, near_tex) == 64); }
        for (int i = 0; i < 64; ++i) {
            REQUIRE(t.rgba[i * 4 + 0] == 255);
            REQUIRE(t.rgba[i * 4 + 2] == 0);
            REQUIRE(t.depth[i] == 2.0f);
        }
    }
}

TEST_CASE("cpu_raster: the alpha cut removes texels below the cutoff and leaves depth untouched", "[cpu_raster]") {
    // 2x1 texture: left texel opaque, right texel alpha 0. Quad over an 8x8
    // target with u in [0,1]: the left half (x < 4) survives at cutoff 128.
    const std::array<uint8_t, 8> tex_px{ 10, 20, 30, 255,   40, 50, 60, 0 };
    const cr::texture_view tex{ tex_px.data(), 2, 1 };
    cr::target t(8, 8);
    const int n = cr::draw_triangle(t, v(0, 0, 1, 0, 0), v(8, 0, 1, 1, 0), v(8, 8, 1, 1, 1), tex, 128)
                + cr::draw_triangle(t, v(0, 0, 1, 0, 0), v(8, 8, 1, 1, 1), v(0, 8, 1, 0, 1), tex, 128);
    REQUIRE(n == 32);
    REQUIRE(covered_count(t) == 32);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            INFO("x=" << x << " y=" << y);
            REQUIRE(t.covered(x, y) == (x < 4));
            REQUIRE((t.depth[size_t(y) * 8 + x] == cr::target::k_empty_depth) == (x >= 4));
        }
    // Cutoff 0 keeps every texel: the cut is the only thing removing the right half.
    cr::target all(8, 8);
    REQUIRE(cr::draw_triangle(all, v(0, 0, 1, 0, 0), v(8, 0, 1, 1, 0), v(8, 8, 1, 1, 1), tex, 0)
          + cr::draw_triangle(all, v(0, 0, 1, 0, 0), v(8, 8, 1, 1, 1), v(0, 8, 1, 0, 1), tex, 0) == 64);
}

TEST_CASE("cpu_raster: a degenerate triangle and an off-target triangle draw nothing", "[cpu_raster]") {
    const cr::texture_view none{};
    cr::target t(8, 8);
    REQUIRE(cr::draw_triangle(t, v(1, 1, 1), v(5, 5, 1), v(3, 3, 1), none, 1) == 0);
    REQUIRE(cr::draw_triangle(t, v(20, 20, 1), v(30, 20, 1), v(20, 30, 1), none, 1) == 0);
    REQUIRE(covered_count(t) == 0);
}
