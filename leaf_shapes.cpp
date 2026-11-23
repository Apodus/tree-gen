// /fp:precise required — every angle/sin/cos contributes to byte-hash
// determinism for downstream GLB writes. See treegen.sharpmake.cs.
#include "leaf_shapes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace treegen {

namespace {

constexpr double k_pi      = 3.1415926535897932384626433832795;
constexpr double k_two_pi  = 6.283185307179586476925286766559;
constexpr double k_half_pi = 1.5707963267948966192313216916398;

inline void push_tri(std::vector<uint16_t>& tris, int a, int b, int c) {
    tris.push_back(static_cast<uint16_t>(a));
    tris.push_back(static_cast<uint16_t>(b));
    tris.push_back(static_cast<uint16_t>(c));
}

void fan_triangulate(LeafShapeMesh& m, int perim_count) {
    for (int i = 0; i < perim_count; ++i) {
        const int a = 1 + i;
        const int b = 1 + ((i + 1) % perim_count);
        push_tri(m.tris, 0, a, b);
    }
}

inline double gauss(double x, double mu, double sigma) {
    const double d = (x - mu) / sigma;
    return std::exp(-d * d);
}

// Build a bilateral symmetric leaf perimeter in CCW order starting at base (-Y),
// walking up the right edge (+X) to apex (+Y), then down the left edge (-X).
// Enforces strictly increasing polar angle around (0, 0) so every fan triangle
// is guaranteed positive CCW area with zero self-intersections.
template <typename WidthFn>
LeafShapeMesh build_bilateral_leaf(int half_steps, double y_base, double y_apex, WidthFn width_fn) {
    std::vector<vec2> right_pts;
    right_pts.reserve(static_cast<size_t>(half_steps));
    for (int i = 1; i <= half_steps; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(half_steps + 1);
        const double y = y_base + (y_apex - y_base) * t;
        const double x = std::max(0.02, std::min(0.98, width_fn(t)));
        right_pts.push_back(vec2{static_cast<float>(x), static_cast<float>(y)});
    }

    // Star-shapedness guard: ensure polar angle theta = atan2(y, x) is strictly
    // monotonically increasing along the right edge from -pi/2 to +pi/2.
    double prev_theta = -k_half_pi + 1e-4;
    for (size_t i = 0; i < right_pts.size(); ++i) {
        const double max_theta = k_half_pi - 1e-4 * static_cast<double>(right_pts.size() - i);
        double x = right_pts[i].x;
        double y = right_pts[i].y;
        double th = std::atan2(y, x);
        if (th <= prev_theta + 0.004) {
            const double target_th = std::min(max_theta, prev_theta + 0.008);
            const double r = std::sqrt(x * x + y * y);
            x = std::max(0.02, r * std::cos(target_th));
            y = r * std::sin(target_th);
            right_pts[i] = vec2{static_cast<float>(x), static_cast<float>(y)};
            th = std::atan2(y, x);
        }
        prev_theta = th;
    }

    const int N = 2 * half_steps + 2;
    LeafShapeMesh m;
    m.verts.reserve(static_cast<size_t>(N + 1));
    m.tris.reserve(static_cast<size_t>(N * 3));

    m.verts.push_back(vec2{0.0f, 0.0f});                     // 0: centroid
    m.verts.push_back(vec2{0.0f, static_cast<float>(y_base)}); // 1: base (-Y)
    for (const vec2& p : right_pts) m.verts.push_back(p);
    m.verts.push_back(vec2{0.0f, static_cast<float>(y_apex)}); // apex (+Y)
    for (int i = half_steps - 1; i >= 0; --i) {
        m.verts.push_back(vec2{-right_pts[static_cast<size_t>(i)].x,
                                right_pts[static_cast<size_t>(i)].y});
    }

    fan_triangulate(m, N);
    return m;
}

}  // namespace

// ----- Oak: classic broad rounded 9-lobe English/White oak (Quercus robur/alba)
// with obovate silhouette, basal auricles, deep rounded U-sinuses, and rounded
// terminal lobe. 64 perimeter verts + centroid = 65 verts, 64 fan tris.
LeafShapeMesh oak_lobed() {
    auto width_fn = [](double t) -> double {
        // Broad obovate base envelope: narrow petiole base, widest around t=0.58.
        const double env = 0.60 * std::pow(std::sin(k_pi * std::pow(t, 0.72)), 0.62);
        // 4 broad rounded lateral lobe pairs + basal auricles + rounded terminal lobe.
        double lobes = 0.0;
        lobes += 0.14 * gauss(t, 0.14, 0.048);  // basal auricle
        lobes += 0.24 * gauss(t, 0.34, 0.068);  // lower lateral lobe
        lobes += 0.32 * gauss(t, 0.56, 0.078);  // widest mid-upper lateral lobe
        lobes += 0.26 * gauss(t, 0.77, 0.068);  // upper lateral lobe
        lobes += 0.12 * gauss(t, 0.92, 0.055);  // rounded terminal lobe shoulder
        // Rounded U-sinuses between lobes (kept smooth so lobes stay plump and rounded).
        lobes -= 0.09 * gauss(t, 0.23, 0.036);
        lobes -= 0.15 * gauss(t, 0.45, 0.044);
        lobes -= 0.16 * gauss(t, 0.67, 0.044);
        lobes -= 0.10 * gauss(t, 0.86, 0.036);

        const double end_fade = std::pow(std::sin(k_pi * t), 0.28);
        const double w = (env * 0.72 + lobes) * end_fade;
        return std::max(0.05, std::min(0.84, w));
    };
    return build_bilateral_leaf(/*half_steps=*/31, /*y_base=*/-0.96, /*y_apex=*/+1.00, width_fn);
}

// ----- Pine needle: tapered needle strip with 4 axial rows + pointed tip
// (9 verts, 7 tris). Runs from -Y (sheath base) to +Y (needle point).
LeafShapeMesh pine_needle() {
    LeafShapeMesh m;
    m.verts.reserve(9);
    m.tris.reserve(7 * 3);

    m.verts.push_back(vec2{-0.055f, -1.00f});  // 0 base-left
    m.verts.push_back(vec2{+0.055f, -1.00f});  // 1 base-right
    m.verts.push_back(vec2{-0.095f, -0.35f});  // 2 lower-left
    m.verts.push_back(vec2{+0.095f, -0.35f});  // 3 lower-right
    m.verts.push_back(vec2{-0.090f, +0.35f});  // 4 upper-mid-left
    m.verts.push_back(vec2{+0.090f, +0.35f});  // 5 upper-mid-right
    m.verts.push_back(vec2{-0.048f, +0.78f});  // 6 sub-tip-left
    m.verts.push_back(vec2{+0.048f, +0.78f});  // 7 sub-tip-right
    m.verts.push_back(vec2{ 0.000f, +1.00f});  // 8 apex tip

    push_tri(m.tris, 0, 1, 3);
    push_tri(m.tris, 0, 3, 2);
    push_tri(m.tris, 2, 3, 5);
    push_tri(m.tris, 2, 5, 4);
    push_tri(m.tris, 4, 5, 7);
    push_tri(m.tris, 4, 7, 6);
    push_tri(m.tris, 6, 7, 8);
    return m;
}

// ----- Birch serrated: doubly-serrated ovate-deltoid silver birch silhouette
// with slightly cordate base, crisp double teeth, and acuminate drip-tip.
// 64 perimeter verts + centroid = 65 verts, 64 fan tris.
LeafShapeMesh birch_serrated() {
    auto width_fn = [](double t) -> double {
        // Broad ovate-deltoid profile peaking around t=0.26, tapering gracefully to drip-tip.
        const double rise = std::pow(std::sin(k_half_pi * std::min(1.0, t / 0.26)), 0.65);
        const double fall = std::pow(std::max(0.0, (1.0 - t) / 0.74), 1.15);
        const double base_env = (t < 0.26) ? (0.78 * rise) : (0.78 * fall);
        const double tip_pull = (t > 0.76) ? (1.0 - 0.16 * (t - 0.76) / 0.24) : 1.0;

        // Crisp double serration on both sides for t in [0.14, 0.92].
        double serr = 1.0;
        if (t > 0.14 && t < 0.93) {
            const double mask = std::pow(std::sin(k_pi * (t - 0.14) / 0.79), 0.75);
            const double ph1 = (t - 0.14) * 8.0;
            const double f1  = ph1 - std::floor(ph1);
            const double tooth1 = (f1 < 0.70) ? (f1 / 0.70) : ((1.0 - f1) / 0.30);
            const double ph2 = (t - 0.14) * 16.0;
            const double f2  = ph2 - std::floor(ph2);
            const double tooth2 = (f2 < 0.65) ? (f2 / 0.65) : ((1.0 - f2) / 0.35);
            serr = 1.0 + mask * (0.145 * (tooth1 - 0.5) + 0.065 * (tooth2 - 0.5));
        }

        const double w = base_env * tip_pull * serr;
        return std::max(0.03, std::min(0.86, w));
    };
    return build_bilateral_leaf(/*half_steps=*/31, /*y_base=*/-0.88, /*y_apex=*/+1.00, width_fn);
}

// ----- Maple star: broad 5-lobe palmate sugar maple (Acer saccharum) silhouette
// with wide lobes, rounded U-sinuses, and pointed secondary shoulder teeth.
// Polar sampling around (0, 0) guarantees strict star-shapedness and positive
// area on every fan triangle. 72 perimeter verts + centroid = 73 verts, 72 fan tris.
LeafShapeMesh maple_star() {
    LeafShapeMesh m;
    constexpr int N = 72;
    m.verts.reserve(N + 1);
    m.tris.reserve(N * 3);

    m.verts.push_back(vec2{0.0f, 0.0f});  // 0: palm center

    for (int i = 0; i < N; ++i) {
        // Walk CCW starting at -Y (theta = -pi/2) so apex lands exactly at i = N/2 (+Y).
        const double theta = -k_half_pi + (k_two_pi * static_cast<double>(i)) / static_cast<double>(N);
        double phi = std::abs(theta - k_half_pi);
        if (phi > k_pi) phi = k_two_pi - phi;

        // Broad palmate body (r_base = 0.44) + 5 broad sugar-maple lobes + U-sinuses.
        double r = 0.44;
        // Central terminal lobe (phi = 0 deg) + broad side shoulder teeth (phi = 21 deg).
        r += 0.56 * gauss(phi, 0.00, 0.25);
        r += 0.34 * gauss(phi, 0.36, 0.13);
        // Rounded U-sinus between central & upper-lateral lobes (phi = 41 deg).
        r -= 0.14 * gauss(phi, 0.72, 0.09);
        // Broad upper-lateral main lobes (phi = 64 deg) + shoulder points (48 & 82 deg).
        r += 0.54 * gauss(phi, 1.12, 0.24);
        r += 0.24 * gauss(phi, 0.85, 0.11);
        r += 0.28 * gauss(phi, 1.44, 0.12);
        // Rounded lower U-sinus (phi = 100 deg).
        r -= 0.12 * gauss(phi, 1.74, 0.10);
        // Basal lateral lobes (phi = 122 deg).
        r += 0.36 * gauss(phi, 2.13, 0.20);
        // Cordate basal notch around petiole (phi = 180 deg).
        r -= 0.18 * gauss(phi, k_pi, 0.38);

        r = std::max(0.24, std::min(1.00, r));
        const float x = static_cast<float>(r * std::cos(theta));
        const float y = static_cast<float>(r * std::sin(theta));
        m.verts.push_back(vec2{x, y});
    }

    fan_triangulate(m, N);
    return m;
}

}  // namespace treegen
