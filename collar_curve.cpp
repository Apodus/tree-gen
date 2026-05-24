// Hermite collar curve evaluator. /fp:precise (treegen.sharpmake.cs).
#include "collar_curve.hpp"
#include "vec3.hpp"

#include <cmath>

namespace treegen {

namespace {

inline float smoothstep01(float t) {
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    return t * t * (3.0f - 2.0f * t);
}

constexpr float k_eps = 1e-7f;

} // anonymous namespace

CollarSample eval_collar_curve(const CollarCurve& c, float t) {
    const vec3 chord = c.p1 - c.p0;
    const float L = length(chord);

    // Degenerate chord: endpoints coincide.
    if (L < k_eps) {
        const float len0 = length(c.t0_dir);
        const vec3 fallback = (len0 > k_eps) ? normalized(c.t0_dir) : vec3{0.0f, 0.0f, 1.0f};
        return { c.p0, fallback, c.r0 };
    }

    const vec3 chord_dir = chord * (1.0f / L);

    // Tangent directions — fall back to chord if near-zero.
    const vec3 d0 = (length(c.t0_dir) > k_eps) ? normalized(c.t0_dir) : chord_dir;
    const vec3 d1 = (length(c.t1_dir) > k_eps) ? normalized(c.t1_dir) : chord_dir;

    // Chord-length scaled tangents.
    const vec3 T0s = d0 * L;
    const vec3 T1s = d1 * L;

    // Hermite basis coefficients.
    const float t2 = t * t;
    const float t3 = t2 * t;

    const float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
    const float h10 =         t3 - 2.0f * t2 + t;
    const float h01 = -2.0f * t3 + 3.0f * t2;
    const float h11 =         t3 -        t2;

    const vec3 pos = c.p0 * h00 + T0s * h10 + c.p1 * h01 + T1s * h11;

    // Analytical derivative H'(t).
    const float dh00 =  6.0f * t2 - 6.0f * t;
    const float dh10 =  3.0f * t2 - 4.0f * t + 1.0f;
    const float dh01 = -6.0f * t2 + 6.0f * t;
    const float dh11 =  3.0f * t2 - 2.0f * t;

    const vec3 deriv = c.p0 * dh00 + T0s * dh10 + c.p1 * dh01 + T1s * dh11;
    const vec3 tang = normalized(deriv);

    // Smoothstep radius blend.
    const float s = smoothstep01(t);
    const float radius = c.r0 + (c.r1 - c.r0) * s;

    return { pos, tang, radius };
}

} // namespace treegen
