// Hermite collar curve — cubic spline between parent rim and child branch.
// Chord-length tangent scaling; smoothstep radius blend.
#pragma once

#include "vec3.hpp"

namespace treegen {

struct CollarCurve {
    vec3  p0;       // start position (parent rim)
    vec3  p1;       // end position (child branch)
    vec3  t0_dir;   // start tangent direction (normalized internally)
    vec3  t1_dir;   // end tangent direction (normalized internally)
    float r0;       // start radius
    float r1;       // end radius
    float shoulder_hold = 0.0f;
};

struct CollarSample {
    vec3  position; // centerline point
    vec3  tangent;  // unit tangent at this point
    float radius;   // smoothstep-blended radius
};

CollarSample eval_collar_curve(const CollarCurve& c, float t);

} // namespace treegen
