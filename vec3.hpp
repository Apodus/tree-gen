// Tool-local vec3 — avoids engine rynx::vec3 dependency on system/assert.hpp +
// serialization. Minimal POD with the few ops C3+ envelopes / space
// colonization need. /fp:precise (see rynx/generate/treegen.sharpmake.cs).
#pragma once

#include <cmath>

namespace treegen {

struct vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline vec3 operator-(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline vec3 operator-(vec3 a)         { return {-a.x, -a.y, -a.z}; }
inline vec3 operator*(vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline vec3 operator*(float s, vec3 a) { return a * s; }
inline vec3 operator/(vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }

inline float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline vec3  cross(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
inline float length_squared(vec3 a) { return dot(a, a); }
inline float length(vec3 a)         { return std::sqrt(length_squared(a)); }
inline vec3  normalized(vec3 a) {
    float l = length(a);
    return l > 0.0f ? a / l : vec3{0.0f, 0.0f, 1.0f};
}

} // namespace treegen
