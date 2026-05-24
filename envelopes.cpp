#include "envelopes.hpp"

#include <stdexcept>

namespace treegen {

namespace {

// Generic axis-aligned oblate-spheroid (or prolate, when half_height > half_w)
// centred at (0, 0, center_z). a = horizontal half-axis (width_m/2),
// b = vertical half-axis (= H * height_ratio / 2 * top_height_ratio).
// (x/a)^2 + (y/a)^2 + (z'/b)^2 <= 1 where z' = z - center_z.
bool spheroid_contains(vec3 p, float a, float b, float center_z) {
    if (a <= 0.0f || b <= 0.0f) return false;
    const float zr = (p.z - center_z) / b;
    const float xr = p.x / a;
    const float yr = p.y / a;
    return xr * xr + yr * yr + zr * zr <= 1.0f;
}

Aabb spheroid_aabb(float a, float b, float center_z) {
    Aabb out;
    out.min = vec3{-a, -a, center_z - b};
    out.max = vec3{ a,  a, center_z + b};
    return out;
}

// Cone: base disc at z=0 radius=base_radius, apex at z=apex_height. Closed.
bool cone_contains(vec3 p, float base_radius, float apex_height) {
    if (apex_height <= 0.0f || base_radius <= 0.0f) return false;
    if (p.z < 0.0f || p.z > apex_height) return false;
    const float t = p.z / apex_height;        // 0 at base, 1 at apex
    const float r_at_z = base_radius * (1.0f - t);
    return p.x * p.x + p.y * p.y <= r_at_z * r_at_z;
}

Aabb cone_aabb(float base_radius, float apex_height) {
    Aabb out;
    out.min = vec3{-base_radius, -base_radius, 0.0f};
    out.max = vec3{ base_radius,  base_radius, apex_height};
    return out;
}

// Cylinder along Z from 0 to height, radius = r.
bool cylinder_contains(vec3 p, float r, float h) {
    if (r <= 0.0f || h <= 0.0f) return false;
    if (p.z < 0.0f || p.z > h) return false;
    return p.x * p.x + p.y * p.y <= r * r;
}

Aabb cylinder_aabb(float r, float h) {
    Aabb out;
    out.min = vec3{-r, -r, 0.0f};
    out.max = vec3{ r,  r, h};
    return out;
}

// Resolve the spheroid params used by OblateSpheroid/Weeping/Fan. Crown Z
// extent = H * height_ratio; half-height Z = (extent / 2) * top_height_ratio.
struct spheroid_params {
    float a;        // horizontal half-axis
    float b;        // vertical half-axis
    float center_z; // crown centre on Z
};

spheroid_params resolve_spheroid(const TreeDescriptor::Envelope& env, float H) {
    const float crown_extent = H * env.height_ratio;
    spheroid_params sp;
    sp.a = env.width_m * 0.5f;
    sp.b = (crown_extent * 0.5f) * env.top_height_ratio;
    sp.center_z = crown_extent * 0.5f;
    return sp;
}

} // anonymous namespace

bool envelope_contains(EnvelopeShape shape,
                       const TreeDescriptor::Envelope& env,
                       vec3 candidate,
                       float tree_height_m) {
    switch (shape) {
        case EnvelopeShape::OblateSpheroid:
        case EnvelopeShape::Weeping:
        case EnvelopeShape::Fan: {
            // Shape differentiation is via descriptor params (Weeping = low
            // height_ratio + strong gravitropism in tropisms; Fan = very low
            // top_height_ratio). The envelope volume math is shared.
            const auto sp = resolve_spheroid(env, tree_height_m);
            return spheroid_contains(candidate, sp.a, sp.b, sp.center_z);
        }
        case EnvelopeShape::Conical: {
            // Cone from base (z=0) to apex at z = H * height_ratio.
            // top_height_ratio is unused for cones (the canonical cone has
            // no upper-half stretch); deliberately documented in the schema.
            const float apex = tree_height_m * env.height_ratio;
            return cone_contains(candidate, env.width_m * 0.5f, apex);
        }
        case EnvelopeShape::Fastigiate: {
            // Columnar cylinder from base to crown top.
            const float top = tree_height_m * env.height_ratio;
            return cylinder_contains(candidate, env.width_m * 0.5f, top);
        }
    }
    throw std::runtime_error("envelope_contains: unreachable enum");
}

Aabb envelope_aabb(EnvelopeShape shape,
                   const TreeDescriptor::Envelope& env,
                   float tree_height_m) {
    switch (shape) {
        case EnvelopeShape::OblateSpheroid:
        case EnvelopeShape::Weeping:
        case EnvelopeShape::Fan: {
            const auto sp = resolve_spheroid(env, tree_height_m);
            return spheroid_aabb(sp.a, sp.b, sp.center_z);
        }
        case EnvelopeShape::Conical: {
            const float apex = tree_height_m * env.height_ratio;
            return cone_aabb(env.width_m * 0.5f, apex);
        }
        case EnvelopeShape::Fastigiate: {
            const float top = tree_height_m * env.height_ratio;
            return cylinder_aabb(env.width_m * 0.5f, top);
        }
    }
    throw std::runtime_error("envelope_aabb: unreachable enum");
}

} // namespace treegen
