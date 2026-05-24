// /fp:precise required — every angle/sin/cos contributes to the GLB
// byte-hash determinism gate. See treegen.sharpmake.cs.
#include "trivial_cylinder.hpp"

#include <cmath>
#include <cstdint>

namespace treegen {

cpu_mesh_out build_trivial_cylinder(float radius,
                                    float height,
                                    int   radial_segments,
                                    int   axial_segments)
{
    cpu_mesh_out out;

    const int R = radial_segments;
    const int A = axial_segments;

    // Vertex count: side grid (R+1)*(A+1) + top cap (R+2) + bottom cap (R+2).
    const int side_verts   = (R + 1) * (A + 1);
    const int cap_verts    = R + 2; // 1 center + (R+1) rim
    const int total_verts  = side_verts + 2 * cap_verts;
    const int total_tris   = R * A * 2 + R + R;

    out.positions.reserve(total_verts * 3);
    out.normals.reserve(total_verts * 3);
    out.uvs.reserve(total_verts * 2);
    out.indices.reserve(total_tris * 3);

    constexpr double k_two_pi = 6.283185307179586476925286766559;

    // ---- Precompute per-radial sin/cos so wrap seam (i=R) shares values
    // with i=0 to the precision of double. /fp:precise keeps the
    // float-narrowing predictable across runs.
    std::vector<float> cosT(R + 1), sinT(R + 1);
    for (int i = 0; i <= R; ++i) {
        const double t = (k_two_pi * static_cast<double>(i)) / static_cast<double>(R);
        cosT[i] = static_cast<float>(std::cos(t));
        sinT[i] = static_cast<float>(std::sin(t));
    }
    // Snap the seam exactly to i=0 — eliminates any residual sin/cos drift
    // at theta=2pi vs theta=0.
    cosT[R] = cosT[0];
    sinT[R] = sinT[0];

    // ---- Side wall (z runs 0..height, theta runs 0..2pi). ----
    const int side_base = 0;
    for (int j = 0; j <= A; ++j) {
        const float z = (static_cast<float>(j) / static_cast<float>(A)) * height;
        const float v = static_cast<float>(j) / static_cast<float>(A);
        for (int i = 0; i <= R; ++i) {
            const float cx = cosT[i];
            const float sy = sinT[i];
            const float u  = static_cast<float>(i) / static_cast<float>(R);

            out.positions.push_back(radius * cx);
            out.positions.push_back(radius * sy);
            out.positions.push_back(z);

            // Side normal is purely radial.
            out.normals.push_back(cx);
            out.normals.push_back(sy);
            out.normals.push_back(0.0f);

            out.uvs.push_back(u);
            out.uvs.push_back(v);
        }
    }

    auto side_idx = [&](int i, int j) -> uint16_t {
        return static_cast<uint16_t>(side_base + j * (R + 1) + i);
    };

    // CCW outward-facing tris for the side strip.
    for (int j = 0; j < A; ++j) {
        for (int i = 0; i < R; ++i) {
            const uint16_t a = side_idx(i,     j    );
            const uint16_t b = side_idx(i + 1, j    );
            const uint16_t c = side_idx(i,     j + 1);
            const uint16_t d = side_idx(i + 1, j + 1);
            // Triangle 1: a, b, d
            out.indices.push_back(a);
            out.indices.push_back(b);
            out.indices.push_back(d);
            // Triangle 2: a, d, c
            out.indices.push_back(a);
            out.indices.push_back(d);
            out.indices.push_back(c);
        }
    }

    // ---- Top cap (z = height, normal = +Z). ----
    const int top_base = static_cast<int>(out.positions.size() / 3);
    // Center
    out.positions.push_back(0.0f);
    out.positions.push_back(0.0f);
    out.positions.push_back(height);
    out.normals.push_back(0.0f);
    out.normals.push_back(0.0f);
    out.normals.push_back(1.0f);
    out.uvs.push_back(0.5f);
    out.uvs.push_back(0.5f);
    // Rim
    for (int i = 0; i <= R; ++i) {
        const float cx = cosT[i];
        const float sy = sinT[i];
        out.positions.push_back(radius * cx);
        out.positions.push_back(radius * sy);
        out.positions.push_back(height);
        out.normals.push_back(0.0f);
        out.normals.push_back(0.0f);
        out.normals.push_back(1.0f);
        // Cap UVs: radial unwrap centered at (0.5, 0.5).
        out.uvs.push_back(0.5f + 0.5f * cx);
        out.uvs.push_back(0.5f + 0.5f * sy);
    }
    // Triangulate fan (center, i, i+1) — CCW when viewed from +Z.
    for (int i = 0; i < R; ++i) {
        out.indices.push_back(static_cast<uint16_t>(top_base));
        out.indices.push_back(static_cast<uint16_t>(top_base + 1 + i));
        out.indices.push_back(static_cast<uint16_t>(top_base + 1 + i + 1));
    }

    // ---- Bottom cap (z = 0, normal = -Z). ----
    const int bot_base = static_cast<int>(out.positions.size() / 3);
    // Center
    out.positions.push_back(0.0f);
    out.positions.push_back(0.0f);
    out.positions.push_back(0.0f);
    out.normals.push_back(0.0f);
    out.normals.push_back(0.0f);
    out.normals.push_back(-1.0f);
    out.uvs.push_back(0.5f);
    out.uvs.push_back(0.5f);
    // Rim
    for (int i = 0; i <= R; ++i) {
        const float cx = cosT[i];
        const float sy = sinT[i];
        out.positions.push_back(radius * cx);
        out.positions.push_back(radius * sy);
        out.positions.push_back(0.0f);
        out.normals.push_back(0.0f);
        out.normals.push_back(0.0f);
        out.normals.push_back(-1.0f);
        out.uvs.push_back(0.5f + 0.5f * cx);
        out.uvs.push_back(0.5f - 0.5f * sy);
    }
    // Reversed winding so the outward face is -Z.
    for (int i = 0; i < R; ++i) {
        out.indices.push_back(static_cast<uint16_t>(bot_base));
        out.indices.push_back(static_cast<uint16_t>(bot_base + 1 + i + 1));
        out.indices.push_back(static_cast<uint16_t>(bot_base + 1 + i));
    }

    return out;
}

// C2 P2: trunk + leaves disk. Two distinct primitives to exercise the
// multi-primitive write/read path end-to-end.
std::vector<cpu_mesh_out> build_trivial_two_primitive_cylinder(float radius, float height) {
    std::vector<cpu_mesh_out> out;
    out.reserve(2);

    // Trunk: pinned trivial cylinder (matches single-prim case for sanity).
    out.push_back(build_trivial_cylinder(radius, height, 12, 4));

    // Leaves: flat radial-fan disk at z=height, larger radius. 12 segments
    // (R=12) + 1 center vertex = 13 verts, 12 triangles.
    constexpr int R = 12;
    constexpr double k_two_pi = 6.283185307179586476925286766559;
    const float leaf_r = radius * 3.0f;
    const float z = height;

    cpu_mesh_out leaves;
    leaves.positions.reserve((R + 1) * 3);
    leaves.normals.reserve((R + 1) * 3);
    leaves.uvs.reserve((R + 1) * 2);
    leaves.indices.reserve(R * 3);

    // Center vertex.
    leaves.positions.push_back(0.0f);
    leaves.positions.push_back(0.0f);
    leaves.positions.push_back(z);
    leaves.normals.push_back(0.0f);
    leaves.normals.push_back(0.0f);
    leaves.normals.push_back(1.0f);
    leaves.uvs.push_back(0.5f);
    leaves.uvs.push_back(0.5f);

    // Rim — closes via i=0 at the end of the fan (no duplicated seam).
    for (int i = 0; i < R; ++i) {
        const double t = (k_two_pi * static_cast<double>(i)) / static_cast<double>(R);
        const float cx = static_cast<float>(std::cos(t));
        const float sy = static_cast<float>(std::sin(t));
        leaves.positions.push_back(leaf_r * cx);
        leaves.positions.push_back(leaf_r * sy);
        leaves.positions.push_back(z);
        leaves.normals.push_back(0.0f);
        leaves.normals.push_back(0.0f);
        leaves.normals.push_back(1.0f);
        leaves.uvs.push_back(0.5f + 0.5f * cx);
        leaves.uvs.push_back(0.5f + 0.5f * sy);
    }

    // Fan triangles: (center, i+1, ((i+1)%R)+1).
    for (int i = 0; i < R; ++i) {
        const uint16_t a = 0;
        const uint16_t b = static_cast<uint16_t>(1 + i);
        const uint16_t c = static_cast<uint16_t>(1 + ((i + 1) % R));
        leaves.indices.push_back(a);
        leaves.indices.push_back(b);
        leaves.indices.push_back(c);
    }

    out.push_back(std::move(leaves));
    return out;
}

} // namespace treegen
