// Trivial cylinder primitive — first-write calibration geometry for C1.
// Deterministic: output depends only on the four input parameters; no RNG.
//
// Topology:
//   Side wall:  (radial_segments + 1) x (axial_segments + 1) grid
//               (duplicated seam column so UV runs [0,1] across the wrap).
//   Top cap:    1 center vertex + (radial_segments + 1) rim, fan-triangulated.
//   Bottom cap: same as top, fan with REVERSED winding so outward face = -Z.
//
// For the pinned default (radial=12, axial=4):
//   verts = 13*5 + 14 + 14 = 93
//   tris  = 12*4*2 + 12 + 12 = 120
#pragma once

#include <cstdint>
#include <vector>

namespace treegen {

struct cpu_mesh_out {
    std::vector<float>    positions;  // xyz packed
    std::vector<float>    normals;    // xyz packed
    std::vector<float>    uvs;        // xy packed (may be empty)
    std::vector<uint16_t> indices;
};

cpu_mesh_out build_trivial_cylinder(float radius,
                                    float height,
                                    int   radial_segments,
                                    int   axial_segments);

// Pinned counts for the default scenario (radius=0.5, height=2.0,
// radial=12, axial=4). [treegen_glb_roundtrip] asserts against these.
inline constexpr int K_TRIVIAL_VERT_COUNT = 93;
inline constexpr int K_TRIVIAL_TRI_COUNT  = 120;

// C2 P2 — multi-primitive test fixture. Returns two cpu_mesh_out instances:
//   [0] trunk: a default trivial cylinder (radius=`radius`, height=`height`,
//              radial=12, axial=4) — pinned counts.
//   [1] leaves: a flat radial fan disk centered at (0, 0, height) with
//               radius=`radius * 3.0f`, 12 segments + 1 center vertex.
// Vertices are local-mesh coordinates (no global translation), so the
// disk is offset by translating its z to `height` directly.
std::vector<cpu_mesh_out> build_trivial_two_primitive_cylinder(float radius,
                                                               float height);

} // namespace treegen
