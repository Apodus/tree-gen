// C5 P2 — analytic 2D leaf shape polygons. Each species function returns a
// unit-half-extent polygon (verts in [-1, 1]) + a pre-computed triangle index
// list. Consumed by leaf_geometry.cpp's ProceduralVeined emitter. Pure POD;
// no RNG, no engine deps. /fp:precise (treegen.sharpmake.cs).
//
// Coordinate convention: +Y is the leaf "tip" direction (apex along +Y), -Y
// the petiole. Vertex 0 is always the polygon centroid (used as fan pivot).
#pragma once

#include "vec3.hpp"  // for treegen::vec3 -- but we use a local vec2 here

#include <cstdint>
#include <vector>

namespace treegen {

struct vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct LeafShapeMesh {
    // Verts in unit-half-extent: x ∈ [-1, 1], y ∈ [-1, 1]. Vertex 0 is the
    // polygon centroid; perimeter follows. Triangle list is a center-fan for
    // every shape except the maple star (multi-ring stitch).
    std::vector<vec2>     verts;
    std::vector<uint16_t> tris;   // triangle list (3 indices per tri)
};

// Per-shape polygons. Triangle counts (P2 actuals — emitter is ground truth;
// tris_per_leaf_shape() in leaf_geometry.hpp exposes the same numbers at
// runtime so consumers never duplicate this table):
//   oak_lobed       : 14 tris, 15 verts (1 center + 14 perimeter)
//   pine_needle     :  4 tris,  6 verts (2 rows × 3 cols)
//   birch_serrated  : 15 tris, 16 verts (1 center + 15 perimeter)
//   maple_star      : 20 tris, 16 verts (1 center + 5 inner + 5 mid + 5 outer)
LeafShapeMesh oak_lobed();
LeafShapeMesh pine_needle();
LeafShapeMesh birch_serrated();
LeafShapeMesh maple_star();

}  // namespace treegen
