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

// Per-shape botanical polygons. Triangle counts (emitter is ground truth;
// tris_per_leaf_shape() in leaf_geometry.hpp exposes the same numbers at
// runtime so consumers never duplicate this table):
//   oak_lobed       : 64 tris, 65 verts (1 center + 64 perimeter; 7-9 lobe + basal auricles)
//   pine_needle     :  7 tris,  9 verts (4-row tapered strip + pointed tip)
//   birch_serrated  : 64 tris, 65 verts (1 center + 64 perimeter; doubly-serrated ovate/cordate)
//   maple_star      : 72 tris, 73 verts (1 center + 72 perimeter; 5-lobe palmate with U-sinuses)
LeafShapeMesh oak_lobed();
LeafShapeMesh pine_needle();
LeafShapeMesh birch_serrated();
LeafShapeMesh maple_star();

}  // namespace treegen
