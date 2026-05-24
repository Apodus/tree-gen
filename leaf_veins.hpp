// C6 P3b — per-species vein endpoint derivation. Pure POD; no engine deps.
// /fp:precise (treegen.sharpmake.cs) — endpoint coords feed pixel-space line
// stamps that contribute to atlas byte hash.
#pragma once

#include "leaf_shapes.hpp"      // vec2, LeafShapeMesh
#include "tree_descriptor.hpp"  // LeafShape

#include <utility>
#include <vector>

namespace treegen {

    // One vein segment in shape space ([-1, 1]², apex +Y). `from` is the root
    // end (full width); `to` is the tip end (half width). 2-px-wide line is
    // stamped by the rasterizer in atlas pixel space.
    struct VeinSegment {
        vec2 from;
        vec2 to;
    };

    // Per-species vein endpoints. Derivation follows the C6 P3 plan:
    //   Oak  / Birch — central (0,0)→(0,+1) + 4 lateral verts from perimeter
    //                  (skip the perimeter vert closest to θ=π/2; even-pick the
    //                  remaining), scaled inward by 0.85.
    //   Pine         — central spine (0,-1)→(0,+1); no laterals.
    //   Maple        — central (0,0)→verts[11] (outer +Y point); 4 laterals
    //                  from verts[12..15] scaled inward by 0.90.
    std::vector<VeinSegment> veins_for_species(LeafShape s, const LeafShapeMesh& mesh);

}
