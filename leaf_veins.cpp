// /fp:precise — vein endpoint coordinates feed the pixel-space line stamper,
// which decides per-pixel vein-height bytes. See treegen.sharpmake.cs.
#include "leaf_veins.hpp"

#include <cmath>
#include <cstdint>

namespace treegen {

std::vector<VeinSegment> veins_for_species(LeafShape s, const LeafShapeMesh& /*mesh*/) {
    std::vector<VeinSegment> out;
    switch (s) {
        case LeafShape::PineNeedle: {
            // Single central spine — no laterals (needle silhouette).
            out.push_back(VeinSegment{vec2{0.0f, -1.0f}, vec2{0.0f, +1.0f}});
            break;
        }
        case LeafShape::OakLobed: {
            // Strong central midrib + 4 pinnate lateral pairs running into the lobes.
            out.push_back(VeinSegment{vec2{0.0f, -0.92f}, vec2{0.0f, +0.94f}});
            const float roots[4]  = { -0.45f, -0.10f, +0.25f, +0.55f };
            const float tip_x[4]  = {  0.34f,  0.54f,  0.46f,  0.26f };
            const float tip_y[4]  = { -0.26f, +0.16f, +0.52f, +0.76f };
            for (int k = 0; k < 4; ++k) {
                out.push_back(VeinSegment{vec2{0.0f, roots[k]}, vec2{+tip_x[k], tip_y[k]}});
                out.push_back(VeinSegment{vec2{0.0f, roots[k]}, vec2{-tip_x[k], tip_y[k]}});
            }
            break;
        }
        case LeafShape::BirchSerrated: {
            // Central midrib + 5 ascending pinnate lateral pairs.
            out.push_back(VeinSegment{vec2{0.0f, -0.84f}, vec2{0.0f, +0.92f}});
            const float roots[5]  = { -0.58f, -0.32f, -0.05f, +0.22f, +0.48f };
            const float tip_x[5]  = {  0.52f,  0.56f,  0.44f,  0.30f,  0.16f };
            const float tip_y[5]  = { -0.34f, -0.02f, +0.26f, +0.52f, +0.74f };
            for (int k = 0; k < 5; ++k) {
                out.push_back(VeinSegment{vec2{0.0f, roots[k]}, vec2{+tip_x[k], tip_y[k]}});
                out.push_back(VeinSegment{vec2{0.0f, roots[k]}, vec2{-tip_x[k], tip_y[k]}});
            }
            break;
        }
        case LeafShape::MapleStar: {
            // 5 palmate primary veins radiating from the basal palm junction into the 5 lobes,
            // plus secondary side veins off the central lobe.
            const vec2 base{0.0f, -0.18f};
            out.push_back(VeinSegment{base, vec2{ 0.00f, +0.94f}});  // central lobe
            out.push_back(VeinSegment{base, vec2{+0.72f, +0.44f}});  // upper-right lobe
            out.push_back(VeinSegment{base, vec2{-0.72f, +0.44f}});  // upper-left lobe
            out.push_back(VeinSegment{base, vec2{+0.50f, -0.22f}});  // basal-right lobe
            out.push_back(VeinSegment{base, vec2{-0.50f, -0.22f}});  // basal-left lobe
            // Secondary veins off central midrib.
            out.push_back(VeinSegment{vec2{0.0f, +0.25f}, vec2{+0.24f, +0.52f}});
            out.push_back(VeinSegment{vec2{0.0f, +0.25f}, vec2{-0.24f, +0.52f}});
            break;
        }
    }
    return out;
}

}  // namespace treegen
