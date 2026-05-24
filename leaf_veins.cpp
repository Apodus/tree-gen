// /fp:precise — vein endpoint coordinates feed the pixel-space line stamper,
// which decides per-pixel vein-height bytes. See treegen.sharpmake.cs.
#include "leaf_veins.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace treegen {

namespace {

constexpr float k_pi      = 3.14159265358979323846f;
constexpr float k_half_pi = 1.57079632679489661923f;

// Lateral picker for oak/birch: from `n_perim` perimeter verts (indices
// 1..n_perim in the mesh), drop the vert nearest θ=π/2 (the apex on +Y) and
// evenly stride the rest into 4 laterals. Returns mesh indices.
void pick_4_laterals_skip_apex(int n_perim, std::vector<int>& out_indices,
                               const LeafShapeMesh& mesh) {
    // Find apex index (perimeter vert with angle closest to +π/2 = atan2(y, x)).
    int apex_local = 0;
    float best_da = 1e9f;
    for (int i = 0; i < n_perim; ++i) {
        const vec2 v = mesh.verts[1 + i];
        const float a = std::atan2(v.y, v.x);
        const float da = std::abs(a - k_half_pi);
        if (da < best_da) { best_da = da; apex_local = i; }
    }
    // Build the candidate ring skipping apex; stride evenly into 4.
    std::vector<int> ring;
    ring.reserve(static_cast<size_t>(n_perim - 1));
    for (int i = 0; i < n_perim; ++i) {
        if (i == apex_local) continue;
        ring.push_back(1 + i);
    }
    const int m = static_cast<int>(ring.size());
    for (int k = 0; k < 4; ++k) {
        const int j = (k * m) / 4;  // 0, m/4, m/2, 3m/4 (truncation; deterministic)
        out_indices.push_back(ring[j]);
    }
}

}  // namespace

std::vector<VeinSegment> veins_for_species(LeafShape s, const LeafShapeMesh& mesh) {
    std::vector<VeinSegment> out;
    switch (s) {
        case LeafShape::PineNeedle: {
            // Single central spine — no laterals (needle silhouette).
            out.push_back(VeinSegment{vec2{0.0f, -1.0f}, vec2{0.0f, +1.0f}});
            break;
        }
        case LeafShape::OakLobed: {
            const int n_perim = static_cast<int>(mesh.verts.size()) - 1;  // 14
            out.push_back(VeinSegment{vec2{0.0f, 0.0f}, vec2{0.0f, +1.0f}});
            std::vector<int> lat;
            pick_4_laterals_skip_apex(n_perim, lat, mesh);
            for (int idx : lat) {
                const vec2 tip = mesh.verts[idx];
                out.push_back(VeinSegment{vec2{0.0f, 0.0f}, vec2{tip.x * 0.85f, tip.y * 0.85f}});
            }
            break;
        }
        case LeafShape::BirchSerrated: {
            const int n_perim = static_cast<int>(mesh.verts.size()) - 1;  // 15
            out.push_back(VeinSegment{vec2{0.0f, 0.0f}, vec2{0.0f, +1.0f}});
            std::vector<int> lat;
            pick_4_laterals_skip_apex(n_perim, lat, mesh);
            for (int idx : lat) {
                const vec2 tip = mesh.verts[idx];
                out.push_back(VeinSegment{vec2{0.0f, 0.0f}, vec2{tip.x * 0.85f, tip.y * 0.85f}});
            }
            break;
        }
        case LeafShape::MapleStar: {
            // Outer ring verts live at indices 11..15. verts[11] is the +Y star
            // point (phase shift in leaf_shapes::maple_star).
            const vec2 apex = mesh.verts[11];
            out.push_back(VeinSegment{vec2{0.0f, 0.0f}, apex});
            for (int i = 12; i <= 15; ++i) {
                const vec2 tip = mesh.verts[i];
                out.push_back(VeinSegment{vec2{0.0f, 0.0f}, vec2{tip.x * 0.90f, tip.y * 0.90f}});
            }
            break;
        }
    }
    return out;
}

}  // namespace treegen
