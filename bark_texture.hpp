// Procedural bark texture maps. 1024x512 RGBA8.
//   C6 P1/P2: 4 species x {diffuse, normal, AO, roughness}.
//   C2 P1: simplex noise FBM + analytic derivatives (replaces value noise).
//   C2 P2: 3-key color ramp per species (crevice/mid/ridge replaces baseColor).
#pragma once

#include "tree_descriptor.hpp"   // for treegen::LeafShape

#include <cstdint>
#include <string_view>
#include <vector>

namespace treegen {

    // ---- Diffuse (sRGB-encoded RGBA8).
    // Returns 1024*512*4 bytes. Stride = w*4.
    std::vector<uint8_t> bake_bark_diffuse(LeafShape species, uint64_t seed);
    std::vector<uint8_t> encode_bark_png  (LeafShape species, uint64_t seed);

    // ---- C6 P2 — Normal map (RGB encoded tangent-space normal, +Y up; A=255).
    //   (nx, ny) = normalize(-dh/dx * strength, -dh/dy * strength, 1.0)
    // Encoded n*0.5+0.5 → sRGB-free byte (linear UNORM by GL/glTF convention for
    // normal maps — sampler is non-color).
    std::vector<uint8_t> bake_bark_normal(LeafShape species, uint64_t seed);
    std::vector<uint8_t> encode_bark_normal_png(LeafShape species, uint64_t seed);

    // ---- C6 P2 — AO map. 7×7 neighborhood mean-of-heightfield → grayscale AO.
    // R = AO byte (LINEAR UNORM — engine samples as non-color); G=R, B=R; A=255.
    std::vector<uint8_t> bake_bark_ao(LeafShape species, uint64_t seed);
    std::vector<uint8_t> encode_bark_ao_png(LeafShape species, uint64_t seed);

    // ---- C6 P2 — Roughness map (glTF metallic-roughness layout, LINEAR UNORM).
    // R=255 (unused), G=roughness (LINEAR byte), B=0 (metallic=0), A=255.
    std::vector<uint8_t> bake_bark_roughness(LeafShape species, uint64_t seed);
    std::vector<uint8_t> encode_bark_roughness_png(LeafShape species, uint64_t seed);

    // ---- Per-species seed mixing helper. Wraps fnv1a64(leaf_shape_to_string(s)).
    // Single source of truth for seed-derivation; callers must use this rather
    // than ad-hoc string literals to keep the per-species streams non-aliased.
    uint64_t species_fnv_name(LeafShape species);

}
