// Procedural bark texture maps. 1024x512 RGBA8.
//   C6 P1/P2: 4 species x {diffuse, normal, AO, roughness}.
//   Bark B-1: single-pass exact integer-periodic 2D torus heightfield + Worley
//   species morphology, physically scaled normals (18-38 deg), true multi-radius
//   cavity AO, full-V birch lenticels, and per-species bundle dedup.
#pragma once

#include "tree_descriptor.hpp"   // for treegen::LeafShape

#include <cstdint>
#include <string_view>
#include <vector>

namespace treegen {

    constexpr int K_BARK_TEX_W = 1024;
    constexpr int K_BARK_TEX_H = 512;

    // Single-pass encoded PNG bundle (avoids 4x redundant heightfield synthesis).
    struct BarkPngBundle {
        std::vector<uint8_t> diffuse_png;
        std::vector<uint8_t> normal_png;
        std::vector<uint8_t> ao_png;
        std::vector<uint8_t> roughness_png;
    };

    BarkPngBundle encode_bark_bundle_pngs(LeafShape species, uint64_t seed);

    // ---- Diffuse (sRGB-encoded RGBA8).
    // Returns 1024*512*4 bytes. Stride = w*4.
    std::vector<uint8_t> bake_bark_diffuse(LeafShape species, uint64_t seed);
    std::vector<uint8_t> encode_bark_png  (LeafShape species, uint64_t seed);

    // ---- Normal map (RGB encoded tangent-space normal, +Y up; A=255).
    // Physically scaled (-dh/du_m, -dh/dv_m, 1) normalized -> linear UNORM.
    std::vector<uint8_t> bake_bark_normal(LeafShape species, uint64_t seed);
    std::vector<uint8_t> encode_bark_normal_png(LeafShape species, uint64_t seed);

    // ---- AO map. Multi-radius wrap-aware cavity AO -> grayscale linear UNORM.
    // R = AO byte (LINEAR UNORM — engine samples as non-color); G=R, B=R; A=255.
    std::vector<uint8_t> bake_bark_ao(LeafShape species, uint64_t seed);
    std::vector<uint8_t> encode_bark_ao_png(LeafShape species, uint64_t seed);

    // ---- Roughness map (glTF metallic-roughness layout, LINEAR UNORM).
    // R=255 (unused), G=roughness (LINEAR byte), B=0 (metallic=0), A=255.
    std::vector<uint8_t> bake_bark_roughness(LeafShape species, uint64_t seed);
    std::vector<uint8_t> encode_bark_roughness_png(LeafShape species, uint64_t seed);

    // ---- Per-species seed mixing helper. Wraps fnv1a64(leaf_shape_to_string(s)).
    // Single source of truth for seed-derivation; callers must use this rather
    // than ad-hoc string literals to keep the per-species streams non-aliased.
    uint64_t species_fnv_name(LeafShape species);

}
