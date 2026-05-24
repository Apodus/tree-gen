#pragma once

// C1 P4 — header-only 8x8 DCT perceptual hash (pHash) for PNG comparison.
//
// Pipeline:
//   1. Decode PNG to RGBA via stb_image (NEVER define STB_IMAGE_IMPLEMENTATION
//      here — the engine has the anchor; this header includes the decls only).
//   2. Bilinearly downscale to 32x32 grayscale (luminance).
//   3. Apply 2D DCT-II separably (cosine table cached per call — 32x32 is
//      cheap enough for tests).
//   4. Take the top-left 8x8 block, drop the DC term, threshold against the
//      median to produce a 64-bit hash.
//
// Hamming distance: popcount(a ^ b). C1's tolerance is <= 6 over 64 bits.
//
// Self-contained: depends only on stb_image.h (header-only-decl), <cmath>,
// <bit>, <cstdint>, <span>, <vector>, <algorithm>. No engine deps so the
// test TU links without dragging Graphics/Application in.

#include <stb_image.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace treegen::phash {

namespace detail {

// Downscale src (w x h RGBA8) to dst_size x dst_size grayscale (float) via
// bilinear filtering. Luminance via Rec. 709 weights.
inline void bilinear_to_gray(const std::uint8_t* src, int w, int h,
                             float* dst, int dst_size) {
    if (w <= 0 || h <= 0) return;
    const float sx = static_cast<float>(w) / static_cast<float>(dst_size);
    const float sy = static_cast<float>(h) / static_cast<float>(dst_size);
    for (int y = 0; y < dst_size; ++y) {
        const float fy = (static_cast<float>(y) + 0.5f) * sy - 0.5f;
        int y0 = static_cast<int>(std::floor(fy));
        int y1 = y0 + 1;
        float ty = fy - static_cast<float>(y0);
        if (y0 < 0) { y0 = 0; ty = 0.0f; }
        if (y1 >= h) y1 = h - 1;
        for (int x = 0; x < dst_size; ++x) {
            const float fx = (static_cast<float>(x) + 0.5f) * sx - 0.5f;
            int x0 = static_cast<int>(std::floor(fx));
            int x1 = x0 + 1;
            float tx = fx - static_cast<float>(x0);
            if (x0 < 0) { x0 = 0; tx = 0.0f; }
            if (x1 >= w) x1 = w - 1;

            auto sample = [&](int xx, int yy) -> float {
                const std::uint8_t* px = src + (yy * w + xx) * 4;
                // Rec. 709 luma.
                return 0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
            };
            const float v00 = sample(x0, y0);
            const float v10 = sample(x1, y0);
            const float v01 = sample(x0, y1);
            const float v11 = sample(x1, y1);
            const float vx0 = v00 + (v10 - v00) * tx;
            const float vx1 = v01 + (v11 - v01) * tx;
            dst[y * dst_size + x] = vx0 + (vx1 - vx0) * ty;
        }
    }
}

// Separable 2D DCT-II on an N x N grid. Standard formulation:
//   X[k] = sum_{n=0..N-1} x[n] * cos(pi * (2n+1) * k / (2N))
// 32x32: cheap. We only need the top-left 8x8 of the output, but emitting
// the full 32x32 keeps the code simple and tests aren't perf-critical.
inline void dct2d(const float* in, float* out, int N) {
    // Cosine table cos(pi*(2n+1)*k/(2N)).
    std::vector<float> cos_table(static_cast<std::size_t>(N) * N);
    const float pi = 3.14159265358979323846f;
    for (int k = 0; k < N; ++k) {
        for (int n = 0; n < N; ++n) {
            cos_table[k * N + n] =
                std::cos(pi * (2.0f * n + 1.0f) * k / (2.0f * N));
        }
    }
    // Row pass.
    std::vector<float> tmp(static_cast<std::size_t>(N) * N);
    for (int y = 0; y < N; ++y) {
        for (int k = 0; k < N; ++k) {
            float s = 0.0f;
            for (int n = 0; n < N; ++n) {
                s += in[y * N + n] * cos_table[k * N + n];
            }
            tmp[y * N + k] = s;
        }
    }
    // Column pass.
    for (int x = 0; x < N; ++x) {
        for (int k = 0; k < N; ++k) {
            float s = 0.0f;
            for (int n = 0; n < N; ++n) {
                s += tmp[n * N + x] * cos_table[k * N + n];
            }
            out[k * N + x] = s;
        }
    }
}

} // namespace detail

// Compute 8x8 DCT pHash of an in-memory PNG. Returns 0 on decode failure.
inline std::uint64_t compute(std::span<const std::uint8_t> png_bytes) {
    if (png_bytes.empty()) return 0;
    int w = 0, h = 0, n = 0;
    stbi_uc* rgba = stbi_load_from_memory(
        png_bytes.data(), static_cast<int>(png_bytes.size()),
        &w, &h, &n, 4);
    if (!rgba || w <= 0 || h <= 0) {
        if (rgba) stbi_image_free(rgba);
        return 0;
    }
    constexpr int kDownsample = 32;
    std::vector<float> small(static_cast<std::size_t>(kDownsample) * kDownsample);
    detail::bilinear_to_gray(rgba, w, h, small.data(), kDownsample);
    stbi_image_free(rgba);

    std::vector<float> dct(static_cast<std::size_t>(kDownsample) * kDownsample);
    detail::dct2d(small.data(), dct.data(), kDownsample);

    // Extract top-left 8x8 (skip DC at [0,0] when computing median; include
    // it in the hash but threshold by median of remaining 63 coeffs).
    float coeffs[64];
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            coeffs[y * 8 + x] = dct[y * kDownsample + x];
        }
    }
    // Median of the 63 non-DC coefficients.
    float scratch[63];
    for (int i = 0; i < 63; ++i) scratch[i] = coeffs[i + 1];
    std::nth_element(scratch, scratch + 31, scratch + 63);
    const float median = scratch[31];

    std::uint64_t hash = 0;
    for (int i = 0; i < 64; ++i) {
        if (coeffs[i] > median) {
            hash |= (1ULL << i);
        }
    }
    return hash;
}

// Hamming distance between two pHashes (popcount of XOR).
inline int hamming(std::uint64_t a, std::uint64_t b) {
    return static_cast<int>(std::popcount(a ^ b));
}

} // namespace treegen::phash
