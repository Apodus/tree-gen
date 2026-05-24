// C2 P1 — 2D simplex gradient noise with analytic derivatives.
// Standard Perlin simplex algorithm (patent expired 2022). Deterministic given
// seed; uses mix64 (splitmix) for gradient selection.
#pragma once

#include <cmath>
#include <cstdint>

namespace treegen {

    struct simplex_result { float value; float dx; float dy; };

    namespace detail {

        inline uint64_t mix64(uint64_t x) {
            x += 0x9E3779B97F4A7C15ULL;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            return x ^ (x >> 31);
        }

        // 8 unit-circle gradients (45-degree spacing). Sufficient for 2D simplex;
        // more directions don't improve isotropy measurably.
        inline void grad2(uint64_t hash, float& gx, float& gy) {
            constexpr float k_grads[8][2] = {
                { 1.0f,  0.0f}, { 0.707107f,  0.707107f},
                { 0.0f,  1.0f}, {-0.707107f,  0.707107f},
                {-1.0f,  0.0f}, {-0.707107f, -0.707107f},
                { 0.0f, -1.0f}, { 0.707107f, -0.707107f},
            };
            int idx = static_cast<int>(hash & 7);
            gx = k_grads[idx][0];
            gy = k_grads[idx][1];
        }

        inline uint64_t corner_hash(int ix, int iy, uint64_t seed) {
            uint64_t h = seed;
            h = mix64(h ^ static_cast<uint64_t>(static_cast<uint32_t>(ix)));
            h = mix64(h ^ (static_cast<uint64_t>(static_cast<uint32_t>(iy)) << 1));
            return h;
        }

    } // namespace detail

    // 2D simplex noise at (x, y) with analytic derivatives.
    // Returns value in approximately [-1, 1] (un-normalized; typical range ~[-0.7, 0.7]).
    inline simplex_result simplex_noise_2d(float x, float y, uint64_t seed) {
        // Skew to simplex grid.
        constexpr float F2 = 0.3660254037844386f;  // 0.5*(sqrt(3)-1)
        constexpr float G2 = 0.21132486540518713f; // (3-sqrt(3))/6

        const float s = (x + y) * F2;
        const float xs = x + s;
        const float ys = y + s;
        const int i = static_cast<int>(std::floor(xs));
        const int j = static_cast<int>(std::floor(ys));

        // Unskew back.
        const float t = static_cast<float>(i + j) * G2;
        const float X0 = static_cast<float>(i) - t;
        const float Y0 = static_cast<float>(j) - t;
        const float x0 = x - X0;
        const float y0 = y - Y0;

        // Determine which simplex triangle we're in.
        int i1, j1;
        if (x0 > y0) { i1 = 1; j1 = 0; }
        else          { i1 = 0; j1 = 1; }

        const float x1 = x0 - static_cast<float>(i1) + G2;
        const float y1 = y0 - static_cast<float>(j1) + G2;
        const float x2 = x0 - 1.0f + 2.0f * G2;
        const float y2 = y0 - 1.0f + 2.0f * G2;

        // Gradient hashes for 3 corners.
        uint64_t h0 = detail::corner_hash(i,      j,      seed);
        uint64_t h1 = detail::corner_hash(i + i1, j + j1, seed);
        uint64_t h2 = detail::corner_hash(i + 1,  j + 1,  seed);

        float gx0, gy0, gx1, gy1, gx2, gy2;
        detail::grad2(h0, gx0, gy0);
        detail::grad2(h1, gx1, gy1);
        detail::grad2(h2, gx2, gy2);

        // Radial falloff kernel: (0.5 - d^2)^4. Exponent 4 gives C1 continuity
        // and smooth analytic derivatives.
        float value = 0.0f;
        float dnoise_dx = 0.0f;
        float dnoise_dy = 0.0f;

        auto contribute = [&](float cx, float cy, float gx, float gy) {
            float t_val = 0.5f - cx * cx - cy * cy;
            if (t_val > 0.0f) {
                float t2 = t_val * t_val;
                float t4 = t2 * t2;
                float gdot = gx * cx + gy * cy;
                value += t4 * gdot;
                // d/dx of t^4 * (gx*cx + gy*cy):
                //   = 4*t^3 * (-2*cx) * gdot + t^4 * gx
                float dt = -8.0f * t_val * t2 * gdot;
                dnoise_dx += dt * cx + t4 * gx;
                dnoise_dy += dt * cy + t4 * gy;
            }
        };

        contribute(x0, y0, gx0, gy0);
        contribute(x1, y1, gx1, gy1);
        contribute(x2, y2, gx2, gy2);

        // Scale to approximately [-1, 1]. The theoretical max of 2D simplex with
        // this kernel is ~0.0225; scale by 70.0 per reference implementations.
        constexpr float k_scale = 70.0f;
        return { value * k_scale, dnoise_dx * k_scale, dnoise_dy * k_scale };
    }

} // namespace treegen
