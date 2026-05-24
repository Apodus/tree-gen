// Halton(2,3,5) 3D low-discrepancy sampler. Deterministic, integer-keyed —
// the i-th sample depends only on i, so attractor positions are reproducible
// regardless of insertion order. Used by space_colonization.cpp to scatter
// attractor candidates inside the envelope AABB; candidates outside the actual
// envelope shape are rejected and the next halton index is tried.
#pragma once

#include "vec3.hpp"

#include <cstdint>

namespace treegen {

// Van der Corput radical-inverse for the given prime `base`. Returns a value
// in [0, 1). Pure integer math + a single FP division at the end — exact under
// /fp:precise.
inline float halton_radical_inverse(uint32_t i, uint32_t base) {
    float f      = 1.0f;
    float result = 0.0f;
    while (i > 0) {
        f      /= static_cast<float>(base);
        result += f * static_cast<float>(i % base);
        i      /= base;
    }
    return result;
}

// (2, 3, 5) — well-distributed in 3D for small N (< ~10k). For larger N the
// 5-base direction shows striping; not a concern at our attractor counts.
inline vec3 halton_3d(uint32_t i) {
    // i+1 because Halton(0) == (0,0,0) which would always cluster at the AABB
    // corner — shift the sequence start by one.
    return {
        halton_radical_inverse(i + 1u, 2u),
        halton_radical_inverse(i + 1u, 3u),
        halton_radical_inverse(i + 1u, 5u),
    };
}

} // namespace treegen
