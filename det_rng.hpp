// /fp:precise is required for byte-hash determinism of any tool consuming
// this RNG. See rynx/generate/treegen.sharpmake.cs.
#pragma once

#include <cstdint>
#include <string_view>

namespace treegen {

// Canonical PCG32-XSH-RR (state advance + XOR-shift + right rotation). Seed
// `state` and `inc` per the reference; `inc` must be odd. Two RNGs with the
// same (state, inc) produce identical streams across runs and platforms.
struct pcg32 {
    uint64_t state = 0;
    uint64_t inc   = 1;

    // Initialise per the reference seed routine: zero state, advance once,
    // add user seed, advance again. `stream` selects an independent sequence.
    void seed(uint64_t init_state, uint64_t stream) {
        state = 0;
        inc   = (stream << 1u) | 1u;
        (void)next_u32();
        state += init_state;
        (void)next_u32();
    }

    uint32_t next_u32() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-static_cast<int>(rot)) & 31));
    }

    // 24-bit mantissa float in [0, 1). Division by 2^24 is exact under
    // /fp:precise.
    float next_float_01() {
        constexpr float k_scale = 1.0f / static_cast<float>(1u << 24);
        return static_cast<float>(next_u32() >> 8) * k_scale;
    }
};

// Standard FNV-1a 64-bit hash. Pure integer math; deterministic regardless of
// FP mode but kept alongside pcg32 since downstream uses both for the
// `seed_effective = cli_seed ^ fnv1a64(scenario_name)` pattern.
inline uint64_t fnv1a64(std::string_view s) {
    constexpr uint64_t k_offset = 14695981039346656037ULL;
    constexpr uint64_t k_prime  = 1099511628211ULL;
    uint64_t h = k_offset;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= k_prime;
    }
    return h;
}

} // namespace treegen
