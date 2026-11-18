// T0 ledger baselines for the 16 shipped trees and the 36-seed set, filled from
// the [ledger] cases' own output on their first run (treegen aafcb1c, Release).
// The t0_ / k_t0_ values are FROZEN: no later phase edits them. `skeleton_fnv`
// is the column later phases update.
#pragma once

#include <array>
#include <cstdint>

namespace treegen::shipped_table {

struct Row {
    const char* species;
    uint64_t    seed;                          // CLI seed; seed_effective = seed ^ scenario_fnv
    uint64_t    t0_skeleton_fnv;               // frozen
    uint64_t    skeleton_fnv;                  // current generator
    int         t0_coincident_siblings_exact;  // frozen
    int         t0_iterations_run;             // frozen
};

// Species oak, maple, birch, pine x seeds 7, 42, 123, 999, as the stage builds them.
inline constexpr std::array<Row, 16> k_rows = {{
    {"oak",   7,   0x7c6ee27aa66327e6ull, 0x7c6ee27aa66327e6ull, 573, 300},
    {"oak",   42,  0xfa26535e68a1ab45ull, 0xfa26535e68a1ab45ull, 861, 300},
    {"oak",   123, 0x66ea3a5112662cacull, 0x66ea3a5112662cacull, 576, 300},
    {"oak",   999, 0xde947af36995086eull, 0xde947af36995086eull, 295, 300},
    {"maple", 7,   0x508dfc7308d02bb3ull, 0x508dfc7308d02bb3ull, 917, 250},
    {"maple", 42,  0x91333d1a878949bcull, 0x91333d1a878949bcull, 0, 29},
    {"maple", 123, 0x5fe21759efbcf5bdull, 0x5fe21759efbcf5bdull, 226, 250},
    {"maple", 999, 0x290fcc0c7da3227aull, 0x290fcc0c7da3227aull, 458, 250},
    {"birch", 7,   0x082a206d80c4edb2ull, 0x082a206d80c4edb2ull, 0, 35},
    {"birch", 42,  0x9e42489002fa1bfaull, 0x9e42489002fa1bfaull, 0, 35},
    {"birch", 123, 0x4baa1f375efc8676ull, 0x4baa1f375efc8676ull, 0, 35},
    {"birch", 999, 0x62a2492c822b0970ull, 0x62a2492c822b0970ull, 0, 35},
    {"pine",  7,   0x8bd8d3d63c0f0789ull, 0x8bd8d3d63c0f0789ull, 0, 37},
    {"pine",  42,  0xe8cf86c455cd8090ull, 0xe8cf86c455cd8090ull, 291, 300},
    {"pine",  123, 0xc8594dcbcf821f9full, 0xc8594dcbcf821f9full, 291, 300},
    {"pine",  999, 0x179cefb6aeef41feull, 0x179cefb6aeef41feull, 0, 37},
}};

// bark_vertices + leaf_vertices summed over the 16 shipped trees, per level 0..2.
inline constexpr std::array<long long, 3> k_t0_vertex_sum      = {303508, 97676, 39968};
inline constexpr std::array<long long, 3> k_t0_bark_vertex_sum = {165052, 51500, 13784};
inline constexpr std::array<long long, 3> k_t0_leaf_vertex_sum = {138456, 46176, 26184};

// Largest leaf_max_vertices_on_one_host at any level on a clean shipped tree:
// [0] card species (birch, maple, oak), [1] strip species (pine).
inline constexpr std::array<int, 2> k_t0_clean_max_host_vertices = {12, 12};

// Per species over the 36-seed set (0..32, 42, 123, 999), skeleton section only.
// max_children / max_near_pairs are maxima over that species' stall-free seeds.
struct SpeciesRow {
    const char* species;
    int         k_t0_stalled_seeds;  // seeds with coincident_siblings_exact > 0
    int         k_t0_max_children;
    int         k_t0_max_near_pairs;
    long long   k_t0_consumed_sum;   // attractors_consumed summed over the set
};

inline constexpr std::array<SpeciesRow, 4> k_species = {{
    {"oak",   36, 3, 243, 21348},  // no stall-free seed: maxima are the other three species' largest
    {"maple", 29, 3, 243, 24863},
    {"birch", 5,  3, 179, 17994},
    {"pine",  11, 3, 49,  7703},
}};

inline constexpr std::array<uint64_t, 36> k_seed_set = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 42, 123, 999,
};

}  // namespace treegen::shipped_table
