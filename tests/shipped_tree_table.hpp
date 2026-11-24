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
    {"oak",   7,   0x7c6ee27aa66327e6ull, 0x931f2180c1b1b18aull, 573, 300},  // T1: stall removed from the generator
    {"oak",   42,  0xfa26535e68a1ab45ull, 0x6d3c5a8ab2ece9f0ull, 861, 300},  // T1: stall removed from the generator
    {"oak",   123, 0x66ea3a5112662cacull, 0x24f3fb2b4b408d91ull, 576, 300},  // T1: stall removed from the generator
    {"oak",   999, 0xde947af36995086eull, 0x759d5f88e2140ec2ull, 295, 300},  // T1: stall removed from the generator
    {"maple", 7,   0x508dfc7308d02bb3ull, 0xc126d05697b67c03ull, 917, 250},  // T1: stall removed from the generator
    {"maple", 42,  0x91333d1a878949bcull, 0x91333d1a878949bcull, 0, 29},
    {"maple", 123, 0x5fe21759efbcf5bdull, 0x7d04a13615c975c7ull, 226, 250},  // T1: stall removed from the generator
    {"maple", 999, 0x290fcc0c7da3227aull, 0xb179f4ebd0caa717ull, 458, 250},  // T1: stall removed from the generator
    {"birch", 7,   0x082a206d80c4edb2ull, 0x082a206d80c4edb2ull, 0, 35},
    {"birch", 42,  0x9e42489002fa1bfaull, 0x9e42489002fa1bfaull, 0, 35},
    {"birch", 123, 0x4baa1f375efc8676ull, 0x4baa1f375efc8676ull, 0, 35},
    {"birch", 999, 0x62a2492c822b0970ull, 0x62a2492c822b0970ull, 0, 35},
    {"pine",  7,   0x8bd8d3d63c0f0789ull, 0x27f514f0abc2de8full, 0, 37},  // Pine Uplift: conifer skeleton redesigned
    {"pine",  42,  0xe8cf86c455cd8090ull, 0x00d746fc4855d5b8ull, 291, 300},  // Pine Uplift: conifer skeleton redesigned
    {"pine",  123, 0xc8594dcbcf821f9full, 0x94cf92cd85e971a0ull, 291, 300},  // Pine Uplift: conifer skeleton redesigned
    {"pine",  999, 0x179cefb6aeef41feull, 0x428b039aeb627631ull, 0, 37},  // Pine Uplift: conifer skeleton redesigned
}};

// bark_vertices + leaf_vertices summed over the 16 shipped trees, per level 0..2.
inline constexpr std::array<long long, 3> k_t0_vertex_sum      = {303508, 97676, 39968};
inline constexpr std::array<long long, 3> k_t0_bark_vertex_sum = {165052, 51500, 13784};
inline constexpr std::array<long long, 3> k_t0_leaf_vertex_sum = {138456, 46176, 26184};

// Botanical Leaf Sprigs + Pine Uplift Pass (ClusterCard 3x4-vert bouquets across all 4 species):
// well under frozen k_t0_vertex_sum {303508, 97676, 39968} on all 3 LODs.
inline constexpr std::array<long long, 3> k_t1_vertex_sum      = {277142, 85940, 32242};

// Largest leaf_max_vertices_on_one_host at any level on a clean shipped tree:
// [0] card species (birch, maple, oak, pine), [1] strip species.
inline constexpr std::array<int, 2> k_t0_clean_max_host_vertices = {12, 12};

// Per species over the 36-seed set (0..32, 42, 123, 999), skeleton section only.
// max_children / max_near_pairs are maxima over that species' stall-free seeds.
struct SpeciesRow {
    const char* species;
    int         k_t0_stalled_seeds;  // seeds with coincident_siblings_exact > 0
    int         k_t0_max_children;
    int         k_t0_max_near_pairs;   // no longer read by a gate (R-T1-near): see k_t0_stall_free_near_pairs
    long long   k_t0_consumed_sum;   // attractors_consumed summed over the set
    // The shipped trees' _RYNX_COLLISION capsule (scenario constants), recorded at T1
    // from the T0-tip GLBs (t0_ frozen; capsule_half_length tracks scenario updates).
    float       t0_capsule_half_length;
    float       t0_capsule_radius;
    float       capsule_half_length;
};

inline constexpr std::array<SpeciesRow, 4> k_species = {{
    {"oak",   36, 3, 243, 21348, 2.0999999f, 0.300000012f, 2.0999999f},  // no stall-free seed: maxima are the other three species' largest
    {"maple", 29, 3, 243, 24863, 1.80000007f, 0.300000012f, 1.80000007f},
    {"birch", 5,  3, 179, 17994, 1.75f, 0.200000003f, 1.75f},
    {"pine",  11, 3, 49,  7703, 1.35000002f, 0.400000006f, 1.80000007f},  // Pine Uplift: crown_base_fraction 0.15 -> 0.20
}};

// near_sibling_pairs of every seed that was stall-free on main (coincident_siblings_exact == 0
// at T0): their skeletons are unchanged by T1, so G1a asserts the same totals.
struct StallFreeNearPairs {
    const char* species;
    uint64_t    seed;
    int         near_sibling_pairs;
};
inline constexpr StallFreeNearPairs k_t0_stall_free_near_pairs[] = {
    {"maple", 0, 243}, {"maple", 5, 240}, {"maple", 8, 234}, {"maple", 23, 238}, {"maple", 28, 242},
    {"maple", 29, 243}, {"maple", 42, 241},
    {"birch", 0, 163}, {"birch", 1, 163}, {"birch", 2, 151}, {"birch", 3, 152}, {"birch", 4, 159},
    {"birch", 5, 160}, {"birch", 7, 160}, {"birch", 8, 156}, {"birch", 11, 179}, {"birch", 12, 161},
    {"birch", 13, 154}, {"birch", 14, 159}, {"birch", 16, 152}, {"birch", 17, 164}, {"birch", 18, 169},
    {"birch", 19, 152}, {"birch", 20, 149}, {"birch", 21, 159}, {"birch", 22, 154}, {"birch", 23, 163},
    {"birch", 24, 152}, {"birch", 26, 164}, {"birch", 27, 158}, {"birch", 28, 163}, {"birch", 29, 151},
    {"birch", 30, 157}, {"birch", 31, 159}, {"birch", 32, 168}, {"birch", 42, 163}, {"birch", 123, 160},
    {"birch", 999, 159},
    {"pine", 0, 46}, {"pine", 1, 47}, {"pine", 2, 40}, {"pine", 3, 47}, {"pine", 4, 42}, {"pine", 5, 46},
    {"pine", 6, 43}, {"pine", 7, 42}, {"pine", 8, 48}, {"pine", 9, 44}, {"pine", 10, 44}, {"pine", 11, 44},
    {"pine", 12, 44}, {"pine", 13, 40}, {"pine", 15, 49}, {"pine", 17, 40}, {"pine", 19, 47}, {"pine", 20, 39},
    {"pine", 23, 43}, {"pine", 24, 46}, {"pine", 25, 41}, {"pine", 26, 40}, {"pine", 30, 46}, {"pine", 32, 46},
    {"pine", 999, 47},
};

inline constexpr std::array<uint64_t, 36> k_seed_set = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 42, 123, 999,
};

}  // namespace treegen::shipped_table
