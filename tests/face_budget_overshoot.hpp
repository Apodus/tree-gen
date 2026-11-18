// Known bark-budget overshoot (tree-budget T1, gate G1g, ruling R-T1-budget 2026-11-18).
//
// Defect: face_budget's estimator under-counts some skeletons, so the emitted
// bark exceeds LodBudget. With the grower stall removed (T1) the regrown oaks show
// it; oak_7 L0, oak_999 L0 and maple_7 L0 (25,033 > 24,000; 22,015 after T1, so
// no row) already overshot on main, unseen because the budget cases built seed 42
// only. Owner: epic 01M7QJGBAQ3CXEJTSCNR31KHCP. Deleted by
// tree-budget T2 (exact budget): its gate is "this table is gone and every tree
// is within budget".
//
// The one rule, applied by BarkBudgetCheck: a listed (tree, options, level) must
// EQUAL its measured value and be over budget (growth, shrinkage, or a row that is
// within budget are all red); an unlisted one must be within budget.
#pragma once

#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <cstring>
#include <string>
#include <vector>

namespace treegen::face_budget_overshoot {

// Options sets: "tool" = as rynx-treegen and the stage build (seam offset, root
// flare, shoulder); "test" = the budget cases' BarkMeshOptions (seam 0, defaults).
// Fork-blend stitch counts follow vertex angles, so the two differ by a few triangles.
struct Row {
    const char* species;
    uint64_t    seed;
    const char* options;
    int         level;
    int         bark_triangles;  // measured at the T1 tip
};

inline constexpr Row k_face_budget_overshoot[] = {
    // T1 tip. Main (T0) for comparison, tool options: oak_7 L0 11986, oak_999 L0 10181 (both
    // already over), oak_42 L0 9469, oak_123 L0 9515 / L1 under 4000.
    {"oak", 7,   "test", 0, 11386},
    {"oak", 42,  "test", 0, 14582},
    {"oak", 123, "test", 0, 10750},
    {"oak", 123, "test", 1, 4092},
    {"oak", 999, "test", 0, 10916},
    {"oak", 7,   "tool", 0, 11358},
    {"oak", 42,  "tool", 0, 14542},
    {"oak", 123, "tool", 0, 10750},
    {"oak", 123, "tool", 1, 4081},
    {"oak", 999, "tool", 0, 10877},
};

class BarkBudgetCheck {
public:
    explicit BarkBudgetCheck(const char* options) : m_options(options) {}

    void check(const std::string& species, uint64_t seed, int level, int measured, int budget) {
        INFO(species << "_" << seed << " options=" << m_options << " L" << level << " bark_triangles=" << measured
                     << " budget=" << budget);
        for (size_t i = 0; i < std::size(k_face_budget_overshoot); ++i) {
            const Row& r = k_face_budget_overshoot[i];
            if (species != r.species || seed != r.seed || level != r.level || std::strcmp(r.options, m_options) != 0)
                continue;
            m_visited.push_back(i);
            std::printf("G1g overshoot %s_%llu %s L%d measured %d listed %d budget %d\n", r.species,
                        (unsigned long long)seed, m_options, level, measured, r.bark_triangles, budget);
            CHECK(measured == r.bark_triangles);  // the listed value is exact
            CHECK(r.bark_triangles > budget);     // a listed row within budget is a stale exemption
            return;
        }
        if (measured > budget)
            std::printf("G1g UNLISTED OVER BUDGET %s_%llu %s L%d measured %d budget %d\n", species.c_str(),
                        (unsigned long long)seed, m_options, level, measured, budget);
        CHECK(measured <= budget);
    }

    // For a case that builds every tree of its options set: every row of that set was seen.
    void require_every_row_visited() const {
        int rows = 0;
        for (size_t i = 0; i < std::size(k_face_budget_overshoot); ++i) {
            if (std::strcmp(k_face_budget_overshoot[i].options, m_options) != 0) continue;
            ++rows;
            INFO("row " << i << " " << k_face_budget_overshoot[i].species << "_" << k_face_budget_overshoot[i].seed
                        << " L" << k_face_budget_overshoot[i].level << " was never built");
            CHECK(std::find(m_visited.begin(), m_visited.end(), i) != m_visited.end());
        }
        CHECK(rows > 0);  // while the table exists it is non-empty (T2 deletes it)
    }

private:
    const char*         m_options;
    std::vector<size_t> m_visited;
};

}  // namespace treegen::face_budget_overshoot
