// The shipped trees (every tracked tree scenario at the stage's seeds) as the
// tool generates them, built once per test process and shared by every case
// that needs the tool's exact output (extraction table, ledger, crown probe).
#pragma once

#include "test_support_paths.hpp"

#include "../generate_tree.hpp"
#include "../scenario.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace rynx::test_support {

inline treegen::Scenario load_species_scenario(const std::string& species) {
    const std::string path = treegen_file("scenarios/c3_" + species + ".json");
    if (path.empty()) throw std::runtime_error("no scenarios/c3_" + species + ".json");
    return treegen::load_scenario(path);
}

// generate_tree(c3_<species>.json, seed) with default options, cached.
inline const treegen::GeneratedTree& generated_tree(const std::string& species, uint64_t seed) {
    static std::map<std::pair<std::string, uint64_t>, std::unique_ptr<treegen::GeneratedTree>> cache;
    auto& slot = cache[{species, seed}];
    if (!slot)
        slot = std::make_unique<treegen::GeneratedTree>(treegen::generate_tree(load_species_scenario(species), seed));
    return *slot;
}

} // namespace rynx::test_support
