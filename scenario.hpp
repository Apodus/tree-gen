#pragma once

#include "tree_descriptor.hpp"

#include <cstdint>
#include <string>

namespace treegen {

struct TrivialCylinder {
    float radius          = 0.0f;
    float height          = 0.0f;
    int   radial_segments = 0;
    int   axial_segments  = 0;
};

struct Scenario {
    std::string kind;          // discriminator for the generator switch (C2+)
    std::string name;          // FNV-hashed for per-scenario seed split
    uint64_t    scenario_fnv = 0;
    TrivialCylinder cyl;       // populated when kind == "trivial_cylinder"
    TreeDescriptor  tree;      // populated when kind == "tree"
};

// Reads JSON from `path`, parses into Scenario. Throws std::runtime_error on
// any failure (missing file, malformed JSON, missing required field). CLI
// `main` catches and reports to stderr.
Scenario load_scenario(const std::string& path);

} // namespace treegen
