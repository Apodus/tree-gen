// The tool's scenario-to-tree pipeline as library calls (crown-shadows S-P0a,
// K17 "Pipeline"), one per stage, each a strict extension of the last:
// grow_tree (skeleton, leaf sites, capsule) -> emit_levels (every LOD) ->
// generate_tree (the atlases' PNG bytes and the GLB as bytes). The rynx-treegen
// tree branch, the crown probe, the ledger and the tests that need the tool's
// exact output all call these; there is no second copy of the pipeline and no
// options struct whose defaults could drift (review R-CR5d): a caller names the
// stage it needs, and each stage's output is gated by its consumer (the
// extraction table for the GLB, the ledger for the levels).
#pragma once

#include "collision_fit.hpp"
#include "crown_volume.hpp"
#include "glb_writer.hpp"
#include "lod_emitter.hpp"
#include "scenario.hpp"
#include "skeleton.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace treegen {

struct GrownTree {
    uint64_t     scenario_fnv   = 0;  // Scenario::scenario_fnv that grew this skeleton (D4)
    uint64_t     seed_effective = 0;  // cli_seed ^ fnv1a64(scenario.name)
    TreeSkeleton skeleton;            // leaf_sites filled
    TrunkCapsule capsule;
};

struct TreeLevels : GrownTree {
    std::vector<LodOutput> lods;      // L0..L2; per-LOD kept-site lists in leaf_kept_sites
};

struct GeneratedTree : TreeLevels {
    std::vector<ImageData>                  images;        // GLB image order: bark 0-3, leaf 4-7 when leafy
    std::optional<crown::PackedCrownVolume> crown_volume;  // S-P1 F2_tensor Option 2A (present iff leafy)
    std::vector<uint8_t>                    glb;           // what --out writes, byte for byte
};

// `scenario.kind` must be "tree" (throws std::runtime_error otherwise).
GrownTree grow_tree(const Scenario& scenario, uint64_t cli_seed);
// Every LOD of a grown tree (no image or GLB byte).
TreeLevels emit_levels(const Scenario& scenario, GrownTree grown);
// The whole pipeline; throws std::runtime_error on a GLB build failure.
GeneratedTree generate_tree(const Scenario& scenario, uint64_t cli_seed);

}  // namespace treegen
