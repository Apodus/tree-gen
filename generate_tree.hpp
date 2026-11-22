// The tool's scenario-to-tree pipeline as one library call (crown-shadows
// S-P0a, K17 "Pipeline"): skeleton, leaf sites, every LOD, the atlases' PNG
// bytes and the GLB as bytes. The rynx-treegen tree branch, the crown probe,
// the fixture-freshness gate and the tests that need the tool's exact output
// all call this; there is no second copy of the pipeline.
#pragma once

#include "collision_fit.hpp"
#include "glb_writer.hpp"
#include "lod_emitter.hpp"
#include "scenario.hpp"
#include "skeleton.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace treegen {

struct GenerateOptions {
    // false: skeleton, leaf sites and capsule only (no LODs, images or GLB) --
    // the tool without --out, which never built levels.
    bool levels = true;
    // false: levels without the images or the GLB (a consumer of the meshes
    // alone, e.g. the ledger's in-process mirror).
    bool glb = true;
};
// K17: every field changes what default options produce, which the extraction
// table ([treegen_generate_tree]) pins; a field added, removed or retyped is a
// deliberate act that re-records that table and updates these checks. The
// binding fails to compile on any field count but two, whatever the size; the
// decltypes catch a same-size type swap; the return pins both defaults. Still
// open: a field whose meaning changes under the same name, type and default.
consteval bool generate_options_fields_are_pinned() {
    const auto [levels, glb] = GenerateOptions{};
    static_assert(std::is_same_v<decltype(levels), const bool>, "GenerateOptions::levels retyped: re-record the extraction table");
    static_assert(std::is_same_v<decltype(glb), const bool>, "GenerateOptions::glb retyped: re-record the extraction table");
    return levels && glb;
}
static_assert(generate_options_fields_are_pinned(), "GenerateOptions changed: re-record the extraction table");
static_assert(sizeof(GenerateOptions) == 2, "GenerateOptions changed: re-record the extraction table");

struct GeneratedTree {
    uint64_t               seed_effective = 0;  // cli_seed ^ fnv1a64(scenario.name)
    TreeSkeleton           skeleton;            // leaf_sites filled
    TrunkCapsule           capsule;
    std::vector<LodOutput> lods;                // L0..L2; per-LOD kept-site lists in leaf_kept_sites
    std::vector<ImageData> images;              // GLB image order: bark 0-3, leaf 4-7 when leafy (options.glb)
    std::vector<uint8_t>   glb;                 // what --out writes, byte for byte (options.glb)
};

// `scenario.kind` must be "tree" (throws std::runtime_error otherwise, and on a
// GLB build failure).
GeneratedTree generate_tree(const Scenario& scenario, uint64_t cli_seed, const GenerateOptions& options = {});

}  // namespace treegen
