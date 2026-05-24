// Tropism evaluator — blends raw growth direction (from attractor mean) with
// gravitropism (downward bias for weeping shapes; signed → positive value
// pulls UP — the "negative gravitropism" upward growth biologists use) and
// phototropism (toward `light_dir`). Both are scaled by the descriptor's
// global tropism_strength so a single species knob can soften / amplify the
// total bias without retuning each axis.
//
// Header-only so radius_solver/space_colonization can both consume without an
// extra TU. /fp:precise.
#pragma once

#include "tree_descriptor.hpp"
#include "vec3.hpp"

namespace treegen {

// Default light_dir convention: (0,0,1) = sun directly overhead. A descriptor
// with phototropism > 0 + light_dir = (0,0,1) biases growth upward — biologically
// equivalent to positive phototropism in nature.
inline vec3 apply_tropisms(vec3 raw_dir, const TreeDescriptor::Tropisms& tr,
                           float strength) {
    vec3 result = raw_dir;
    // Gravitropism: positive value biases UP (against gravity). Some species
    // (weeping willow) want negative — that's just gravitropism < 0.
    result.z += tr.gravitropism * strength;
    // Phototropism: bias toward light_dir.
    result   = result + (tr.light_dir * (tr.phototropism * strength));
    return normalized(result);
}

} // namespace treegen
