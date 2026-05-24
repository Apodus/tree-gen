// Space colonization (Runions/Lane/Prusinkiewicz 2007). Attractor-driven
// growth: Halton-sample candidate points inside the envelope AABB → reject
// outside the envelope shape → iteratively grow branch tips toward nearby
// attractors → kill consumed attractors. Tropisms blended per-tip via
// apply_tropisms(). Radius solved via radius_solver after growth.
//
// Determinism: stable_sort of attractor-tip pairs by (distance², halton_idx)
// makes neighbour ties resolution-independent of attractor enumeration order.
// Combined with /fp:precise + pcg32 + Halton(2,3,5), output is byte-identical
// across runs given identical (descriptor, seed_effective).
#pragma once

#include <cstdint>

namespace treegen {

struct TreeDescriptor;
struct TreeSkeleton;

TreeSkeleton grow_skeleton(const TreeDescriptor& desc, uint64_t seed_effective);

} // namespace treegen
