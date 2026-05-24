// Envelope evaluators — the 5 shipped crown-volume shapes.
//
// Coordinate convention: +Z up. Tree base at origin (0, 0, 0). Trunk
// extends to (0, 0, height_m). The envelope is the crown volume — for
// centred shapes (OblateSpheroid, Weeping, Fan) it is centred at
// (0, 0, height_m/2); for ground-rooted shapes (Conical, Fastigiate) it
// sits from base to apex.
//
// `top_height_ratio` is a vertical stretch factor on the upper half of the
// crown (anisotropic for species that taper asymmetrically above the centre);
// `height_ratio` scales the crown's total Z extent relative to height_m.
#pragma once

#include "tree_descriptor.hpp"
#include "vec3.hpp"

namespace treegen {

struct Aabb {
    vec3 min;
    vec3 max;
};

// True if `candidate` (world-space) lies inside the crown volume given
// `env` + `tree_height_m`. Boundary points count as inside (closed volume).
bool envelope_contains(EnvelopeShape shape,
                       const TreeDescriptor::Envelope& env,
                       vec3 candidate,
                       float tree_height_m);

// Smallest axis-aligned box containing the envelope. envelope_contains(p) →
// p is within envelope_aabb (strict containment, modulo float rounding).
Aabb envelope_aabb(EnvelopeShape shape,
                   const TreeDescriptor::Envelope& env,
                   float tree_height_m);

} // namespace treegen
