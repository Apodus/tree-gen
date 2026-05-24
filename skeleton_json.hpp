// Pretty-printed deterministic JSON dump of a TreeSkeleton. Node-index order;
// %.9g float precision (1-9 sig digits; round-trip float through double w/o
// loss). NO hash containers anywhere — the dump is byte-identical across runs
// given identical input, which is the determinism gate for C3 P2.
#pragma once

#include <string>

namespace treegen {

struct TreeSkeleton;

std::string dump_skeleton_json(const TreeSkeleton& skel);

} // namespace treegen
