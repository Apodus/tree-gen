// C4 P4 + C5 P3 — emit all 4 LODs from one TreeSkeleton.
//
// Per LOD i in {0,1,2}: allocate per-order radial via face_budget, build a
// bark mesh + fork blend + wind-weight bake. LOD 3 = 4-tri vertical billboard
// stub (impostor placeholder; C9 bakes the actual impostor texture).
//
// C5 P3: the leaf-aware overload also runs allocate_leaves_all_lods once and,
// per LOD, materialises per-LOD leaf positions/normals/uvs/indices/wind/material
// in the same `LodOutput`. main.cpp packages them as a 2nd primitive (leaves)
// alongside primitive[0] (bark) inside each MeshData.
//
// Output is a vector of 4 `LodOutput`s; caller (main.cpp) packs them into a
// single multi-mesh GLB via glb_writer::write_glb_multi_mesh, one MeshData per
// LOD with the matching `_RYNX_LOD` extension payload.
#pragma once

#include "branch_mesh.hpp"
#include "face_budget.hpp"
#include "leaf_budget.hpp"
#include "leaf_geometry.hpp"
#include "skeleton.hpp"

#include <cstdint>
#include <vector>

namespace treegen {

struct LodOutput {
    cpu_mesh_out          mesh;                   // bark: positions/normals/uvs populated
    std::vector<uint32_t> indices_u32;            // bark: triangle list
    std::vector<uint8_t>  wind_weights_packed;    // bark: 4 bytes/vert; empty for L3 (all zeros baked anyway)
    std::vector<float>    bark_tangents;           // C1 P6 — xyzw packed, 4 floats/vert
    int                   lod_index           = 0;
    float                 lod_max_distance_m  = 0.0f;

    // C5 P3 — per-LOD leaf primitive payload. `has_leaves` gates main.cpp's
    // primitive[1] emission; the leaf-blind 5-arg overload leaves them empty.
    std::vector<float>    leaf_positions;
    std::vector<float>    leaf_normals;
    std::vector<float>    leaf_uvs;
    std::vector<uint32_t> leaf_indices_u32;
    std::vector<uint8_t>  leaf_wind_weights_packed;
    std::vector<int32_t>  leaf_material_slots;
    std::vector<float>    leaf_tangents;            // C1 P6 — xyzw packed, 4 floats/vert
    bool                  has_leaves = false;
};

// Emit 3 LODs (L0-L2) from one skeleton. base_opts supplies bark_repeat_m, seam
// offset, fork zone factor — radial counts are overridden per LOD by the
// budget allocator. `tree_height_m` propagates into the per-LOD wind bake.
std::vector<LodOutput> emit_all_lods(const TreeSkeleton&     skel,
                                     const BarkMeshOptions&  base_opts,
                                     const LodBudget&        budget,
                                     float                   tree_height_m,
                                     uint64_t                seed_effective);

// C5 P3 — leaf-aware overload. Same bark pipeline as above; additionally runs
// allocate_leaves_all_lods over `all_leaf_sites` and materialises per-LOD leaf
// primitive payloads (positions/normals/uvs/indices/wind/material) inside each
// LodOutput. L3 stays leafless (matches LeafBudget.l3_tris default = 0 plus the
// billboard-stub L3 emit).
//
// Distinct arity (8 args) — the leaf-blind 5-arg overload is preserved
// byte-identical for callers that don't need leaves.
std::vector<LodOutput> emit_all_lods(const TreeSkeleton&            skel,
                                     const std::vector<LeafSite>&   all_leaf_sites,
                                     const BarkMeshOptions&         bark_opts,
                                     const LodBudget&               bark_budget,
                                     const LeafBudget&              leaf_budget,
                                     const LeafMeshOptions&         leaf_geom_opts,
                                     float                          tree_height_m,
                                     uint64_t                       seed_effective);

}  // namespace treegen
