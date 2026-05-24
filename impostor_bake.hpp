// C9 P3 -- CPU-side 8-azimuth impostor atlas bake.
// Projects L0 mesh triangles from 8 equidistant camera directions
// (0, 45, ..., 315 degrees around +Z), rasterizes with barycentric
// UV interpolation, samples the bark/leaf diffuse texture, and
// produces a 2048x1024 RGBA8 atlas (8 cells of 256x1024).
#pragma once

#include <cstdint>
#include <vector>

namespace treegen {

struct ImpostorInput {
    // L0 mesh data (positions xyz packed, uvs xy packed, indices).
    const float*    positions     = nullptr;
    int             vertex_count  = 0;
    const uint32_t* indices       = nullptr;
    int             index_count   = 0;
    const float*    uvs           = nullptr;

    // Bark diffuse texture (decoded RGBA8).
    const uint8_t*  bark_rgba     = nullptr;
    int             bark_w        = 0;
    int             bark_h        = 0;

    // Leaf diffuse texture (decoded RGBA8). Null if no leaves.
    const uint8_t*  leaf_rgba     = nullptr;
    int             leaf_w        = 0;
    int             leaf_h        = 0;

    // Per-vertex material slot (0 = bark, >= 1 = leaf). Null = all bark.
    const int32_t*  material_slots = nullptr;

    // Leaf primitive data (separate mesh, appended after bark verts in L0).
    const float*    leaf_positions    = nullptr;
    int             leaf_vertex_count = 0;
    const uint32_t* leaf_indices      = nullptr;
    int             leaf_index_count  = 0;
    const float*    leaf_uvs          = nullptr;

    // Tree bounding box for framing the orthographic projection.
    float aabb_min[3] = {0, 0, 0};
    float aabb_max[3] = {0, 0, 0};
};

struct ImpostorAtlas {
    std::vector<uint8_t> rgba;   // 2048 * 1024 * 4 bytes
    int width  = 2048;
    int height = 1024;
    int cell_count = 8;          // 8 azimuths
};

// Bake 8-azimuth impostor atlas from L0 mesh + textures.
ImpostorAtlas bake_impostor_atlas(const ImpostorInput& input);

}  // namespace treegen
