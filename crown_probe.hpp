// The crown probe core (crown-shadows S-P0a, clause K17): one CPU measurement
// core for a tree crown, read from GLB bytes. Exact alpha-tested rays against
// the leaf cards and the opaque bark, triangle GEOMETRIC normals only (the
// vertex NORMAL is never read: K16). The oracle and sunlit modes
// (crown_oracle.hpp) and crown-lighting's normals mode stand on it.
//
// Units: metres, the tree's object space (Z up, trunk base near the origin).
#pragma once

#include "vec3.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace treegen::crown {

// The leaf atlas's alpha channel, decoded from the PNG the GLB carries.
// Row 0 is the image's top row (glTF UV v = 0). Sampled nearest-texel, mip 0;
// a texel passes the material's alphaMode MASK test when alpha/255 >= cutoff.
struct AlphaAtlas {
    int                  width  = 0;
    int                  height = 0;
    std::vector<uint8_t> alpha;
    float                cutoff = 0.5f;

    bool empty() const { return alpha.empty(); }
    uint8_t texel(int x, int y) const { return alpha[size_t(y) * size_t(width) + size_t(x)]; }
    bool passes_texel(int x, int y) const { return float(texel(x, y)) >= cutoff * 255.0f; }
    // Nearest texel at (u, v), clamped to the image (the engine's leaf sampler
    // clamps; leaf UVs stay inside their cell).
    bool passes(float u, float v) const;
};

struct Triangle {
    std::array<vec3, 3>                  p;
    std::array<std::array<float, 2>, 3>  uv{};
    int32_t                              site = -1;   // -1: bark (opaque); else leaf site id
};

// One leaf site (K17): consecutive cards whose vertex centroids agree within
// 1 mm; a card is a connected component of leaf triangles under shared
// vertex indices.
struct Site {
    int32_t               id = 0;
    vec3                  centroid;    // mean of the member cards' vertex centroids
    vec3                  normal;      // area-weighted geometric normal, each triangle signed toward the first
    int                   cards = 0;
    std::vector<uint32_t> triangles;   // indices into CrownMesh::triangles
};

struct SiteRule {
    std::vector<Site>    sites;        // ids in index order
    std::vector<int32_t> triangle_site;  // per leaf triangle (input order)
};

// K17's site rule on a leaf primitive (positions xyz-packed, triangle list).
SiteRule derive_sites(std::span<const float> positions, std::span<const uint32_t> indices);

struct CrownMesh {
    std::vector<Triangle> triangles;   // bark first, then leaf triangles in index order
    std::vector<Site>     sites;
    AlphaAtlas            atlas;       // empty: leaf cards are opaque
    uint32_t              bark_triangles = 0;
    uint32_t              leaf_triangles = 0;

    // Appends a leaf primitive (sites derived by K17's rule; ids continue).
    void add_leaves(std::span<const float> positions, std::span<const float> uvs, std::span<const uint32_t> indices);
    void add_bark(std::span<const float> positions, std::span<const uint32_t> indices);
};

// --- Loader (crown_probe_glb.cpp) -------------------------------------------

struct LoadedCrown {
    CrownMesh   mesh;
    int         lod = 0;
    int         meshes_in_glb = 0;
    float       alpha_cutoff = 0.5f;
};

// Reads mesh `lod` of a treegen GLB: the leaf primitive is the one whose
// material is alphaMode MASK with _RYNX_LEAF_TRANSLUCENCY; every other
// primitive of the mesh is bark (opaque). The leaf atlas is the leaf
// material's baseColorTexture image, decoded with stb_image. Throws
// std::runtime_error with the reason on any malformed input. The scenario
// path and --glb read through this one function.
LoadedCrown load_crown_from_glb(std::span<const uint8_t> glb, int lod);

// --- Ray queries --------------------------------------------------------------

// The two oracle populations (commander ruling R-CR1). Every occlusion query
// names one; there is no default.
// - leaves_and_bark: the whole crown, as every shadow path sees it (transmittance
//   profile, regions, stop rule, sunlit modes).
// - leaves_only: what a leaf-medium volume represents (bark stays real geometry),
//   so the population the format rule scores the volume against.
enum class Occluders { leaves_and_bark, leaves_only };
const char* occluders_name(Occluders o);

// Single population membership predicate (D5): leaves always block; bark blocks
// only in leaves_and_bark. Used by LineHits::occluded and every Probe query.
inline bool blocks_category(bool is_leaf, Occluders o) {
    return is_leaf || o == Occluders::leaves_and_bark;
}
inline bool blocks(const Triangle& t, Occluders o) {
    return blocks_category(t.site >= 0, o);
}

struct LineHits {
    int  leaf_geometric = 0;   // leaf triangles the ray intersects, alpha ignored
    int  leaf_opaque    = 0;   // ... that pass the alpha test there
    int  bark           = 0;
    bool occluded(Occluders o) const {
        return (leaf_opaque > 0 && blocks_category(true, o)) ||
               (bark > 0 && blocks_category(false, o));
    }
};

class Probe {
public:
    explicit Probe(CrownMesh mesh);

    const CrownMesh& mesh() const { return m_mesh; }

    // Any opaque hit of population `occ` on the ray origin + t * dir,
    // t > 1e-4 m (to infinity), ignoring the triangles of site `skip_site` (-1: none).
    bool occluded(vec3 origin, vec3 dir, int32_t skip_site, Occluders occ) const;
    // Any opaque hit of population `occ` strictly between origin and target.
    bool occluded_segment(vec3 origin, vec3 target, int32_t skip_site, Occluders occ) const;
    // Every hit on origin + t * dir, t in (1e-4 m, t_max); the caller picks the population.
    LineHits line_hits(vec3 origin, vec3 dir, float t_max) const;

    // Alpha coverage of one triangle (area-weighted share of its UV footprint's
    // texels that pass; nearest texel, mip 0); 1 for bark or with no atlas.
    float triangle_coverage(uint32_t triangle) const { return m_coverage[triangle]; }

    vec3 bounds_min() const { return m_min; }
    vec3 bounds_max() const { return m_max; }

private:
    struct Node {
        vec3     lo, hi;
        uint32_t first = 0;   // leaf: first index into m_order; inner: right child
        uint32_t count = 0;   // leaf: triangle count; inner: 0
    };
    template <class Visit> void traverse(vec3 origin, vec3 dir, float t_max, Visit&& visit) const;
    bool hit_opaque(uint32_t tri, float b1, float b2) const;

    CrownMesh             m_mesh;
    std::vector<float>    m_coverage;
    std::vector<Node>     m_nodes;
    std::vector<uint32_t> m_order;
    vec3                  m_min, m_max;
};

// Alpha coverage of one UV triangle: exact over the texels whose centres lie
// inside it (nearest texel, mip 0). 1 when the atlas is empty.
float uv_triangle_coverage(const AlphaAtlas& atlas, const std::array<std::array<float, 2>, 3>& uv);

// Runs fn(i) for i in [0, n) on worker threads. Each i writes only its own
// output slot, so results never depend on scheduling.
void parallel_for(size_t n, const std::function<void(size_t)>& fn);

}  // namespace treegen::crown
