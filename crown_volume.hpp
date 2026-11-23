// The crown density volume (crown-shadows S-P1, K8, K16): symmetric 2nd-moment
// leaf-area tensor F2_tensor (M_xx, M_yy, M_zz, M_xy, M_xz, M_yz) splatted
// into a 3D grid from triangle GEOMETRIC normals (K16) and the leaf atlas's
// mip-0 nearest-texel alpha coverage, plus Option 2A hardware-trilinear
// quantization (RGBA8_UNORM + RGBA8_SNORM) and K4/K8 march helpers.
//
// Extinction along a unit direction w through a cell (per metre):
//   scalar:  sigma = clumping * G * rho,  rho = tr(M) = M_xx + M_yy + M_zz
//   tensor:  sigma = clumping * sqrt(3/4) * sqrt(max(0, rho * w^T M w))
#pragma once

#include "crown_probe.hpp"
#include "vec3.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace treegen::crown {

// Physical twig-bouquet clumping index Omega:
// - Broadleaf 3-card sprig bouquets (ClusterCard) straddling real twigs exhibit
//   shoot-level clumping ~0.887 (matches temperate broadleaf shoot clumping and
//   balances GPU hardware-trilinear vs DDA extinction across multi-normal bouquets).
// - Conifer BranchStrip shoots distribute uniformly along branches (~1.05).
inline constexpr float k_default_crown_clumping   = 0.887f;
inline constexpr float k_strip_crown_clumping     = 1.05f;
inline constexpr float k_default_target_cell_m    = 0.20f;
inline constexpr int   k_default_max_l0_dim       = 64;
inline constexpr int   k_max_volume_dim           = 128;
inline constexpr int   k_default_trilinear_steps  = 48;
// K4 receiver exclusion: half-cell exit from receiver's own [-0.5*cell, +0.5*cell]
// voxel so the 1st midpoint sample lands at 1.0 * cell_m (centered on 1st neighbor).
inline constexpr float k_receiver_exclude_cells   = 0.5f;

enum class ExtinctionKind { scalar, tensor };

struct ExtinctionModel {
    ExtinctionKind kind     = ExtinctionKind::tensor;
    float          g        = 0.5f;   // scalar model's projection factor
    float          clumping = 1.0f;
};

// Option 2A quantized unblurred (sigma = 0) F2_tensor volume ready for
// _RYNX_CROWN_VOLUME serialization and GPU 3D atlas upload:
//   tex0 (RGBA8_UNORM): { M_xx, M_yy, M_zz, 0 } / rho_cap
//   tex1 (RGBA8_SNORM): { 2*M_xy, 2*M_xz, 2*M_yz, 0 } / rho_cap
// Empty voxels are {0, 0, 0, 0} in both textures (SNORM byte 0 == exact 0.0f).
struct PackedCrownVolume {
    std::array<int, 3>   dims{0, 0, 0};     // {nx, ny, nz}
    vec3                 box_min{0, 0, 0};
    vec3                 box_max{0, 0, 0};
    float                cell_m   = 0.0f;
    float                rho_cap  = 0.0f;
    float                clumping = k_default_crown_clumping;
    std::vector<uint8_t> tex0;              // nx * ny * nz * 4, x-fastest
    std::vector<int8_t>  tex1;              // nx * ny * nz * 4, x-fastest

    // Diagnostics (not serialized into GLB):
    uint32_t             occupied_voxels = 0;
    uint32_t             clipped_voxels  = 0;
    float                max_rho         = 0.0f;

    bool empty() const { return tex0.empty() || tex1.empty(); }

    // Hardware-trilinear reference decode at normalized brick coordinates uvw in [0, 1]^3
    // with 1-texel zero border outside [0, dims). Matches GPU GLSL shader decode exactly.
    float sample_trilinear_sigma(vec3 uvw, vec3 dir) const;
    // Discrete per-voxel decode at integer index (ix, iy, iz).
    float sample_voxel_sigma(int ix, int iy, int iz, vec3 dir) const;

    // Fixed-step midpoint march with hardware-trilinear sampling (GPU reference).
    // When `exclude_receiver_cell` is true (K4), excludes t < cell_m from p.
    float optical_depth_trilinear(vec3 p, vec3 dir, int max_steps = k_default_trilinear_steps,
                                  bool exclude_receiver_cell = false) const;
    float transmittance_trilinear(vec3 p, vec3 dir, int max_steps = k_default_trilinear_steps,
                                  bool exclude_receiver_cell = false) const;

    // Exact Amanatides-Woo DDA march over the quantized Option 2A cells.
    float optical_depth_dda(vec3 p, vec3 dir, bool exclude_receiver_cell = false) const;
    float transmittance_dda(vec3 p, vec3 dir, bool exclude_receiver_cell = false) const;
};

class CrownVolume {
public:
    // `cells_longest`: cells on the longest axis of the leaf AABB; every axis
    // gets at least 2 cells (a degenerate extent is padded to one cell).
    static CrownVolume build(const Probe& probe, int cells_longest);

    // Shipped L0-AABB sizing + union-of-LODs zero-padded outer shell (S-P1 §2.3.1):
    //   cell_m = max(target_cell_m, longest(L0_ext) / max_l0_dim)
    //   dims   = L0_dims + pad_lo + pad_hi (asserted <= k_max_volume_dim)
    // Outer padded shell voxels are strictly zero.
    static CrownVolume build_shipped(const Probe& l0_probe, vec3 union_min, vec3 union_max,
                                     float target_cell_m = k_default_target_cell_m,
                                     int max_l0_dim = k_default_max_l0_dim);
    static CrownVolume build_shipped(const Probe& l0_probe,
                                     float target_cell_m = k_default_target_cell_m,
                                     int max_l0_dim = k_default_max_l0_dim);

    float cell_size() const { return m_grid.cell; }
    std::array<int, 3> dims() const { return {m_grid.nx, m_grid.ny, m_grid.nz}; }
    std::array<int, 3> l0_dims() const { return m_l0_dims; }
    std::array<int, 3> pad_lo() const { return m_pad_lo; }
    std::array<int, 3> pad_hi() const { return m_pad_hi; }
    vec3 origin() const { return m_grid.origin; }
    vec3 box_min() const { return m_grid.origin; }
    vec3 box_max() const {
        return m_grid.origin + vec3{float(m_grid.nx) * m_grid.cell,
                                    float(m_grid.ny) * m_grid.cell,
                                    float(m_grid.nz) * m_grid.cell};
    }
    const ExtinctionModel& model() const { return m_model; }
    void set_model(const ExtinctionModel& m) { m_model = m; }

    // Unprojected opaque leaf-area density rho = tr(M) (m2 / m3) at p; 0 outside.
    float density(vec3 p) const;
    // Integral of extinction along p + t * dir (unit dir), t >= (exclude ? cell : 0),
    // through the whole grid, under `model`. lowpass_sigma_m > 0 marches a lazily-cached
    // expanded 6-channel FP32 copy convolved with a separable Gaussian of sigma_m (K8).
    float optical_depth(vec3 p, vec3 dir, float lowpass_sigma_m, const ExtinctionModel& model,
                        bool exclude_receiver_cell = false) const;
    float optical_depth(vec3 p, vec3 dir, float lowpass_sigma_m,
                        bool exclude_receiver_cell = false) const {
        return optical_depth(p, dir, lowpass_sigma_m, m_model, exclude_receiver_cell);
    }
    float transmittance(vec3 p, vec3 dir, bool exclude_receiver_cell = false) const;

    // Both channels integrated along a line, at sigma 0:
    //   [0] = integral of rho = tr(M)
    //   [1] = integral of unclumped tensor extinction sqrt(3/4 * rho * w^T M w)
    std::array<float, 2> integrals(vec3 p, vec3 dir) const;

    // Option 2A quantization of the unblurred (sigma = 0) FP32 grid.
    PackedCrownVolume pack_option2a(float clumping = k_default_crown_clumping) const;

    // Testing / evidence: a volume from explicit cells (6 floats per cell:
    // M_xx, M_yy, M_zz, M_xy, M_xz, M_yz; x fastest).
    static CrownVolume from_cells(vec3 origin, float cell, int nx, int ny, int nz, std::vector<float> cells);

    double total_area_m2() const;   // sum of tr(M) * cell volume
    const std::vector<float>& raw_cells() const { return m_grid.c; }

private:
    struct Grid {
        vec3               origin;
        float              cell = 0.0f;
        int                nx = 0, ny = 0, nz = 0;
        std::vector<float> c;   // 6 per cell: M_xx, M_yy, M_zz, M_xy, M_xz, M_yz
    };
    template <class Fn> static void march(const Grid& g, vec3 p, vec3 dir, double t_min, Fn&& fn);
    const Grid& ensure_lowpass(float sigma_m) const;

    struct LowpassCache {
        std::mutex                             mutex;
        std::map<float, std::unique_ptr<Grid>> grids;
    };

    Grid                          m_grid;
    std::array<int, 3>            m_l0_dims{0, 0, 0};
    std::array<int, 3>            m_pad_lo{0, 0, 0};
    std::array<int, 3>            m_pad_hi{0, 0, 0};
    ExtinctionModel               m_model;
    std::shared_ptr<LowpassCache> m_cache = std::make_shared<LowpassCache>();
};

}  // namespace treegen::crown
