// The crown density volume (crown-shadows 5.1, K8): leaf area splatted into a
// grid from triangle GEOMETRIC normals (K16) and the leaf atlas's alpha
// coverage, and the analytic transmittance march through it. S-P0a lands the
// library (both candidate grids, the march, the low-pass) for the format rule
// and for crown-lighting's normals mode; the GLB extension and everything
// engine-side are S-P1.
//
// Extinction along a unit direction w through a cell (per metre):
//   scalar:          sigma = G * rho,                rho = opaque leaf area / cell volume
//   axis_projected:  sigma = dot(|w|, rho_xyz),      rho_k = sum(|n_k| * opaque area) / cell volume
// times `clumping` (1 unless the format rule fits one).
#pragma once

#include "crown_probe.hpp"
#include "vec3.hpp"

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace treegen::crown {

enum class ExtinctionKind { scalar, axis_projected };

struct ExtinctionModel {
    ExtinctionKind kind     = ExtinctionKind::axis_projected;
    float          g        = 0.5f;   // scalar model's projection factor
    float          clumping = 1.0f;
};

class CrownVolume {
public:
    // `cells_longest`: cells on the longest axis of the leaf AABB; every axis
    // gets at least 2 cells (a degenerate extent is padded to one cell).
    // Cell coverage: the per-triangle alpha coverage the probe computed.
    static CrownVolume build(const Probe& probe, int cells_longest);

    float cell_size() const { return m_grid.cell; }
    std::array<int, 3> dims() const { return {m_grid.nx, m_grid.ny, m_grid.nz}; }
    vec3 origin() const { return m_grid.origin; }
    const ExtinctionModel& model() const { return m_model; }
    void set_model(const ExtinctionModel& m) { m_model = m; }

    // Unprojected opaque leaf-area density (m2 / m3) at p; 0 outside.
    float density(vec3 p) const;
    // Integral of extinction along origin + t * dir (unit dir), t >= 0, through
    // the whole grid, under `model`. lowpass_sigma_m > 0 marches a copy of the
    // grid convolved with a separable Gaussian of that sigma in METRES.
    float optical_depth(vec3 p, vec3 dir, float lowpass_sigma_m, const ExtinctionModel& model) const;
    float optical_depth(vec3 p, vec3 dir, float lowpass_sigma_m) const { return optical_depth(p, dir, lowpass_sigma_m, m_model); }
    float transmittance(vec3 p, vec3 dir) const;

    // Both channels integrated along a line, at sigma 0: x = integral of rho,
    // y = integral of dot(|dir|, rho_xyz). One DDA for both models.
    std::array<float, 2> integrals(vec3 p, vec3 dir) const;

    // Testing / evidence: a volume from explicit cells (4 floats per cell:
    // rho, rho_x, rho_y, rho_z; x fastest).
    static CrownVolume from_cells(vec3 origin, float cell, int nx, int ny, int nz, std::vector<float> cells);

    double total_area_m2() const;   // sum of rho * cell volume

private:
    struct Grid {
        vec3               origin;
        float              cell = 0.0f;
        int                nx = 0, ny = 0, nz = 0;
        std::vector<float> c;   // 4 per cell
    };
    template <class Fn> static void march(const Grid& g, vec3 p, vec3 dir, Fn&& fn);
    const Grid& lowpassed(float sigma_m) const;

    struct LowpassCache {
        std::mutex                                 mutex;
        std::map<float, std::unique_ptr<Grid>>     grids;
    };

    Grid                          m_grid;
    ExtinctionModel               m_model;
    std::shared_ptr<LowpassCache> m_cache = std::make_shared<LowpassCache>();
};

}  // namespace treegen::crown
