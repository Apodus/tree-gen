// NOTE: /fp:precise like the rest of TreegenCore.
#include "crown_volume.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace treegen::crown {

namespace {

constexpr int k_channels = 4;  // rho, rho_x, rho_y, rho_z

float comp(vec3 v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }

// Single projected-density sum shared by integrals() and optical_depth() (D1).
inline double projected_density(const float* c, double ax, double ay, double az) {
    return ax * c[1] + ay * c[2] + az * c[3];
}

}  // namespace

CrownVolume CrownVolume::build(const Probe& probe, int cells_longest) {
    if (cells_longest < 2) throw std::runtime_error("crown volume: need at least 2 cells on the longest axis");
    const CrownMesh& m = probe.mesh();
    vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const Triangle& t : m.triangles) {
        if (t.site < 0) continue;
        for (const vec3& p : t.p) {
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
        }
    }
    if (m.leaf_triangles == 0) throw std::runtime_error("crown volume: the mesh has no leaf triangles");

    const vec3 ext = hi - lo;
    const float longest = std::max({ext.x, ext.y, ext.z});
    if (!(longest > 0.0f)) throw std::runtime_error("crown volume: leaf AABB is degenerate");
    Grid g;
    g.cell = longest / static_cast<float>(cells_longest);
    int n[3];
    for (int a = 0; a < 3; ++a) {
        // A degenerate extent is padded to one cell; every axis gets at least 2.
        const float e = std::max(comp(ext, a), g.cell);
        n[a] = std::max(2, static_cast<int>(std::ceil(e / g.cell - 1e-4f)));
    }
    g.nx = n[0]; g.ny = n[1]; g.nz = n[2];
    const vec3 centre = (lo + hi) * 0.5f;
    g.origin = centre - vec3{g.nx * g.cell, g.ny * g.cell, g.nz * g.cell} * 0.5f;
    g.c.assign(size_t(g.nx) * g.ny * g.nz * k_channels, 0.0f);

    const double inv_vol = 1.0 / (double(g.cell) * g.cell * g.cell);
    std::vector<double> acc(g.c.size(), 0.0);
    for (uint32_t ti = 0; ti < m.triangles.size(); ++ti) {
        const Triangle& t = m.triangles[ti];
        if (t.site < 0) continue;
        const vec3 a2 = cross(t.p[1] - t.p[0], t.p[2] - t.p[0]);
        const float area = 0.5f * length(a2);
        if (!(area > 0.0f)) continue;
        const vec3 nrm = a2 / (2.0f * area);   // geometric normal (K16)
        const double opaque = double(area) * probe.triangle_coverage(ti);
        const float max_edge = std::max({length(t.p[1] - t.p[0]), length(t.p[2] - t.p[1]), length(t.p[0] - t.p[2])});
        // Uniform subdivision into s*s sub-triangles of equal area, each
        // deposited at its centroid; sub-triangle edges <= a quarter cell.
        const int s = std::max(1, static_cast<int>(std::ceil(max_edge / (0.25f * g.cell))));
        const double w = opaque / (double(s) * s) * inv_vol;
        const double wx = w * std::abs(nrm.x), wy = w * std::abs(nrm.y), wz = w * std::abs(nrm.z);
        auto deposit = [&](float b1, float b2) {
            const vec3 p = t.p[0] + (t.p[1] - t.p[0]) * b1 + (t.p[2] - t.p[0]) * b2;
            const int ix = std::clamp(static_cast<int>(std::floor((p.x - g.origin.x) / g.cell)), 0, g.nx - 1);
            const int iy = std::clamp(static_cast<int>(std::floor((p.y - g.origin.y) / g.cell)), 0, g.ny - 1);
            const int iz = std::clamp(static_cast<int>(std::floor((p.z - g.origin.z) / g.cell)), 0, g.nz - 1);
            const size_t c = ((size_t(iz) * g.ny + iy) * g.nx + ix) * k_channels;
            acc[c] += w; acc[c + 1] += wx; acc[c + 2] += wy; acc[c + 3] += wz;
        };
        const float inv_s = 1.0f / static_cast<float>(s);
        for (int i = 0; i < s; ++i)
            for (int j = 0; j < s - i; ++j) {
                deposit((i + 1.0f / 3.0f) * inv_s, (j + 1.0f / 3.0f) * inv_s);
                if (i + j <= s - 2) deposit((i + 2.0f / 3.0f) * inv_s, (j + 2.0f / 3.0f) * inv_s);
            }
    }
    for (size_t i = 0; i < acc.size(); ++i) g.c[i] = static_cast<float>(acc[i]);

    CrownVolume v;
    v.m_grid = std::move(g);
    return v;
}

CrownVolume CrownVolume::from_cells(vec3 origin, float cell, int nx, int ny, int nz, std::vector<float> cells) {
    if (cells.size() != size_t(nx) * ny * nz * k_channels) throw std::runtime_error("crown volume: cell count");
    CrownVolume v;
    v.m_grid.origin = origin;
    v.m_grid.cell = cell;
    v.m_grid.nx = nx; v.m_grid.ny = ny; v.m_grid.nz = nz;
    v.m_grid.c = std::move(cells);
    return v;
}

double CrownVolume::total_area_m2() const {
    double s = 0.0;
    for (size_t i = 0; i < m_grid.c.size(); i += k_channels) s += m_grid.c[i];
    return s * double(m_grid.cell) * m_grid.cell * m_grid.cell;
}

float CrownVolume::density(vec3 p) const {
    const Grid& g = m_grid;
    const int ix = static_cast<int>(std::floor((p.x - g.origin.x) / g.cell));
    const int iy = static_cast<int>(std::floor((p.y - g.origin.y) / g.cell));
    const int iz = static_cast<int>(std::floor((p.z - g.origin.z) / g.cell));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= g.nx || iy >= g.ny || iz >= g.nz) return 0.0f;
    return g.c[((size_t(iz) * g.ny + iy) * g.nx + ix) * k_channels];
}

// Exact traversal (Amanatides-Woo) of the cells the ray p + t * dir, t >= 0,
// crosses; fn(cell_index, segment_length) per cell.
template <class Fn>
void CrownVolume::march(const Grid& g, vec3 p, vec3 dir, Fn&& fn) {
    const double o[3] = {p.x, p.y, p.z}, d[3] = {dir.x, dir.y, dir.z};
    const double lo[3] = {g.origin.x, g.origin.y, g.origin.z};
    const int n[3] = {g.nx, g.ny, g.nz};
    double t0 = 0.0, t1 = std::numeric_limits<double>::infinity();
    for (int a = 0; a < 3; ++a) {
        const double hi = lo[a] + n[a] * double(g.cell);
        if (d[a] == 0.0) {
            if (o[a] < lo[a] || o[a] >= hi) return;
            continue;
        }
        double ta = (lo[a] - o[a]) / d[a], tb = (hi - o[a]) / d[a];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
    }
    if (!(t0 < t1)) return;
    int idx[3], step[3];
    double t_next[3], t_delta[3];
    const double tm = 0.5 * (t0 + std::min(t1, t0 + 1e-6 * g.cell));
    for (int a = 0; a < 3; ++a) {
        const double x = (o[a] + d[a] * tm - lo[a]) / g.cell;
        idx[a] = std::clamp(static_cast<int>(std::floor(x)), 0, n[a] - 1);
        if (d[a] > 0.0) {
            step[a] = 1;
            t_next[a] = (lo[a] + (idx[a] + 1) * double(g.cell) - o[a]) / d[a];
            t_delta[a] = g.cell / d[a];
        } else if (d[a] < 0.0) {
            step[a] = -1;
            t_next[a] = (lo[a] + idx[a] * double(g.cell) - o[a]) / d[a];
            t_delta[a] = -g.cell / d[a];
        } else {
            step[a] = 0;
            t_next[a] = std::numeric_limits<double>::infinity();
            t_delta[a] = std::numeric_limits<double>::infinity();
        }
    }
    double t = t0;
    for (;;) {
        const int a = (t_next[0] <= t_next[1] && t_next[0] <= t_next[2]) ? 0 : (t_next[1] <= t_next[2] ? 1 : 2);
        const double te = std::min(t_next[a], t1);
        if (te > t) fn((size_t(idx[2]) * g.ny + idx[1]) * g.nx + idx[0], te - t);
        if (t_next[a] >= t1) return;
        t = t_next[a];
        idx[a] += step[a];
        if (idx[a] < 0 || idx[a] >= n[a]) return;
        t_next[a] += t_delta[a];
    }
}

std::array<float, 2> CrownVolume::integrals(vec3 p, vec3 dir) const {
    const double ax = std::abs(dir.x), ay = std::abs(dir.y), az = std::abs(dir.z);
    double rho = 0.0, proj = 0.0;
    march(m_grid, p, dir, [&](size_t cell, double len) {
        const float* c = &m_grid.c[cell * k_channels];
        rho  += c[0] * len;
        proj += projected_density(c, ax, ay, az) * len;
    });
    return {static_cast<float>(rho), static_cast<float>(proj)};
}

const CrownVolume::Grid& CrownVolume::lowpassed(float sigma_m) const {
    std::lock_guard<std::mutex> lock(m_cache->mutex);
    auto& slot = m_cache->grids[sigma_m];
    if (slot) return *slot;

    const Grid& g = m_grid;
    const double sc = double(sigma_m) / g.cell;   // sigma in cells
    const int r = std::max(1, static_cast<int>(std::ceil(3.0 * sc)));
    std::vector<double> k(size_t(2 * r + 1));
    double ksum = 0.0;
    for (int i = -r; i <= r; ++i) ksum += (k[size_t(i + r)] = std::exp(-0.5 * (i * i) / (sc * sc)));
    for (double& v : k) v /= ksum;

    auto out = std::make_unique<Grid>();
    out->cell = g.cell;
    out->nx = g.nx + 2 * r; out->ny = g.ny + 2 * r; out->nz = g.nz + 2 * r;
    out->origin = g.origin - vec3{r * g.cell, r * g.cell, r * g.cell};
    const size_t N = size_t(out->nx) * out->ny * out->nz;
    std::vector<double> a(N * k_channels, 0.0), b(N * k_channels, 0.0);
    auto at = [&](int x, int y, int z) { return ((size_t(z) * out->ny + y) * out->nx + x) * k_channels; };
    for (int z = 0; z < g.nz; ++z)
        for (int y = 0; y < g.ny; ++y)
            for (int x = 0; x < g.nx; ++x)
                for (int ch = 0; ch < k_channels; ++ch)
                    a[at(x + r, y + r, z + r) + ch] = g.c[((size_t(z) * g.ny + y) * g.nx + x) * k_channels + ch];
    // Separable passes x, y, z (a -> b -> a -> b).
    const int dims[3] = {out->nx, out->ny, out->nz};
    std::vector<double>* src = &a;
    std::vector<double>* dst = &b;
    for (int axis = 0; axis < 3; ++axis) {
        std::fill(dst->begin(), dst->end(), 0.0);
        for (int z = 0; z < out->nz; ++z)
            for (int y = 0; y < out->ny; ++y)
                for (int x = 0; x < out->nx; ++x) {
                    const int pos[3] = {x, y, z};
                    for (int i = -r; i <= r; ++i) {
                        int q[3] = {x, y, z};
                        q[axis] = pos[axis] + i;
                        if (q[axis] < 0 || q[axis] >= dims[axis]) continue;
                        const double w = k[size_t(i + r)];
                        const size_t s = at(q[0], q[1], q[2]), d = at(x, y, z);
                        for (int ch = 0; ch < k_channels; ++ch) (*dst)[d + ch] += w * (*src)[s + ch];
                    }
                }
        std::swap(src, dst);
    }
    out->c.resize(N * k_channels);
    for (size_t i = 0; i < out->c.size(); ++i) out->c[i] = static_cast<float>((*src)[i]);
    slot = std::move(out);
    return *slot;
}

float CrownVolume::optical_depth(vec3 p, vec3 dir, float lowpass_sigma_m, const ExtinctionModel& model) const {
    const Grid& g = lowpass_sigma_m > 0.0f ? lowpassed(lowpass_sigma_m) : m_grid;
    const double ax = std::abs(dir.x), ay = std::abs(dir.y), az = std::abs(dir.z);
    double tau = 0.0;
    march(g, p, dir, [&](size_t cell, double len) {
        const float* c = &g.c[cell * k_channels];
        tau += (model.kind == ExtinctionKind::scalar ? model.g * c[0] : projected_density(c, ax, ay, az)) * len;
    });
    return static_cast<float>(tau * model.clumping);
}

float CrownVolume::transmittance(vec3 p, vec3 dir) const {
    return std::exp(-optical_depth(p, dir, 0.0f));
}

}  // namespace treegen::crown
