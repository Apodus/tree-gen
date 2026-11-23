// NOTE: /fp:precise like the rest of TreegenCore.
#include "crown_volume.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace treegen::crown {

namespace {

constexpr int    k_channels    = 6;  // M_xx, M_yy, M_zz, M_xy, M_xz, M_yz
constexpr double k_sqrt_3_over_4 = 0.86602540378443864676372317075294;
constexpr float  k_sqrt_3_over_4_f = 0.866025403784f;

float comp(vec3 v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }

// Deterministic round-to-nearest ties-to-even (independent of CRT/FENV state).
inline int round_ties_to_even(double x) {
    const double fl = std::floor(x);
    const double frac = x - fl;
    if (frac > 0.5) return static_cast<int>(fl) + 1;
    if (frac < 0.5) return static_cast<int>(fl);
    const int i = static_cast<int>(fl);
    return (i & 1) ? (i + 1) : i;
}

// Single F2_tensor unclumped extinction evaluator shared by integrals() and optical_depth().
inline double tensor_extinction_unclumped(const float* c, double wx, double wy, double wz) {
    const double rho = double(c[0]) + double(c[1]) + double(c[2]);
    if (!(rho > 0.0)) return 0.0;
    const double quad = wx * wx * double(c[0]) + wy * wy * double(c[1]) + wz * wz * double(c[2])
                      + 2.0 * (wx * wy * double(c[3]) + wx * wz * double(c[4]) + wy * wz * double(c[5]));
    if (!(quad > 0.0)) return 0.0;
    return k_sqrt_3_over_4 * std::sqrt(rho * quad);
}

void leaf_aabb(const CrownMesh& m, vec3& lo, vec3& hi) {
    lo = {1e30f, 1e30f, 1e30f};
    hi = {-1e30f, -1e30f, -1e30f};
    for (const Triangle& t : m.triangles) {
        if (t.site < 0) continue;
        for (const vec3& p : t.p) {
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
        }
    }
}

void splat_leaves(const Probe& probe, vec3 origin, float cell, int nx, int ny, int nz,
                  std::array<int, 3> clamp_lo, std::array<int, 3> clamp_hi,
                  std::vector<float>& out_c) {
    const CrownMesh& m = probe.mesh();
    out_c.assign(size_t(nx) * size_t(ny) * size_t(nz) * k_channels, 0.0f);
    const double inv_vol = 1.0 / (double(cell) * double(cell) * double(cell));
    std::vector<double> acc(out_c.size(), 0.0);

    struct Sample { float b1, b2; };
    std::vector<Sample> samples;

    for (uint32_t ti = 0; ti < m.triangles.size(); ++ti) {
        const Triangle& t = m.triangles[ti];
        if (t.site < 0) continue;
        const vec3 a2 = cross(t.p[1] - t.p[0], t.p[2] - t.p[0]);
        const float area = 0.5f * length(a2);
        if (!(area > 0.0f)) continue;
        const vec3 nrm = a2 / (2.0f * area);   // geometric normal (K16)
        const double opaque = double(area) * probe.triangle_coverage(ti);
        if (!(opaque > 0.0)) continue;
        const float max_edge = std::max({length(t.p[1] - t.p[0]),
                                         length(t.p[2] - t.p[1]),
                                         length(t.p[0] - t.p[2])});
        // Uniform subdivision into s*s sub-triangles of equal area, each
        // deposited at its centroid; sub-triangle edges <= a quarter cell.
        // When an alpha atlas is present, weight sub-triangles by the atlas's
        // opaque alpha mask at each sub-triangle centroid (normalized to preserve
        // exact triangle opaque area) so recessed non-square sprigs deposit their
        // extinction strictly where opaque foliage exists, not into empty borders.
        const bool use_alpha = !m.atlas.empty();
        const int s_min = use_alpha ? 8 : 1;
        const int s = std::max(s_min, static_cast<int>(std::ceil(max_edge / (0.25f * cell))));
        const double nx_d = double(nrm.x), ny_d = double(nrm.y), nz_d = double(nrm.z);

        samples.clear();
        samples.reserve(static_cast<size_t>(s) * static_cast<size_t>(s));
        const float inv_s = 1.0f / static_cast<float>(s);
        auto consider = [&](float b1, float b2) {
            if (use_alpha) {
                const float u = t.uv[0][0] + (t.uv[1][0] - t.uv[0][0]) * b1 + (t.uv[2][0] - t.uv[0][0]) * b2;
                const float v = t.uv[0][1] + (t.uv[1][1] - t.uv[0][1]) * b1 + (t.uv[2][1] - t.uv[0][1]) * b2;
                if (!m.atlas.passes(u, v)) return;
            }
            samples.push_back({b1, b2});
        };
        for (int i = 0; i < s; ++i)
            for (int j = 0; j < s - i; ++j) {
                consider((i + 1.0f / 3.0f) * inv_s, (j + 1.0f / 3.0f) * inv_s);
                if (i + j <= s - 2) consider((i + 2.0f / 3.0f) * inv_s, (j + 2.0f / 3.0f) * inv_s);
            }
        // Fallback when a tiny opaque sliver missed all sub-triangle centroids:
        // deposit uniformly so exact trace conservation (< 1e-4) always holds.
        if (samples.empty()) {
            for (int i = 0; i < s; ++i)
                for (int j = 0; j < s - i; ++j) {
                    samples.push_back({(i + 1.0f / 3.0f) * inv_s, (j + 1.0f / 3.0f) * inv_s});
                    if (i + j <= s - 2) samples.push_back({(i + 2.0f / 3.0f) * inv_s, (j + 2.0f / 3.0f) * inv_s});
                }
        }
        const double w = opaque / static_cast<double>(samples.size()) * inv_vol;
        const double mxx = w * nx_d * nx_d;
        const double myy = w * ny_d * ny_d;
        const double mzz = w * nz_d * nz_d;
        const double mxy = w * nx_d * ny_d;
        const double mxz = w * nx_d * nz_d;
        const double myz = w * ny_d * nz_d;

        for (const Sample& sp : samples) {
            const vec3 p = t.p[0] + (t.p[1] - t.p[0]) * sp.b1 + (t.p[2] - t.p[0]) * sp.b2;
            const int ix = std::clamp(static_cast<int>(std::floor((p.x - origin.x) / cell)), clamp_lo[0], clamp_hi[0]);
            const int iy = std::clamp(static_cast<int>(std::floor((p.y - origin.y) / cell)), clamp_lo[1], clamp_hi[1]);
            const int iz = std::clamp(static_cast<int>(std::floor((p.z - origin.z) / cell)), clamp_lo[2], clamp_hi[2]);
            const size_t c = ((size_t(iz) * size_t(ny) + size_t(iy)) * size_t(nx) + size_t(ix)) * k_channels;
            acc[c + 0] += mxx;
            acc[c + 1] += myy;
            acc[c + 2] += mzz;
            acc[c + 3] += mxy;
            acc[c + 4] += mxz;
            acc[c + 5] += myz;
        }
    }
    for (size_t i = 0; i < acc.size(); ++i) out_c[i] = static_cast<float>(acc[i]);
}

inline float decode_snorm8(int8_t v) {
    return std::max(-1.0f, float(v) / 127.0f);
}

bool ray_box_intersect(vec3 p, vec3 dir, vec3 bmin, vec3 bmax, double t_min, double& out_t0, double& out_t1) {
    const double o[3]  = {p.x, p.y, p.z};
    const double d[3]  = {dir.x, dir.y, dir.z};
    const double lo[3] = {bmin.x, bmin.y, bmin.z};
    const double hi[3] = {bmax.x, bmax.y, bmax.z};
    double t0 = t_min, t1 = std::numeric_limits<double>::infinity();
    for (int a = 0; a < 3; ++a) {
        if (d[a] == 0.0) {
            if (o[a] < lo[a] || o[a] >= hi[a]) return false;
            continue;
        }
        double ta = (lo[a] - o[a]) / d[a];
        double tb = (hi[a] - o[a]) / d[a];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
    }
    if (!(t0 < t1)) return false;
    out_t0 = t0;
    out_t1 = t1;
    return true;
}

}  // namespace

CrownVolume CrownVolume::build(const Probe& probe, int cells_longest) {
    if (cells_longest < 2) throw std::runtime_error("crown volume: need at least 2 cells on the longest axis");
    const CrownMesh& m = probe.mesh();
    if (m.leaf_triangles == 0) throw std::runtime_error("crown volume: the mesh has no leaf triangles");
    vec3 lo, hi;
    leaf_aabb(m, lo, hi);

    const vec3 ext = hi - lo;
    const float longest = std::max({ext.x, ext.y, ext.z});
    if (!(longest > 0.0f)) throw std::runtime_error("crown volume: leaf AABB is degenerate");

    Grid g;
    g.cell = longest / static_cast<float>(cells_longest);
    int n[3];
    for (int a = 0; a < 3; ++a) {
        const float e = std::max(comp(ext, a), g.cell);
        n[a] = std::max(2, static_cast<int>(std::ceil(e / g.cell - 1e-4f)));
    }
    g.nx = n[0]; g.ny = n[1]; g.nz = n[2];
    const vec3 centre = (lo + hi) * 0.5f;
    g.origin = centre - vec3{g.nx * g.cell, g.ny * g.cell, g.nz * g.cell} * 0.5f;
    splat_leaves(probe, g.origin, g.cell, g.nx, g.ny, g.nz, {0, 0, 0}, {g.nx - 1, g.ny - 1, g.nz - 1}, g.c);

    CrownVolume v;
    v.m_grid    = std::move(g);
    v.m_l0_dims = {n[0], n[1], n[2]};
    v.m_pad_lo  = {0, 0, 0};
    v.m_pad_hi  = {0, 0, 0};
    return v;
}

CrownVolume CrownVolume::build_shipped(const Probe& l0_probe, vec3 union_min, vec3 union_max,
                                       float target_cell_m, int max_l0_dim) {
    if (max_l0_dim < 2) throw std::runtime_error("crown volume: max_l0_dim must be >= 2");
    if (!(target_cell_m > 0.0f)) throw std::runtime_error("crown volume: target_cell_m must be > 0");
    const CrownMesh& m = l0_probe.mesh();
    if (m.leaf_triangles == 0) throw std::runtime_error("crown volume: L0 mesh has no leaf triangles");

    vec3 l0_min, l0_max;
    leaf_aabb(m, l0_min, l0_max);
    const vec3 l0_ext = l0_max - l0_min;
    const float longest_l0 = std::max({l0_ext.x, l0_ext.y, l0_ext.z});
    if (!(longest_l0 > 0.0f)) throw std::runtime_error("crown volume: L0 leaf AABB is degenerate");

    const float cell_m = std::max(target_cell_m, longest_l0 / static_cast<float>(max_l0_dim));

    std::array<int, 3> l0_dims{};
    std::array<int, 3> pad_lo{};
    std::array<int, 3> pad_hi{};
    std::array<int, 3> dims{};
    float box_min_a[3]{};

    for (int a = 0; a < 3; ++a) {
        const float l0_lo_a = comp(l0_min, a);
        const float l0_hi_a = comp(l0_max, a);
        const float u_lo_a  = std::min(l0_lo_a, comp(union_min, a));
        const float u_hi_a  = std::max(l0_hi_a, comp(union_max, a));

        const float ext_a = std::max(comp(l0_ext, a), cell_m);
        l0_dims[a] = std::max(2, static_cast<int>(std::ceil(ext_a / cell_m - 1e-4f)));
        const float l0_snapped_max_a = l0_lo_a + static_cast<float>(l0_dims[a]) * cell_m;

        pad_lo[a] = std::max(0, static_cast<int>(std::ceil((l0_lo_a - u_lo_a) / cell_m - 1e-5f)));
        pad_hi[a] = std::max(0, static_cast<int>(std::ceil((u_hi_a - l0_snapped_max_a) / cell_m - 1e-5f)));

        box_min_a[a] = l0_lo_a - static_cast<float>(pad_lo[a]) * cell_m;
        dims[a]      = l0_dims[a] + pad_lo[a] + pad_hi[a];

        // Guard against 1-ULP boundary miss when u_lo/u_hi sits just outside snapped max.
        while (box_min_a[a] > u_lo_a) {
            ++pad_lo[a];
            box_min_a[a] = l0_lo_a - static_cast<float>(pad_lo[a]) * cell_m;
            dims[a]      = l0_dims[a] + pad_lo[a] + pad_hi[a];
        }
        while (box_min_a[a] + static_cast<float>(dims[a]) * cell_m < u_hi_a) {
            ++pad_hi[a];
            dims[a] = l0_dims[a] + pad_lo[a] + pad_hi[a];
        }
        if (dims[a] > k_max_volume_dim)
            throw std::runtime_error("crown volume: padded grid dimension exceeds 128");
    }

    Grid g;
    g.origin = vec3{box_min_a[0], box_min_a[1], box_min_a[2]};
    g.cell   = cell_m;
    g.nx     = dims[0];
    g.ny     = dims[1];
    g.nz     = dims[2];

    // Clamp splats strictly inside the L0 interior [pad_lo, pad_lo + l0_dims - 1]
    // so every outer union-of-LODs padded shell voxel stays bit-exact zero.
    const std::array<int, 3> clamp_lo = pad_lo;
    const std::array<int, 3> clamp_hi = {
        pad_lo[0] + l0_dims[0] - 1,
        pad_lo[1] + l0_dims[1] - 1,
        pad_lo[2] + l0_dims[2] - 1
    };
    splat_leaves(l0_probe, g.origin, g.cell, g.nx, g.ny, g.nz, clamp_lo, clamp_hi, g.c);

    CrownVolume v;
    v.m_grid    = std::move(g);
    v.m_l0_dims = l0_dims;
    v.m_pad_lo  = pad_lo;
    v.m_pad_hi  = pad_hi;
    return v;
}

CrownVolume CrownVolume::build_shipped(const Probe& l0_probe, float target_cell_m, int max_l0_dim) {
    vec3 l0_min, l0_max;
    leaf_aabb(l0_probe.mesh(), l0_min, l0_max);
    return build_shipped(l0_probe, l0_min, l0_max, target_cell_m, max_l0_dim);
}

CrownVolume CrownVolume::from_cells(vec3 origin, float cell, int nx, int ny, int nz, std::vector<float> cells) {
    if (nx < 1 || ny < 1 || nz < 1 || cells.size() != size_t(nx) * size_t(ny) * size_t(nz) * k_channels)
        throw std::runtime_error("crown volume: cell count");
    CrownVolume v;
    v.m_grid.origin = origin;
    v.m_grid.cell   = cell;
    v.m_grid.nx     = nx;
    v.m_grid.ny     = ny;
    v.m_grid.nz     = nz;
    v.m_grid.c      = std::move(cells);
    v.m_l0_dims     = {nx, ny, nz};
    return v;
}

double CrownVolume::total_area_m2() const {
    double s = 0.0;
    for (size_t i = 0; i < m_grid.c.size(); i += k_channels)
        s += double(m_grid.c[i + 0]) + double(m_grid.c[i + 1]) + double(m_grid.c[i + 2]);
    return s * double(m_grid.cell) * double(m_grid.cell) * double(m_grid.cell);
}

float CrownVolume::density(vec3 p) const {
    const Grid& g = m_grid;
    const int ix = static_cast<int>(std::floor((p.x - g.origin.x) / g.cell));
    const int iy = static_cast<int>(std::floor((p.y - g.origin.y) / g.cell));
    const int iz = static_cast<int>(std::floor((p.z - g.origin.z) / g.cell));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= g.nx || iy >= g.ny || iz >= g.nz) return 0.0f;
    const size_t base = ((size_t(iz) * size_t(g.ny) + size_t(iy)) * size_t(g.nx) + size_t(ix)) * k_channels;
    return g.c[base + 0] + g.c[base + 1] + g.c[base + 2];
}

// Exact traversal (Amanatides-Woo) of the cells the ray p + t * dir, t >= t_min,
// crosses; fn(cell_index, segment_length) per cell.
template <class Fn>
void CrownVolume::march(const Grid& g, vec3 p, vec3 dir, double t_min, Fn&& fn) {
    const double o[3] = {p.x, p.y, p.z}, d[3] = {dir.x, dir.y, dir.z};
    const double lo[3] = {g.origin.x, g.origin.y, g.origin.z};
    const int n[3] = {g.nx, g.ny, g.nz};
    double t0 = t_min, t1 = std::numeric_limits<double>::infinity();
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
        if (te > t) fn((size_t(idx[2]) * size_t(g.ny) + size_t(idx[1])) * size_t(g.nx) + size_t(idx[0]), te - t);
        if (t_next[a] >= t1) return;
        t = t_next[a];
        idx[a] += step[a];
        if (idx[a] < 0 || idx[a] >= n[a]) return;
        t_next[a] += t_delta[a];
    }
}

std::array<float, 2> CrownVolume::integrals(vec3 p, vec3 dir) const {
    const double wx = double(dir.x), wy = double(dir.y), wz = double(dir.z);
    double rho_sum = 0.0, tensor_sum = 0.0;
    march(m_grid, p, dir, 0.0, [&](size_t cell, double len) {
        const float* c = &m_grid.c[cell * k_channels];
        const double rho = double(c[0]) + double(c[1]) + double(c[2]);
        rho_sum    += rho * len;
        tensor_sum += tensor_extinction_unclumped(c, wx, wy, wz) * len;
    });
    return {static_cast<float>(rho_sum), static_cast<float>(tensor_sum)};
}

// K8: separable Gaussian low-pass over all 6 linear FP32 tensor channels on an
// expanded (nx + 2r, ny + 2r, nz + 2r) grid. Cached lazily; NEVER alters m_grid.
const CrownVolume::Grid& CrownVolume::ensure_lowpass(float sigma_m) const {
    std::lock_guard<std::mutex> lock(m_cache->mutex);
    auto& slot = m_cache->grids[sigma_m];
    if (slot) return *slot;

    const Grid& g = m_grid;
    const double sc = double(sigma_m) / double(g.cell);   // sigma in cells
    const int r = std::max(1, static_cast<int>(std::ceil(3.0 * sc)));
    std::vector<double> k(size_t(2 * r + 1));
    double ksum = 0.0;
    for (int i = -r; i <= r; ++i) ksum += (k[size_t(i + r)] = std::exp(-0.5 * (double(i) * double(i)) / (sc * sc)));
    for (double& v : k) v /= ksum;

    auto out = std::make_unique<Grid>();
    out->cell   = g.cell;
    out->nx     = g.nx + 2 * r;
    out->ny     = g.ny + 2 * r;
    out->nz     = g.nz + 2 * r;
    out->origin = g.origin - vec3{float(r) * g.cell, float(r) * g.cell, float(r) * g.cell};
    const size_t N = size_t(out->nx) * size_t(out->ny) * size_t(out->nz);
    std::vector<double> a(N * k_channels, 0.0), b(N * k_channels, 0.0);
    auto at = [&](int x, int y, int z) {
        return ((size_t(z) * size_t(out->ny) + size_t(y)) * size_t(out->nx) + size_t(x)) * k_channels;
    };
    for (int z = 0; z < g.nz; ++z)
        for (int y = 0; y < g.ny; ++y)
            for (int x = 0; x < g.nx; ++x)
                for (int ch = 0; ch < k_channels; ++ch)
                    a[at(x + r, y + r, z + r) + ch] =
                        g.c[((size_t(z) * size_t(g.ny) + size_t(y)) * size_t(g.nx) + size_t(x)) * k_channels + ch];

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
                    const size_t d = at(x, y, z);
                    for (int i = -r; i <= r; ++i) {
                        int q[3] = {x, y, z};
                        q[axis] = pos[axis] + i;
                        if (q[axis] < 0 || q[axis] >= dims[axis]) continue;
                        const double w = k[size_t(i + r)];
                        const size_t s = at(q[0], q[1], q[2]);
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

float CrownVolume::optical_depth(vec3 p, vec3 dir, float lowpass_sigma_m, const ExtinctionModel& model,
                                 bool exclude_receiver_cell) const {
    const Grid& g = lowpass_sigma_m > 0.0f ? ensure_lowpass(lowpass_sigma_m) : m_grid;
    const double t_min = exclude_receiver_cell ? double(k_receiver_exclude_cells) * double(g.cell) : 0.0;
    const double wx = double(dir.x), wy = double(dir.y), wz = double(dir.z);
    double tau = 0.0;
    march(g, p, dir, t_min, [&](size_t cell, double len) {
        const float* c = &g.c[cell * k_channels];
        if (model.kind == ExtinctionKind::scalar) {
            const double rho = double(c[0]) + double(c[1]) + double(c[2]);
            tau += double(model.g) * rho * len;
        } else {
            tau += tensor_extinction_unclumped(c, wx, wy, wz) * len;
        }
    });
    return static_cast<float>(tau * double(model.clumping));
}

float CrownVolume::transmittance(vec3 p, vec3 dir, bool exclude_receiver_cell) const {
    return std::exp(-optical_depth(p, dir, 0.0f, m_model, exclude_receiver_cell));
}

PackedCrownVolume CrownVolume::pack_option2a(float clumping) const {
    const Grid& g = m_grid;
    if (g.nx < 2 || g.ny < 2 || g.nz < 2)
        throw std::runtime_error("crown volume: dims must be >= 2 before packing");
    if (g.nx > k_max_volume_dim || g.ny > k_max_volume_dim || g.nz > k_max_volume_dim)
        throw std::runtime_error("crown volume: dims exceed 128 before packing");
    if (!(g.cell > 0.0f) || !(clumping > 0.0f))
        throw std::runtime_error("crown volume: invalid cell_m or clumping");

    const size_t n_vox = size_t(g.nx) * size_t(g.ny) * size_t(g.nz);
    double max_rho = 0.0;
    for (size_t i = 0; i < n_vox; ++i) {
        const float* c = &g.c[i * k_channels];
        const double rho = double(c[0]) + double(c[1]) + double(c[2]);
        if (rho > max_rho) max_rho = rho;
    }

    const float rho_ceil = 6.0f / g.cell;
    const float rho_cap  = std::max(1e-4f, std::min(static_cast<float>(max_rho), rho_ceil));

    PackedCrownVolume out;
    out.dims     = {g.nx, g.ny, g.nz};
    out.box_min  = g.origin;
    out.box_max  = box_max();
    out.cell_m   = g.cell;
    out.rho_cap  = rho_cap;
    out.clumping = clumping;
    out.max_rho  = static_cast<float>(max_rho);
    out.tex0.assign(n_vox * 4u, 0u);
    out.tex1.assign(n_vox * 4u, 0);

    for (size_t i = 0; i < n_vox; ++i) {
        const float* c = &g.c[i * k_channels];
        const double mxx = double(c[0]);
        const double myy = double(c[1]);
        const double mzz = double(c[2]);
        const double mxy = double(c[3]);
        const double mxz = double(c[4]);
        const double myz = double(c[5]);
        const double rho = mxx + myy + mzz;
        if (!(rho > 0.0)) continue;

        ++out.occupied_voxels;
        double inv_cap = 1.0 / double(rho_cap);
        if (rho > double(rho_cap)) {
            ++out.clipped_voxels;
            // Isotropic scaling M <- M * (rho_cap / rho), then divide by rho_cap:
            // identical to dividing M directly by rho.
            inv_cap = 1.0 / rho;
        }

        auto enc_u8 = [&](double val) -> uint8_t {
            const double clamped = std::clamp(val, 0.0, 1.0);
            return static_cast<uint8_t>(std::clamp(round_ties_to_even(clamped * 255.0), 0, 255));
        };
        auto enc_s8 = [&](double val) -> int8_t {
            const double clamped = std::clamp(val, -1.0, 1.0);
            return static_cast<int8_t>(std::clamp(round_ties_to_even(clamped * 127.0), -127, 127));
        };

        const size_t b = i * 4u;
        out.tex0[b + 0] = enc_u8(mxx * inv_cap);
        out.tex0[b + 1] = enc_u8(myy * inv_cap);
        out.tex0[b + 2] = enc_u8(mzz * inv_cap);
        out.tex0[b + 3] = 0u;

        out.tex1[b + 0] = enc_s8(2.0 * mxy * inv_cap);
        out.tex1[b + 1] = enc_s8(2.0 * mxz * inv_cap);
        out.tex1[b + 2] = enc_s8(2.0 * myz * inv_cap);
        out.tex1[b + 3] = 0;
    }

    return out;
}

float PackedCrownVolume::sample_voxel_sigma(int ix, int iy, int iz, vec3 dir) const {
    if (ix < 0 || iy < 0 || iz < 0 || ix >= dims[0] || iy >= dims[1] || iz >= dims[2]) return 0.0f;
    const size_t b = ((size_t(iz) * size_t(dims[1]) + size_t(iy)) * size_t(dims[0]) + size_t(ix)) * 4u;
    const float dx = float(tex0[b + 0]) / 255.0f;
    const float dy = float(tex0[b + 1]) / 255.0f;
    const float dz = float(tex0[b + 2]) / 255.0f;
    const float ox = decode_snorm8(tex1[b + 0]) * 0.5f;
    const float oy = decode_snorm8(tex1[b + 1]) * 0.5f;
    const float oz = decode_snorm8(tex1[b + 2]) * 0.5f;
    const float rho_n = dx + dy + dz;
    if (!(rho_n > 0.0f)) return 0.0f;
    const float quad_n = std::max(0.0f,
        dir.x * dir.x * dx + dir.y * dir.y * dy + dir.z * dir.z * dz
        + 2.0f * (dir.x * dir.y * ox + dir.x * dir.z * oy + dir.y * dir.z * oz));
    const float sigma_scale = clumping * k_sqrt_3_over_4_f * rho_cap;
    return sigma_scale * std::sqrt(rho_n * quad_n);
}

float PackedCrownVolume::sample_trilinear_sigma(vec3 uvw, vec3 dir) const {
    const float fx = uvw.x * float(dims[0]) - 0.5f;
    const float fy = uvw.y * float(dims[1]) - 0.5f;
    const float fz = uvw.z * float(dims[2]) - 0.5f;
    const int x0 = static_cast<int>(std::floor(fx));
    const int y0 = static_cast<int>(std::floor(fy));
    const int z0 = static_cast<int>(std::floor(fz));
    const float tx = fx - float(x0);
    const float ty = fy - float(y0);
    const float tz = fz - float(z0);

    float diag[3] = {0.0f, 0.0f, 0.0f};
    float offd[3] = {0.0f, 0.0f, 0.0f};

    for (int dz = 0; dz <= 1; ++dz) {
        const int iz = z0 + dz;
        if (iz < 0 || iz >= dims[2]) continue;  // 1-texel zero border
        const float wz = dz ? tz : (1.0f - tz);
        for (int dy = 0; dy <= 1; ++dy) {
            const int iy = y0 + dy;
            if (iy < 0 || iy >= dims[1]) continue;
            const float wyz = wz * (dy ? ty : (1.0f - ty));
            for (int dx = 0; dx <= 1; ++dx) {
                const int ix = x0 + dx;
                if (ix < 0 || ix >= dims[0]) continue;
                const float w = wyz * (dx ? tx : (1.0f - tx));
                const size_t b = ((size_t(iz) * size_t(dims[1]) + size_t(iy)) * size_t(dims[0]) + size_t(ix)) * 4u;
                diag[0] += w * (float(tex0[b + 0]) / 255.0f);
                diag[1] += w * (float(tex0[b + 1]) / 255.0f);
                diag[2] += w * (float(tex0[b + 2]) / 255.0f);
                offd[0] += w * (decode_snorm8(tex1[b + 0]) * 0.5f);
                offd[1] += w * (decode_snorm8(tex1[b + 1]) * 0.5f);
                offd[2] += w * (decode_snorm8(tex1[b + 2]) * 0.5f);
            }
        }
    }

    const float rho_n = diag[0] + diag[1] + diag[2];
    if (!(rho_n > 0.0f)) return 0.0f;
    const float quad_n = std::max(0.0f,
        dir.x * dir.x * diag[0] + dir.y * dir.y * diag[1] + dir.z * dir.z * diag[2]
        + 2.0f * (dir.x * dir.y * offd[0] + dir.x * dir.z * offd[1] + dir.y * dir.z * offd[2]));
    const float sigma_scale = clumping * k_sqrt_3_over_4_f * rho_cap;
    return sigma_scale * std::sqrt(rho_n * quad_n);
}

float PackedCrownVolume::optical_depth_trilinear(vec3 p, vec3 dir, int max_steps,
                                                 bool exclude_receiver_cell) const {
    if (empty() || max_steps <= 0) return 0.0f;
    const double t_min = exclude_receiver_cell ? double(k_receiver_exclude_cells) * double(cell_m) : 0.0;
    double t0 = 0.0, t1 = 0.0;
    if (!ray_box_intersect(p, dir, box_min, box_max, t_min, t0, t1)) return 0.0f;

    const double span = t1 - t0;
    const int steps = std::clamp(static_cast<int>(std::ceil(span / double(cell_m))), 1, max_steps);
    const double dt = span / double(steps);
    const vec3 ext = box_max - box_min;
    const vec3 inv_ext{1.0f / ext.x, 1.0f / ext.y, 1.0f / ext.z};

    double tau = 0.0;
    for (int i = 0; i < steps; ++i) {
        const float t = static_cast<float>(t0 + (double(i) + 0.5) * dt);
        const vec3 q = p + dir * t;
        const vec3 uvw{(q.x - box_min.x) * inv_ext.x,
                       (q.y - box_min.y) * inv_ext.y,
                       (q.z - box_min.z) * inv_ext.z};
        tau += double(sample_trilinear_sigma(uvw, dir)) * dt;
    }
    return static_cast<float>(tau);
}

float PackedCrownVolume::transmittance_trilinear(vec3 p, vec3 dir, int max_steps,
                                                 bool exclude_receiver_cell) const {
    return std::exp(-optical_depth_trilinear(p, dir, max_steps, exclude_receiver_cell));
}

float PackedCrownVolume::optical_depth_dda(vec3 p, vec3 dir, bool exclude_receiver_cell) const {
    if (empty()) return 0.0f;
    const double t_min = exclude_receiver_cell ? double(k_receiver_exclude_cells) * double(cell_m) : 0.0;
    const double o[3]  = {p.x, p.y, p.z};
    const double d[3]  = {dir.x, dir.y, dir.z};
    const double lo[3] = {box_min.x, box_min.y, box_min.z};
    const int n[3]     = {dims[0], dims[1], dims[2]};
    double t0 = t_min, t1 = std::numeric_limits<double>::infinity();
    for (int a = 0; a < 3; ++a) {
        const double hi = lo[a] + n[a] * double(cell_m);
        if (d[a] == 0.0) {
            if (o[a] < lo[a] || o[a] >= hi) return 0.0f;
            continue;
        }
        double ta = (lo[a] - o[a]) / d[a], tb = (hi - o[a]) / d[a];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
    }
    if (!(t0 < t1)) return 0.0f;

    int idx[3], step[3];
    double t_next[3], t_delta[3];
    const double tm = 0.5 * (t0 + std::min(t1, t0 + 1e-6 * double(cell_m)));
    for (int a = 0; a < 3; ++a) {
        const double x = (o[a] + d[a] * tm - lo[a]) / double(cell_m);
        idx[a] = std::clamp(static_cast<int>(std::floor(x)), 0, n[a] - 1);
        if (d[a] > 0.0) {
            step[a] = 1;
            t_next[a] = (lo[a] + (idx[a] + 1) * double(cell_m) - o[a]) / d[a];
            t_delta[a] = double(cell_m) / d[a];
        } else if (d[a] < 0.0) {
            step[a] = -1;
            t_next[a] = (lo[a] + idx[a] * double(cell_m) - o[a]) / d[a];
            t_delta[a] = -double(cell_m) / d[a];
        } else {
            step[a] = 0;
            t_next[a] = std::numeric_limits<double>::infinity();
            t_delta[a] = std::numeric_limits<double>::infinity();
        }
    }

    double tau = 0.0;
    double t = t0;
    for (;;) {
        const int a = (t_next[0] <= t_next[1] && t_next[0] <= t_next[2]) ? 0 : (t_next[1] <= t_next[2] ? 1 : 2);
        const double te = std::min(t_next[a], t1);
        if (te > t) tau += double(sample_voxel_sigma(idx[0], idx[1], idx[2], dir)) * (te - t);
        if (t_next[a] >= t1) break;
        t = t_next[a];
        idx[a] += step[a];
        if (idx[a] < 0 || idx[a] >= n[a]) break;
        t_next[a] += t_delta[a];
    }
    return static_cast<float>(tau);
}

float PackedCrownVolume::transmittance_dda(vec3 p, vec3 dir, bool exclude_receiver_cell) const {
    return std::exp(-optical_depth_dda(p, dir, exclude_receiver_cell));
}

}  // namespace treegen::crown
