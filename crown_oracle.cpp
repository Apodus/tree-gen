// NOTE: /fp:precise like the rest of TreegenCore: the report is compared byte
// for byte between the scenario path and --glb.
#include "crown_oracle.hpp"

#include "det_rng.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace treegen::crown {

namespace {

constexpr double k_pi = 3.14159265358979323846;
constexpr float  k_edge_bin_m  = 0.25f;
constexpr float  k_trunk_bin_m = 0.5f;
constexpr float  k_depth_bin_m = 0.5f;

// ---- JSON writer: deterministic, ordered keys, %.9g numbers -----------------
class Json {
public:
    Json& obj()  { open('{'); return *this; }
    Json& arr()  { open('['); return *this; }
    Json& end()  {
        const char c = m_stack.back().first;
        const bool empty = m_stack.back().second == 0;
        m_stack.pop_back();
        if (!empty) { m_s += '\n'; indent(); }
        m_s += (c == '{' ? '}' : ']');
        return *this;
    }
    Json& key(const char* k) { item(); m_s += '"'; m_s += k; m_s += "\": "; m_pending_key = true; return *this; }
    Json& num(double v) {
        if (!std::isfinite(v)) throw std::runtime_error("crown probe: non-finite number in report");
        if (v == 0.0) v = 0.0;   // no "-0"
        char buf[40];
        std::snprintf(buf, sizeof buf, "%.9g", v);
        value(buf);
        return *this;
    }
    Json& integer(long long v) { char buf[32]; std::snprintf(buf, sizeof buf, "%lld", v); value(buf); return *this; }
    Json& boolean(bool b) { value(b ? "true" : "false"); return *this; }
    Json& null() { value("null"); return *this; }
    Json& str(const std::string& s) { value(json_string(s).c_str()); return *this; }
    Json& raw(const std::string& s) { value(s.c_str()); return *this; }
    Json& opt(const std::optional<float>& v) { return v ? num(*v) : null(); }
    Json& vec(vec3 v) { arr(); num(v.x); num(v.y); num(v.z); return end(); }
    std::string take() { m_s += '\n'; return std::move(m_s); }

private:
    void indent() { m_s.append(m_stack.size() * 2, ' '); }
    void item() {
        if (m_stack.empty()) return;
        if (m_stack.back().second++ > 0) m_s += ',';
        m_s += '\n';
        indent();
    }
    void value(const char* v) {
        if (m_pending_key) m_pending_key = false; else item();
        m_s += v;
    }
    void open(char c) {
        if (m_pending_key) m_pending_key = false; else item();
        m_s += c;
        m_stack.push_back({c, 0});
    }
    std::string m_s;
    std::vector<std::pair<char, int>> m_stack;
    bool m_pending_key = false;
};

float percentile(std::vector<float> v, double q) {
    std::sort(v.begin(), v.end());
    const size_t k = static_cast<size_t>(std::ceil(q * double(v.size())));
    return v[std::min(v.size() - 1, k == 0 ? 0 : k - 1)];
}

// ---- SE helpers (R-CR6 / P1 / P2 / P3) -----------------------------------------
template <class T> double sample_variance_over_n(std::span<const T> x) {
    if (x.size() < 2) return 0.0;
    double m = 0.0;
    for (const T& v : x) m += double(v);
    m /= double(x.size());
    double s = 0.0;
    for (const T& v : x) s += (double(v) - m) * (double(v) - m);
    return s / double(x.size() - 1) / double(x.size());
}

// Exact Binomial(R, p) probability P(lo <= X <= hi).
double binom_prob_range(int R, double p, int lo, int hi) {
    if (R <= 0 || lo > hi) return 0.0;
    if (p <= 0.0) return (lo <= 0 && 0 <= hi) ? 1.0 : 0.0;
    if (p >= 1.0) return (lo <= R && R <= hi) ? 1.0 : 0.0;
    const int a = std::max(0, lo);
    const int b = std::min(R, hi);
    if (a > b) return 0.0;
    const double lgr = std::lgamma(double(R) + 1.0);
    const double lp = std::log(p);
    const double lq = std::log(1.0 - p);
    double sum = 0.0;
    for (int x = a; x <= b; ++x) {
        const double lpmf = lgr - std::lgamma(double(x) + 1.0) - std::lgamma(double(R - x) + 1.0)
                          + double(x) * lp + double(R - x) * lq;
        sum += std::exp(lpmf);
    }
    return std::clamp(sum, 0.0, 1.0);
}

int max_silhouette_open(int R) {
    int hi = -1;
    for (int x = 0; x <= R; ++x)
        if (float(x) / float(R) < k_silhouette_t) hi = x;
    return hi;
}

std::pair<int, int> unsaturated_open_range(int R) {
    int lo = R + 1, hi = -1;
    for (int x = 0; x <= R; ++x) {
        const float t = float(x) / float(R);
        if (t >= k_opaque_t && t <= k_unsaturated_hi) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
    }
    return {lo, hi};
}

double footprint_open_prob(const Footprint& f, int& out_rays) {
    if (f.ray_count > 0) {
        out_rays = int(f.ray_count);
        const int open = std::min(int(std::popcount(f.open_mask)), out_rays);
        return double(open) / double(out_rays);
    }
    out_rays = rays_per_side(RayStream::selection) * rays_per_side(RayStream::selection);
    return std::clamp(double(f.t), 0.0, 1.0);
}

vec3 bounds_centre(const Probe& p) { return (p.bounds_min() + p.bounds_max()) * 0.5f; }
float bounds_diag(const Probe& p) { return length(p.bounds_max() - p.bounds_min()); }

// Golden-section minimum of a unimodal f on [a, b].
template <class F> double golden_min(F&& f, double a, double b) {
    const double r = 0.5 * (std::sqrt(5.0) - 1.0);
    double c = b - r * (b - a), d = a + r * (b - a);
    double fc = f(c), fd = f(d);
    for (int it = 0; it < 80; ++it) {
        if (fc < fd) { b = d; d = c; fd = fc; c = b - r * (b - a); fc = f(c); }
        else         { a = c; c = d; fc = fd; d = a + r * (b - a); fd = f(d); }
    }
    return 0.5 * (a + b);
}

double t_scalar(const FormatFootprint& f, double g) {
    double s = 0.0;
    for (const auto& r : f.integrals) s += std::exp(-g * r[0]);
    return s / double(f.integrals.size());
}
double t_axis(const FormatFootprint& f, double c) {
    double s = 0.0;
    for (const auto& r : f.integrals) s += std::exp(-c * r[1]);
    return s / double(f.integrals.size());
}

std::string format_probe_input(const ProbeInput& input, std::span<const uint8_t> glb, int lod) {
    if (input.kind == ProbeInputKind::scenario) {
        return "{\"source\": \"scenario\", \"scenario\": " + json_string(input.path) + ", \"seed\": " +
               std::to_string(input.seed) + ", \"glb_fnv1a64\": " + json_string(glb_digest(glb)) + ", \"probe_lod\": " +
               std::to_string(lod) + "}";
    }
    return "{\"source\": \"glb\", \"glb\": " + json_string(input.path) + ", \"glb_fnv1a64\": " +
           json_string(glb_digest(glb)) + ", \"probe_lod\": " + std::to_string(lod) + "}";
}

}  // namespace

namespace detail {

double se_mean_t(std::span<const float> t, int rays) {
    if (t.empty() || rays <= 0) return 0.0;
    double v = 0.0;
    for (float x : t) v += double(x) * (1.0 - x);
    return std::sqrt(std::max(0.0, v) / double(rays)) / double(t.size());
}

double se_share(double p, size_t n) {
    return n ? std::sqrt(std::max(0.0, p * (1.0 - p)) / double(n)) : 0.0;
}

double se_mean(std::span<const double> x) { return std::sqrt(sample_variance_over_n(x)); }
double se_mean(std::span<const float> x)  { return std::sqrt(sample_variance_over_n(x)); }

// Quantile q of n values: half the spread of the order statistics one binomial
// SE (sqrt(n q (1 - q)) ranks) either side of its rank. When one side clamps at
// the sample boundary (q*n - d <= 0 or q*n + d > n) or equals the quantile's
// own rank (exact discrete handoff such as n=19, q=0.95), use the full
// one-sided spread instead of halving it (P1).
double se_quantile(std::vector<float> v, double q) {
    if (v.size() < 2) return 0.0;
    q = std::clamp(q, 0.0, 1.0);
    std::sort(v.begin(), v.end());
    const double n = double(v.size()), d = std::sqrt(n * q * (1.0 - q));
    const double lo_pos = q * n - d;
    const double hi_pos = q * n + d;
    auto rank_idx = [&](double rank) { return size_t(std::clamp(rank, 0.0, n - 1.0)); };
    const size_t k_q  = rank_idx(std::ceil(q * n) - 1.0);
    const size_t k_lo = rank_idx(std::ceil(lo_pos) - 1.0);
    const size_t k_hi = rank_idx(std::ceil(hi_pos) - 1.0);
    const bool one_sided = (lo_pos <= 0.0) || (hi_pos > n) || (k_lo == k_q) || (k_hi == k_q);
    const double spread = double(v[k_hi]) - double(v[k_lo]);
    return one_sided ? spread : 0.5 * spread;
}

double mae_se(std::span<const double> abs_errors) {
    return std::sqrt(sample_variance_over_n(abs_errors));
}

double fit_se(std::span<const double> residuals, std::span<const double> derivatives) {
    if (residuals.size() != derivatives.size())
        throw std::logic_error("fit_se: residuals and derivatives size mismatch");
    const size_t n = residuals.size();
    if (n < 2) return 0.0;
    double s = 0.0, j2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s  += residuals[i] * residuals[i];
        j2 += derivatives[i] * derivatives[i];
    }
    return j2 > 0.0 ? std::sqrt(s / double(n - 1) / j2) : 0.0;
}

double dt_scalar(const FormatFootprint& f, double g) {
    if (f.integrals.empty()) return 0.0;
    double s = 0.0;
    for (const auto& r : f.integrals) s -= r[0] * std::exp(-g * r[0]);
    return s / double(f.integrals.size());
}

double dt_axis(const FormatFootprint& f, double c) {
    if (f.integrals.empty()) return 0.0;
    double s = 0.0;
    for (const auto& r : f.integrals) s -= r[1] * std::exp(-c * r[1]);
    return s / double(f.integrals.size());
}

}  // namespace detail

double pin_band(double se, double resolution_floor) {
    return std::max(3.0 * se, resolution_floor);
}
double pin_band_share(double se, size_t n) {
    return pin_band(se, n > 0 ? 1.0 / double(n) : 1.0);
}
double pin_band_mean_t(double se, size_t footprints, int rays_per_footprint) {
    const double floor = (footprints > 0 && rays_per_footprint > 0)
                             ? 1.0 / (double(rays_per_footprint) * double(footprints))
                             : 1.0;
    return pin_band(se, floor);
}
double pin_band_quantile(double se, int step_count) {
    return pin_band(se, step_count > 0 ? 1.0 / double(step_count) : 1.0);
}
double pin_band_count(double se) {
    return pin_band(se, 1.0);
}

std::string json_string(const std::string& s) {
    std::string q = "\"";
    for (const char c : s) {
        switch (c) {
            case '"':  q += "\\\""; break;
            case '\\': q += "\\\\"; break;
            case '\n': q += "\\n"; break;
            case '\r': q += "\\r"; break;
            case '\t': q += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(static_cast<unsigned char>(c)));
                    q += buf;
                } else q += c;
        }
    }
    q += '"';
    return q;
}

std::string glb_digest(std::span<const uint8_t> glb) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "0x%016llx",
                  static_cast<unsigned long long>(fnv1a64(std::string_view(reinterpret_cast<const char*>(glb.data()), glb.size()))));
    return buf;
}

SunDirection sun_direction(float elevation_deg, float azimuth_deg) {
    const double e = elevation_deg * k_pi / 180.0, a = azimuth_deg * k_pi / 180.0;
    SunDirection s;
    s.elevation_deg = elevation_deg;
    s.azimuth_deg = azimuth_deg;
    s.dir = elevation_deg >= 90.0f ? vec3{0.0f, 0.0f, 1.0f}
                                   : normalized(vec3{float(std::cos(e) * std::cos(a)), float(std::cos(e) * std::sin(a)),
                                                     float(std::sin(e))});
    return s;
}

std::vector<SunDirection> sun_directions(const std::vector<float>& elevations_deg, int azimuths) {
    std::vector<SunDirection> out;
    for (float e : elevations_deg) {
        // Deduplicate zenith (elevation >= 89.99 deg) to a single azimuth (0 deg)
        // so zenith footprints are never 4x weighted (Carbon Note #1 / S-P1 §2.3.4).
        const int n_az = (e >= 89.99f) ? 1 : azimuths;
        for (int k = 0; k < n_az; ++k) out.push_back(sun_direction(e, 360.0f * float(k) / float(n_az)));
    }
    return out;
}

int rays_per_side(RayStream s) { return s == RayStream::format ? k_format_rays_per_side : k_rays_per_side; }
uint64_t stream_id(RayStream s) {
    switch (s) {
        case RayStream::selection:     return k_footprint_ray_stream;
        case RayStream::selection_alt: return k_footprint_alt_ray_stream;
        case RayStream::format:        return k_format_ray_stream;
    }
    return k_footprint_ray_stream;
}

void FootprintSet::rays(size_t k, std::vector<vec3>& out, RayStream s) const {
    const Footprint& f = footprints[k];
    const int per_side = rays_per_side(s);
    out.clear();
    pcg32 rng;
    rng.seed((uint64_t(uint32_t(f.i + 1000000)) << 32) | uint64_t(uint32_t(f.j + 1000000)), stream_id(s));
    const float sub = k_footprint_m / float(per_side);
    for (int a = 0; a < per_side; ++a)
        for (int b = 0; b < per_side; ++b) {
            const float du = (float(a) + rng.next_float_01()) * sub - 0.5f * k_footprint_m;
            const float dv = (float(b) + rng.next_float_01()) * sub - 0.5f * k_footprint_m;
            out.push_back(origin + e1 * (f.u + du) + e2 * (f.v + dv) - sun.dir * ray_back);
        }
}

FootprintSet measure_footprints(const Probe& probe, const SunDirection& sun, PlaneKind kind, Occluders occ,
                                RayStream selection_stream) {
    if (selection_stream == RayStream::format)
        throw std::logic_error("crown probe: measure_footprints expects a selection RayStream, not RayStream::format");
    FootprintSet set;
    set.sun = sun;
    set.kind = kind;
    set.occluders = occ;
    set.selection_stream = selection_stream;
    const vec3 s = sun.dir;
    const vec3 c = bounds_centre(probe);
    const float diag = bounds_diag(probe) + 1.0f;
    const float zmin = probe.bounds_min().z, zmax = probe.bounds_max().z;
    const bool beam = kind == PlaneKind::beam;
    if (beam) {
        const double a = sun.azimuth_deg * k_pi / 180.0;
        set.origin = c;
        set.e1 = vec3{float(-std::sin(a)), float(std::cos(a)), 0.0f};
        set.e2 = normalized(cross(s, set.e1));
        set.ray_back = diag;
        set.t_max = 2.0f * diag;
    } else {
        if (!(s.z > 0.0f)) throw std::runtime_error("crown probe: ground plane needs a sun above the horizon");
        set.origin = vec3{c.x, c.y, zmin};
        set.e1 = vec3{1.0f, 0.0f, 0.0f};
        set.e2 = vec3{0.0f, 1.0f, 0.0f};
        set.ray_back = 1.0f;
        set.t_max = 1.0f + (zmax - zmin) / s.z + 2.0f * diag;
    }
    const float t_max = set.t_max;
    // Plane coordinates of a point's line along the sun.
    auto project = [&](vec3 p) {
        if (beam) {
            const vec3 d = p - set.origin;
            return std::array<float, 2>{dot(d, set.e1), dot(d, set.e2)};
        }
        const vec3 q = p - s * ((p.z - zmin) / s.z);
        return std::array<float, 2>{q.x - set.origin.x, q.y - set.origin.y};
    };

    float umin = 1e30f, umax = -1e30f, vmin = 1e30f, vmax = -1e30f;
    for (const Triangle& t : probe.mesh().triangles)
        for (const vec3& p : t.p) {
            const auto uv = project(p);
            umin = std::min(umin, uv[0]); umax = std::max(umax, uv[0]);
            vmin = std::min(vmin, uv[1]); vmax = std::max(vmax, uv[1]);
        }
    if (umin > umax) return set;   // empty mesh
    const int i0 = int(std::floor(umin / k_footprint_m)) - 1, i1 = int(std::ceil(umax / k_footprint_m)) + 1;
    const int j0 = int(std::floor(vmin / k_footprint_m)) - 1, j1 = int(std::ceil(vmax / k_footprint_m)) + 1;
    set.i0 = i0; set.i1 = i1; set.j0 = j0; set.j1 = j1;
    for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {
            Footprint f;
            f.i = i; f.j = j;
            f.u = (float(i) + 0.5f) * k_footprint_m;
            f.v = (float(j) + 0.5f) * k_footprint_m;
            f.centre = set.origin + set.e1 * f.u + set.e2 * f.v;
            set.footprints.push_back(f);
        }

    parallel_for(set.footprints.size(), [&](size_t k) {
        std::vector<vec3> rays;
        set.rays(k, rays, selection_stream);
        int open = 0;
        uint64_t mask = 0;
        long geo = 0, opq = 0;
        for (size_t ri = 0; ri < rays.size(); ++ri) {
            const LineHits h = probe.line_hits(rays[ri], s, t_max);
            if (!h.occluded(occ)) {
                ++open;
                mask |= (uint64_t(1) << ri);
            }
            geo += h.leaf_geometric;
            opq += h.leaf_opaque;
        }
        Footprint& f = set.footprints[k];
        f.t = float(open) / float(rays.size());
        f.open_mask = mask;
        f.ray_count = static_cast<uint8_t>(rays.size());
        f.crossings_geometric = float(geo) / float(rays.size());
        f.crossings_opaque    = float(opq) / float(rays.size());
    });

    // Edge distances: the nearest outside centre is always one with a
    // silhouette 8-neighbour, so only those are searched.
    const int w = i1 - i0;
    auto idx = [&](int i, int j) { return size_t(j - j0) * size_t(w) + size_t(i - i0); };
    std::vector<std::array<float, 2>> boundary;
    for (const Footprint& f : set.footprints) {
        if (f.silhouette()) continue;
        bool touches = false;
        for (int dj = -1; dj <= 1 && !touches; ++dj)
            for (int di = -1; di <= 1 && !touches; ++di) {
                const int ii = f.i + di, jj = f.j + dj;
                if (ii < i0 || ii >= i1 || jj < j0 || jj >= j1) continue;
                touches = set.footprints[idx(ii, jj)].silhouette();
            }
        if (touches) boundary.push_back({f.u, f.v});
    }
    for (Footprint& f : set.footprints) {
        if (!f.silhouette()) continue;
        float best = std::numeric_limits<float>::infinity();
        for (const auto& b : boundary) best = std::min(best, std::hypot(f.u - b[0], f.v - b[1]));
        f.edge_distance = best;
    }
    return set;
}

CountSEs footprint_count_se(const FootprintSet& set, float min_edge_m, bool include_spatial_bootstrap) {
    CountSEs out;
    const size_t n = set.footprints.size();
    if (n == 0) return out;

    // Independent binomial crossing variance for silhouette count (P2):
    // every footprint with 0 < open < R contributes q_k(1 - q_k), not only +-1/R.
    // A footprint with open == R whose rays still intersected leaf card geometry
    // (crossings_geometric > 0, all transparent on this draw) uses a 0.5-hit
    // continuity estimate p = (R - 0.5) / R so zero-hit boundary cards are not
    // treated as zero-variance empty sky.
    std::vector<double> p_open(n, 1.0);
    std::vector<int>    r_cnt(n, 0);
    std::vector<size_t> uncertain;
    double sil_var = 0.0;
    for (size_t k = 0; k < n; ++k) {
        int R = 0;
        double p = footprint_open_prob(set.footprints[k], R);
        if (p >= 1.0 && set.footprints[k].crossings_geometric > 0.0f && R > 1)
            p = (double(R) - 0.5) / double(R);
        p_open[k] = p;
        r_cnt[k] = R;
        const double q = binom_prob_range(R, p, 0, max_silhouette_open(R));
        if (p > 0.0 && p < 1.0) {
            sil_var += q * (1.0 - q);
            uncertain.push_back(k);
        }
    }
    out.silhouette_se = std::sqrt(sil_var);
    if (!include_spatial_bootstrap || uncertain.empty()) return out;

    // Spatially coupled counts and body stats (body.footprints, rim, body.mean,
    // p50, p95, share_opaque): bootstrap over the 64 stored ray hits with exact
    // edge_distance >= min_edge_m recomputation. Resampling open_boot ~ Binomial(R, p_k)
    // per uncertain footprint has the exact joint distribution of an i.i.d. 64-ray
    // nonparametric bootstrap (a conservative upper bound on 8x8 stratified
    // Poisson-binomial variance by Hoeffding's theorem) and captures both ray
    // jitter and membership cascades.
    const int w = set.i1 - set.i0, h = set.j1 - set.j0;
    if (w <= 0 || h <= 0 || size_t(w) * size_t(h) != n) return out;

    const int rad = std::max(1, int(std::ceil(min_edge_m / k_footprint_m)));
    std::vector<std::pair<int, int>> offsets;
    for (int dj = -rad; dj <= rad; ++dj)
        for (int di = -rad; di <= rad; ++di) {
            if (di == 0 && dj == 0) continue;
            if (std::hypot(float(di) * k_footprint_m, float(dj) * k_footprint_m) < min_edge_m)
                offsets.push_back({di, dj});
        }

    // Build exact Binomial(R, p_k) CDFs keyed on exact (R, p_k) so arbitrary
    // synthetic probabilities never alias.
    std::map<std::pair<int, double>, std::vector<double>> cdf_cache;
    std::vector<const std::vector<double>*> cdf_of(uncertain.size(), nullptr);
    for (size_t ui = 0; ui < uncertain.size(); ++ui) {
        const size_t k = uncertain[ui];
        const int R = r_cnt[k];
        auto [it, fresh] = cdf_cache.try_emplace({R, p_open[k]});
        if (fresh) {
            auto& cdf = it->second;
            cdf.resize(size_t(R + 1));
            const double p = p_open[k];
            const double lgr = std::lgamma(double(R) + 1.0);
            const double lp = std::log(p), lq = std::log(1.0 - p);
            double acc = 0.0;
            for (int x = 0; x <= R; ++x) {
                acc += std::exp(lgr - std::lgamma(double(x) + 1.0) - std::lgamma(double(R - x) + 1.0)
                                + double(x) * lp + double(R - x) * lq);
                cdf[size_t(x)] = acc;
            }
            cdf.back() = 1.0;
        }
        cdf_of[ui] = &it->second;
    }

    std::vector<float> t_base(n, 1.0f);
    std::vector<uint8_t> sil_base(n, 0);
    for (size_t k = 0; k < n; ++k) {
        t_base[k] = set.footprints[k].t;
        sil_base[k] = set.footprints[k].silhouette() ? 1 : 0;
    }

    constexpr int k_boot = 256;
    std::vector<double> body_counts, rim_counts, body_means, body_p50s, body_p95s, body_opaques;
    body_counts.reserve(k_boot);
    rim_counts.reserve(k_boot);
    body_means.reserve(k_boot);
    body_p50s.reserve(k_boot);
    body_p95s.reserve(k_boot);
    body_opaques.reserve(k_boot);

    std::vector<float> t_cur(n);
    std::vector<uint8_t> sil(n);
    std::vector<float> body_t;
    body_t.reserve(n);
    pcg32 rng;
    rng.seed(0xB00757A9ull, 0xC00475Eull);
    for (int b = 0; b < k_boot; ++b) {
        t_cur = t_base;
        sil = sil_base;
        for (size_t ui = 0; ui < uncertain.size(); ++ui) {
            const size_t k = uncertain[ui];
            const auto& cdf = *cdf_of[ui];
            const double u = (double(rng.next_u32()) + 0.5) / 4294967296.0;
            const auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
            const int x = int(std::min<size_t>(size_t(it - cdf.begin()), cdf.size() - 1));
            const float tv = float(x) / float(r_cnt[k]);
            t_cur[k] = tv;
            sil[k] = tv < k_silhouette_t ? 1 : 0;
        }
        int body_n = 0, rim_n = 0, opq_n = 0;
        double sum_t = 0.0;
        body_t.clear();
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i) {
                const size_t k = size_t(j) * size_t(w) + size_t(i);
                if (!sil[k]) continue;
                bool near_out = false;
                for (const auto& [di, dj] : offsets) {
                    const int ii = i + di, jj = j + dj;
                    if (ii < 0 || ii >= w || jj < 0 || jj >= h || !sil[size_t(jj) * size_t(w) + size_t(ii)]) {
                        near_out = true;
                        break;
                    }
                }
                if (near_out) {
                    ++rim_n;
                } else {
                    ++body_n;
                    const float tv = t_cur[k];
                    body_t.push_back(tv);
                    sum_t += tv;
                    if (tv < k_opaque_t) ++opq_n;
                }
            }
        body_counts.push_back(double(body_n));
        rim_counts.push_back(double(rim_n));
        if (!body_t.empty()) {
            body_means.push_back(sum_t / double(body_t.size()));
            body_opaques.push_back(double(opq_n) / double(body_t.size()));
            std::sort(body_t.begin(), body_t.end());
            auto pct = [&](double q) {
                const size_t idx = static_cast<size_t>(std::ceil(q * double(body_t.size())));
                return double(body_t[std::min(body_t.size() - 1, idx == 0 ? 0 : idx - 1)]);
            };
            body_p50s.push_back(pct(0.50));
            body_p95s.push_back(pct(0.95));
        }
    }
    auto sample_sd = [](const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        double m = 0.0;
        for (double x : v) m += x;
        m /= double(v.size());
        double s = 0.0;
        for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / double(v.size() - 1));
    };
    out.body_footprints_se   = sample_sd(body_counts);
    out.rim_se               = sample_sd(rim_counts);
    out.body_mean_se         = sample_sd(body_means);
    out.body_p50_se          = sample_sd(body_p50s);
    out.body_p95_se          = sample_sd(body_p95s);
    out.body_share_opaque_se = sample_sd(body_opaques);
    return out;
}

namespace {
std::optional<RegionStats> body_stats_impl(const FootprintSet& set, float min_edge_m, const CountSEs* precomputed_se, bool with_count_se) {
    std::vector<float> t;
    for (const Footprint& f : set.footprints)
        if (f.in_body(min_edge_m)) t.push_back(f.t);
    if (t.empty()) return std::nullopt;
    RegionStats r;
    r.footprints = int(t.size());
    double sum = 0.0;
    int opaque = 0;
    for (float x : t) { sum += x; if (x < k_opaque_t) ++opaque; }
    r.mean = float(sum / double(t.size()));
    r.share_opaque = float(double(opaque) / double(t.size()));
    r.p50 = percentile(t, 0.50);
    r.p95 = percentile(t, 0.95);
    const double analytic_mean_se   = detail::se_mean_t(t, rays_per_side(RayStream::selection) * rays_per_side(RayStream::selection));
    const double analytic_share_se  = detail::se_share(r.share_opaque, t.size());
    const double analytic_p50_se    = detail::se_quantile(t, 0.50);
    const double analytic_p95_se    = detail::se_quantile(t, 0.95);
    if (precomputed_se || with_count_se) {
        const CountSEs owned = precomputed_se ? CountSEs{} : footprint_count_se(set, min_edge_m);
        const CountSEs& cse = precomputed_se ? *precomputed_se : owned;
        r.footprints_se   = float(cse.body_footprints_se);
        r.mean_se         = float(std::max(analytic_mean_se, cse.body_mean_se));
        r.p50_se          = float(std::max(analytic_p50_se, cse.body_p50_se));
        r.p95_se          = float(std::max(analytic_p95_se, cse.body_p95_se));
        r.share_opaque_se = float(std::max(analytic_share_se, cse.body_share_opaque_se));
    } else {
        r.mean_se         = float(analytic_mean_se);
        r.share_opaque_se = float(analytic_share_se);
        r.p50_se          = float(analytic_p50_se);
        r.p95_se          = float(analytic_p95_se);
    }
    return r;
}
}  // namespace

std::optional<RegionStats> body_stats(const FootprintSet& set, float min_edge_m) {
    return body_stats_impl(set, min_edge_m, nullptr, true);
}

bool body_opaque(const FootprintSet& beam) {
    const auto body = body_stats_impl(beam, k_body_edge_m, nullptr, false);
    return body && body->share_opaque >= k_stop_body_share;
}

StopRule stop_rule(const std::vector<std::pair<float, bool>>& body_opaque_by_direction) {
    std::map<float, bool> by_e;
    for (const auto& [e, opaque] : body_opaque_by_direction) {
        auto [it, fresh] = by_e.try_emplace(e, true);
        it->second = it->second && opaque;   // every azimuth of this elevation
    }
    StopRule r;
    bool any = false, all = true;
    for (const auto& [e, opaque] : by_e) {
        r.opaque_by_elevation.push_back({e, opaque});
        if (e >= k_stop_min_elevation_deg) { any = true; all = all && opaque; }
    }
    r.crown_opaque_to_direct_sun = any && all;
    return r;
}

void add_format_footprints(const Probe& probe, const FootprintSet& set, const CrownVolume& volume, FormatElevation& into,
                           const std::function<bool(const Footprint&)>& keep) {
    if (set.occluders != Occluders::leaves_only)
        throw std::logic_error("crown probe: the format rule scores a leaves-only oracle (R-CR1); this footprint set was measured on "
                               + std::string(occluders_name(set.occluders)));
    into.elevation_deg = set.sun.elevation_deg;
    std::vector<size_t> chosen;
    for (size_t k = 0; k < set.footprints.size(); ++k) {
        const Footprint& f = set.footprints[k];
        if (keep && !keep(f)) continue;
        int R = 0;
        const double p = footprint_open_prob(f, R);
        const auto [ulo, uhi] = unsaturated_open_range(R);
        const double qu = binom_prob_range(R, p, ulo, uhi);
        into.unsaturated_var += qu * (1.0 - qu);
        if (f.silhouette()) {
            ++into.silhouette;
            if (f.t < k_opaque_t) ++into.saturated;
        }
        if (f.t < k_opaque_t || f.t > k_unsaturated_hi) continue;
        chosen.push_back(k);
    }
    std::vector<FormatFootprint> out(chosen.size());
    parallel_for(chosen.size(), [&](size_t c) {
        std::vector<vec3> rays;
        set.rays(chosen[c], rays, RayStream::format);
        int open = 0;
        for (const vec3& o : rays) {
            open += probe.occluded(o, set.sun.dir, -1, set.occluders) ? 0 : 1;   // the crown lies wholly ahead of o
            out[c].integrals.push_back(volume.integrals(o, set.sun.dir));
        }
        out[c].t_oracle = float(open) / float(rays.size());
    });
    for (auto& f : out) into.unsaturated.push_back(std::move(f));
}

FormatDecision decide_format(const std::vector<FormatElevation>& elevations) {
    FormatDecision d;
    std::vector<const FormatElevation*> measurable;
    for (const FormatElevation& e : elevations) {
        FormatRow r;
        r.elevation_deg = e.elevation_deg;
        r.unsaturated = int(e.unsaturated.size());
        r.unsaturated_se = float(std::sqrt(std::max(0.0, e.unsaturated_var)));
        r.measurable = r.unsaturated >= k_min_unsaturated;
        r.saturated_share = e.silhouette > 0 ? float(double(e.saturated) / e.silhouette) : 0.0f;
        r.saturated_share_se = float(detail::se_share(r.saturated_share, size_t(e.silhouette)));
        d.rows.push_back(r);
        if (r.measurable) measurable.push_back(&e);
    }
    float lo = 1e9f, hi = -1e9f;
    for (const auto* e : measurable) { lo = std::min(lo, e->elevation_deg); hi = std::max(hi, e->elevation_deg); }
    if (int(measurable.size()) < k_min_measurable || hi - lo < k_min_span_deg) {
        d.decision = "undecidable";
        return d;
    }

    auto sse = [](const std::vector<const FormatElevation*>& es, auto&& model) {
        double s = 0.0;
        for (const auto* e : es) for (const auto& f : e->unsaturated) { const double r = model(f) - f.t_oracle; s += r * r; }
        return s;
    };
    auto mae = [](const FormatElevation& e, auto&& model) {
        double s = 0.0;
        for (const auto& f : e.unsaturated) s += std::abs(model(f) - f.t_oracle);
        return float(s / double(e.unsaturated.size()));
    };
    auto mae_se = [](const FormatElevation& e, auto&& model) {
        std::vector<double> err;
        err.reserve(e.unsaturated.size());
        for (const auto& f : e.unsaturated) err.push_back(std::abs(model(f) - f.t_oracle));
        return float(detail::mae_se(err));
    };
    // Linearised least-squares SE of a one-parameter fit T(p) (definition at FormatRow::g_eff_se).
    auto fit_se = [](const std::vector<const FormatElevation*>& es, auto&& model, auto&& dmodel) {
        std::vector<double> res, der;
        for (const auto* e : es)
            for (const auto& f : e->unsaturated) {
                res.push_back(model(f) - f.t_oracle);
                der.push_back(dmodel(f));
            }
        return float(detail::fit_se(res, der));
    };

    d.g = float(golden_min([&](double g) { return sse(measurable, [&](const FormatFootprint& f) { return t_scalar(f, g); }); },
                           0.01, 2.0));
    {
        const double g = d.g;
        d.g_se = fit_se(measurable, [&](const FormatFootprint& f) { return t_scalar(f, g); },
                        [&](const FormatFootprint& f) { return detail::dt_scalar(f, g); });
    }
    float scalar_worst = 0.0f, axis_worst = 0.0f;
    bool scalar_within = true;
    for (FormatRow& r : d.rows) {
        if (!r.measurable) continue;
        const FormatElevation& e = *std::find_if(elevations.begin(), elevations.end(),
                                                 [&](const FormatElevation& x) { return x.elevation_deg == r.elevation_deg; });
        const double g = d.g;
        r.scalar_mae = mae(e, [&](const FormatFootprint& f) { return t_scalar(f, g); });
        r.axis_mae   = mae(e, [&](const FormatFootprint& f) { return t_axis(f, 1.0); });
        r.scalar_mae_se = mae_se(e, [&](const FormatFootprint& f) { return t_scalar(f, g); });
        r.axis_mae_se   = mae_se(e, [&](const FormatFootprint& f) { return t_axis(f, 1.0); });
        const std::vector<const FormatElevation*> one{&e};
        r.g_eff = float(golden_min([&](double gg) { return sse(one, [&](const FormatFootprint& f) { return t_scalar(f, gg); }); },
                                   0.01, 2.0));
        const double ge = r.g_eff;
        r.g_eff_se = fit_se(one, [&](const FormatFootprint& f) { return t_scalar(f, ge); },
                            [&](const FormatFootprint& f) { return detail::dt_scalar(f, ge); });
        scalar_worst = std::max(scalar_worst, r.scalar_mae);
        axis_worst   = std::max(axis_worst, r.axis_mae);
        if (r.scalar_mae > r.axis_mae + k_format_scalar_slack) scalar_within = false;
    }
    if (scalar_worst <= k_format_max_mae && scalar_within) { d.decision = "scalar"; return d; }
    if (axis_worst <= k_format_max_mae) { d.decision = "axis_projected"; return d; }

    // Step 5: one clumping factor fitted to the oracle on the axis-projected grid.
    d.clumping = float(golden_min([&](double c) { return sse(measurable, [&](const FormatFootprint& f) { return t_axis(f, c); }); },
                                  0.05, 4.0));
    {
        const double c = d.clumping;
        d.clumping_se = fit_se(measurable, [&](const FormatFootprint& f) { return t_axis(f, c); },
                               [&](const FormatFootprint& f) { return detail::dt_axis(f, c); });
    }
    float clumped_worst = 0.0f;
    for (const auto* e : measurable) {
        const double c = d.clumping;
        clumped_worst = std::max(clumped_worst, mae(*e, [&](const FormatFootprint& f) { return t_axis(f, c); }));
    }
    d.decision = clumped_worst <= k_format_max_mae ? "axis_projected" : "escalate";
    return d;
}

// ---- Normals mode (crown-lighting L-P0a, §5.3, §5.5, K17) -------------------

NormalsReport measure_normals(const Probe& probe, const CrownVolume& volume, const ExtinctionModel& model,
                              float lowpass_sigma_m) {
    NormalsReport report;
    const CrownMesh& m = probe.mesh();
    const size_t n = m.sites.size();
    report.sites = int(n);
    report.lowpass_sigma_m = lowpass_sigma_m;
    report.directions = k_normals_directions;
    report.neighbours = k_normals_neighbours;
    report.model = model;
    if (n == 0) return report;

    // Bounding ellipsoid fitted to the leaf AABB (crown-lighting §5.3).
    vec3 llo{1e30f, 1e30f, 1e30f}, lhi{-1e30f, -1e30f, -1e30f};
    for (const Triangle& tr : m.triangles) {
        if (tr.site < 0) continue;
        for (const vec3& p : tr.p) {
            llo = {std::min(llo.x, p.x), std::min(llo.y, p.y), std::min(llo.z, p.z)};
            lhi = {std::max(lhi.x, p.x), std::max(lhi.y, p.y), std::max(lhi.z, p.z)};
        }
    }
    const vec3 centre = (llo + lhi) * 0.5f;
    const vec3 semi{
        std::max(0.5f * (lhi.x - llo.x), 1e-4f),
        std::max(0.5f * (lhi.y - llo.y), 1e-4f),
        std::max(0.5f * (lhi.z - llo.z), 1e-4f),
    };

    report.site_card.resize(n);
    report.site_ellipsoid.resize(n);
    report.site_march.resize(n);
    report.site_neg_grad.resize(n);
    report.site_inner_half.assign(n, 0);

    std::vector<float> ell_r(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        const vec3 p = m.sites[i].centroid;
        const vec3 q{(p.x - centre.x) / semi.x, (p.y - centre.y) / semi.y, (p.z - centre.z) / semi.z};
        ell_r[i] = length(q);
        const vec3 g{q.x / semi.x, q.y / semi.y, q.z / semi.z};
        report.site_ellipsoid[i] = length(g) > 1e-8f ? normalized(g) : vec3{0.0f, 0.0f, 1.0f};
        const vec3 cn = m.sites[i].normal;
        report.site_card[i] = length(cn) > 1e-8f ? normalized(cn) : vec3{0.0f, 0.0f, 1.0f};
    }

    // Inner half: the floor(N / 2) sites with smallest normalised ellipsoid radius
    // (tie-broken by site index for exact determinism).
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return ell_r[a] < ell_r[b] || (ell_r[a] == ell_r[b] && a < b);
    });
    const size_t inner_n = n / 2;
    report.inner_half_sites = int(inner_n);
    for (size_t r = 0; r < inner_n; ++r) report.site_inner_half[order[r]] = 1;

    // 64 full-sphere Fibonacci directions (deterministic golden-angle spiral).
    std::array<vec3, k_normals_directions> dirs;
    const double golden = k_pi * (3.0 - std::sqrt(5.0));
    for (int k = 0; k < k_normals_directions; ++k) {
        const double z = 1.0 - 2.0 * (double(k) + 0.5) / double(k_normals_directions);
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double ph = golden * double(k);
        dirs[size_t(k)] = vec3{float(r * std::cos(ph)), float(r * std::sin(ph)), float(z)};
    }

    // Warm the low-pass cache on the main thread before parallel site marches.
    if (lowpass_sigma_m > 0.0f)
        (void)volume.optical_depth(m.sites[0].centroid, dirs[0], lowpass_sigma_m, model);

    // 4 sources: 0 = card, 1 = ellipsoid, 2 = march_lowpass_0_6m, 3 = neg_grad.
    std::array<std::vector<uint8_t>, 4> in_occluded;
    for (auto& v : in_occluded) v.assign(n, 0);

    const float h = 0.5f * volume.cell_size();
    parallel_for(n, [&](size_t i) {
        const vec3 p = m.sites[i].centroid;
        std::array<double, k_normals_directions> tau{};
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int k = 0; k < k_normals_directions; ++k) {
            const double t = double(volume.optical_depth(p, dirs[size_t(k)], lowpass_sigma_m, model));
            tau[size_t(k)] = t;
            const double w = 1.0 / (1.0 + t);
            sx += w * double(dirs[size_t(k)].x);
            sy += w * double(dirs[size_t(k)].y);
            sz += w * double(dirs[size_t(k)].z);
        }
        const double sl = std::sqrt(sx * sx + sy * sy + sz * sz);
        report.site_march[i] = sl > 1e-12
            ? vec3{float(sx / sl), float(sy / sl), float(sz / sl)}
            : report.site_ellipsoid[i];

        // -grad: central finite difference of CrownVolume::density(p) at half a cell (§5.5).
        const float gx = volume.density(vec3{p.x + h, p.y, p.z}) - volume.density(vec3{p.x - h, p.y, p.z});
        const float gy = volume.density(vec3{p.x, p.y + h, p.z}) - volume.density(vec3{p.x, p.y - h, p.z});
        const float gz = volume.density(vec3{p.x, p.y, p.z + h}) - volume.density(vec3{p.x, p.y, p.z - h});
        const vec3 ng{-gx, -gy, -gz};
        const float ngl = length(ng);
        report.site_neg_grad[i] = ngl > 1e-8f ? (ng / ngl) : vec3{0.0f, 0.0f, 0.0f};

        const std::array<vec3, 4> cand = {
            report.site_card[i],
            report.site_ellipsoid[i],
            report.site_march[i],
            report.site_neg_grad[i],
        };
        for (size_t s = 0; s < 4; ++s) {
            const vec3 nd = cand[s];
            if (dot(nd, nd) < 0.5f) continue;
            double front_sum = 0.0, back_sum = 0.0;
            int front_n = 0, back_n = 0;
            for (int k = 0; k < k_normals_directions; ++k) {
                const float d = dot(dirs[size_t(k)], nd);
                if (d > 0.0f)      { front_sum += tau[size_t(k)]; ++front_n; }
                else if (d < 0.0f) { back_sum  += tau[size_t(k)]; ++back_n; }
            }
            if (front_n > 0 && back_n > 0 && (front_sum / double(front_n)) > (back_sum / double(back_n)))
                in_occluded[s][i] = 1;
        }
    });

    // Neighbour agreement: mean dot with k = min(8, N - 1) nearest other sites.
    const size_t k_nbr = std::min<size_t>(size_t(k_normals_neighbours), n > 0 ? n - 1 : 0);
    std::array<std::vector<float>, 4> agree;
    for (auto& v : agree) v.assign(n, 0.0f);
    if (k_nbr > 0) {
        const std::array<const std::vector<vec3>*, 4> src_vecs = {
            &report.site_card,
            &report.site_ellipsoid,
            &report.site_march,
            &report.site_neg_grad,
        };
        parallel_for(n, [&](size_t i) {
            const vec3 pi = m.sites[i].centroid;
            std::array<std::pair<float, size_t>, k_normals_neighbours> best;
            size_t count = 0;
            for (size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                const vec3 d = m.sites[j].centroid - pi;
                const float d2 = dot(d, d);
                if (count < k_nbr) {
                    best[count++] = {d2, j};
                    size_t c = count - 1;
                    while (c > 0 && (best[c].first < best[c - 1].first ||
                                     (best[c].first == best[c - 1].first && best[c].second < best[c - 1].second))) {
                        std::swap(best[c], best[c - 1]);
                        --c;
                    }
                } else if (d2 < best[k_nbr - 1].first ||
                           (d2 == best[k_nbr - 1].first && j < best[k_nbr - 1].second)) {
                    best[k_nbr - 1] = {d2, j};
                    size_t c = k_nbr - 1;
                    while (c > 0 && (best[c].first < best[c - 1].first ||
                                     (best[c].first == best[c - 1].first && best[c].second < best[c - 1].second))) {
                        std::swap(best[c], best[c - 1]);
                        --c;
                    }
                }
            }
            for (size_t s = 0; s < 4; ++s) {
                const auto& nv = *src_vecs[s];
                double sum = 0.0;
                for (size_t b = 0; b < count; ++b) sum += double(dot(nv[i], nv[best[b].second]));
                agree[s][i] = float(sum / double(count));
            }
        });
    }

    auto summarize = [&](size_t s, const std::vector<vec3>& nv) -> NormalSourceStats {
        NormalSourceStats st;
        size_t against = 0, inner_against = 0, hollow = 0, inner_hollow = 0, below = 0;
        double agree_sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const bool ag = dot(nv[i], report.site_ellipsoid[i]) < 0.0f;
            const bool hl = in_occluded[s][i] != 0;
            if (ag) ++against;
            if (hl) ++hollow;
            if (nv[i].z < 0.0f) ++below;
            if (report.site_inner_half[i]) {
                if (ag) ++inner_against;
                if (hl) ++inner_hollow;
            }
            agree_sum += double(agree[s][i]);
        }
        const double p_ag = double(against) / double(n);
        const double p_hl = double(hollow) / double(n);
        const double p_bl = double(below) / double(n);
        st.against_outward_share    = float(p_ag);
        st.against_outward_share_se = float(detail::se_share(p_ag, n));
        st.hollow_shell_share       = float(p_hl);
        st.hollow_shell_share_se    = float(detail::se_share(p_hl, n));
        st.below_horizontal_share    = float(p_bl);
        st.below_horizontal_share_se = float(detail::se_share(p_bl, n));
        if (inner_n > 0) {
            const double p_iag = double(inner_against) / double(inner_n);
            const double p_ihl = double(inner_hollow) / double(inner_n);
            st.inner_half_against_outward_share    = float(p_iag);
            st.inner_half_against_outward_share_se = float(detail::se_share(p_iag, inner_n));
            st.inner_half_hollow_shell_share       = float(p_ihl);
            st.inner_half_hollow_shell_share_se    = float(detail::se_share(p_ihl, inner_n));
        }
        st.neighbour_agreement    = k_nbr > 0 ? float(agree_sum / double(n)) : 1.0f;
        st.neighbour_agreement_se = k_nbr > 0 ? float(detail::se_mean(agree[s])) : 0.0f;
        return st;
    };

    report.card               = summarize(0, report.site_card);
    report.ellipsoid          = summarize(1, report.site_ellipsoid);
    report.march_lowpass_0_6m = summarize(2, report.site_march);
    report.neg_grad           = summarize(3, report.site_neg_grad);

    std::vector<float> angles(n, 0.0f);
    double ang_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double c = std::clamp(double(dot(report.site_ellipsoid[i], report.site_march[i])), -1.0, 1.0);
        const float deg = float(std::acos(c) * (180.0 / k_pi));
        angles[i] = deg;
        ang_sum += double(deg);
    }
    report.ellipsoid_vs_march_angle_deg_p50     = percentile(angles, 0.50);
    report.ellipsoid_vs_march_angle_deg_p50_se  = float(detail::se_quantile(angles, 0.50));
    report.ellipsoid_vs_march_angle_deg_p90     = percentile(angles, 0.90);
    report.ellipsoid_vs_march_angle_deg_p90_se  = float(detail::se_quantile(angles, 0.90));
    report.ellipsoid_vs_march_angle_deg_mean    = float(ang_sum / double(n));
    report.ellipsoid_vs_march_angle_deg_mean_se = float(detail::se_mean(angles));

    const bool angle_ok = report.ellipsoid_vs_march_angle_deg_p50 < k_normals_ellipsoid_angle_deg;
    const bool hollow_ok =
        std::abs(report.ellipsoid.hollow_shell_share - report.march_lowpass_0_6m.hollow_shell_share) <= k_normals_ellipsoid_pp &&
        std::abs(report.ellipsoid.inner_half_hollow_shell_share - report.march_lowpass_0_6m.inner_half_hollow_shell_share) <= k_normals_ellipsoid_pp;
    const bool skirt_ok =
        std::abs(report.ellipsoid.below_horizontal_share - report.march_lowpass_0_6m.below_horizontal_share) <= k_normals_ellipsoid_pp;
    report.decision = (angle_ok && hollow_ok && skirt_ok) ? "ellipsoid" : "march_lowpass_0_6m";
    return report;
}

// ---- The report ----------------------------------------------------------------

namespace {

struct SiteLight {
    std::vector<uint8_t> lit;   // per site: sees the sun (own site excluded)
};

SiteLight light_sites(const Probe& probe, vec3 sun) {
    const auto& sites = probe.mesh().sites;
    SiteLight r;
    r.lit.assign(sites.size(), 0);
    parallel_for(sites.size(), [&](size_t i) {
        r.lit[i] = probe.occluded(sites[i].centroid, sun, sites[i].id, Occluders::leaves_and_bark) ? 0 : 1;
    });
    return r;
}

// Sites ordered nearest the sun first (largest projection on the sun
// direction), ties by id; split into k groups of near-equal size.
struct RankGroup {
    std::optional<float> share;   // nullopt: an empty group
    int                  sites = 0;
};
std::vector<RankGroup> lit_share_by_rank(const Probe& probe, const SiteLight& L, vec3 sun, int k) {
    const auto& sites = probe.mesh().sites;
    std::vector<uint32_t> order(sites.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return dot(sites[a].centroid, sun) > dot(sites[b].centroid, sun);
    });
    std::vector<int> n(size_t(k), 0), lit(size_t(k), 0);
    for (size_t r = 0; r < order.size(); ++r) {
        const size_t g = r * size_t(k) / order.size();
        ++n[g];
        lit[g] += L.lit[order[r]];
    }
    std::vector<RankGroup> out;
    for (int g = 0; g < k; ++g)
        out.push_back({n[size_t(g)] ? std::optional<float>(float(lit[size_t(g)]) / float(n[size_t(g)])) : std::nullopt, n[size_t(g)]});
    return out;
}

void write_body(Json& j, const std::optional<RegionStats>& b) {
    if (!b) { j.null(); return; }
    j.obj();
    j.key("footprints").integer(b->footprints);
    j.key("footprints_se").num(b->footprints_se);
    j.key("mean").num(b->mean);
    j.key("mean_se").num(b->mean_se);
    j.key("p50").num(b->p50);
    j.key("p50_se").num(b->p50_se);
    j.key("p95").num(b->p95);
    j.key("p95_se").num(b->p95_se);
    j.key("share_below_0_05").num(b->share_opaque);
    j.key("share_below_0_05_se").num(b->share_opaque_se);
    j.end();
}

// Precompute body stats at the 4 min-edge thresholds once (reusing `cse_1m` for d == k_body_edge_m).
std::array<std::optional<RegionStats>, 4> body_by_edge_stats(const FootprintSet& set, const CountSEs& cse_1m) {
    constexpr float k_edges[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    std::array<std::optional<RegionStats>, 4> out;
    for (size_t i = 0; i < 4; ++i)
        out[i] = body_stats_impl(set, k_edges[i], k_edges[i] == k_body_edge_m ? &cse_1m : nullptr, true);
    return out;
}

// {"<d>": mean} and its {"<d>": se} sibling.
void write_body_by_edge(Json& j, const std::array<std::optional<RegionStats>, 4>& stats, bool se) {
    constexpr const char* k_names[4] = {"0.5", "1.0", "1.5", "2.0"};
    j.obj();
    for (size_t i = 0; i < 4; ++i)
        j.key(k_names[i]).opt(stats[i] ? std::optional<float>(se ? stats[i]->mean_se : stats[i]->mean) : std::nullopt);
    j.end();
}

void write_binned(Json& j, const std::vector<std::pair<float, float>>& value_t, float bin, const char* lo_key, const char* hi_key) {
    std::map<int, std::vector<float>> bins;
    for (const auto& [x, t] : value_t) bins[int(std::floor(x / bin))].push_back(t);
    j.arr();
    for (const auto& [k, ts] : bins) {
        double sum = 0.0;
        for (float t : ts) sum += t;
        j.obj();
        j.key(lo_key).num(k * bin);
        j.key(hi_key).num((k + 1) * bin);
        j.key("footprints").integer(static_cast<long long>(ts.size()));
        j.key("mean").num(sum / double(ts.size()));
        j.key("mean_se").num(detail::se_mean_t(ts, rays_per_side(RayStream::selection) * rays_per_side(RayStream::selection)));
        j.end();
    }
    j.end();
}

}  // namespace

std::string probe_report_json(const Probe& probe, const ProbeOptions& opt, const std::string& input_json) {
    const CrownMesh& m = probe.mesh();
    const auto suns = sun_directions(opt.elevations_deg, opt.azimuths);
    Json j;
    j.obj();
    j.key("schema").integer(k_report_schema);
    j.key("input").raw(input_json);

    // ---- tree ------------------------------------------------------------------
    double card_area = 0.0, opaque_area = 0.0;
    vec3 llo{1e30f, 1e30f, 1e30f}, lhi{-1e30f, -1e30f, -1e30f};
    float bark_zmin = 1e30f;
    for (uint32_t t = 0; t < m.triangles.size(); ++t) {
        const Triangle& tr = m.triangles[t];
        if (tr.site < 0) { for (const vec3& p : tr.p) bark_zmin = std::min(bark_zmin, p.z); continue; }
        const double a = 0.5 * length(cross(tr.p[1] - tr.p[0], tr.p[2] - tr.p[0]));
        card_area += a;
        opaque_area += a * probe.triangle_coverage(t);
        for (const vec3& p : tr.p) {
            llo = {std::min(llo.x, p.x), std::min(llo.y, p.y), std::min(llo.z, p.z)};
            lhi = {std::max(lhi.x, p.x), std::max(lhi.y, p.y), std::max(lhi.z, p.z)};
        }
    }
    const bool leafy = m.leaf_triangles > 0;
    const FootprintSet zenith = measure_footprints(probe, sun_direction(90.0f, 0.0f), PlaneKind::beam,
                                                   Occluders::leaves_and_bark, opt.selection_stream);
    int zenith_sil = 0;
    for (const Footprint& f : zenith.footprints) zenith_sil += f.silhouette() ? 1 : 0;
    const double fp_area = double(k_footprint_m) * k_footprint_m;
    const double ground_area = zenith_sil * fp_area;
    const double zenith_sil_se = footprint_count_se(zenith, k_body_edge_m, false).silhouette_se;
    const double ground_area_se = zenith_sil_se * fp_area;
    std::map<int, int> cards_hist;
    for (const Site& s : m.sites) ++cards_hist[s.cards];

    j.key("tree").obj();
    j.key("bark_triangles").integer(m.bark_triangles);
    j.key("leaf_triangles").integer(m.leaf_triangles);
    j.key("sites").integer(static_cast<long long>(m.sites.size()));
    j.key("cards_per_site").obj();
    for (const auto& [c, n] : cards_hist) j.key(std::to_string(c).c_str()).integer(n);
    j.end();
    j.key("alpha_coverage");     if (leafy) j.num(opaque_area / card_area); else j.null();
    j.key("card_area_m2").num(card_area);
    j.key("opaque_leaf_area_m2").num(opaque_area);
    j.key("crown_ground_projected_area_m2").num(ground_area);
    j.key("crown_ground_projected_area_m2_se").num(ground_area_se);
    j.key("leaf_area_index");    if (ground_area > 0.0) j.num(opaque_area / ground_area); else j.null();
    j.key("leaf_area_index_se"); if (ground_area > 0.0) j.num((opaque_area / (ground_area * ground_area)) * ground_area_se); else j.null();
    if (leafy) {
        const vec3 e = lhi - llo;
        j.key("leaf_aabb_min_m").vec(llo);
        j.key("leaf_aabb_max_m").vec(lhi);
        j.key("leaf_area_density_m2_per_m3").num(opaque_area / (double(e.x) * e.y * e.z));
    } else {
        j.key("leaf_aabb_min_m").null();
        j.key("leaf_aabb_max_m").null();
        j.key("leaf_area_density_m2_per_m3").null();
    }
    j.key("trunk_base_z_m"); if (m.bark_triangles) j.num(bark_zmin); else j.null();
    j.key("lowest_leaf_height_m"); if (leafy && m.bark_triangles) j.num(llo.z - bark_zmin); else j.null();
    j.end();

    // Per-direction site lighting, shared by oracle and sunlit modes.
    std::vector<SiteLight> light;
    if (opt.oracle || opt.sunlit)
        for (const SunDirection& s : suns) light.push_back(light_sites(probe, s.dir));

    std::optional<CrownVolume> shared_volume;
    ExtinctionModel normals_model;
    bool have_decided_model = false;

    j.key("modes").obj();
    if (opt.oracle) {
        j.key("oracle").obj();
        j.key("footprint_m").num(k_footprint_m);
        j.key("rays_per_footprint").integer(k_rays_per_side * k_rays_per_side);
        j.key("trunk_axis").str("object z axis through the origin");
        j.key("occluders").str(occluders_name(Occluders::leaves_and_bark));
        std::vector<FormatElevation> fmt;
        if (leafy) shared_volume = CrownVolume::build(probe, k_format_cells);
        std::optional<CrownVolume>& volume = shared_volume;
        std::vector<std::pair<float, bool>> opaque_by_direction;

        j.key("directions").arr();
        for (size_t si = 0; si < suns.size(); ++si) {
            const SunDirection& s = suns[si];
            const FootprintSet beam = measure_footprints(probe, s, PlaneKind::beam, Occluders::leaves_and_bark, opt.selection_stream);
            const FootprintSet ground = measure_footprints(probe, s, PlaneKind::ground, Occluders::leaves_and_bark, opt.selection_stream);
            int sil = 0, rim = 0;
            std::vector<float> cg_s, co_s, cg_b, co_b;   // per-footprint mean crossings
            std::vector<std::pair<float, float>> by_edge, by_trunk;
            for (const Footprint& f : beam.footprints) {
                if (!f.silhouette()) continue;
                ++sil;
                if (!f.body()) ++rim;
                cg_s.push_back(f.crossings_geometric); co_s.push_back(f.crossings_opaque);
                if (f.body()) { cg_b.push_back(f.crossings_geometric); co_b.push_back(f.crossings_opaque); }
                by_edge.push_back({f.edge_distance, f.t});
            }
            auto mean_of = [](const std::vector<float>& v) {
                double s = 0.0;
                for (float x : v) s += x;
                return s / double(v.size());
            };
            auto crossings = [&](const std::vector<float>& g, const std::vector<float>& o) {
                if (g.empty()) { j.null(); return; }
                j.obj();
                j.key("geometric").num(mean_of(g)); j.key("geometric_se").num(detail::se_mean(g));
                j.key("opaque").num(mean_of(o));    j.key("opaque_se").num(detail::se_mean(o));
                j.end();
            };
            if (s.elevation_deg >= 90.0f)
                for (const Footprint& f : beam.footprints) by_trunk.push_back({std::hypot(f.centre.x, f.centre.y), f.t});
            const CountSEs beam_cse = footprint_count_se(beam, k_body_edge_m);
            const auto beam_by_edge = body_by_edge_stats(beam, beam_cse);
            const std::optional<RegionStats>& body = beam_by_edge[1];
            opaque_by_direction.push_back({s.elevation_deg, body && body->share_opaque >= k_stop_body_share});

            j.obj();
            j.key("elevation_deg").num(s.elevation_deg);
            j.key("azimuth_deg").num(s.azimuth_deg);
            j.key("footprints").integer(static_cast<long long>(beam.footprints.size()));
            j.key("silhouette").integer(sil);
            j.key("silhouette_se").num(beam_cse.silhouette_se);
            j.key("rim").integer(rim);
            j.key("rim_se").num(beam_cse.rim_se);
            j.key("body"); write_body(j, body);
            j.key("body_opaque"); if (body) j.boolean(opaque_by_direction.back().second); else j.null();
            j.key("body_mean_by_min_edge_m"); write_body_by_edge(j, beam_by_edge, false);
            j.key("body_mean_by_min_edge_m_se"); write_body_by_edge(j, beam_by_edge, true);
            j.key("t_by_edge_distance"); write_binned(j, by_edge, k_edge_bin_m, "d_min_m", "d_max_m");
            j.key("crossings").obj();
            j.key("silhouette"); crossings(cg_s, co_s);
            j.key("body");       crossings(cg_b, co_b);
            j.end();
            j.key("ground_plane").obj();
            {
                int gsil = 0;
                for (const Footprint& f : ground.footprints) gsil += f.silhouette() ? 1 : 0;
                const CountSEs ground_cse = footprint_count_se(ground, k_body_edge_m);
                const auto ground_by_edge = body_by_edge_stats(ground, ground_cse);
                j.key("footprints").integer(static_cast<long long>(ground.footprints.size()));
                j.key("silhouette").integer(gsil);
                j.key("silhouette_se").num(ground_cse.silhouette_se);
                j.key("body"); write_body(j, ground_by_edge[1]);
                j.key("body_mean_by_min_edge_m"); write_body_by_edge(j, ground_by_edge, false);
                j.key("body_mean_by_min_edge_m_se"); write_body_by_edge(j, ground_by_edge, true);
            }
            j.end();
            j.key("t_by_trunk_distance");
            if (!by_trunk.empty()) write_binned(j, by_trunk, k_trunk_bin_m, "r_min_m", "r_max_m"); else j.null();
            // Sunlit share of sites by depth into the crown along the sun:
            // depth = max over leaf vertices of dot(p, sun) - dot(site, sun).
            j.key("sunlit_by_depth");
            if (!m.sites.empty()) {
                float top = -1e30f;
                for (const Triangle& t : m.triangles) if (t.site >= 0) for (const vec3& p : t.p) top = std::max(top, dot(p, s.dir));
                std::map<int, std::pair<int, int>> bins;
                for (const Site& st : m.sites) {
                    auto& b = bins[int(std::floor((top - dot(st.centroid, s.dir)) / k_depth_bin_m))];
                    ++b.first;
                    b.second += light[si].lit[size_t(st.id)];
                }
                j.arr();
                for (const auto& [k, b] : bins) {
                    j.obj();
                    j.key("depth_min_m").num(k * k_depth_bin_m);
                    j.key("depth_max_m").num((k + 1) * k_depth_bin_m);
                    j.key("sites").integer(b.first);
                    j.key("sunlit_share").num(double(b.second) / b.first);
                    j.key("sunlit_share_se").num(detail::se_share(double(b.second) / b.first, size_t(b.first)));
                    j.end();
                }
                j.end();
            } else j.null();
            j.end();

            if (volume) {
                auto it = std::find_if(fmt.begin(), fmt.end(), [&](const FormatElevation& e) { return e.elevation_deg == s.elevation_deg; });
                if (it == fmt.end()) { fmt.push_back({}); it = fmt.end() - 1; }
                // The format rule's own population (R-CR1): selected and scored with bark removed.
                const FootprintSet leaves = measure_footprints(probe, s, PlaneKind::beam, Occluders::leaves_only, opt.selection_stream);
                add_format_footprints(probe, leaves, *volume, *it);
            }
        }
        j.end();

        // Stop rule (plan S-P0a; stop_rule() states it).
        const StopRule rule = stop_rule(opaque_by_direction);
        j.key("stop_rule").obj();
        j.key("body_opaque_by_elevation").arr();
        for (const auto& [e, all] : rule.opaque_by_elevation) {
            j.obj(); j.key("elevation_deg").num(e); j.key("body_opaque_every_azimuth").boolean(all); j.end();
        }
        j.end();
        j.key("crown_opaque_to_direct_sun").boolean(rule.crown_opaque_to_direct_sun);
        j.end();

        j.key("format");
        if (volume) {
            const FormatDecision d = decide_format(fmt);
            if (d.decision != "undecidable") {
                if (d.decision == "scalar") {
                    normals_model = ExtinctionModel{ExtinctionKind::scalar, d.g, 1.0f};
                } else {
                    normals_model = ExtinctionModel{ExtinctionKind::tensor, 0.5f, d.clumping};
                }
                have_decided_model = true;
            }
            j.obj();
            j.key("cells_longest_axis").integer(k_format_cells);
            j.key("rays_per_footprint").integer(k_format_rays_per_side * k_format_rays_per_side);
            j.key("occluders").str(occluders_name(Occluders::leaves_only));
            j.key("cell_m").num(volume->cell_size());
            j.key("decision").str(d.decision);
            j.key("g");    if (d.decision != "undecidable") j.num(d.g); else j.null();
            j.key("g_se"); if (d.decision != "undecidable") j.num(d.g_se); else j.null();
            j.key("clumping").num(d.clumping);
            j.key("clumping_se").num(d.clumping_se);
            j.key("rows").arr();
            for (const FormatRow& r : d.rows) {
                j.obj();
                j.key("elevation_deg").num(r.elevation_deg);
                j.key("unsaturated_footprints").integer(r.unsaturated);
                j.key("unsaturated_footprints_se").num(r.unsaturated_se);
                j.key("measurable").boolean(r.measurable);
                j.key("saturated_share").num(r.saturated_share);
                j.key("saturated_share_se").num(r.saturated_share_se);
                const bool fitted = r.measurable && d.decision != "undecidable";
                j.key("scalar_mae");    if (fitted) j.num(r.scalar_mae);    else j.null();
                j.key("scalar_mae_se"); if (fitted) j.num(r.scalar_mae_se); else j.null();
                j.key("axis_mae");      if (fitted) j.num(r.axis_mae);      else j.null();
                j.key("axis_mae_se");   if (fitted) j.num(r.axis_mae_se);   else j.null();
                j.key("g_eff");         if (fitted) j.num(r.g_eff);         else j.null();
                j.key("g_eff_se");      if (fitted) j.num(r.g_eff_se);      else j.null();
                j.end();
            }
            j.end();
            j.end();
        } else j.null();
        j.end();
    }

    if (opt.sunlit) {
        j.key("sunlit").obj();
        j.key("sites").integer(static_cast<long long>(m.sites.size()));
        // Share of directions (64, Fibonacci over the upper hemisphere) in which
        // a site sees open sky, own site excluded.
        j.key("open_sky");
        if (!m.sites.empty()) {
            std::vector<vec3> dirs;
            const double golden = k_pi * (3.0 - std::sqrt(5.0));
            for (int k = 0; k < k_open_sky_directions; ++k) {
                const double z = (k + 0.5) / k_open_sky_directions, r = std::sqrt(1.0 - z * z), ph = golden * k;
                dirs.push_back(vec3{float(r * std::cos(ph)), float(r * std::sin(ph)), float(z)});
            }
            std::vector<float> share(m.sites.size());
            parallel_for(m.sites.size(), [&](size_t i) {
                int open = 0;
                for (const vec3& d : dirs) open += probe.occluded(m.sites[i].centroid, d, m.sites[i].id, Occluders::leaves_and_bark) ? 0 : 1;
                share[i] = float(open) / float(dirs.size());
            });
            double sum = 0.0;
            for (float x : share) sum += x;
            j.obj();
            j.key("directions").integer(k_open_sky_directions);
            j.key("mean").num(sum / double(share.size()));
            j.key("mean_se").num(detail::se_mean(share));
            j.key("p10").num(percentile(share, 0.10));
            j.key("p10_se").num(detail::se_quantile(share, 0.10));
            j.key("p50").num(percentile(share, 0.50));
            j.key("p50_se").num(detail::se_quantile(share, 0.50));
            j.key("p90").num(percentile(share, 0.90));
            j.key("p90_se").num(detail::se_quantile(share, 0.90));
            j.end();
        } else j.null();

        j.key("directions").arr();
        for (size_t si = 0; si < suns.size(); ++si) {
            const SunDirection& s = suns[si];
            j.obj();
            j.key("elevation_deg").num(s.elevation_deg);
            j.key("azimuth_deg").num(s.azimuth_deg);
            int lit = 0;
            for (uint8_t x : light[si].lit) lit += x;
            const double share = m.sites.empty() ? 0.0 : double(lit) / double(m.sites.size());
            j.key("share");    if (!m.sites.empty()) j.num(share); else j.null();
            j.key("share_se"); if (!m.sites.empty()) j.num(detail::se_share(share, m.sites.size())); else j.null();
            for (const auto& [name, k] : std::vector<std::pair<const char*, int>>{{"quartiles", 4}, {"deciles", 10}}) {
                const auto groups = lit_share_by_rank(probe, light[si], s.dir, k);
                j.key(name).arr(); for (const auto& g : groups) j.opt(g.share); j.end();
                j.key((std::string(name) + "_se").c_str()).arr();
                for (const auto& g : groups) j.opt(g.share ? std::optional<float>(float(detail::se_share(*g.share, size_t(g.sites)))) : std::nullopt);
                j.end();
            }
            j.key("eyes").arr();
            for (const vec3& eye : opt.eyes) {
                std::vector<uint8_t> vis(m.sites.size(), 0);
                parallel_for(m.sites.size(), [&](size_t i) {
                    vis[i] = probe.occluded_segment(eye, m.sites[i].centroid, m.sites[i].id, Occluders::leaves_and_bark) ? 0 : 1;
                });
                int nv = 0, nl = 0;
                for (size_t i = 0; i < vis.size(); ++i) if (vis[i]) { ++nv; nl += light[si].lit[i]; }
                j.obj();
                j.key("eye_m").vec(eye);
                j.key("visible_sites").integer(nv);
                j.key("share");    if (nv) j.num(double(nl) / nv); else j.null();
                j.key("share_se"); if (nv) j.num(detail::se_share(double(nl) / nv, size_t(nv))); else j.null();
                j.end();
            }
            j.end();
            j.end();
        }
        j.end();
        j.end();
    }

    if (opt.normals) {
        auto write_normal_source = [&](const NormalSourceStats& st) {
            j.obj();
            j.key("against_outward_share").num(st.against_outward_share);
            j.key("against_outward_share_se").num(st.against_outward_share_se);
            j.key("inner_half_against_outward_share").num(st.inner_half_against_outward_share);
            j.key("inner_half_against_outward_share_se").num(st.inner_half_against_outward_share_se);
            j.key("neighbour_agreement").num(st.neighbour_agreement);
            j.key("neighbour_agreement_se").num(st.neighbour_agreement_se);
            j.key("hollow_shell_share").num(st.hollow_shell_share);
            j.key("hollow_shell_share_se").num(st.hollow_shell_share_se);
            j.key("inner_half_hollow_shell_share").num(st.inner_half_hollow_shell_share);
            j.key("inner_half_hollow_shell_share_se").num(st.inner_half_hollow_shell_share_se);
            j.key("below_horizontal_share").num(st.below_horizontal_share);
            j.key("below_horizontal_share_se").num(st.below_horizontal_share_se);
            j.end();
        };
        j.key("normals").obj();
        j.key("sites").integer(static_cast<long long>(m.sites.size()));
        j.key("lowpass_sigma_m").num(k_normals_lowpass_sigma_m);
        j.key("directions").integer(k_normals_directions);
        j.key("neighbours").integer(k_normals_neighbours);
        if (leafy && !m.sites.empty()) {
            if (!shared_volume) shared_volume = CrownVolume::build(probe, k_format_cells);
            const ExtinctionModel model = have_decided_model ? normals_model : shared_volume->model();
            const NormalsReport nr = measure_normals(probe, *shared_volume, model, k_normals_lowpass_sigma_m);
            j.key("inner_half_sites").integer(nr.inner_half_sites);
            j.key("extinction_kind").str(nr.model.kind == ExtinctionKind::scalar ? "scalar" : "tensor");
            j.key("extinction_g").num(nr.model.g);
            j.key("extinction_clumping").num(nr.model.clumping);
            j.key("sources").obj();
            j.key("card");               write_normal_source(nr.card);
            j.key("ellipsoid");          write_normal_source(nr.ellipsoid);
            j.key("march_lowpass_0_6m"); write_normal_source(nr.march_lowpass_0_6m);
            j.key("neg_grad");           write_normal_source(nr.neg_grad);
            j.end();
            j.key("ellipsoid_vs_march_angle_deg").obj();
            j.key("p50").num(nr.ellipsoid_vs_march_angle_deg_p50);
            j.key("p50_se").num(nr.ellipsoid_vs_march_angle_deg_p50_se);
            j.key("p90").num(nr.ellipsoid_vs_march_angle_deg_p90);
            j.key("p90_se").num(nr.ellipsoid_vs_march_angle_deg_p90_se);
            j.key("mean").num(nr.ellipsoid_vs_march_angle_deg_mean);
            j.key("mean_se").num(nr.ellipsoid_vs_march_angle_deg_mean_se);
            j.end();
            j.key("decision").str(nr.decision);
        } else {
            j.key("inner_half_sites").integer(0);
            j.key("extinction_kind").null();
            j.key("extinction_g").null();
            j.key("extinction_clumping").null();
            j.key("sources").null();
            j.key("ellipsoid_vs_march_angle_deg").null();
            j.key("decision").null();
        }
        j.end();
    }
    j.end();   // modes
    j.end();
    return j.take();
}

std::string probe_glb_report(std::span<const uint8_t> glb, int lod, const ProbeOptions& options,
                             const ProbeInput& input) {
    LoadedCrown loaded = load_crown_from_glb(glb, lod);
    const Probe probe(std::move(loaded.mesh));
    return probe_report_json(probe, options, format_probe_input(input, glb, lod));
}

}  // namespace treegen::crown
