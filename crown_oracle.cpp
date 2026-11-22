// NOTE: /fp:precise like the rest of TreegenCore: the report is compared byte
// for byte between the scenario path and --glb.
#include "crown_oracle.hpp"

#include "det_rng.hpp"

#include <algorithm>
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

// ---- SE (R-CR6; the definitions are stated at RegionStats) ----------------------
// Mean of footprint T, each the open share of `rays` jittered rays.
double se_mean_t(const std::vector<float>& t, int rays) {
    if (t.empty()) return 0.0;
    double v = 0.0;
    for (float x : t) v += double(x) * (1.0 - x);
    return std::sqrt(v / rays) / double(t.size());
}
// A share p over n footprints or sites.
double se_share(double p, size_t n) { return n ? std::sqrt(std::max(0.0, p * (1.0 - p)) / double(n)) : 0.0; }
// Mean over n independent items.
template <class T> double se_mean(const std::vector<T>& x) {
    if (x.size() < 2) return 0.0;
    double m = 0.0;
    for (const T& v : x) m += double(v);
    m /= double(x.size());
    double s = 0.0;
    for (const T& v : x) s += (double(v) - m) * (double(v) - m);
    return std::sqrt(s / double(x.size() - 1) / double(x.size()));
}
// Quantile q of n values: half the spread of the order statistics one binomial
// SE (sqrt(n q (1 - q)) ranks) either side of its rank.
double se_quantile(std::vector<float> v, double q) {
    if (v.size() < 2) return 0.0;
    std::sort(v.begin(), v.end());
    const double n = double(v.size()), d = std::sqrt(n * q * (1.0 - q));
    auto at = [&](double rank) { return double(v[size_t(std::clamp(rank, 0.0, n - 1.0))]); };
    return 0.5 * (at(std::ceil(q * n + d) - 1.0) - at(std::ceil(q * n - d) - 1.0));
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

}  // namespace

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

std::string probe_input_scenario(const std::string& scenario_path, uint64_t seed, std::span<const uint8_t> glb, int lod) {
    return "{\"source\": \"scenario\", \"scenario\": " + json_string(scenario_path) + ", \"seed\": " +
           std::to_string(seed) + ", \"glb_fnv1a64\": " + json_string(glb_digest(glb)) + ", \"probe_lod\": " +
           std::to_string(lod) + "}";
}

std::string probe_input_glb(const std::string& glb_path, std::span<const uint8_t> glb, int lod) {
    return "{\"source\": \"glb\", \"glb\": " + json_string(glb_path) + ", \"glb_fnv1a64\": " + json_string(glb_digest(glb)) +
           ", \"probe_lod\": " + std::to_string(lod) + "}";
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
    for (float e : elevations_deg)
        for (int k = 0; k < azimuths; ++k) out.push_back(sun_direction(e, 360.0f * float(k) / float(azimuths)));
    return out;
}

int rays_per_side(RayStream s) { return s == RayStream::format ? k_format_rays_per_side : k_rays_per_side; }
uint64_t stream_id(RayStream s) { return s == RayStream::format ? k_format_ray_stream : k_footprint_ray_stream; }

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

FootprintSet measure_footprints(const Probe& probe, const SunDirection& sun, PlaneKind kind, Occluders occ) {
    FootprintSet set;
    set.sun = sun;
    set.kind = kind;
    set.occluders = occ;
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
        set.rays(k, rays, RayStream::selection);
        int open = 0;
        long geo = 0, opq = 0;
        for (const vec3& o : rays) {
            const LineHits h = probe.line_hits(o, s, t_max);
            if (!h.occluded(occ)) ++open;
            geo += h.leaf_geometric;
            opq += h.leaf_opaque;
        }
        Footprint& f = set.footprints[k];
        f.t = float(open) / float(rays.size());
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

std::optional<RegionStats> body_stats(const FootprintSet& set, float min_edge_m) {
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
    r.mean_se = float(se_mean_t(t, rays_per_side(RayStream::selection) * rays_per_side(RayStream::selection)));
    r.share_opaque_se = float(se_share(r.share_opaque, t.size()));
    r.p50_se = float(se_quantile(t, 0.50));
    r.p95_se = float(se_quantile(t, 0.95));
    return r;
}

bool body_opaque(const FootprintSet& beam) {
    const auto body = body_stats(beam, k_body_edge_m);
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
        r.measurable = r.unsaturated >= k_min_unsaturated;
        r.saturated_share = e.silhouette > 0 ? float(double(e.saturated) / e.silhouette) : 0.0f;
        r.saturated_share_se = float(se_share(r.saturated_share, size_t(e.silhouette)));
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
        for (const auto& f : e.unsaturated) err.push_back(std::abs(model(f) - f.t_oracle));
        return float(se_mean(err));
    };
    // Linearised least-squares SE of a one-parameter fit T(p) (definition at FormatRow::g_eff_se).
    auto fit_se = [&](const std::vector<const FormatElevation*>& es, auto&& model, auto&& dmodel) {
        double s = 0.0, j2 = 0.0;
        size_t n = 0;
        for (const auto* e : es)
            for (const auto& f : e->unsaturated) { const double r = model(f) - f.t_oracle; s += r * r; j2 += dmodel(f) * dmodel(f); ++n; }
        return n > 1 && j2 > 0.0 ? float(std::sqrt(s / double(n - 1) / j2)) : 0.0f;
    };
    auto dt_scalar = [](const FormatFootprint& f, double g) {
        double s = 0.0;
        for (const auto& r : f.integrals) s -= r[0] * std::exp(-g * r[0]);
        return s / double(f.integrals.size());
    };
    auto dt_axis = [](const FormatFootprint& f, double c) {
        double s = 0.0;
        for (const auto& r : f.integrals) s -= r[1] * std::exp(-c * r[1]);
        return s / double(f.integrals.size());
    };

    d.g = float(golden_min([&](double g) { return sse(measurable, [&](const FormatFootprint& f) { return t_scalar(f, g); }); },
                           0.01, 2.0));
    {
        const double g = d.g;
        d.g_se = fit_se(measurable, [&](const FormatFootprint& f) { return t_scalar(f, g); },
                        [&](const FormatFootprint& f) { return dt_scalar(f, g); });
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
                            [&](const FormatFootprint& f) { return dt_scalar(f, ge); });
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
                               [&](const FormatFootprint& f) { return dt_axis(f, c); });
    }
    float clumped_worst = 0.0f;
    for (const auto* e : measurable) {
        const double c = d.clumping;
        clumped_worst = std::max(clumped_worst, mae(*e, [&](const FormatFootprint& f) { return t_axis(f, c); }));
    }
    d.decision = clumped_worst <= k_format_max_mae ? "axis_projected" : "escalate";
    return d;
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

// {"<d>": mean} and its {"<d>": se} sibling.
void write_body_by_edge(Json& j, const FootprintSet& set, bool se) {
    j.obj();
    for (const auto& [name, d] : std::vector<std::pair<const char*, float>>{{"0.5", 0.5f}, {"1.0", 1.0f}, {"1.5", 1.5f}, {"2.0", 2.0f}}) {
        const auto b = body_stats(set, d);
        j.key(name).opt(b ? std::optional<float>(se ? b->mean_se : b->mean) : std::nullopt);
    }
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
        j.key("mean_se").num(se_mean_t(ts, rays_per_side(RayStream::selection) * rays_per_side(RayStream::selection)));
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
    const FootprintSet zenith = measure_footprints(probe, sun_direction(90.0f, 0.0f), PlaneKind::beam, Occluders::leaves_and_bark);
    int zenith_sil = 0;
    for (const Footprint& f : zenith.footprints) zenith_sil += f.silhouette() ? 1 : 0;
    const double ground_area = zenith_sil * double(k_footprint_m) * k_footprint_m;
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
    j.key("leaf_area_index");    if (ground_area > 0.0) j.num(opaque_area / ground_area); else j.null();
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

    // Per-direction site lighting, shared by both modes.
    std::vector<SiteLight> light;
    if (opt.oracle || opt.sunlit)
        for (const SunDirection& s : suns) light.push_back(light_sites(probe, s.dir));

    j.key("modes").obj();
    if (opt.oracle) {
        j.key("oracle").obj();
        j.key("footprint_m").num(k_footprint_m);
        j.key("rays_per_footprint").integer(k_rays_per_side * k_rays_per_side);
        j.key("trunk_axis").str("object z axis through the origin");
        j.key("occluders").str(occluders_name(Occluders::leaves_and_bark));
        std::vector<FormatElevation> fmt;
        std::optional<CrownVolume> volume;
        if (leafy) volume = CrownVolume::build(probe, k_format_cells);
        std::vector<std::pair<float, bool>> opaque_by_direction;

        j.key("directions").arr();
        for (size_t si = 0; si < suns.size(); ++si) {
            const SunDirection& s = suns[si];
            const FootprintSet beam = measure_footprints(probe, s, PlaneKind::beam, Occluders::leaves_and_bark);
            const FootprintSet ground = measure_footprints(probe, s, PlaneKind::ground, Occluders::leaves_and_bark);
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
                j.key("geometric").num(mean_of(g)); j.key("geometric_se").num(se_mean(g));
                j.key("opaque").num(mean_of(o));    j.key("opaque_se").num(se_mean(o));
                j.end();
            };
            if (s.elevation_deg >= 90.0f)
                for (const Footprint& f : beam.footprints) by_trunk.push_back({std::hypot(f.centre.x, f.centre.y), f.t});
            const auto body = body_stats(beam, k_body_edge_m);
            opaque_by_direction.push_back({s.elevation_deg, crown::body_opaque(beam)});

            j.obj();
            j.key("elevation_deg").num(s.elevation_deg);
            j.key("azimuth_deg").num(s.azimuth_deg);
            j.key("footprints").integer(static_cast<long long>(beam.footprints.size()));
            j.key("silhouette").integer(sil);
            j.key("rim").integer(rim);
            j.key("body"); write_body(j, body);
            j.key("body_opaque"); if (body) j.boolean(opaque_by_direction.back().second); else j.null();
            j.key("body_mean_by_min_edge_m"); write_body_by_edge(j, beam, false);
            j.key("body_mean_by_min_edge_m_se"); write_body_by_edge(j, beam, true);
            j.key("t_by_edge_distance"); write_binned(j, by_edge, k_edge_bin_m, "d_min_m", "d_max_m");
            j.key("crossings").obj();
            j.key("silhouette"); crossings(cg_s, co_s);
            j.key("body");       crossings(cg_b, co_b);
            j.end();
            j.key("ground_plane").obj();
            {
                int gsil = 0;
                for (const Footprint& f : ground.footprints) gsil += f.silhouette() ? 1 : 0;
                j.key("footprints").integer(static_cast<long long>(ground.footprints.size()));
                j.key("silhouette").integer(gsil);
                j.key("body"); write_body(j, body_stats(ground, k_body_edge_m));
                j.key("body_mean_by_min_edge_m"); write_body_by_edge(j, ground, false);
                j.key("body_mean_by_min_edge_m_se"); write_body_by_edge(j, ground, true);
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
                    j.key("sunlit_share_se").num(se_share(double(b.second) / b.first, size_t(b.first)));
                    j.end();
                }
                j.end();
            } else j.null();
            j.end();

            if (volume) {
                auto it = std::find_if(fmt.begin(), fmt.end(), [&](const FormatElevation& e) { return e.elevation_deg == s.elevation_deg; });
                if (it == fmt.end()) { fmt.push_back({}); it = fmt.end() - 1; }
                // The format rule's own population (R-CR1): selected and scored with bark removed.
                const FootprintSet leaves = measure_footprints(probe, s, PlaneKind::beam, Occluders::leaves_only);
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
            j.key("mean_se").num(se_mean(share));
            j.key("p10").num(percentile(share, 0.10));
            j.key("p10_se").num(se_quantile(share, 0.10));
            j.key("p50").num(percentile(share, 0.50));
            j.key("p50_se").num(se_quantile(share, 0.50));
            j.key("p90").num(percentile(share, 0.90));
            j.key("p90_se").num(se_quantile(share, 0.90));
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
            j.key("share_se"); if (!m.sites.empty()) j.num(se_share(share, m.sites.size())); else j.null();
            for (const auto& [name, k] : std::vector<std::pair<const char*, int>>{{"quartiles", 4}, {"deciles", 10}}) {
                const auto groups = lit_share_by_rank(probe, light[si], s.dir, k);
                j.key(name).arr(); for (const auto& g : groups) j.opt(g.share); j.end();
                j.key((std::string(name) + "_se").c_str()).arr();
                for (const auto& g : groups) j.opt(g.share ? std::optional<float>(float(se_share(*g.share, size_t(g.sites)))) : std::nullopt);
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
                j.key("share_se"); if (nv) j.num(se_share(double(nl) / nv, size_t(nv))); else j.null();
                j.end();
            }
            j.end();
            j.end();
        }
        j.end();
        j.end();
    }
    j.end();   // modes
    j.end();
    return j.take();
}

std::string probe_glb_report(std::span<const uint8_t> glb, int lod, const ProbeOptions& options,
                             const std::string& input_json) {
    LoadedCrown loaded = load_crown_from_glb(glb, lod);
    const Probe probe(std::move(loaded.mesh));
    return probe_report_json(probe, options, input_json);
}

}  // namespace treegen::crown
