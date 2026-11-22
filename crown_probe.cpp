// NOTE: /fp:precise like the rest of TreegenCore: the probe's reports are
// compared byte for byte between its two input paths.
#include "crown_probe.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

namespace treegen::crown {

namespace {

constexpr float k_site_merge_m = 0.001f;   // K17: card centroids within 1 mm form one site
constexpr float k_ray_t_min    = 1e-4f;    // self-intersection guard, metres
constexpr uint32_t k_leaf_size = 4;

vec3 vmin(vec3 a, vec3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
vec3 vmax(vec3 a, vec3 b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
float axis_of(vec3 v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }

vec3 tri_normal_area2(const std::array<vec3, 3>& p) { return cross(p[1] - p[0], p[2] - p[0]); }

struct UnionFind {
    std::vector<uint32_t> parent;
    explicit UnionFind(size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0u); }
    uint32_t find(uint32_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void unite(uint32_t a, uint32_t b) {
        a = find(a); b = find(b);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }
};

}  // namespace

bool AlphaAtlas::passes(float u, float v) const {
    const int x = std::clamp(static_cast<int>(std::floor(u * static_cast<float>(width))), 0, width - 1);
    const int y = std::clamp(static_cast<int>(std::floor(v * static_cast<float>(height))), 0, height - 1);
    return passes_texel(x, y);
}

float uv_triangle_coverage(const AlphaAtlas& atlas, const std::array<std::array<float, 2>, 3>& uv) {
    if (atlas.empty()) return 1.0f;
    const double W = atlas.width, H = atlas.height;
    // Texel space (x right, y down): a texel (x, y) is counted when its centre
    // (x + 0.5, y + 0.5) lies inside the triangle (edges inclusive).
    double px[3], py[3];
    for (int i = 0; i < 3; ++i) { px[i] = uv[i][0] * W; py[i] = uv[i][1] * H; }
    const double area2 = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]);
    if (area2 == 0.0) {
        const float cu = (uv[0][0] + uv[1][0] + uv[2][0]) / 3.0f, cv = (uv[0][1] + uv[1][1] + uv[2][1]) / 3.0f;
        return atlas.passes(cu, cv) ? 1.0f : 0.0f;
    }
    const double sgn = area2 > 0.0 ? 1.0 : -1.0;
    const int x0 = std::max(0, static_cast<int>(std::floor(std::min({px[0], px[1], px[2]}) - 0.5)));
    const int x1 = std::min(atlas.width - 1, static_cast<int>(std::ceil(std::max({px[0], px[1], px[2]}) - 0.5)));
    const int y0 = std::max(0, static_cast<int>(std::floor(std::min({py[0], py[1], py[2]}) - 0.5)));
    const int y1 = std::min(atlas.height - 1, static_cast<int>(std::ceil(std::max({py[0], py[1], py[2]}) - 0.5)));
    uint64_t inside = 0, pass = 0;
    for (int y = y0; y <= y1; ++y) {
        const double cy = y + 0.5;
        for (int x = x0; x <= x1; ++x) {
            const double cx = x + 0.5;
            bool in = true;
            for (int e = 0; e < 3 && in; ++e) {
                const int a = e, b = (e + 1) % 3;
                const double ef = (px[b] - px[a]) * (cy - py[a]) - (py[b] - py[a]) * (cx - px[a]);
                in = ef * sgn >= 0.0;
            }
            if (!in) continue;
            ++inside;
            if (atlas.passes_texel(x, y)) ++pass;
        }
    }
    if (inside == 0) {
        const float cu = (uv[0][0] + uv[1][0] + uv[2][0]) / 3.0f, cv = (uv[0][1] + uv[1][1] + uv[2][1]) / 3.0f;
        return atlas.passes(cu, cv) ? 1.0f : 0.0f;
    }
    return static_cast<float>(static_cast<double>(pass) / static_cast<double>(inside));
}

SiteRule derive_sites(std::span<const float> positions, std::span<const uint32_t> indices) {
    if (positions.size() % 3 != 0 || indices.size() % 3 != 0)
        throw std::runtime_error("crown probe: leaf primitive is not VEC3 positions / TRIANGLES");
    const size_t tri_count = indices.size() / 3;
    const size_t vert_count = positions.size() / 3;
    auto pos = [&](uint32_t v) {
        if (v >= vert_count) throw std::runtime_error("crown probe: leaf index out of range");
        return vec3{positions[size_t(v) * 3], positions[size_t(v) * 3 + 1], positions[size_t(v) * 3 + 2]};
    };

    // Cards: connected components of triangles under shared vertex indices.
    UnionFind uf(tri_count);
    std::vector<int64_t> first_tri_of_vertex(vert_count, -1);
    for (size_t t = 0; t < tri_count; ++t) {
        for (int k = 0; k < 3; ++k) {
            const uint32_t v = indices[t * 3 + size_t(k)];
            pos(v);  // range check
            if (first_tri_of_vertex[v] < 0) first_tri_of_vertex[v] = int64_t(t);
            else uf.unite(uint32_t(first_tri_of_vertex[v]), uint32_t(t));
        }
    }
    // Card ids in index order (by each component's first triangle).
    std::vector<int32_t> card_of_root(tri_count, -1);
    std::vector<int32_t> card_of_tri(tri_count);
    int32_t cards = 0;
    for (size_t t = 0; t < tri_count; ++t) {
        const uint32_t r = uf.find(uint32_t(t));
        if (card_of_root[r] < 0) card_of_root[r] = cards++;
        card_of_tri[t] = card_of_root[r];
    }
    std::vector<std::vector<uint32_t>> card_tris(static_cast<size_t>(cards));
    for (size_t t = 0; t < tri_count; ++t) card_tris[size_t(card_of_tri[t])].push_back(uint32_t(t));

    // Card vertex centroids (each distinct vertex once).
    std::vector<vec3> card_centroid(static_cast<size_t>(cards));
    {
        std::vector<int32_t> seen(vert_count, -1);
        for (int32_t c = 0; c < cards; ++c) {
            double sx = 0, sy = 0, sz = 0;
            int n = 0;
            for (uint32_t t : card_tris[size_t(c)]) {
                for (int k = 0; k < 3; ++k) {
                    const uint32_t v = indices[size_t(t) * 3 + size_t(k)];
                    if (seen[v] == c) continue;
                    seen[v] = c;
                    const vec3 p = pos(v);
                    sx += p.x; sy += p.y; sz += p.z; ++n;
                }
            }
            card_centroid[size_t(c)] = vec3{float(sx / n), float(sy / n), float(sz / n)};
        }
    }

    // Sites: runs of consecutive cards whose centroids agree with the run's
    // first card within 1 mm.
    SiteRule out;
    out.triangle_site.assign(tri_count, -1);
    std::vector<int32_t> site_first_card;
    for (int32_t c = 0; c < cards; ++c) {
        const bool join = !site_first_card.empty() &&
            length(card_centroid[size_t(c)] - card_centroid[size_t(site_first_card.back())]) <= k_site_merge_m;
        if (!join) {
            Site s;
            s.id = int32_t(out.sites.size());
            out.sites.push_back(std::move(s));
            site_first_card.push_back(c);
        }
        ++out.sites.back().cards;
    }
    for (Site& s : out.sites) {
        // Centroid: mean of the member cards' centroids (they agree within 1 mm).
        // Normal: area-weighted geometric normal, each triangle signed toward the
        // site's first triangle (cards are two-sided; crossed cards keep their sum).
        double sx = 0, sy = 0, sz = 0, nx = 0, ny = 0, nz = 0;
        vec3 ref{};
        bool have_ref = false;
        const int32_t c0 = site_first_card[size_t(s.id)];
        for (int32_t c = c0; c < c0 + s.cards; ++c) {
            const vec3 cc = card_centroid[size_t(c)];
            sx += cc.x; sy += cc.y; sz += cc.z;
            for (uint32_t t : card_tris[size_t(c)]) {
                s.triangles.push_back(t);
                out.triangle_site[t] = s.id;
                const std::array<vec3, 3> p = {pos(indices[size_t(t) * 3]), pos(indices[size_t(t) * 3 + 1]),
                                               pos(indices[size_t(t) * 3 + 2])};
                vec3 a2 = tri_normal_area2(p);
                if (!have_ref) { ref = a2; have_ref = true; }
                if (dot(a2, ref) < 0.0f) a2 = -a2;
                nx += a2.x; ny += a2.y; nz += a2.z;
            }
        }
        std::sort(s.triangles.begin(), s.triangles.end());
        s.centroid = vec3{float(sx / s.cards), float(sy / s.cards), float(sz / s.cards)};
        s.normal   = normalized(vec3{float(nx), float(ny), float(nz)});
    }
    return out;
}

void CrownMesh::add_bark(std::span<const float> positions, std::span<const uint32_t> indices) {
    const size_t vc = positions.size() / 3;
    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        Triangle tri;
        for (int k = 0; k < 3; ++k) {
            const uint32_t v = indices[t + size_t(k)];
            if (v >= vc) throw std::runtime_error("crown probe: bark index out of range");
            tri.p[size_t(k)] = vec3{positions[size_t(v) * 3], positions[size_t(v) * 3 + 1], positions[size_t(v) * 3 + 2]};
        }
        triangles.push_back(tri);
        ++bark_triangles;
    }
}

void CrownMesh::add_leaves(std::span<const float> positions, std::span<const float> uvs,
                           std::span<const uint32_t> indices) {
    const size_t vc = positions.size() / 3;
    if (!uvs.empty() && uvs.size() != vc * 2) throw std::runtime_error("crown probe: leaf UV count mismatch");
    SiteRule rule = derive_sites(positions, indices);
    const int32_t site_base = int32_t(sites.size());
    const uint32_t tri_base = uint32_t(triangles.size());
    for (size_t t = 0; t < indices.size() / 3; ++t) {
        Triangle tri;
        for (int k = 0; k < 3; ++k) {
            const uint32_t v = indices[t * 3 + size_t(k)];
            tri.p[size_t(k)] = vec3{positions[size_t(v) * 3], positions[size_t(v) * 3 + 1], positions[size_t(v) * 3 + 2]};
            if (!uvs.empty()) tri.uv[size_t(k)] = {uvs[size_t(v) * 2], uvs[size_t(v) * 2 + 1]};
        }
        tri.site = site_base + rule.triangle_site[t];
        triangles.push_back(tri);
        ++leaf_triangles;
    }
    for (Site& s : rule.sites) {
        s.id += site_base;
        for (uint32_t& t : s.triangles) t += tri_base;
        sites.push_back(std::move(s));
    }
}

Probe::Probe(CrownMesh mesh) : m_mesh(std::move(mesh)) {
    const size_t n = m_mesh.triangles.size();
    m_coverage.assign(n, 1.0f);
    for (size_t t = 0; t < n; ++t)
        if (m_mesh.triangles[t].site >= 0) m_coverage[t] = uv_triangle_coverage(m_mesh.atlas, m_mesh.triangles[t].uv);

    m_min = vec3{1e30f, 1e30f, 1e30f};
    m_max = vec3{-1e30f, -1e30f, -1e30f};
    std::vector<vec3> centroid(n), lo(n), hi(n);
    for (size_t t = 0; t < n; ++t) {
        const auto& p = m_mesh.triangles[t].p;
        lo[t] = vmin(p[0], vmin(p[1], p[2]));
        hi[t] = vmax(p[0], vmax(p[1], p[2]));
        centroid[t] = (p[0] + p[1] + p[2]) / 3.0f;
        m_min = vmin(m_min, lo[t]);
        m_max = vmax(m_max, hi[t]);
    }
    m_order.resize(n);
    std::iota(m_order.begin(), m_order.end(), 0u);
    if (n == 0) return;

    // Median-split BVH (deterministic: stable order + nth_element on a total order).
    struct Job { uint32_t node, begin, end; };
    m_nodes.reserve(2 * n / k_leaf_size + 2);
    m_nodes.push_back({});
    std::vector<Job> stack{{0, 0, uint32_t(n)}};
    while (!stack.empty()) {
        const Job j = stack.back();
        stack.pop_back();
        vec3 blo{1e30f, 1e30f, 1e30f}, bhi{-1e30f, -1e30f, -1e30f};
        vec3 clo = blo, chi = bhi;
        for (uint32_t i = j.begin; i < j.end; ++i) {
            const uint32_t t = m_order[i];
            blo = vmin(blo, lo[t]); bhi = vmax(bhi, hi[t]);
            clo = vmin(clo, centroid[t]); chi = vmax(chi, centroid[t]);
        }
        m_nodes[j.node].lo = blo;
        m_nodes[j.node].hi = bhi;
        const uint32_t count = j.end - j.begin;
        const vec3 ext = chi - clo;
        const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
        if (count <= k_leaf_size || axis_of(ext, axis) <= 0.0f) {
            m_nodes[j.node].first = j.begin;
            m_nodes[j.node].count = count;
            continue;
        }
        const uint32_t mid = j.begin + count / 2;
        std::nth_element(m_order.begin() + j.begin, m_order.begin() + mid, m_order.begin() + j.end,
                         [&](uint32_t a, uint32_t b) {
                             const float ca = axis_of(centroid[a], axis), cb = axis_of(centroid[b], axis);
                             return ca < cb || (ca == cb && a < b);
                         });
        // Children are created as a pair: left at `first - 1`, right at `first`.
        const uint32_t left = uint32_t(m_nodes.size());
        m_nodes.push_back({});
        m_nodes.push_back({});
        m_nodes[j.node].first = left + 1;
        m_nodes[j.node].count = 0;
        stack.push_back({left + 1, mid, j.end});
        stack.push_back({left, j.begin, mid});
    }
}

bool Probe::hit_opaque(uint32_t tri, float b1, float b2) const {
    const Triangle& T = m_mesh.triangles[tri];
    if (T.site < 0 || m_mesh.atlas.empty()) return true;
    const float b0 = 1.0f - b1 - b2;
    const float u = b0 * T.uv[0][0] + b1 * T.uv[1][0] + b2 * T.uv[2][0];
    const float v = b0 * T.uv[0][1] + b1 * T.uv[1][1] + b2 * T.uv[2][1];
    return m_mesh.atlas.passes(u, v);
}

template <class Visit>
void Probe::traverse(vec3 origin, vec3 dir, float t_max, Visit&& visit) const {
    if (m_nodes.empty()) return;
    auto safe = [](float d) { return std::abs(d) < 1e-12f ? (d < 0.0f ? -1e-12f : 1e-12f) : d; };
    const vec3 inv{1.0f / safe(dir.x), 1.0f / safe(dir.y), 1.0f / safe(dir.z)};
    uint32_t stack[128];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const Node& nd = m_nodes[stack[--sp]];
        float t0 = 0.0f, t1 = t_max;
        for (int a = 0; a < 3; ++a) {
            const float o = axis_of(origin, a), iv = axis_of(inv, a);
            float ta = (axis_of(nd.lo, a) - o) * iv, tb = (axis_of(nd.hi, a) - o) * iv;
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
        }
        if (t0 > t1) continue;
        if (nd.count == 0) {
            stack[sp++] = nd.first;        // right
            stack[sp++] = nd.first - 1;    // left
            continue;
        }
        for (uint32_t i = nd.first; i < nd.first + nd.count; ++i) {
            const uint32_t t = m_order[i];
            const auto& p = m_mesh.triangles[t].p;
            const vec3 e1 = p[1] - p[0], e2 = p[2] - p[0];
            const vec3 pv = cross(dir, e2);
            const float det = dot(e1, pv);
            if (std::abs(det) < 1e-14f) continue;
            const float id = 1.0f / det;
            const vec3 tv = origin - p[0];
            const float b1 = dot(tv, pv) * id;
            if (b1 < 0.0f || b1 > 1.0f) continue;
            const vec3 qv = cross(tv, e1);
            const float b2 = dot(dir, qv) * id;
            if (b2 < 0.0f || b1 + b2 > 1.0f) continue;
            const float th = dot(e2, qv) * id;
            if (th <= k_ray_t_min || th >= t_max) continue;
            if (visit(t, b1, b2)) return;
        }
    }
}

const char* occluders_name(Occluders o) { return o == Occluders::leaves_only ? "leaves_only" : "leaves_and_bark"; }

bool Probe::occluded(vec3 origin, vec3 dir, int32_t skip_site, Occluders occ) const {
    bool hit = false;
    traverse(origin, dir, 1e30f, [&](uint32_t t, float b1, float b2) {
        if (occ == Occluders::leaves_only && m_mesh.triangles[t].site < 0) return false;
        if (skip_site >= 0 && m_mesh.triangles[t].site == skip_site) return false;
        if (!hit_opaque(t, b1, b2)) return false;
        hit = true;
        return true;
    });
    return hit;
}

bool Probe::occluded_segment(vec3 origin, vec3 target, int32_t skip_site, Occluders occ) const {
    const vec3 d = target - origin;
    const float len = length(d);
    if (len <= 2.0f * k_ray_t_min) return false;
    bool hit = false;
    traverse(origin, d / len, len - k_ray_t_min, [&](uint32_t t, float b1, float b2) {
        if (occ == Occluders::leaves_only && m_mesh.triangles[t].site < 0) return false;
        if (skip_site >= 0 && m_mesh.triangles[t].site == skip_site) return false;
        if (!hit_opaque(t, b1, b2)) return false;
        hit = true;
        return true;
    });
    return hit;
}

LineHits Probe::line_hits(vec3 origin, vec3 dir, float t_max) const {
    LineHits h;
    traverse(origin, dir, t_max, [&](uint32_t t, float b1, float b2) {
        if (m_mesh.triangles[t].site < 0) { ++h.bark; return false; }
        ++h.leaf_geometric;
        if (hit_opaque(t, b1, b2)) ++h.leaf_opaque;
        return false;
    });
    return h;
}

void parallel_for(size_t n, const std::function<void(size_t)>& fn) {
    if (n == 0) return;
    const size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t workers = std::min<size_t>({hw, size_t(16), n});
    std::atomic<size_t> next{0};
    std::exception_ptr err;
    std::mutex err_mutex;
    auto work = [&] {
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= n) return;
            try { fn(i); }
            catch (...) {
                std::lock_guard<std::mutex> lock(err_mutex);
                if (!err) err = std::current_exception();
                next.store(n);
                return;
            }
        }
    };
    std::vector<std::thread> pool;
    for (size_t w = 1; w < workers; ++w) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    if (err) std::rethrow_exception(err);
}

}  // namespace treegen::crown
