// The crown probe's oracle and sunlit modes and its one JSON report
// (crown-shadows S-P0a; definitions in the plan's section 5.1, interface K17).
//
// Definitions used by every rule and gate:
// - Footprint plane ("beam plane"): perpendicular to the sun direction through
//   the crown box centre. Footprints are 0.5 m squares on a grid in it, each
//   sampled by 8 x 8 jittered-stratified parallel rays along the sun direction;
//   T is the unoccluded share. Ground-plane footprints (0.5 m squares on the
//   horizontal plane through the lowest vertex, stretched by 1/sin(e) in the
//   beam) are reported in addition and never used by a rule.
// - Silhouette: footprints with T < 0.98. Edge distance d: Euclidean distance in
//   the plane from a footprint centre to the nearest footprint centre outside
//   the silhouette. Rim: d < 1 m. Body: d >= 1 m (none: "body": null).
// - Crossings per ray: leaf triangles the ray's line intersects (alpha ignored)
//   and those that pass the alpha test.
#pragma once

#include "crown_probe.hpp"
#include "crown_volume.hpp"
#include "vec3.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace treegen::crown {

inline constexpr int   k_report_schema       = 1;
inline constexpr float k_footprint_m         = 0.5f;
inline constexpr int   k_rays_per_side       = 8;      // 64 rays per footprint
static_assert(k_rays_per_side * k_rays_per_side <= 64, "Footprint::open_mask stores up to 64 selection rays");
inline constexpr float k_silhouette_t        = 0.98f;
inline constexpr float k_body_edge_m         = 1.0f;
inline constexpr float k_opaque_t            = 0.05f;  // stop rule and format-rule floor
inline constexpr float k_unsaturated_hi      = 0.95f;
inline constexpr int   k_min_unsaturated     = 30;
inline constexpr int   k_min_measurable      = 4;
inline constexpr float k_min_span_deg        = 40.0f;
inline constexpr float k_format_max_mae      = 0.05f;
inline constexpr float k_format_scalar_slack = 0.01f;
inline constexpr int   k_format_cells        = 32;     // the review's grid; S-P1 chooses the shipped N
// The format rule re-measures its unsaturated footprints with 32 x 32 rays: at
// 64 rays a footprint's binomial error alone (SE 0.0625 at T = 0.5) gives an
// MAE of 0.05, the rule's own threshold, for an exact model; at 1024 it is 0.0125.
inline constexpr int   k_format_rays_per_side = 32;
// Jitter streams: the format rule scores footprints on rays independent of the
// 64 that selected them, so selecting on a noisy T cannot bias its MAE.
// selection_alt is a test-only 8 x 8 stream for re-jitter stability checks (P4).
inline constexpr uint64_t k_footprint_ray_stream     = 0xC20A5EEDull;
inline constexpr uint64_t k_footprint_alt_ray_stream = 0xA175EED1ull;
inline constexpr uint64_t k_format_ray_stream        = 0xF0E3A7EDull;
inline constexpr int   k_open_sky_directions = 64;
inline constexpr float k_normals_lowpass_sigma_m     = 0.6f;   // crown-lighting §5.3, §5.5 (K8)
inline constexpr int   k_normals_directions          = 64;     // full-sphere Fibonacci directions
inline constexpr int   k_normals_neighbours          = 8;      // nearest-site agreement window
inline constexpr float k_normals_ellipsoid_angle_deg = 15.0f;  // §5.3 ellipsoid-vs-march median angle cut
inline constexpr float k_normals_ellipsoid_pp        = 0.02f;  // §5.3 failure-mode gap cut (2 pp)

struct SunDirection {
    float elevation_deg = 0.0f;
    float azimuth_deg   = 0.0f;
    vec3  dir;   // unit, toward the sun
};
SunDirection sun_direction(float elevation_deg, float azimuth_deg);
std::vector<SunDirection> sun_directions(const std::vector<float>& elevations_deg, int azimuths);

enum class PlaneKind { beam, ground };

struct Footprint {
    int      i = 0, j = 0;
    float    u = 0.0f, v = 0.0f;   // centre, plane coordinates (metres)
    vec3     centre;               // world
    float    t = 1.0f;             // unoccluded share of its rays
    float    edge_distance = 0.0f; // 0 outside the silhouette
    float    crossings_geometric = 0.0f;  // mean over its rays
    float    crossings_opaque    = 0.0f;
    uint64_t open_mask = 0;        // bit r = 1 iff selection ray r was unoccluded (P2)
    uint8_t  ray_count = 0;        // selection rays traced (64 from measure_footprints; 0 if synthetic)
    bool     silhouette() const { return t < k_silhouette_t; }
    bool     body() const { return in_body(k_body_edge_m); }
    // The one body predicate: silhouette footprints at least min_edge_m from an
    // outside centre. Edge distances are centre to centre on the footprint
    // lattice, d = pitch * sqrt(n) for n a sum of two squares (0.5, 0.707, 1.0,
    // 1.118, 1.414, ... m at a 0.5 m pitch): a threshold is only meaningful at
    // those distances, and every threshold in (0.707, 1.0] is the same rule.
    bool     in_body(float min_edge_m) const { return silhouette() && edge_distance >= min_edge_m; }
};
// The pitch the rim/body split assumes (k_body_edge_m = 2 pitches: the 0.707 m
// ring is rim, the 1.0 m ring body) and the exact-lattice plate gate derives its
// integers from. A new pitch moves which ring is body: re-derive both.
static_assert(k_footprint_m == 0.5f, "footprint pitch changed: re-derive the body threshold against the lattice distances");

// The footprint sampler's jitter streams (one definition site: rays per
// side and stream id both follow from the purpose; there is no default).
// - selection: the 8 x 8 rays that measure a footprint's T;
// - selection_alt: test-only 8 x 8 stream with independent jitter (P4);
// - format: the 32 x 32 rays, independent of selection's, that re-measure the
//   format rule's unsaturated footprints, so selecting on a noisy T cannot bias its MAE.
enum class RayStream { selection, selection_alt, format };
int      rays_per_side(RayStream s);
uint64_t stream_id(RayStream s);

struct FootprintSet {
    SunDirection           sun;
    PlaneKind              kind = PlaneKind::beam;
    Occluders              occluders = Occluders::leaves_and_bark;   // set by measure_footprints
    RayStream              selection_stream = RayStream::selection;  // set by measure_footprints
    vec3                   origin, e1, e2;   // plane frame
    float                  ray_back = 0.0f;  // rays start this far behind the plane
    float                  t_max = 0.0f;     // ... and end this far along the sun direction
    int                    i0 = 0, i1 = 0, j0 = 0, j1 = 0;  // lattice index bounds [i0,i1) x [j0,j1)
    std::vector<Footprint> footprints;
    // The rays_per_side(s)^2 jittered-stratified ray origins of footprint k
    // (deterministic in s; direction = sun.dir).
    void rays(size_t k, std::vector<vec3>& out, RayStream s) const;
};

// Measures every footprint of the plane's grid (the crown's projection plus one
// footprint of margin) against population `occ` and assigns edge distances.
// `selection_stream` defaults to RayStream::selection; pass RayStream::selection_alt
// in re-jitter tests (P4). RayStream::format throws std::logic_error.
FootprintSet measure_footprints(const Probe& probe, const SunDirection& sun, PlaneKind kind, Occluders occ,
                                RayStream selection_stream = RayStream::selection);

// Standard errors (R-CR6 / P2; one definition site, crown_oracle.cpp "SE").
// Every sampled statistic of the report carries `<name>_se` beside it:
// - a mean of footprint T: ray sampling, sqrt(sum T (1 - T) / R) / n over its n
//   footprints of R rays (the geometry fixed, the jitter re-drawn);
// - a share or a quantile over footprints or sites: the population's, binomial
//   sqrt(p (1 - p) / n), and for a quantile the half-width of the order
//   statistics one binomial SE either side of its rank (full one-sided spread
//   when one rank clamps at 0 or n or equals the quantile's own rank);
// - a mean over footprints or sites of anything else (crossings, open sky,
//   |error| for an MAE): std / sqrt(n), footprints treated as independent;
// - RegionStats::{mean_se, p50_se, p95_se, share_opaque_se}: when FootprintSet
//   carries lattice bounds (measured grids), body_stats takes max(analytic,
//   spatial_bootstrap) so boundary-flip membership cascades are included; on
//   gridless synthetic sets, the exact analytic formulas are returned.
// Exact functions of the GLB bytes (carry no `_se`):
//   bark_triangles, leaf_triangles, sites, cards_per_site, alpha_coverage,
//   card_area_m2, opaque_leaf_area_m2, leaf_aabb_min_m, leaf_aabb_max_m,
//   leaf_area_density_m2_per_m3, trunk_base_z_m, lowest_leaf_height_m.
// Jitter-dependent counts and derived areas (depend on the 64-ray selection
// stream jitter; carry `_se` via exact binomial crossing probabilities for
// independent counts and ray-hit bootstrap with edge_distance recomputation
// for spatially coupled counts):
//   crown_ground_projected_area_m2, leaf_area_index, silhouette, rim,
//   body.footprints, ground_plane.silhouette, ground_plane.body.footprints,
//   format.rows[].unsaturated_footprints.
struct CountSEs {
    double silhouette_se        = 0.0;
    double rim_se               = 0.0;
    double body_footprints_se   = 0.0;
    double body_mean_se         = 0.0;
    double body_p50_se          = 0.0;
    double body_p95_se          = 0.0;
    double body_share_opaque_se = 0.0;
};
CountSEs footprint_count_se(const FootprintSet& set, float min_edge_m = k_body_edge_m,
                            bool include_spatial_bootstrap = true);

struct RegionStats {
    int   footprints = 0;
    float footprints_se = 0.0f;
    float mean = 0.0f, p50 = 0.0f, p95 = 0.0f, share_opaque = 0.0f;
    float mean_se = 0.0f, p50_se = 0.0f, p95_se = 0.0f, share_opaque_se = 0.0f;
};
// Body statistics over the footprints Footprint::in_body(min_edge_m) admits (nullopt: none).
std::optional<RegionStats> body_stats(const FootprintSet& set, float min_edge_m);

// Pin bands (P1; one definition site in crown_oracle.cpp):
// max(3 * se, resolution_floor), so zero-SE degenerate statistics never collapse
// to a zero-width band. Pin tests call these wrappers instead of literal bands.
double pin_band(double se, double resolution_floor);
double pin_band_share(double se, size_t n);
double pin_band_mean_t(double se, size_t footprints, int rays_per_footprint);
double pin_band_quantile(double se, int step_count);
double pin_band_count(double se);

// --- Stop rule (plan S-P0a) ---------------------------------------------------
// One direction: the body is opaque when the share of its footprints with
// T < k_opaque_t is at least k_stop_body_share; no body is not opaque.
inline constexpr float k_stop_body_share        = 0.5f;
inline constexpr float k_stop_min_elevation_deg = 30.0f;
bool body_opaque(const FootprintSet& beam);
// Per elevation: opaque at every azimuth sampled there. The crown is opaque to
// direct sun when at least one elevation >= k_stop_min_elevation_deg was
// sampled and the body is opaque at every such elevation.
struct StopRule {
    std::vector<std::pair<float, bool>> opaque_by_elevation;   // ascending elevation
    bool                                crown_opaque_to_direct_sun = false;
};
StopRule stop_rule(const std::vector<std::pair<float, bool>>& body_opaque_by_direction);

// --- Format rule (plan 5.1, R10) ---------------------------------------------

struct FormatFootprint {
    float t_oracle = 1.0f;
    std::vector<std::array<float, 2>> integrals;   // per ray: {integral rho, integral dot(|w|, rho_xyz)}
};
struct FormatElevation {
    float  elevation_deg = 0.0f;
    int    silhouette = 0;                         // for the saturated share
    int    saturated  = 0;                         // silhouette footprints with T < 0.05
    double unsaturated_var = 0.0;                  // binomial crossing variance of unsaturated count (P2)
    std::vector<FormatFootprint> unsaturated;      // T in [0.05, 0.95]
};
struct FormatRow {
    float elevation_deg = 0.0f;
    int   unsaturated = 0;
    float unsaturated_se = 0.0f;
    bool  measurable = false;
    float scalar_mae = 0.0f, axis_mae = 0.0f, g_eff = 0.0f, saturated_share = 0.0f;
    float scalar_mae_se = 0.0f, axis_mae_se = 0.0f, saturated_share_se = 0.0f;
    float g_eff_se = 0.0f;   // a fitted parameter: sqrt(SSE / (n - 1) / sum (dT/dg)^2), the linearised least-squares SE
};
struct FormatDecision {
    std::string            decision;   // scalar | axis_projected | escalate | undecidable
    float                  g = 0.0f;   // scalar fit (0 when undecidable)
    float                  clumping = 1.0f;
    float                  g_se = 0.0f, clumping_se = 0.0f;   // as FormatRow::g_eff_se (0 when not fitted)
    std::vector<FormatRow> rows;
};
FormatDecision decide_format(const std::vector<FormatElevation>& elevations);
// The format-rule input for one footprint set (beam plane): footprints whose
// selection T lies in [0.05, 0.95], re-measured on the format stream, each ray
// marched through `volume`. The set must be measured on Occluders::leaves_only
// (R-CR1: the volume holds leaves only, so selection and score both exclude
// bark); any other set throws.
void add_format_footprints(const Probe& probe, const FootprintSet& set, const CrownVolume& volume, FormatElevation& into,
                           const std::function<bool(const Footprint&)>& keep = {});

// --- Exposed SE primitives for direct construction tests (P3) ------------------

namespace detail {
double se_mean(std::span<const double> x);
double se_mean(std::span<const float> x);
double se_mean_t(std::span<const float> t, int rays);
double se_share(double p, size_t n);
double se_quantile(std::vector<float> v, double q);
double mae_se(std::span<const double> abs_errors);
double fit_se(std::span<const double> residuals, std::span<const double> derivatives);
double dt_scalar(const FormatFootprint& f, double g);
double dt_axis(const FormatFootprint& f, double c);
}  // namespace detail

// --- Normals mode (crown-lighting L-P0a, §5.3, §5.5, K17) --------------------

struct NormalSourceStats {
    float against_outward_share                 = 0.0f;
    float against_outward_share_se              = 0.0f;
    float inner_half_against_outward_share      = 0.0f;
    float inner_half_against_outward_share_se   = 0.0f;
    float neighbour_agreement                   = 0.0f;
    float neighbour_agreement_se                = 0.0f;
    float hollow_shell_share                    = 0.0f;
    float hollow_shell_share_se                 = 0.0f;
    float inner_half_hollow_shell_share         = 0.0f;
    float inner_half_hollow_shell_share_se      = 0.0f;
    float below_horizontal_share                = 0.0f;
    float below_horizontal_share_se             = 0.0f;
};

struct NormalsReport {
    int               sites            = 0;
    int               inner_half_sites = 0;
    float             lowpass_sigma_m  = k_normals_lowpass_sigma_m;
    int               directions       = k_normals_directions;
    int               neighbours       = k_normals_neighbours;
    ExtinctionModel   model;
    NormalSourceStats card;
    NormalSourceStats ellipsoid;
    NormalSourceStats march_lowpass_0_6m;
    NormalSourceStats neg_grad;
    float             ellipsoid_vs_march_angle_deg_p50     = 0.0f;
    float             ellipsoid_vs_march_angle_deg_p50_se  = 0.0f;
    float             ellipsoid_vs_march_angle_deg_p90     = 0.0f;
    float             ellipsoid_vs_march_angle_deg_p90_se  = 0.0f;
    float             ellipsoid_vs_march_angle_deg_mean    = 0.0f;
    float             ellipsoid_vs_march_angle_deg_mean_se = 0.0f;
    std::string       decision;   // ellipsoid | march_lowpass_0_6m

    // Per-site unit directions (in Probe::mesh().sites order) for direct unit-test checks.
    std::vector<vec3>    site_card;
    std::vector<vec3>    site_ellipsoid;
    std::vector<vec3>    site_march;
    std::vector<vec3>    site_neg_grad;
    std::vector<uint8_t> site_inner_half;
};

// Measures the four candidate leaf-normal sources of crown-lighting §5.3
// (card, per-tree ellipsoid, 0.6 m low-passed march, and -grad of density(p))
// over all mesh-derived leaf sites of `probe` using `volume` and `model`.
NormalsReport measure_normals(const Probe& probe, const CrownVolume& volume, const ExtinctionModel& model,
                              float lowpass_sigma_m = k_normals_lowpass_sigma_m);

// --- The report ----------------------------------------------------------------

struct ProbeOptions {
    bool               oracle  = true;
    bool               sunlit  = true;
    bool               normals = true;
    std::vector<float> elevations_deg = {10, 20, 30, 45, 60, 70, 75, 90};
    int                azimuths = 4;
    std::vector<vec3>  eyes;   // object space
    RayStream          selection_stream = RayStream::selection;   // test-only re-jitter hook (P4)
};

// `s` as a quoted, escaped JSON string.
std::string json_string(const std::string& s);

// The probed GLB bytes' digest (fnv1a64, det_rng.hpp), "0x%016x".
std::string glb_digest(std::span<const uint8_t> glb);

// Typed provenance for probe_glb_report (D4): callers pass only source metadata;
// probe_glb_report computes glb_fnv1a64 and probe_lod from the actual GLB bytes
// and LOD it probes, so caller/payload mismatch is impossible by API.
enum class ProbeInputKind { scenario, glb };

struct ProbeInput {
    ProbeInputKind kind = ProbeInputKind::glb;
    std::string    path;
    uint64_t       seed = 0;

    static ProbeInput scenario(std::string scenario_path, uint64_t cli_seed) {
        return {ProbeInputKind::scenario, std::move(scenario_path), cli_seed};
    }
    static ProbeInput glb(std::string glb_path) {
        return {ProbeInputKind::glb, std::move(glb_path), 0};
    }
};

// The versioned JSON document: {"schema", "input", "tree", "modes": {...}}.
// `input_json` is the loader's echo, already a JSON object (used directly by
// synthetic Probe unit tests that have no GLB bytes).
std::string probe_report_json(const Probe& probe, const ProbeOptions& options, const std::string& input_json);

// The front end's whole path for one input: GLB bytes (from generate_tree or a
// file) -> load_crown_from_glb(lod) -> Probe -> report. Computes glb_fnv1a64
// and probe_lod internally from `glb` and `lod`. Both inputs call this.
std::string probe_glb_report(std::span<const uint8_t> glb, int lod, const ProbeOptions& options,
                             const ProbeInput& input);

}  // namespace treegen::crown
