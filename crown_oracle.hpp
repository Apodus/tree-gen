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
#include <string>
#include <vector>

namespace treegen::crown {

inline constexpr int   k_report_schema       = 1;
inline constexpr float k_footprint_m         = 0.5f;
inline constexpr int   k_rays_per_side       = 8;      // 64 rays per footprint
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
inline constexpr uint64_t k_footprint_ray_stream = 0xC20A5EEDull;
inline constexpr uint64_t k_format_ray_stream    = 0xF0E3A7EDull;
inline constexpr int   k_open_sky_directions = 64;

struct SunDirection {
    float elevation_deg = 0.0f;
    float azimuth_deg   = 0.0f;
    vec3  dir;   // unit, toward the sun
};
SunDirection sun_direction(float elevation_deg, float azimuth_deg);
std::vector<SunDirection> sun_directions(const std::vector<float>& elevations_deg, int azimuths);

enum class PlaneKind { beam, ground };

struct Footprint {
    int   i = 0, j = 0;
    float u = 0.0f, v = 0.0f;   // centre, plane coordinates (metres)
    vec3  centre;               // world
    float t = 1.0f;             // unoccluded share of its rays
    float edge_distance = 0.0f; // 0 outside the silhouette
    float crossings_geometric = 0.0f;  // mean over its rays
    float crossings_opaque    = 0.0f;
    bool  silhouette() const { return t < k_silhouette_t; }
    bool  body() const { return in_body(k_body_edge_m); }
    // The one body predicate: silhouette footprints at least min_edge_m from an
    // outside centre. Edge distances are centre-to-centre on the 0.5 m lattice
    // (0.5, 0.707, 1.0, 1.118, ...), so every threshold in (0.707, 1.0] is the same rule.
    bool  in_body(float min_edge_m) const { return silhouette() && edge_distance >= min_edge_m; }
};

// The footprint sampler's two jitter streams (one definition site: rays per
// side and stream id both follow from the purpose; there is no default).
// - selection: the 8 x 8 rays that measure a footprint's T;
// - format: the 32 x 32 rays, independent of selection's, that re-measure the
//   format rule's unsaturated footprints, so selecting on a noisy T cannot bias its MAE.
enum class RayStream { selection, format };
int      rays_per_side(RayStream s);
uint64_t stream_id(RayStream s);

struct FootprintSet {
    SunDirection           sun;
    PlaneKind              kind = PlaneKind::beam;
    Occluders              occluders = Occluders::leaves_and_bark;   // set by measure_footprints
    vec3                   origin, e1, e2;   // plane frame
    float                  ray_back = 0.0f;  // rays start this far behind the plane
    float                  t_max = 0.0f;     // ... and end this far along the sun direction
    std::vector<Footprint> footprints;
    // The rays_per_side(s)^2 jittered-stratified ray origins of footprint k
    // (deterministic in s; direction = sun.dir).
    void rays(size_t k, std::vector<vec3>& out, RayStream s) const;
};

// Measures every footprint of the plane's grid (the crown's projection plus one
// footprint of margin) against population `occ` and assigns edge distances.
FootprintSet measure_footprints(const Probe& probe, const SunDirection& sun, PlaneKind kind, Occluders occ);

// Standard errors (R-CR6; one definition site, crown_oracle.cpp "SE"). Every
// sampled statistic of the report carries `<name>_se` beside it:
// - a mean of footprint T: ray sampling, sqrt(sum T (1 - T) / R) / n over its n
//   footprints of R rays (the geometry fixed, the jitter re-drawn);
// - a share or a quantile over footprints or sites: the population's, binomial
//   sqrt(p (1 - p) / n), and for a quantile the half-width of the order
//   statistics one binomial SE either side of its rank;
// - a mean over footprints or sites of anything else (crossings, open sky,
//   |error| for an MAE): std / sqrt(n), footprints treated as independent.
// Counts, areas, coverage and the site histogram are exact functions of the
// bytes and carry none.
struct RegionStats {
    int   footprints = 0;
    float mean = 0.0f, p50 = 0.0f, p95 = 0.0f, share_opaque = 0.0f;
    float mean_se = 0.0f, p50_se = 0.0f, p95_se = 0.0f, share_opaque_se = 0.0f;
};
// Body statistics over the footprints Footprint::in_body(min_edge_m) admits (nullopt: none).
std::optional<RegionStats> body_stats(const FootprintSet& set, float min_edge_m);

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
    float elevation_deg = 0.0f;
    int   silhouette = 0;                          // for the saturated share
    int   saturated  = 0;                          // silhouette footprints with T < 0.05
    std::vector<FormatFootprint> unsaturated;      // T in [0.05, 0.95]
};
struct FormatRow {
    float elevation_deg = 0.0f;
    int   unsaturated = 0;
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

// --- The report ----------------------------------------------------------------

struct ProbeOptions {
    bool               oracle = true;
    bool               sunlit = true;
    std::vector<float> elevations_deg = {10, 20, 30, 45, 60, 70, 75, 90};
    int                azimuths = 4;
    std::vector<vec3>  eyes;   // object space
};

// `s` as a quoted, escaped JSON string.
std::string json_string(const std::string& s);

// The report's "input" object: the loader's arguments, one spelling per input,
// and the probed GLB bytes' digest (fnv1a64, det_rng.hpp), so a report names the
// bytes it measured whichever path produced them.
std::string glb_digest(std::span<const uint8_t> glb);   // "0x%016x"
std::string probe_input_scenario(const std::string& scenario_path, uint64_t seed, std::span<const uint8_t> glb, int lod);
std::string probe_input_glb(const std::string& glb_path, std::span<const uint8_t> glb, int lod);

// The versioned JSON document: {"schema", "input", "tree", "modes": {...}}.
// `input_json` is the loader's echo, already a JSON object.
std::string probe_report_json(const Probe& probe, const ProbeOptions& options, const std::string& input_json);

// The front end's whole path for one input: GLB bytes (from generate_tree or a
// file) -> load_crown_from_glb(lod) -> Probe -> report. Both inputs call this.
std::string probe_glb_report(std::span<const uint8_t> glb, int lod, const ProbeOptions& options,
                             const std::string& input_json);

}  // namespace treegen::crown
