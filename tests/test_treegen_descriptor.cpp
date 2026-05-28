// [treegen_descriptor] — pins the TreeDescriptor IR (POD shape, JSON
// round-trip, envelope-shape string mapping, envelope_contains for all 5
// shapes, envelope_aabb containment, and scenario.cpp loading kind:"tree").
//
// Tool sources (tree_descriptor.cpp + envelopes.cpp + scenario.cpp + extended
// json_reader.cpp) are linked directly into TestTech via
// rynx_tests.sharpmake.cs — same split rationale as C1 P2's trivial_cylinder.

#include "../external/catch2/catch.hpp"

#include "test_support_paths.hpp"

#include "../envelopes.hpp"
#include "../json_reader.hpp"
#include "../scenario.hpp"
#include "../tree_descriptor.hpp"
#include "../vec3.hpp"

#include <cmath>
#include <cstdint>
#include <string>

namespace {

// Compare two floats with a slack matching the round-trip precision of %.9g
// through strtod (~1 ULP for in-range single-precision values).
bool float_eq(float a, float b) {
    const float tol = 1e-6f * std::max(1.0f, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) <= tol;
}

bool descriptor_equal(const treegen::TreeDescriptor& a, const treegen::TreeDescriptor& b) {
    if (a.species != b.species) return false;
    if (a.seed    != b.seed)    return false;
    if (!float_eq(a.height_m, b.height_m)) return false;
    if (!float_eq(a.trunk_base_radius_m, b.trunk_base_radius_m)) return false;
    if (!float_eq(a.taper_exponent, b.taper_exponent)) return false;

    if (a.envelope.shape != b.envelope.shape) return false;
    if (!float_eq(a.envelope.width_m, b.envelope.width_m)) return false;
    if (!float_eq(a.envelope.height_ratio, b.envelope.height_ratio)) return false;
    if (!float_eq(a.envelope.top_height_ratio, b.envelope.top_height_ratio)) return false;

    if (a.branching.attractor_count          != b.branching.attractor_count)          return false;
    if (!float_eq(a.branching.kill_distance,            b.branching.kill_distance))            return false;
    if (!float_eq(a.branching.growth_distance,          b.branching.growth_distance))          return false;
    if (a.branching.max_iterations           != b.branching.max_iterations)           return false;
    if (!float_eq(a.branching.branch_angle_jitter_deg,  b.branching.branch_angle_jitter_deg))  return false;
    if (a.branching.min_attractors_to_split  != b.branching.min_attractors_to_split)  return false;
    if (!float_eq(a.branching.tropism_strength,         b.branching.tropism_strength))         return false;
    if (!float_eq(a.branching.angular_spread_split_deg, b.branching.angular_spread_split_deg)) return false;
    if (!float_eq(a.branching.crown_base_fraction, b.branching.crown_base_fraction)) return false;
    if (!float_eq(a.branching.min_branch_radius_m, b.branching.min_branch_radius_m)) return false;

    if (!float_eq(a.trunk_taper_rate, b.trunk_taper_rate)) return false;
    if (!float_eq(a.root_flare_factor, b.root_flare_factor)) return false;
    if (!float_eq(a.junction_shoulder_factor, b.junction_shoulder_factor)) return false;

    if (!float_eq(a.tropisms.gravitropism, b.tropisms.gravitropism)) return false;
    if (!float_eq(a.tropisms.phototropism, b.tropisms.phototropism)) return false;
    if (!float_eq(a.tropisms.light_dir.x, b.tropisms.light_dir.x)) return false;
    if (!float_eq(a.tropisms.light_dir.y, b.tropisms.light_dir.y)) return false;
    if (!float_eq(a.tropisms.light_dir.z, b.tropisms.light_dir.z)) return false;
    return true;
}

bool aabb_contains(treegen::Aabb box, treegen::vec3 p) {
    const float eps = 1e-5f;
    return p.x >= box.min.x - eps && p.x <= box.max.x + eps
        && p.y >= box.min.y - eps && p.y <= box.max.y + eps
        && p.z >= box.min.z - eps && p.z <= box.max.z + eps;
}

} // anonymous namespace

TEST_CASE("treegen_descriptor: JSON round-trip on c3_oak fixture",
          "[treegen][treegen_descriptor][treegen_descriptor_roundtrip]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    INFO("fixture=" << fixture);
    REQUIRE_FALSE(fixture.empty());

    treegen::Scenario s = treegen::load_scenario(fixture);
    REQUIRE(s.kind == "tree");
    REQUIRE(s.name == "oak_default");

    // Serialize → parse-again → deep-equal POD.
    const std::string serialized = treegen::serialize_tree_descriptor(s.tree);
    auto parsed = treegen::json::parse(serialized);
    REQUIRE(parsed.ok);
    REQUIRE(parsed.root.is_object());

    treegen::TreeDescriptor td2 = treegen::parse_tree_descriptor(parsed.root);
    REQUIRE(descriptor_equal(s.tree, td2));

    // Serialize is itself deterministic across two calls.
    const std::string serialized2 = treegen::serialize_tree_descriptor(td2);
    REQUIRE(serialized == serialized2);

    // Field spot-checks against the fixture values.
    REQUIRE(s.tree.species == "oak");
    REQUIRE(s.tree.seed    == 42u);
    REQUIRE(float_eq(s.tree.height_m, 12.0f));
    REQUIRE(s.tree.envelope.shape == treegen::EnvelopeShape::OblateSpheroid);
    REQUIRE(s.tree.branching.attractor_count == 600);
    REQUIRE(float_eq(s.tree.branching.angular_spread_split_deg, 30.0f));
    REQUIRE(float_eq(s.tree.tropisms.light_dir.z, 1.0f));
}

TEST_CASE("treegen_descriptor: envelope-shape string mapping round-trip",
          "[treegen][treegen_descriptor][treegen_descriptor_envelope_shapes]") {
    using ES = treegen::EnvelopeShape;
    constexpr ES k_all[] = {
        ES::OblateSpheroid, ES::Conical, ES::Weeping, ES::Fastigiate, ES::Fan,
    };
    for (ES s : k_all) {
        const char* name = treegen::envelope_shape_to_string(s);
        INFO("shape name=" << name);
        REQUIRE(name != nullptr);
        REQUIRE(treegen::parse_envelope_shape(name) == s);
    }

    // Unknown name throws.
    REQUIRE_THROWS_AS(treegen::parse_envelope_shape("not_a_real_shape"),
                      std::runtime_error);
}

TEST_CASE("treegen_descriptor: envelope_contains for all 5 shapes",
          "[treegen][treegen_descriptor][treegen_descriptor_envelope_contains]") {
    using ES = treegen::EnvelopeShape;
    using treegen::vec3;
    const float H = 10.0f;

    // OblateSpheroid: width=6 → a=3; height_ratio=1, top_height_ratio=1
    // → center_z=5, b=5. Equation (x/3)^2+(y/3)^2+((z-5)/5)^2 <= 1.
    {
        treegen::TreeDescriptor::Envelope env;
        env.shape = ES::OblateSpheroid;
        env.width_m = 6.0f;
        env.height_ratio = 1.0f;
        env.top_height_ratio = 1.0f;

        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 5}, H));    // center
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 0}, H));    // bottom pole (z=0, (0-5)/5=-1)
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 10}, H));   // top pole
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{3, 0, 5}, H));    // x extreme
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{3.01f, 0, 5}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 11}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{100, 100, 5}, H));
    }

    // Conical: width=4 → base radius=2, apex at z=H*height_ratio=10.
    // At z=0: r=2. At z=5: r=1. At z=10: r=0.
    {
        treegen::TreeDescriptor::Envelope env;
        env.shape = ES::Conical;
        env.width_m = 4.0f;
        env.height_ratio = 1.0f;
        env.top_height_ratio = 1.0f; // unused for cone

        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 0}, H));
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{2, 0, 0}, H));     // base rim
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 10}, H));    // apex
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0.99f, 0, 5}, H)); // mid, inside
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{1.01f, 0, 5}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{0, 0, -0.5f}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 10.5f}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{0.5f, 0, 9.9f}, H)); // near apex, narrow
    }

    // Weeping: spheroid math; pick params that diverge from OblateSpheroid
    // defaults (low top_height_ratio) so we exercise the param path.
    {
        treegen::TreeDescriptor::Envelope env;
        env.shape = ES::Weeping;
        env.width_m = 6.0f;
        env.height_ratio = 1.0f;
        env.top_height_ratio = 0.6f; // squished vertically
        // → a=3, b=3, center_z=5. So inside between z in [2,8] near axis.
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 5}, H));
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 2}, H));
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 8}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 1.5f}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 8.5f}, H));
    }

    // Fastigiate: cylinder width=2 → r=1, height = H*height_ratio = 10.
    {
        treegen::TreeDescriptor::Envelope env;
        env.shape = ES::Fastigiate;
        env.width_m = 2.0f;
        env.height_ratio = 1.0f;
        env.top_height_ratio = 1.0f;

        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 0}, H));
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 10}, H));
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{1, 0, 5}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{1.01f, 0, 5}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 10.1f}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{0, 0, -0.01f}, H));
    }

    // Fan: oblate spheroid with very small top_height_ratio.
    // width=8 → a=4; top_height_ratio=0.2 → b = (1.0 * H/2) * 0.2 = 1 (with
    // height_ratio=1, H=10). center_z = H*height_ratio/2 = 5.
    {
        treegen::TreeDescriptor::Envelope env;
        env.shape = ES::Fan;
        env.width_m = 8.0f;
        env.height_ratio = 1.0f;
        env.top_height_ratio = 0.2f;
        // → a=4, b=1, center=5.
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 5}, H));
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{4, 0, 5}, H));
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 4}, H));
        REQUIRE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 6}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{0, 0, 3.9f}, H));
        REQUIRE_FALSE(treegen::envelope_contains(env.shape, env, vec3{4.01f, 0, 5}, H));
    }
}

TEST_CASE("treegen_descriptor: envelope_aabb contains tested inside points",
          "[treegen][treegen_descriptor][treegen_descriptor_envelope_aabb]") {
    using ES = treegen::EnvelopeShape;
    using treegen::vec3;
    const float H = 10.0f;

    constexpr ES k_all[] = {
        ES::OblateSpheroid, ES::Conical, ES::Weeping, ES::Fastigiate, ES::Fan,
    };

    treegen::TreeDescriptor::Envelope env;
    env.width_m = 6.0f;
    env.height_ratio = 1.0f;
    env.top_height_ratio = 0.8f;

    // Sweep a 7×7×11 grid of candidates per shape. Every point that
    // envelope_contains accepts MUST also be inside the AABB. The AABB itself
    // must be non-empty (max > min on each axis).
    for (ES shape : k_all) {
        env.shape = shape;
        treegen::Aabb box = treegen::envelope_aabb(shape, env, H);
        INFO("shape=" << treegen::envelope_shape_to_string(shape));
        REQUIRE(box.max.x > box.min.x);
        REQUIRE(box.max.y > box.min.y);
        REQUIRE(box.max.z > box.min.z);

        for (int ix = -3; ix <= 3; ++ix) {
            for (int iy = -3; iy <= 3; ++iy) {
                for (int iz = 0; iz <= 10; ++iz) {
                    vec3 p{static_cast<float>(ix), static_cast<float>(iy), static_cast<float>(iz)};
                    if (treegen::envelope_contains(shape, env, p, H)) {
                        INFO("inside p=(" << p.x << "," << p.y << "," << p.z << ")");
                        REQUIRE(aabb_contains(box, p));
                    }
                }
            }
        }
    }
}

TEST_CASE("treegen_descriptor: scenario.cpp loads kind:\"tree\"",
          "[treegen][treegen_descriptor][treegen_descriptor_scenario_load]") {
    namespace ts = rynx::test_support;

    const auto fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/c3_oak.json");
    REQUIRE_FALSE(fixture.empty());

    treegen::Scenario s = treegen::load_scenario(fixture);
    REQUIRE(s.kind == "tree");
    REQUIRE(s.name == "oak_default");
    REQUIRE(s.scenario_fnv != 0u);
    REQUIRE(s.tree.species == "oak");
    REQUIRE(s.tree.envelope.shape == treegen::EnvelopeShape::OblateSpheroid);

    // trivial_cylinder path still works alongside the tree path.
    const auto cyl_fixture = ts::find_repo_file("tools/rynx-treegen/scenarios/trivial_cylinder.json");
    REQUIRE_FALSE(cyl_fixture.empty());
    treegen::Scenario cyl = treegen::load_scenario(cyl_fixture);
    REQUIRE(cyl.kind == "trivial_cylinder");
    REQUIRE(cyl.cyl.radial_segments == 12);
}
