// C1 P4 — end-to-end golden harness for the treegen pipeline: the trivial
// cylinder scenario through rynx-treegen, rendered by the game's --render-glb
// mode, pHash within 6 of tests/golden/treegen/trivial_42.png.
//
// The frame is near-uniform (the cylinder fills it), which makes its pHash
// sensitive to a one-level luminance shift: before the render ran on the
// lockstep timestep, wall time moved the sun by a sliver per run and this
// gate failed about one run in nine at hamming 8, on one of two stable images.

#include "../external/catch2/catch.hpp"

#include "render_golden.hpp"
#include "test_support_paths.hpp"

#include <string>

TEST_CASE("treegen: render trivial cylinder phash within 6 of golden",
          "[treegen_render_phash]") {
    namespace rg = treegen::render_golden;

    const std::string scenario = rynx::test_support::find_repo_file(
        "tools/rynx-treegen/scenarios/trivial_cylinder.json");
    INFO("scenario=" << scenario);
    REQUIRE_FALSE(scenario.empty());

    const auto glb = rg::generate_glb("--scenario=\"" + scenario + "\" --seed=42", "treegen_c1_42.glb");
    const auto png = rg::render(glb, 0, "treegen_c1_42.png");
    const auto g = rg::grade_against_golden(png, "trivial_42.png");
    if (!g) return;
    INFO("rendered_phash=" << std::hex << g->rendered << " golden_phash=" << g->golden
         << " hamming=" << std::dec << g->hamming);
    REQUIRE(g->hamming <= 6);
}
