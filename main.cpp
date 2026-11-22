// NOTE: this exe must stay /fp:precise — byte-hash determinism depends on it.
// See rynx/generate/treegen.sharpmake.cs.

#include "bark_texture.hpp"
#include "branch_mesh.hpp"
#include "crown_oracle.hpp"
#include "collision_fit.hpp"
#include "det_rng.hpp"
#include "envelopes.hpp"
#include "face_budget.hpp"
#include "generate_tree.hpp"
#include "glb_writer.hpp"
#include "leaf_atlas.hpp"
#include "leaf_budget.hpp"
#include "leaf_geometry.hpp"
#include "leaf_placement.hpp"
#include "lod_emitter.hpp"
#include "png_encoder.hpp"
#include "scenario.hpp"
#include "skeleton.hpp"
#include "skeleton_json.hpp"
#include "space_colonization.hpp"
#include "species_registry.hpp"
#include "tree_descriptor.hpp"
#include "treegen_root.hpp"
#include "tree_ledger.hpp"
#include "trivial_cylinder.hpp"
#include "vec3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifndef TREEGEN_ROOT_FROM_EXE
#error "rynx-treegen finds its scenarios from the exe's own location: the build must define TREEGEN_ROOT_FROM_EXE (the relative path from the exe's directory to the repository root)"
#endif

namespace {

// The repository root for this exe; empty when the layout the build promised is absent.
std::optional<std::filesystem::path> tool_root() {
    return treegen::locate_root(treegen::executable_directory(), TREEGEN_ROOT_FROM_EXE);
}

// Single source of truth for shape names enumerated in --list-envelope-shapes.
// EnvelopeShape <-> string conversion lives in tree_descriptor.cpp; this is
// the enumeration of the 5 values, deliberately sorted alphabetically for
// stable script-parseable output.
constexpr std::array<treegen::EnvelopeShape, 5> k_all_envelope_shapes = {
    treegen::EnvelopeShape::Conical,
    treegen::EnvelopeShape::Fan,
    treegen::EnvelopeShape::Fastigiate,
    treegen::EnvelopeShape::OblateSpheroid,
    treegen::EnvelopeShape::Weeping,
};

void print_usage() {
    std::fprintf(stderr,
        "usage: rynx-treegen --scenario=<path> [--seed=<u64>] [--out=<path>] [--dump-skeleton-json=<path>] [--ledger=<path>]\n"
        "       rynx-treegen --species=<name> [--seed=<u64>] [--out=<path>] [--dump-skeleton-json=<path>] [--ledger=<path>]\n"
        "       rynx-treegen (--scenario=<path> | --species=<name>) [--seed=<u64>] --probe-crown=<out.json> [probe flags]\n"
        "       rynx-treegen --glb=<file> --probe-crown=<out.json> [probe flags]\n"
        "       rynx-treegen --list-species\n"
        "       rynx-treegen --print-root\n"
        "       rynx-treegen --list-envelope-shapes\n"
        "       rynx-treegen --help\n");
}

// Enumerate <root>/scenarios/*.json. Best-effort — failure is silently
// swallowed; --help still prints the flag list.
std::vector<std::filesystem::path> find_scenario_files(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> hits;
    namespace fs = std::filesystem;

    auto scan = [&](const fs::path& dir) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) return;
        for (const auto& e : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!e.is_regular_file()) continue;
            if (e.path().extension() == ".json") hits.push_back(e.path());
        }
    };

    scan(root / "scenarios");
    std::sort(hits.begin(), hits.end());
    return hits;
}

void print_help() {
    std::fprintf(stderr,
        "rynx-treegen — procedural tree skeleton generator.\n\n"
        "flags:\n"
        "  --scenario=<path>             Scenario JSON (required unless --species).\n"
        "  --species=<name>              Species preset (resolves to c3_<name>.json).\n"
        "  --seed=<u64>                  CLI seed XOR'd with scenario_fnv (default 0).\n"
        "  --out=<path>                  Mesh output (.glb).\n"
        "  --dump-skeleton-json=<path>   Write TreeSkeleton JSON dump (tree scenarios).\n"
        "  --ledger=<path>               Write the geometry ledger JSON (tree scenarios; needs --out).\n"
        "  --probe-crown=<path>          Crown probe report JSON (schema 1) of the tree or of --glb.\n"
        "  --glb=<file>                  Probe a GLB written by treegen instead of generating one.\n"
        "  --modes=oracle,sunlit         Probe modes (default: all).\n"
        "  --sun-elevations=<deg,...>    Default 10,20,30,45,60,70,75,90.\n"
        "  --sun-azimuths=<count>        Azimuths about the vertical axis, from 0 (default 4).\n"
        "  --eyes=<x,y,z;...>            Eye points (object space) for the sunlit share among visible sites.\n"
        "  --probe-lod=<k>               Mesh index probed (default 0).\n"
        "  --list-species                Print available species names + exit.\n"
        "  --print-root                  Print the repository root found from the exe's location + exit.\n"
        "  --list-envelope-shapes        Print the 5 envelope shape names + exit.\n"
        "  -h, --help                    Print this message + scenario index + exit.\n"
        "\n"
        "Both --key=value and --key value forms accepted.\n"
        "--species and --scenario are mutually exclusive (exit 2 if both given).\n");

    const auto root = tool_root();
    const auto scenarios = root ? find_scenario_files(*root) : std::vector<std::filesystem::path>{};
    if (!root) {
        std::fprintf(stderr, "\nno repository root (%s) at the layout this exe was built for; scenarios unavailable.\n",
                     std::string(treegen::k_root_marker).c_str());
    } else if (scenarios.empty()) {
        std::fprintf(stderr, "\nno scenarios/*.json under %s.\n", root->string().c_str());
    } else {
        std::fprintf(stderr, "\navailable scenarios (%zu):\n", scenarios.size());
        for (const auto& p : scenarios) {
            std::fprintf(stderr, "  %s\n", p.string().c_str());
        }
    }
}

// Accept either `--key=value` or `--key value`. `--key` matches keys with
// matching trailing nul (no false-positive matches like --seeds vs --seed).
// Returns nullptr if not matched. Advances `argi` past the value form when
// the space-separated form is consumed.
const char* match_arg(const char* arg, const char* key, int argc, char** argv, int& argi) {
    size_t key_len = std::strlen(key);
    if (std::strncmp(arg, key, key_len) != 0) return nullptr;
    if (arg[key_len] == '=') return arg + key_len + 1;
    if (arg[key_len] == '\0') {
        if (argi + 1 >= argc) return nullptr;
        ++argi;
        return argv[argi];
    }
    return nullptr;
}

uint64_t parse_u64(const char* s) {
    return static_cast<uint64_t>(std::strtoull(s, nullptr, 10));
}

float vlen(treegen::vec3 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

struct SkeletonStats {
    size_t nodes              = 0;
    size_t leaf_tips          = 0;   // nodes with no child
    int    max_depth          = 0;
    float  total_branch_len_m = 0.0f;
    int    attractors_used    = 0;
    int    attractors_total   = 0;
};

SkeletonStats compute_stats(const treegen::TreeSkeleton& skel, int attractors_total) {
    SkeletonStats s;
    s.nodes              = skel.nodes.size();
    s.attractors_used    = skel.attractors_consumed;
    s.attractors_total   = attractors_total;

    // Count children per node; a node with 0 children is a leaf tip. Depth +
    // segment length from parent are O(N) over the dense node array.
    std::vector<int> child_count(skel.nodes.size(), 0);
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        const auto& n = skel.nodes[i];
        if (n.depth > s.max_depth) s.max_depth = n.depth;
        if (n.parent_index >= 0) {
            const auto& parent = skel.nodes[static_cast<size_t>(n.parent_index)];
            s.total_branch_len_m += vlen(n.position - parent.position);
            ++child_count[static_cast<size_t>(n.parent_index)];
        }
    }
    for (size_t i = 0; i < skel.nodes.size(); ++i) {
        if (child_count[i] == 0) ++s.leaf_tips;
    }
    return s;
}

// Comma-separated floats; false on an empty or malformed item.
bool parse_float_list(const std::string& s, std::vector<float>& out) {
    out.clear();
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t end = std::min(s.find(',', pos), s.size());
        const std::string item = s.substr(pos, end - pos);
        char* stop = nullptr;
        const float v = std::strtof(item.c_str(), &stop);
        if (item.empty() || stop != item.c_str() + item.size() || !std::isfinite(v)) return false;
        out.push_back(v);
        pos = end + 1;
    }
    return !out.empty();
}

// --modes, --sun-elevations, --sun-azimuths, --eyes, --probe-lod (K17).
bool parse_probe_options(const std::string& modes, const std::string& elevations, const std::string& azimuths,
                         const std::string& eyes, const std::string& lod, treegen::crown::ProbeOptions& o,
                         int& lod_index, std::string& err) {
    if (!modes.empty()) {
        o.oracle = o.sunlit = false;
        size_t pos = 0;
        while (pos <= modes.size()) {
            const size_t end = std::min(modes.find(',', pos), modes.size());
            const std::string m = modes.substr(pos, end - pos);
            if (m == "oracle") o.oracle = true;
            else if (m == "sunlit") o.sunlit = true;
            else { err = "--modes: unknown mode '" + m + "' (this build has: oracle, sunlit)"; return false; }
            pos = end + 1;
        }
    }
    if (!elevations.empty()) {
        if (!parse_float_list(elevations, o.elevations_deg)) { err = "--sun-elevations: expected degrees, comma separated"; return false; }
        for (float e : o.elevations_deg)
            if (!(e > 0.0f && e <= 90.0f)) { err = "--sun-elevations: each must be in (0, 90]"; return false; }
    }
    if (!azimuths.empty()) {
        char* stop = nullptr;
        const long n = std::strtol(azimuths.c_str(), &stop, 10);
        if (stop != azimuths.c_str() + azimuths.size() || n < 1 || n > 360) { err = "--sun-azimuths: expected a count in [1, 360]"; return false; }
        o.azimuths = int(n);
    }
    if (!eyes.empty()) {
        size_t pos = 0;
        while (pos <= eyes.size()) {
            const size_t end = std::min(eyes.find(';', pos), eyes.size());
            std::vector<float> xyz;
            if (!parse_float_list(eyes.substr(pos, end - pos), xyz) || xyz.size() != 3) {
                err = "--eyes: expected x,y,z;x,y,z;... in the tree's object space";
                return false;
            }
            o.eyes.push_back(treegen::vec3{xyz[0], xyz[1], xyz[2]});
            pos = end + 1;
        }
    }
    char* stop = nullptr;
    const long l = std::strtol(lod.c_str(), &stop, 10);
    if (lod.empty() || stop != lod.c_str() + lod.size() || l < 0) { err = "--probe-lod: expected a mesh index >= 0"; return false; }
    lod_index = int(l);
    return true;
}

int write_probe_report(const std::string& doc, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "rynx-treegen: cannot open --probe-crown path: %s\n", path.c_str());
        return 1;
    }
    f.write(doc.data(), static_cast<std::streamsize>(doc.size()));
    if (!f) {
        std::fprintf(stderr, "rynx-treegen: write failed: %s\n", path.c_str());
        return 1;
    }
    std::printf("treegen: wrote crown probe report to %s\n", path.c_str());
    return 0;
}

} // anonymous namespace

int main(int argc, char** argv) {
    try {
        std::string scenario_path;
        std::string species_name;
        std::string out_path;
        std::string skeleton_json_path;
        std::string ledger_path;
        uint64_t    cli_seed = 0;
        bool        seed_given = false;
        // Crown probe (crown-shadows S-P0a, K17).
        std::string probe_path;
        std::string glb_path;
        std::string probe_modes;
        std::string probe_elevations;
        std::string probe_azimuths;
        std::string probe_eyes;
        std::string probe_lod = "0";

        // --dump-skeleton-json MUST be matched before --dump (prefix-wise the
        // longer key first), and before --scenario (no prefix conflict but
        // order is the seal).
        for (int i = 1; i < argc; ++i) {
            const char* arg = argv[i];
            if (std::strcmp(arg, "--list-envelope-shapes") == 0) {
                for (const auto s : k_all_envelope_shapes) {
                    std::printf("%s\n", treegen::envelope_shape_to_string(s));
                }
                return 0;
            }
            if (std::strcmp(arg, "--print-root") == 0) {
                const auto root = tool_root();
                if (!root) {
                    std::fprintf(stderr, "rynx-treegen: no %s at the layout this exe was built for\n",
                                 std::string(treegen::k_root_marker).c_str());
                    return 2;
                }
                std::printf("%s\n", root->string().c_str());
                return 0;
            }
            if (std::strcmp(arg, "--list-species") == 0) {
                for (const auto& name : treegen::get_species_names()) {
                    std::printf("%s\n", name.c_str());
                }
                return 0;
            }
            if (const char* v = match_arg(arg, "--dump-skeleton-json", argc, argv, i)) { skeleton_json_path = v; continue; }
            if (const char* v = match_arg(arg, "--scenario", argc, argv, i)) { scenario_path = v; continue; }
            if (const char* v = match_arg(arg, "--species",  argc, argv, i)) { species_name = v; continue; }
            if (const char* v = match_arg(arg, "--seed",     argc, argv, i)) { cli_seed = parse_u64(v); seed_given = true; continue; }
            if (const char* v = match_arg(arg, "--out",      argc, argv, i)) { out_path = v; continue; }
            if (const char* v = match_arg(arg, "--ledger",   argc, argv, i)) { ledger_path = v; continue; }
            if (const char* v = match_arg(arg, "--probe-crown",    argc, argv, i)) { probe_path = v; continue; }
            if (const char* v = match_arg(arg, "--glb",            argc, argv, i)) { glb_path = v; continue; }
            if (const char* v = match_arg(arg, "--modes",          argc, argv, i)) { probe_modes = v; continue; }
            if (const char* v = match_arg(arg, "--sun-elevations", argc, argv, i)) { probe_elevations = v; continue; }
            if (const char* v = match_arg(arg, "--sun-azimuths",   argc, argv, i)) { probe_azimuths = v; continue; }
            if (const char* v = match_arg(arg, "--eyes",           argc, argv, i)) { probe_eyes = v; continue; }
            if (const char* v = match_arg(arg, "--probe-lod",      argc, argv, i)) { probe_lod = v; continue; }
            if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) { print_help(); return 0; }
            std::fprintf(stderr, "rynx-treegen: unknown argument: %s\n", arg);
            print_usage();
            return 2;
        }

        // --species and --scenario are mutually exclusive.
        if (!species_name.empty() && !scenario_path.empty()) {
            std::fprintf(stderr, "rynx-treegen: --species and --scenario are mutually exclusive\n");
            print_usage();
            return 2;
        }

        // Crown probe arguments: parsed (and refused) before any generation.
        const bool probe_flags = !glb_path.empty() || !probe_modes.empty() || !probe_elevations.empty() ||
                                 !probe_azimuths.empty() || !probe_eyes.empty() || probe_lod != "0";
        if (probe_flags && probe_path.empty()) {
            std::fprintf(stderr, "rynx-treegen: --glb, --modes, --sun-elevations, --sun-azimuths, --eyes and --probe-lod need --probe-crown\n");
            print_usage();
            return 2;
        }
        treegen::crown::ProbeOptions probe_options;
        int probe_lod_index = 0;
        if (!probe_path.empty()) {
            std::string err;
            if (!parse_probe_options(probe_modes, probe_elevations, probe_azimuths, probe_eyes, probe_lod,
                                     probe_options, probe_lod_index, err)) {
                std::fprintf(stderr, "rynx-treegen: %s\n", err.c_str());
                return 2;
            }
        }
        if (!glb_path.empty()) {
            if (!scenario_path.empty() || !species_name.empty() || seed_given || !out_path.empty() ||
                !ledger_path.empty() || !skeleton_json_path.empty()) {
                std::fprintf(stderr, "rynx-treegen: --glb is a probe input on its own (no --scenario, --species, --seed, --out, --ledger, --dump-skeleton-json)\n");
                print_usage();
                return 2;
            }
            std::ifstream f(glb_path, std::ios::binary);
            if (!f) {
                std::fprintf(stderr, "rynx-treegen: cannot open --glb: %s\n", glb_path.c_str());
                return 1;
            }
            const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            const std::string input = treegen::crown::probe_input_glb(glb_path, bytes, probe_lod_index);
            return write_probe_report(treegen::crown::probe_glb_report(bytes, probe_lod_index, probe_options, input),
                                      probe_path);
        }

        // The levels are built only when a GLB is written.
        if (!ledger_path.empty() && out_path.empty()) {
            std::fprintf(stderr, "rynx-treegen: --ledger needs --out (the levels are built only when a GLB is written)\n");
            print_usage();
            return 2;
        }

        // Resolve --species to a scenario path.
        if (!species_name.empty()) {
            const auto root = tool_root();
            if (!root) {
                std::fprintf(stderr, "rynx-treegen: --species needs the repository root: no %s at the layout this exe was built for\n",
                             std::string(treegen::k_root_marker).c_str());
                return 2;
            }
            scenario_path = treegen::get_scenario_path(*root, species_name);
            if (scenario_path.empty()) {
                std::fprintf(stderr, "rynx-treegen: unknown species '%s'\n", species_name.c_str());
                return 2;
            }
        }

        if (scenario_path.empty()) {
            std::fprintf(stderr, "rynx-treegen: --scenario or --species is required\n");
            print_usage();
            return 2;
        }

        treegen::Scenario s = treegen::load_scenario(scenario_path);
        uint64_t seed_effective = cli_seed ^ s.scenario_fnv;
        if (!ledger_path.empty() && s.kind != "tree") {
            std::fprintf(stderr, "rynx-treegen: --ledger is for tree scenarios (this one is %s)\n", s.kind.c_str());
            return 2;
        }
        if (!probe_path.empty() && s.kind != "tree") {
            std::fprintf(stderr, "rynx-treegen: --probe-crown is for tree scenarios (this one is %s)\n", s.kind.c_str());
            return 2;
        }

        // %g for floats: deterministic shortest round-trippable form across
        // MSVC stdlib versions. %016llx pins width so seed lines align byte-
        // for-byte across runs.
        if (s.kind == "trivial_cylinder") {
            std::printf(
                "treegen: scenario=%s seed=%llu seed_effective=%016llx radius=%g height=%g radial=%d axial=%d\n",
                s.name.c_str(),
                static_cast<unsigned long long>(cli_seed),
                static_cast<unsigned long long>(seed_effective),
                static_cast<double>(s.cyl.radius),
                static_cast<double>(s.cyl.height),
                s.cyl.radial_segments,
                s.cyl.axial_segments);
        }
        else if (s.kind == "tree") {
            // C3 P2 — grow the skeleton (generate_tree); print summary stats;
            // optional JSON dump. Levels and the GLB are built only for --out
            // and --probe-crown.
            // Without --out or --probe-crown only the grown stage runs (the tool
            // never built levels for a skeleton-only call).
            const bool build = !out_path.empty() || !probe_path.empty();
            const treegen::GeneratedTree gen = build ? treegen::generate_tree(s, cli_seed)
                                                     : treegen::GeneratedTree{{treegen::grow_tree(s, cli_seed)}};
            const treegen::TreeSkeleton& skel    = gen.skeleton;
            const treegen::TrunkCapsule& capsule = gen.capsule;

            int max_depth = 0;
            for (const auto& n : skel.nodes) {
                if (n.depth > max_depth) max_depth = n.depth;
            }
            std::printf(
                "treegen: scenario=%s seed=%llu seed_effective=%016llx kind=tree species=%s height_m=%g shape=%s width_m=%g attractors=%d nodes=%zu max_depth=%d attractors_consumed=%d iterations=%d leaves=%zu\n",
                s.name.c_str(),
                static_cast<unsigned long long>(cli_seed),
                static_cast<unsigned long long>(seed_effective),
                s.tree.species.c_str(),
                static_cast<double>(s.tree.height_m),
                treegen::envelope_shape_to_string(s.tree.envelope.shape),
                static_cast<double>(s.tree.envelope.width_m),
                s.tree.branching.attractor_count,
                skel.nodes.size(),
                max_depth,
                skel.attractors_consumed,
                skel.iterations_run,
                skel.leaf_sites.size());
            std::printf("treegen: capsule half_length=%g radius=%g\n",
                        static_cast<double>(capsule.half_length),
                        static_cast<double>(capsule.radius));

            if (!skeleton_json_path.empty()) {
                const std::string dump = treegen::dump_skeleton_json(skel);
                std::ofstream f(skeleton_json_path, std::ios::binary);
                if (!f) {
                    std::fprintf(stderr, "rynx-treegen: cannot open dump path: %s\n",
                                 skeleton_json_path.c_str());
                    return 1;
                }
                f.write(dump.data(), static_cast<std::streamsize>(dump.size()));
                if (!f) {
                    std::fprintf(stderr, "rynx-treegen: write failed: %s\n",
                                 skeleton_json_path.c_str());
                    return 1;
                }
                std::printf("treegen: wrote skeleton JSON to %s (%zu bytes)\n",
                            skeleton_json_path.c_str(), dump.size());

                // C3 P3 polish: per-dump summary stats for visual eyeball-debug.
                const SkeletonStats st = compute_stats(skel, s.tree.branching.attractor_count);
                std::printf(
                    "treegen: skeleton stats:\n"
                    "  nodes:              %zu\n"
                    "  leaf_tips:          %zu\n"
                    "  max_depth:          %d\n"
                    "  total_branch_len_m: %g\n"
                    "  attractors_used:    %d/%d\n",
                    st.nodes, st.leaf_tips, st.max_depth,
                    static_cast<double>(st.total_branch_len_m),
                    st.attractors_used, st.attractors_total);
            }

            // The probe reads the GLB bytes generate_tree returned (what --out
            // writes), through the same loader as --glb.
            if (!probe_path.empty()) {
                const std::string input = treegen::crown::probe_input_scenario(scenario_path, cli_seed, gen.glb, probe_lod_index);
                const int rc = write_probe_report(
                    treegen::crown::probe_glb_report(gen.glb, probe_lod_index, probe_options, input), probe_path);
                if (rc != 0) return rc;
            }

            if (!out_path.empty()) {
                const std::vector<treegen::LodOutput>& lods = gen.lods;
                std::string err;
                if (!treegen::write_glb_bytes(gen.glb, out_path, &err)) {
                    std::fprintf(stderr, "rynx-treegen: write_glb_multi_mesh failed: %s\n", err.c_str());
                    return 1;
                }
                // The ledger after the GLB: a failed GLB write leaves no ledger behind.
                if (!ledger_path.empty()) {
                    const std::string doc = treegen::ledger_json(treegen::compute_ledger(
                        s.tree, skel, std::span<const treegen::LodOutput>(lods.data(), lods.size())));
                    std::ofstream f(ledger_path, std::ios::binary);
                    if (!f) {
                        std::fprintf(stderr, "rynx-treegen: cannot open ledger path: %s\n", ledger_path.c_str());
                        return 1;
                    }
                    f.write(doc.data(), static_cast<std::streamsize>(doc.size()));
                    if (!f) {
                        std::fprintf(stderr, "rynx-treegen: write failed: %s\n", ledger_path.c_str());
                        return 1;
                    }
                    std::printf("treegen: wrote ledger to %s\n", ledger_path.c_str());
                }
                std::error_code ec;
                auto fsize = std::filesystem::file_size(out_path, ec);
                auto leaf_vcount_for = [](const treegen::LodOutput& lod) {
                    // 4 bytes/vert wind weights → vcount = wind/4. Robust against
                    // any future change to per-leaf vert math.
                    return lod.has_leaves ? lod.leaf_wind_weights_packed.size() / 4 : size_t{0};
                };
                auto leaf_tris_for = [](const treegen::LodOutput& lod) {
                    return lod.has_leaves ? lod.leaf_indices_u32.size() / 3 : size_t{0};
                };
                std::printf(
                    "treegen: wrote %s (%zu LODs: "
                    "L0=%zu bark_t + %zu leaf_t (%zu leaf_v), "
                    "L1=%zu bark_t + %zu leaf_t (%zu leaf_v), "
                    "L2=%zu bark_t + %zu leaf_t (%zu leaf_v), "
                    "%llu bytes)\n",
                    out_path.c_str(),
                    lods.size(),
                    lods[0].indices_u32.size() / 3, leaf_tris_for(lods[0]), leaf_vcount_for(lods[0]),
                    lods[1].indices_u32.size() / 3, leaf_tris_for(lods[1]), leaf_vcount_for(lods[1]),
                    lods[2].indices_u32.size() / 3, leaf_tris_for(lods[2]), leaf_vcount_for(lods[2]),
                    static_cast<unsigned long long>(ec ? 0 : fsize));
                return 0;
            }
            return 0;
        }

        if (!out_path.empty()) {
            if (s.kind != "trivial_cylinder") {
                std::fprintf(stderr, "rynx-treegen: --out only implemented for trivial_cylinder in C1\n");
                return 1;
            }

            treegen::cpu_mesh_out cm = treegen::build_trivial_cylinder(
                s.cyl.radius, s.cyl.height,
                s.cyl.radial_segments, s.cyl.axial_segments);

            treegen::PrimitiveData prim{};
            prim.positions = std::span<const float>(cm.positions.data(), cm.positions.size());
            prim.normals   = std::span<const float>(cm.normals.data(), cm.normals.size());
            prim.uvs       = std::span<const float>(cm.uvs.data(), cm.uvs.size());
            prim.indices   = std::span<const uint16_t>(cm.indices.data(), cm.indices.size());

            std::string err;
            treegen::PrimitiveData prims_arr[1] = { prim };
            if (!treegen::write_glb(std::span<const treegen::PrimitiveData>(prims_arr, 1),
                                    out_path, &err)) {
                std::fprintf(stderr, "rynx-treegen: write_glb failed: %s\n", err.c_str());
                return 1;
            }

            std::error_code ec;
            auto size = std::filesystem::file_size(out_path, ec);
            std::printf("treegen: wrote %s (%zu verts, %zu tris, %llu bytes)\n",
                out_path.c_str(),
                cm.positions.size() / 3,
                cm.indices.size() / 3,
                static_cast<unsigned long long>(ec ? 0 : size));
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::fprintf(stderr, "rynx-treegen: error: %s\n", ex.what());
        return 1;
    }
}
