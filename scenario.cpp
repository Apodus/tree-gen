#include "scenario.hpp"

#include "det_rng.hpp"
#include "json_reader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace treegen {

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("scenario: cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

template <class T>
T require(std::optional<T> v, const char* key) {
    if (!v) throw std::runtime_error(std::string("scenario: missing or wrong-type field: ") + key);
    return *v;
}

} // anonymous namespace

Scenario load_scenario(const std::string& path) {
    std::string text = slurp(path);
    auto result = json::parse(text);
    if (!result.ok) {
        throw std::runtime_error("scenario: JSON parse error at byte " +
                                 std::to_string(result.error_offset) + " in " + path);
    }
    if (!result.root.is_object()) {
        throw std::runtime_error("scenario: top-level JSON must be an object: " + path);
    }

    Scenario s;
    s.kind = require(json::get_string(result.root, "kind"), "kind");
    s.name = require(json::get_string(result.root, "name"), "name");
    s.scenario_fnv = fnv1a64(s.name);

    if (s.kind == "trivial_cylinder") {
        s.cyl.radius          = static_cast<float>(require(json::get_number(result.root, "radius"),         "radius"));
        s.cyl.height          = static_cast<float>(require(json::get_number(result.root, "height"),         "height"));
        s.cyl.radial_segments = require(json::get_int   (result.root, "radial_segments"), "radial_segments");
        s.cyl.axial_segments  = require(json::get_int   (result.root, "axial_segments"),  "axial_segments");
    }
    else if (s.kind == "tree") {
        // The TreeDescriptor body parses the same top-level object. Missing
        // fields fall back to POD defaults; species is permitted-empty here
        // (load_scenario already requires a non-empty `name`).
        s.tree = parse_tree_descriptor(result.root);
    }
    else {
        throw std::runtime_error("scenario: unknown kind '" + s.kind + "' in " + path);
    }

    return s;
}

} // namespace treegen
