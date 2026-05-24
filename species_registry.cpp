#include "species_registry.hpp"

#include <filesystem>

namespace treegen {

std::vector<std::string> get_species_names() {
    return {"birch", "maple", "oak", "pine"};
}

std::string get_scenario_path(const std::string& name) {
    namespace fs = std::filesystem;
    const std::string filename = "c3_" + name + ".json";
    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (ec) return {};

    // Same search dirs as find_scenario_files() in main.cpp.
    const fs::path candidates[] = {
        cwd / "scenarios" / filename,
        cwd / "tools" / "rynx-treegen" / "scenarios" / filename,
    };
    for (const auto& p : candidates) {
        if (fs::exists(p, ec) && !ec) return p.string();
    }
    return {};
}

} // namespace treegen
