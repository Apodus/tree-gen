#include "species_registry.hpp"

#include <filesystem>

namespace treegen {

std::vector<std::string> get_species_names() {
    return {"birch", "maple", "oak", "pine"};
}

std::string get_scenario_path(const std::filesystem::path& root, const std::string& name) {
    namespace fs = std::filesystem;
    const fs::path p = root / "scenarios" / ("c3_" + name + ".json");
    std::error_code ec;
    return fs::exists(p, ec) && !ec ? p.string() : std::string{};
}

} // namespace treegen
