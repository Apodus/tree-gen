#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace treegen {

// Sorted list of the 4 first-ship species.
std::vector<std::string> get_species_names();

// Path of the scenario JSON for `name` under `root` (treegen_root.hpp):
// <root>/scenarios/c3_<name>.json. Empty string if that file does not exist.
std::string get_scenario_path(const std::filesystem::path& root, const std::string& name);

} // namespace treegen
