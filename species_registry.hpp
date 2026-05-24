#pragma once

#include <string>
#include <vector>

namespace treegen {

// Sorted list of the 4 first-ship species.
std::vector<std::string> get_species_names();

// Returns absolute path to the scenario JSON for `name`, searching
// scenarios/ next to CWD and tools/rynx-treegen/scenarios/ (same dirs
// as find_scenario_files). Empty string if not found.
std::string get_scenario_path(const std::string& name);

} // namespace treegen
