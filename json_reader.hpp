// Minimal JSON DOM. Lifted in shape from gltf_loader.cpp's anon-ns parser; no
// engine dependency so rynx-treegen can stay link-isolated.
#pragma once

#include "vec3.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace treegen::json {

struct value {
    enum class kind_t { null_t, bool_t, num_t, str_t, arr_t, obj_t };

    kind_t kind = kind_t::null_t;
    bool   b    = false;
    double n    = 0.0;
    std::string s;
    std::vector<value> a;
    std::vector<std::pair<std::string, value>> o;

    const value* find(std::string_view key) const;
    bool is_object() const { return kind == kind_t::obj_t; }
    bool is_array()  const { return kind == kind_t::arr_t; }
};

struct parse_result {
    value root;
    bool  ok    = false;
    size_t error_offset = 0; // byte offset of failure when !ok
};

parse_result parse(std::string_view input);

std::optional<double>      get_number(const value& obj, std::string_view key);
std::optional<std::string> get_string(const value& obj, std::string_view key);
std::optional<int>         get_int   (const value& obj, std::string_view key);

// Nested-container accessors. Return nullptr if `key` missing OR present-but-
// wrong-type (callers treat both as "not provided" and fall back to defaults).
const value* get_object(const value& obj, std::string_view key);
const value* get_array (const value& obj, std::string_view key);

// Parses [x,y,z] array of 3 numbers into vec3. nullopt on any failure
// (missing key, not array, wrong size, non-number element).
std::optional<vec3>        get_vec3  (const value& obj, std::string_view key);

} // namespace treegen::json
