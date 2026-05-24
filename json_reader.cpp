#include "json_reader.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

namespace treegen::json {

const value* value::find(std::string_view key) const {
    if (kind != kind_t::obj_t) return nullptr;
    for (const auto& kv : o) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

namespace {

class parser {
    const char* p;
    const char* e;
    const char* begin;
    bool ok = true;
    size_t fail_at = 0;

    void skip_ws() {
        while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    void fail() {
        if (ok) {
            ok = false;
            fail_at = static_cast<size_t>(p - begin);
        }
    }

    value parse_string() {
        value v; v.kind = value::kind_t::str_t;
        if (p >= e || *p != '"') { fail(); return v; }
        ++p;
        while (p < e) {
            char c = *p++;
            if (c == '"') return v;
            if (c == '\\' && p < e) {
                char esc = *p++;
                switch (esc) {
                    case '"':  v.s.push_back('"'); break;
                    case '\\': v.s.push_back('\\'); break;
                    case '/':  v.s.push_back('/'); break;
                    case 'n':  v.s.push_back('\n'); break;
                    case 't':  v.s.push_back('\t'); break;
                    case 'r':  v.s.push_back('\r'); break;
                    case 'b':  v.s.push_back('\b'); break;
                    case 'f':  v.s.push_back('\f'); break;
                    case 'u':  // skip the 4 hex digits; we never need unicode here
                        if (p + 4 <= e) p += 4;
                        v.s.push_back('?');
                        break;
                    default:   v.s.push_back(esc); break;
                }
            }
            else {
                v.s.push_back(c);
            }
        }
        fail();
        return v;
    }

    value parse_number() {
        value v; v.kind = value::kind_t::num_t;
        const char* start = p;
        if (p < e && (*p == '-' || *p == '+')) ++p;
        while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) ++p;
        v.n = std::strtod(std::string(start, static_cast<size_t>(p - start)).c_str(), nullptr);
        return v;
    }

    value parse_literal() {
        if (p + 4 <= e && std::memcmp(p, "true", 4) == 0)  { p += 4; value v; v.kind = value::kind_t::bool_t; v.b = true;  return v; }
        if (p + 5 <= e && std::memcmp(p, "false", 5) == 0) { p += 5; value v; v.kind = value::kind_t::bool_t; v.b = false; return v; }
        if (p + 4 <= e && std::memcmp(p, "null", 4) == 0)  { p += 4; return value(); }
        fail();
        return value();
    }

    value parse_array() {
        value v; v.kind = value::kind_t::arr_t;
        ++p;
        skip_ws();
        if (p < e && *p == ']') { ++p; return v; }
        while (p < e && ok) {
            v.a.push_back(parse_value());
            skip_ws();
            if (p < e && *p == ',') { ++p; skip_ws(); continue; }
            if (p < e && *p == ']') { ++p; return v; }
            fail();
            break;
        }
        return v;
    }

    value parse_object() {
        value v; v.kind = value::kind_t::obj_t;
        ++p;
        skip_ws();
        if (p < e && *p == '}') { ++p; return v; }
        while (p < e && ok) {
            skip_ws();
            value key = parse_string();
            skip_ws();
            if (p >= e || *p != ':') { fail(); break; }
            ++p;
            value val = parse_value();
            v.o.emplace_back(std::move(key.s), std::move(val));
            skip_ws();
            if (p < e && *p == ',') { ++p; continue; }
            if (p < e && *p == '}') { ++p; return v; }
            fail();
            break;
        }
        return v;
    }

public:
    parser(const char* b, size_t len) : p(b), e(b + len), begin(b) {}

    value parse_value() {
        skip_ws();
        if (p >= e) { fail(); return value(); }
        char c = *p;
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f' || c == 'n') return parse_literal();
        return parse_number();
    }

    bool good() const { return ok; }
    size_t error_offset() const { return fail_at; }
};

} // anonymous namespace

parse_result parse(std::string_view input) {
    parse_result r;
    parser p(input.data(), input.size());
    r.root = p.parse_value();
    r.ok = p.good();
    r.error_offset = p.error_offset();
    return r;
}

std::optional<double> get_number(const value& obj, std::string_view key) {
    const value* v = obj.find(key);
    if (!v || v->kind != value::kind_t::num_t) return std::nullopt;
    return v->n;
}

std::optional<std::string> get_string(const value& obj, std::string_view key) {
    const value* v = obj.find(key);
    if (!v || v->kind != value::kind_t::str_t) return std::nullopt;
    return v->s;
}

std::optional<int> get_int(const value& obj, std::string_view key) {
    auto n = get_number(obj, key);
    if (!n) return std::nullopt;
    return static_cast<int>(*n);
}

const value* get_object(const value& obj, std::string_view key) {
    const value* v = obj.find(key);
    if (!v || v->kind != value::kind_t::obj_t) return nullptr;
    return v;
}

const value* get_array(const value& obj, std::string_view key) {
    const value* v = obj.find(key);
    if (!v || v->kind != value::kind_t::arr_t) return nullptr;
    return v;
}

std::optional<vec3> get_vec3(const value& obj, std::string_view key) {
    const value* v = get_array(obj, key);
    if (!v || v->a.size() != 3) return std::nullopt;
    for (const auto& el : v->a) {
        if (el.kind != value::kind_t::num_t) return std::nullopt;
    }
    return vec3{static_cast<float>(v->a[0].n),
                static_cast<float>(v->a[1].n),
                static_cast<float>(v->a[2].n)};
}

} // namespace treegen::json
