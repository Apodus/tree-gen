#include "skeleton_json.hpp"

#include "skeleton.hpp"

#include <cstdio>
#include <string>

namespace treegen {

namespace {

void emit_float(std::string& out, float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    out += buf;
}

void emit_int(std::string& out, long long v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", v);
    out += buf;
}

void emit_vec3(std::string& out, vec3 v) {
    out += "[";
    emit_float(out, v.x); out += ", ";
    emit_float(out, v.y); out += ", ";
    emit_float(out, v.z); out += "]";
}

void indent(std::string& out, int n) {
    out.append(static_cast<size_t>(n) * 2, ' ');
}

} // anonymous namespace

std::string dump_skeleton_json(const TreeSkeleton& skel) {
    std::string out;
    out.reserve(64 * skel.nodes.size() + 256);

    out += "{\n";
    indent(out, 1); out += "\"attractors_consumed\": ";
    emit_int(out, skel.attractors_consumed); out += ",\n";
    indent(out, 1); out += "\"iterations_run\": ";
    emit_int(out, skel.iterations_run); out += ",\n";
    indent(out, 1); out += "\"node_count\": ";
    emit_int(out, static_cast<long long>(skel.nodes.size())); out += ",\n";
    indent(out, 1); out += "\"leaf_site_count\": ";
    emit_int(out, static_cast<long long>(skel.leaf_sites.size())); out += ",\n";

    indent(out, 1); out += "\"nodes\": [";
    if (skel.nodes.empty()) {
        out += "]";
    } else {
        out += "\n";
        for (size_t i = 0; i < skel.nodes.size(); ++i) {
            const auto& n = skel.nodes[i];
            indent(out, 2); out += "{\"i\": "; emit_int(out, static_cast<long long>(i));
            out += ", \"parent\": ";     emit_int(out, n.parent_index);
            out += ", \"pos\": ";        emit_vec3(out, n.position);
            out += ", \"axis\": ";       emit_vec3(out, n.axis);
            out += ", \"radius\": ";     emit_float(out, n.radius);
            out += ", \"depth\": ";      emit_int(out, n.depth);
            out += ", \"t\": ";          emit_float(out, n.t_along_parent);
            out += ", \"wind_tier\": ";  emit_int(out, n.wind_tier);
            out += "}";
            if (i + 1 < skel.nodes.size()) out += ",";
            out += "\n";
        }
        indent(out, 1); out += "]";
    }
    out += ",\n";

    indent(out, 1); out += "\"leaf_sites\": [";
    if (skel.leaf_sites.empty()) {
        out += "]";
    } else {
        out += "\n";
        for (size_t i = 0; i < skel.leaf_sites.size(); ++i) {
            const auto& ls = skel.leaf_sites[i];
            // Deterministic key order: position, normal, branch_id, type, age, t.
            // position is the C5 P1 authoritative field; the legacy `t` is
            // kept at the end so dumps don't reorder if a future deletion
            // happens.
            indent(out, 2); out += "{\"pos\": "; emit_vec3(out, ls.position);
            out += ", \"normal\": ";    emit_vec3(out, ls.normal);
            out += ", \"branch_id\": "; emit_int(out, ls.branch_id);
            out += ", \"type\": ";      emit_int(out, ls.type);
            out += ", \"age\": ";       emit_float(out, ls.age);
            out += ", \"t\": ";         emit_float(out, ls.t_along_branch);
            out += ", \"branch_tangent\": ";        emit_vec3(out, ls.branch_tangent);
            out += ", \"branch_normal\": ";         emit_vec3(out, ls.branch_normal);
            out += ", \"branch_depth_fraction\": "; emit_float(out, ls.branch_depth_fraction);
            out += "}";
            if (i + 1 < skel.leaf_sites.size()) out += ",";
            out += "\n";
        }
        indent(out, 1); out += "]";
    }
    out += "\n";
    out += "}\n";
    return out;
}

} // namespace treegen
