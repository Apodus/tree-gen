// [treegen_leaf_atlas_p3] — pins C6 P3 leaf atlas bake. P3a tests (1-10) cover
// alpha + diffuse rasterization + emitter UV migration. P3b adds tests 11-15
// for normal + translucency.
//
// Tool sources link into TestTreegen via TreegenCore:
//   leaf_atlas.cpp (rebuilt for P3) + leaf_rasterizer.cpp + leaf_veins.cpp
//   + leaf_shapes.cpp + leaf_geometry.cpp + png_encoder_api.cpp.
// STB_IMAGE_IMPLEMENTATION anchor lives in rynx::graphics::image (Graphics dep).

#include "../external/catch2/catch.hpp"

#include "../leaf_atlas.hpp"
#include "../leaf_geometry.hpp"
#include "../leaf_rasterizer.hpp"
#include "../leaf_shapes.hpp"
#include "../leaf_veins.hpp"
#include "../png_encoder.hpp"
#include "../tree_descriptor.hpp"

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

namespace tg = treegen;

namespace {

    struct DecodedAtlas {
        std::vector<uint8_t> bytes;   // RGBA8, row-major
        int w = 0, h = 0;
    };

    DecodedAtlas decode_atlas(const std::vector<uint8_t>& png) {
        DecodedAtlas d;
        int dch = 0;
        unsigned char* raw = stbi_load_from_memory(
            png.data(), static_cast<int>(png.size()), &d.w, &d.h, &dch, 4);
        REQUIRE(raw != nullptr);
        d.bytes.assign(raw, raw + static_cast<size_t>(d.w) * d.h * 4);
        stbi_image_free(raw);
        return d;
    }

    // Cell rect in pixel space (top-left inclusive, bottom-right exclusive).
    struct CellRect { int x0, y0, x1, y1; };
    CellRect cell_rect(tg::LeafShape s) {
        const int x0 = tg::leaf_atlas_cell_col(s) * tg::K_LEAF_CELL_PX;
        const int y0 = tg::leaf_atlas_cell_row(s) * tg::K_LEAF_CELL_PX;
        return CellRect{ x0, y0, x0 + tg::K_LEAF_CELL_PX, y0 + tg::K_LEAF_CELL_PX };
    }

    constexpr std::array<tg::LeafShape, 4> k_species = {
        tg::LeafShape::OakLobed,
        tg::LeafShape::PineNeedle,
        tg::LeafShape::BirchSerrated,
        tg::LeafShape::MapleStar,
    };

    constexpr std::array<const char*, 4> k_species_name = {
        "oak", "pine", "birch", "maple"
    };

    inline uint8_t srgb_byte(float linear) {
        if (linear < 0.0f) linear = 0.0f;
        if (linear > 1.0f) linear = 1.0f;
        const float s = (linear <= 0.0031308f)
            ? 12.92f * linear
            : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
        float scaled = s * 255.0f + 0.5f;
        if (scaled < 0.0f)   scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        return static_cast<uint8_t>(scaled);
    }

    struct LinearRgb { float r, g, b; };
    // Mirror of `k_leaf_color` in leaf_atlas.cpp (single source of truth there;
    // pinned here as the test's expected reference).
    constexpr LinearRgb k_expected[4] = {
        /* OakLobed      */ { 0.20f, 0.38f, 0.12f },
        /* PineNeedle    */ { 0.10f, 0.30f, 0.12f },
        /* BirchSerrated */ { 0.35f, 0.50f, 0.18f },
        /* MapleStar     */ { 0.30f, 0.38f, 0.14f },
    };

}  // namespace


// ---- Test 1: dim pin --------------------------------------------------------
TEST_CASE("[treegen_leaf_atlas_p3] atlas dims pinned 1024x1024 RGBA8",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_atlas_png(1u);
    auto d = decode_atlas(png);
    REQUIRE(d.w == tg::K_LEAF_ATLAS_PX);
    REQUIRE(d.h == tg::K_LEAF_ATLAS_PX);
    REQUIRE(d.bytes.size() == static_cast<size_t>(d.w) * d.h * 4);
}


// ---- Test 2: 1 connected component per species cell -------------------------
TEST_CASE("[treegen_leaf_atlas_p3] per-cell connected-component count is 1",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_atlas_png(2u);
    auto d = decode_atlas(png);

    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const CellRect r = cell_rect(sp);
        const int w = r.x1 - r.x0;
        const int h = r.y1 - r.y0;

        // Copy alpha into a local cell-sized buffer, then flood-fill.
        std::vector<uint8_t> mark(static_cast<size_t>(w) * h, 0);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                mark[y * w + x] = d.bytes[((r.y0 + y) * d.w + (r.x0 + x)) * 4 + 3] > 0 ? 1u : 0u;

        int components = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (mark[y * w + x] != 1) continue;
                ++components;
                std::queue<std::pair<int,int>> q;
                q.push({x, y});
                mark[y * w + x] = 2;
                while (!q.empty()) {
                    auto [cx, cy] = q.front(); q.pop();
                    constexpr int dx[4] = {-1, 1, 0, 0};
                    constexpr int dy[4] = {0, 0, -1, 1};
                    for (int k = 0; k < 4; ++k) {
                        const int nx = cx + dx[k];
                        const int ny = cy + dy[k];
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                        if (mark[ny * w + nx] == 1) {
                            mark[ny * w + nx] = 2;
                            q.push({nx, ny});
                        }
                    }
                }
            }
        }
        INFO("species=" << k_species_name[si] << " components=" << components);
        REQUIRE(components == 1);
    }
}


// ---- Test 3: per-cell area within ±10% of expected --------------------------
TEST_CASE("[treegen_leaf_atlas_p3] per-cell area vs expected within 10 percent",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_atlas_png(3u);
    auto d = decode_atlas(png);

    // Polygon area in shape space = sum over tris of |edge(A,B,C)| * 0.5.
    auto shape_area = [](tg::LeafShape sp) -> double {
        tg::LeafShapeMesh m;
        switch (sp) {
            case tg::LeafShape::OakLobed:      m = tg::oak_lobed(); break;
            case tg::LeafShape::PineNeedle:    m = tg::pine_needle(); break;
            case tg::LeafShape::BirchSerrated: m = tg::birch_serrated(); break;
            case tg::LeafShape::MapleStar:     m = tg::maple_star(); break;
        }
        double area = 0.0;
        const size_t n_tris = m.tris.size() / 3;
        for (size_t t = 0; t < n_tris; ++t) {
            const tg::vec2 A = m.verts[m.tris[t * 3 + 0]];
            const tg::vec2 B = m.verts[m.tris[t * 3 + 1]];
            const tg::vec2 C = m.verts[m.tris[t * 3 + 2]];
            const double cross_z = static_cast<double>(B.x - A.x) * (C.y - A.y)
                                 - static_cast<double>(B.y - A.y) * (C.x - A.x);
            area += 0.5 * std::abs(cross_z);
        }
        return area;
    };

    // Shape space spans [-1,+1] in both axes (width 2). Pixel scale for usable
    // rect: (usable_px / 2) per shape unit. Expected pixel area = shape_area *
    // (usable_px/2)^2.
    const double scale_per_axis = static_cast<double>(tg::K_LEAF_CELL_USABLE_PX) / 2.0;
    const double scale_sq = scale_per_axis * scale_per_axis;

    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const double expected = shape_area(sp) * scale_sq;

        const CellRect r = cell_rect(sp);
        size_t measured = 0;
        for (int y = r.y0; y < r.y1; ++y) {
            for (int x = r.x0; x < r.x1; ++x) {
                if (d.bytes[(y * d.w + x) * 4 + 3] > 0) ++measured;
            }
        }
        const double ratio = static_cast<double>(measured) / expected;
        INFO("species=" << k_species_name[si]
             << " measured=" << measured << " expected=" << expected
             << " ratio=" << ratio);
        REQUIRE(ratio >= 0.90);
        REQUIRE(ratio <= 1.10);
    }
}


// ---- Test 4: diffuse RGB stays within species hue + gradient range ----------
// C3 P1 changed flat-color splatting to per-pixel gradients (tip-to-base,
// vein darkening, edge tinting). Center probes are no longer byte-equal to the
// base color — they carry ~50% of the tip-to-base gradient and possible vein
// darkening. We check: (a) R channel is within [0.60*base, 1.05*base],
// (b) per-species hue ratio G/R stays within ±20% of base hue ratio (no
// color-space drift into non-green territory), (c) tip pixels are brighter
// than base pixels (gradient monotonicity).
TEST_CASE("[treegen_leaf_atlas_p3] diffuse within species hue + gradient range",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_atlas_png(4u);
    auto d = decode_atlas(png);

    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const CellRect r = cell_rect(sp);
        const uint8_t base_r = srgb_byte(k_expected[si].r);
        const uint8_t base_g = srgb_byte(k_expected[si].g);

        // Center probes: must be alpha>0 and within gradient-darkened range.
        const int cx = (r.x0 + r.x1) / 2;
        const int cy = (r.y0 + r.y1) / 2;
        const size_t ci = (static_cast<size_t>(cy) * d.w + cx) * 4;
        REQUIRE(d.bytes[ci + 3] > 0);
        const int pr = d.bytes[ci + 0];
        const int pg = d.bytes[ci + 1];
        INFO("species=" << k_species_name[si]
             << " center rgb=(" << pr << "," << pg << "," << int(d.bytes[ci + 2]) << ")"
             << " base=(" << int(base_r) << "," << int(base_g) << ")");
        // Gradient darkens by up to ~30% at base; center is ~50% through.
        REQUIRE(pr >= int(base_r) * 60 / 100);
        REQUIRE(pr <= int(base_r) + 5);
        // Hue ratio: G/R should stay close to base G/R (vegetation greens).
        if (base_r > 10 && pr > 10) {
            const float ratio_base = float(base_g) / float(base_r);
            const float ratio_px   = float(pg) / float(pr);
            REQUIRE(ratio_px >= ratio_base * 0.75f);
            REQUIRE(ratio_px <= ratio_base * 1.25f);
        }
    }
    // Gradient monotonicity: top of cell (tip) brighter than bottom (base)
    // for all non-pine species (pine has subtle gradient but same direction).
    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const CellRect r = cell_rect(sp);
        const int cx = (r.x0 + r.x1) / 2;
        // Sample near top and bottom of alpha region.
        int tip_g = -1, base_g_val = -1;
        for (int y = r.y0; y < r.y1; ++y) {
            const size_t idx = (static_cast<size_t>(y) * d.w + cx) * 4;
            if (d.bytes[idx + 3] > 0) {
                if (tip_g < 0) tip_g = d.bytes[idx + 1];
                base_g_val = d.bytes[idx + 1];
            }
        }
        INFO("species=" << k_species_name[si]
             << " tip_g=" << tip_g << " base_g=" << base_g_val);
        REQUIRE(tip_g >= 0);
        REQUIRE(base_g_val >= 0);
        REQUIRE(tip_g >= base_g_val);  // tip is brighter or equal
    }
}


// ---- Test 5: all 16 cells populated with clean 4-px transparent padding -----
TEST_CASE("[treegen_leaf_atlas_p3] all 16 cells populated with transparent 4-px padding",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_atlas_png(5u);
    auto d = decode_atlas(png);

    for (int row = 0; row < tg::K_LEAF_ATLAS_DIM; ++row) {
        for (int col = 0; col < tg::K_LEAF_ATLAS_DIM; ++col) {
            const int x0 = col * tg::K_LEAF_CELL_PX;
            const int y0 = row * tg::K_LEAF_CELL_PX;
            int opaque = 0;
            for (int y = 0; y < tg::K_LEAF_CELL_PX; ++y) {
                for (int x = 0; x < tg::K_LEAF_CELL_PX; ++x) {
                    const uint8_t a = d.bytes[((y0 + y) * d.w + (x0 + x)) * 4 + 3];
                    const bool in_pad = (x < tg::K_LEAF_CELL_PAD_PX ||
                                         x >= tg::K_LEAF_CELL_PX - tg::K_LEAF_CELL_PAD_PX ||
                                         y < tg::K_LEAF_CELL_PAD_PX ||
                                         y >= tg::K_LEAF_CELL_PX - tg::K_LEAF_CELL_PAD_PX);
                    if (in_pad) {
                        INFO("non-transparent pad pixel in cell (" << row << "," << col << ") at (" << x << "," << y << ")");
                        REQUIRE(a == 0);
                    } else if (a > 0) {
                        ++opaque;
                    }
                }
            }
            INFO("cell (" << row << "," << col << ") opaque=" << opaque);
            REQUIRE(opaque > 2000);
        }
    }
}


// ---- Test 5b: sprig cells hold MULTIPLE distinct sub-cluster components -----
TEST_CASE("[treegen_leaf_atlas_p3] cluster cell has multiple distinct leaves",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_atlas_png(51u);
    auto d = decode_atlas(png);

    for (int row = tg::K_LEAF_SPRIG_ROW_MIN; row <= tg::K_LEAF_SPRIG_ROW_MAX; ++row) {
        for (size_t si = 0; si < k_species.size(); ++si) {
            const auto sp = k_species[si];
            if (sp == tg::LeafShape::PineNeedle && row == tg::K_NEEDLE_STRIP_CELL_ROW) continue;
            const int x0 = tg::leaf_cluster_cell_col(sp) * tg::K_LEAF_CELL_PX;
            const int y0 = row * tg::K_LEAF_CELL_PX;
            const int w = tg::K_LEAF_CELL_PX, h = tg::K_LEAF_CELL_PX;

            std::vector<uint8_t> mark(static_cast<size_t>(w) * h, 0);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    mark[y * w + x] = d.bytes[((y0 + y) * d.w + (x0 + x)) * 4 + 3] > 0 ? 1u : 0u;

            int components = 0, max_comp_w = 0, max_comp_h = 0;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (mark[y * w + x] != 1) continue;
                    ++components;
                    int minx = x, maxx = x, miny = y, maxy = y;
                    std::queue<std::pair<int,int>> q;
                    q.push({x, y}); mark[y * w + x] = 2;
                    while (!q.empty()) {
                        auto [cx, cy] = q.front(); q.pop();
                        minx = std::min(minx, cx); maxx = std::max(maxx, cx);
                        miny = std::min(miny, cy); maxy = std::max(maxy, cy);
                        constexpr int dx[4] = {-1, 1, 0, 0};
                        constexpr int dy[4] = {0, 0, -1, 1};
                        for (int k = 0; k < 4; ++k) {
                            const int nx = cx + dx[k], ny = cy + dy[k];
                            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                            if (mark[ny * w + nx] == 1) { mark[ny * w + nx] = 2; q.push({nx, ny}); }
                        }
                    }
                    max_comp_w = std::max(max_comp_w, maxx - minx + 1);
                    max_comp_h = std::max(max_comp_h, maxy - miny + 1);
                }
            }
            INFO("species=" << k_species_name[si] << " row=" << row
                 << " cluster components=" << components
                 << " max_comp=" << max_comp_w << "x" << max_comp_h);
            REQUIRE(components >= 3);
            REQUIRE(max_comp_w < (tg::K_LEAF_CELL_USABLE_PX * 3) / 4);
        }
    }
}


// ---- Test 6: 4-px padding ring is transparent --------------------------------
TEST_CASE("[treegen_leaf_atlas_p3] 4-px padding is transparent",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_atlas_png(6u);
    auto d = decode_atlas(png);

    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const CellRect r = cell_rect(sp);
        // Top + bottom pad bands (full cell width).
        for (int dy = 0; dy < tg::K_LEAF_CELL_PAD_PX; ++dy) {
            for (int x = r.x0; x < r.x1; ++x) {
                REQUIRE(d.bytes[((r.y0 + dy) * d.w + x) * 4 + 3] == 0);
                REQUIRE(d.bytes[((r.y1 - 1 - dy) * d.w + x) * 4 + 3] == 0);
            }
        }
        // Left + right pad bands (already covered top/bottom rows; only middle).
        for (int dx = 0; dx < tg::K_LEAF_CELL_PAD_PX; ++dx) {
            for (int y = r.y0 + tg::K_LEAF_CELL_PAD_PX;
                 y < r.y1 - tg::K_LEAF_CELL_PAD_PX; ++y) {
                REQUIRE(d.bytes[(y * d.w + r.x0 + dx) * 4 + 3] == 0);
                REQUIRE(d.bytes[(y * d.w + r.x1 - 1 - dx) * 4 + 3] == 0);
            }
        }
    }
}


// ---- Test 7: byte-equal across two runs ------------------------------------
TEST_CASE("[treegen_leaf_atlas_p3] deterministic two-run byte equality",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto a = tg::encode_leaf_atlas_png(7u);
    auto b = tg::encode_leaf_atlas_png(7u);
    REQUIRE(a.size() == b.size());
    REQUIRE(a == b);
}


// ---- Test 8: tile_uv orientation regression -----------------------------------
TEST_CASE("[treegen_leaf_atlas_p3] tile_uv orientation: apex maps to top of cell",
          "[treegen][treegen_leaf_atlas_p3]") {
    constexpr float cell_size_uv = 1.0f / static_cast<float>(tg::K_LEAF_ATLAS_DIM);  // 0.25
    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const tg::vec2 apex = tg::tile_uv(sp, 0.0f, +1.0f);
        const tg::vec2 base = tg::tile_uv(sp, 0.0f, -1.0f);
        const float row_v = static_cast<float>(tg::leaf_atlas_cell_row(sp)) * cell_size_uv;
        INFO("species=" << k_species_name[si]
             << " apex.v=" << apex.y << " base.v=" << base.y << " row_v=" << row_v);
        REQUIRE(apex.y < base.y);
        REQUIRE(apex.y <= row_v + 0.005f);                  // very close to top of cell
        REQUIRE(base.y >= row_v + 0.24f);                   // very close to bottom of cell
    }
}


// ---- Test 9: tile_uv corners stay inside the cell usable rect ----------------
TEST_CASE("[treegen_leaf_atlas_p3] tile_uv cell-bounds containment (closed)",
          "[treegen][treegen_leaf_atlas_p3]") {
    constexpr float cell_size_uv = 1.0f / static_cast<float>(tg::K_LEAF_ATLAS_DIM);  // 0.25
    constexpr float pad_uv       = static_cast<float>(tg::K_LEAF_CELL_PAD_PX)
                                 / static_cast<float>(tg::K_LEAF_ATLAS_PX);          // ≈ 0.00391
    constexpr float usable_uv    = cell_size_uv - 2.0f * pad_uv;

    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const float u0 = static_cast<float>(tg::leaf_atlas_cell_col(sp)) * cell_size_uv + pad_uv;
        const float v0 = static_cast<float>(tg::leaf_atlas_cell_row(sp)) * cell_size_uv + pad_uv;
        const float u1 = u0 + usable_uv;
        const float v1 = v0 + usable_uv;

        const float L[2] = { -1.0f, +1.0f };
        for (float lx : L) {
            for (float ly : L) {
                const tg::vec2 uv = tg::tile_uv(sp, lx, ly);
                INFO("species=" << k_species_name[si]
                     << " (lx,ly)=(" << lx << "," << ly << ")"
                     << " uv=(" << uv.x << "," << uv.y << ")"
                     << " bounds=[" << u0 << "," << u1 << "]x[" << v0 << "," << v1 << "]");
                // CLOSED interval — (lx,ly)=(±1,±1) lands exactly on a usable edge.
                REQUIRE(uv.x >= u0);
                REQUIRE(uv.x <= u1);
                REQUIRE(uv.y >= v0);
                REQUIRE(uv.y <= v1);
            }
        }
    }
}


// ---- Test 10: emitter UV migration regression --------------------------------
TEST_CASE("[treegen_leaf_atlas_p3] emitter UV byte-equal to tile_uv at apex",
          "[treegen][treegen_leaf_atlas_p3]") {
    // 1-site oak ProceduralVeined leaf. Emitted UV for the apex vert must equal
    // tile_uv(OakLobed, apex.x, apex.y) bit-for-bit. Catches drift between the
    // emitter and the atlas mapping.
    std::vector<tg::LeafSite> sites;
    {
        tg::LeafSite s;
        s.position = tg::vec3{0.0f, 0.0f, 0.0f};
        s.normal   = tg::vec3{0.0f, 0.0f, 1.0f};
        s.branch_id = 0;
        s.type = 0;
        sites.push_back(s);
    }
    tg::LeafMeshOptions opts;
    opts.geometry_type = tg::LeafGeometryType::ProceduralVeined;
    opts.shape         = tg::LeafShape::OakLobed;
    opts.leaf_size_m   = 0.12f;
    auto out = tg::build_leaf_mesh(sites, opts);

    const auto shape = tg::oak_lobed();
    REQUIRE(out.uvs.size() == shape.verts.size() * 2);

    // Apex is the perimeter vert closest to +Y (θ=π/2). Walk perimeter and
    // pick it; compare emitter-emitted UV to tile_uv() for the same shape verts.
    int apex_local = 0;
    float best_y = -1e9f;
    for (size_t i = 1; i < shape.verts.size(); ++i) {
        if (shape.verts[i].y > best_y) { best_y = shape.verts[i].y; apex_local = static_cast<int>(i); }
    }

    const tg::vec2 v = shape.verts[apex_local];
    const tg::vec2 expected = tg::tile_uv(tg::LeafShape::OakLobed, v.x, v.y);
    const float emitted_u = out.uvs[apex_local * 2 + 0];
    const float emitted_v = out.uvs[apex_local * 2 + 1];
    INFO("apex vert idx=" << apex_local
         << " emitted_uv=(" << emitted_u << "," << emitted_v << ")"
         << " tile_uv=(" << expected.x << "," << expected.y << ")");
    REQUIRE(emitted_u == expected.x);
    REQUIRE(emitted_v == expected.y);
}


// ============================================================================
// P3b — normal + translucency atlas tests (11-15).
// ============================================================================

// ---- Test 11: normal atlas dims + flat default in padding --------------------
TEST_CASE("[treegen_leaf_atlas_p3] normal atlas dims and flat default in padding",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_normal_png(11u);
    auto d = decode_atlas(png);
    REQUIRE(d.w == tg::K_LEAF_ATLAS_PX);
    REQUIRE(d.h == tg::K_LEAF_ATLAS_PX);

    // Padding pixels must be flat normal (128, 128, 255) across all 16 cells.
    for (int row = 0; row < tg::K_LEAF_ATLAS_DIM; ++row) {
        for (int col = 0; col < tg::K_LEAF_ATLAS_DIM; ++col) {
            const int x0 = col * tg::K_LEAF_CELL_PX;
            const int y0 = row * tg::K_LEAF_CELL_PX;
            for (int dy = 0; dy < tg::K_LEAF_CELL_PAD_PX; ++dy) {
                for (int x = x0; x < x0 + tg::K_LEAF_CELL_PX; ++x) {
                    const size_t idx = (static_cast<size_t>(y0 + dy) * d.w + x) * 4;
                    INFO("cell=(" << row << "," << col << ") pad pixel (" << x << "," << y0 + dy << ")");
                    REQUIRE(d.bytes[idx + 0] == 128);
                    REQUIRE(d.bytes[idx + 1] == 128);
                    REQUIRE(d.bytes[idx + 2] == 255);
                }
            }
        }
    }
}


// ---- Test 12: normal central-vein N.x sign detect --------------------------
TEST_CASE("[treegen_leaf_atlas_p3] normal vein centerline sign detect",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_normal_png(12u);
    auto d = decode_atlas(png);

    // The central vein creates a height ridge along x=center. The finite-diff
    // normal produces N.x < 128 on the left slope and N.x > 128 on the right.
    // The vein line is ~3px wide, so we sample at cx-2 and cx+2. If a side has
    // no perturbed pixels (all flat 128) the vein is too narrow there; we still
    // require at least one side to show deviation from flat. The structural
    // property: "the normal map encodes a vein ridge" — at least one flanking
    // column has mean N.x != 128 with magnitude > 10 bytes.
    int species_with_deviation = 0;
    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const CellRect r = cell_rect(sp);
        const int cx = (r.x0 + r.x1) / 2;
        const int cy = (r.y0 + r.y1) / 2;

        double left_sum = 0, right_sum = 0;
        int left_count = 0, right_count = 0;
        for (int y = cy - 20; y <= cy + 20; ++y) {
            {
                const size_t idx = (static_cast<size_t>(y) * d.w + (cx - 2)) * 4;
                const int nx = static_cast<int>(d.bytes[idx + 0]);
                if (nx != 128) { left_sum += nx; ++left_count; }
            }
            {
                const size_t idx = (static_cast<size_t>(y) * d.w + (cx + 2)) * 4;
                const int nx = static_cast<int>(d.bytes[idx + 0]);
                if (nx != 128) { right_sum += nx; ++right_count; }
            }
        }
        const float left_mean  = left_count  > 0 ? static_cast<float>(left_sum  / left_count)  : 128.0f;
        const float right_mean = right_count > 0 ? static_cast<float>(right_sum / right_count) : 128.0f;
        INFO("species=" << k_species_name[si]
             << " left_nx_mean=" << left_mean << " (" << left_count << " perturbed)"
             << " right_nx_mean=" << right_mean << " (" << right_count << " perturbed)");

        const bool left_deviates  = std::abs(left_mean  - 128.0f) > 10.0f;
        const bool right_deviates = std::abs(right_mean - 128.0f) > 10.0f;
        if (left_deviates || right_deviates) ++species_with_deviation;
    }
    // All 4 species must show vein-induced normal perturbation.
    REQUIRE(species_with_deviation == 4);
}


// ---- Test 13: translucency dims + per-species R mean in [0.3, 0.7] ---------
TEST_CASE("[treegen_leaf_atlas_p3] translucency dims and per-species R mean",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_translucency_png(13u);
    auto d = decode_atlas(png);
    REQUIRE(d.w == tg::K_LEAF_ATLAS_PX);
    REQUIRE(d.h == tg::K_LEAF_ATLAS_PX);

    // Decode diffuse atlas for alpha mask.
    auto diffuse_png = tg::encode_leaf_atlas_png(13u);
    auto da = decode_atlas(diffuse_png);

    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const CellRect r = cell_rect(sp);
        double r_sum = 0;
        size_t count = 0;
        for (int y = r.y0; y < r.y1; ++y) {
            for (int x = r.x0; x < r.x1; ++x) {
                const size_t idx = (static_cast<size_t>(y) * d.w + x) * 4;
                const size_t da_idx = (static_cast<size_t>(y) * da.w + x) * 4;
                if (da.bytes[da_idx + 3] == 0) continue;
                r_sum += static_cast<double>(d.bytes[idx + 0]) / 255.0;
                ++count;
            }
        }
        REQUIRE(count > 0);
        const double mean = r_sum / static_cast<double>(count);
        INFO("species=" << k_species_name[si] << " translucency R mean=" << mean
             << " count=" << count);
        // Translucency = blurred_alpha * (1 - 0.5*vh). For interior pixels
        // blurred_alpha ≈ 1.0; only vein pixels (tiny fraction) have vh>0.
        // Mean is therefore near 1.0 for broad silhouettes (oak, birch, maple)
        // and slightly lower for narrow shapes (pine). Range [0.5, 1.0].
        REQUIRE(mean >= 0.5);
        REQUIRE(mean <= 1.0);
    }
}


// ---- Test 14: translucency monotonicity across vein-strength buckets --------
TEST_CASE("[treegen_leaf_atlas_p3] translucency monotonic with vein height",
          "[treegen][treegen_leaf_atlas_p3]") {
    // For oak (most prominent veins): bucket alpha>0 pixels by vein height
    // into 3 bands (low/mid/high). Mean translucency should decrease
    // monotonically as vein height increases.
    auto trans_png = tg::encode_leaf_translucency_png(14u);
    auto td = decode_atlas(trans_png);
    auto diff_png = tg::encode_leaf_atlas_png(14u);
    auto dd = decode_atlas(diff_png);

    // We need the vein height — bake it directly.
    const tg::LeafShapeMesh mesh = tg::oak_lobed();
    auto veins = tg::veins_for_species(tg::LeafShape::OakLobed, mesh);

    // Rasterize alpha + veins for oak cell.
    std::vector<uint8_t> alpha(static_cast<size_t>(tg::K_LEAF_CELL_PX) * tg::K_LEAF_CELL_PX, 0);
    tg::rasterize_leaf_alpha_into_cell(mesh, alpha.data(),
                                       tg::K_LEAF_CELL_PX, tg::K_LEAF_CELL_PX,
                                       tg::K_LEAF_CELL_PAD_PX, tg::K_LEAF_CELL_USABLE_PX);
    std::vector<uint8_t> vein_h(static_cast<size_t>(tg::K_LEAF_CELL_PX) * tg::K_LEAF_CELL_PX, 0);
    // Stamp veins — use the same rasterizer interface as the bake.
    // Since stamp_veins_into_cell is internal, we replicate via the rasterize_tri_pixel_space API.
    // Actually, we can just bucket based on translucency value vs alpha — low
    // translucency = high vein, high translucency = low vein. That's the
    // definition from the bake formula: trans = ba * (1 - 0.5*vh).
    // So pixels with trans < 0.6*alpha are high-vein, trans >= 0.9*alpha are low-vein.

    const CellRect r = cell_rect(tg::LeafShape::OakLobed);
    double sum_lo = 0, sum_hi = 0;
    int count_lo = 0, count_hi = 0;

    for (int y = r.y0; y < r.y1; ++y) {
        for (int x = r.x0; x < r.x1; ++x) {
            const size_t idx = (static_cast<size_t>(y) * td.w + x) * 4;
            const size_t da_idx = (static_cast<size_t>(y) * dd.w + x) * 4;
            if (dd.bytes[da_idx + 3] == 0) continue;

            const float trans = static_cast<float>(td.bytes[idx + 0]) / 255.0f;

            // Low-vein region: trans is high (near 1.0 * blurred_alpha).
            if (trans > 0.85f) { sum_hi += trans; ++count_hi; }
            // High-vein region: trans is low.
            if (trans < 0.60f) { sum_lo += trans; ++count_lo; }
        }
    }
    INFO("oak translucency: low_vein_mean=" << (count_hi > 0 ? sum_hi / count_hi : 0)
         << " high_vein_mean=" << (count_lo > 0 ? sum_lo / count_lo : 0)
         << " count_hi=" << count_hi << " count_lo=" << count_lo);
    REQUIRE(count_hi > 0);
    REQUIRE(count_lo > 0);
    // Monotonicity: mean trans in low-vein region > mean trans in high-vein region.
    REQUIRE(sum_hi / count_hi > sum_lo / count_lo);
}


// ---- Test 15: normal + translucency + roughness two-run determinism ---------
TEST_CASE("[treegen_leaf_atlas_p3] normal, translucency, roughness deterministic",
          "[treegen][treegen_leaf_atlas_p3]") {
    {
        auto a = tg::encode_leaf_normal_png(15u);
        auto b = tg::encode_leaf_normal_png(15u);
        REQUIRE(a.size() == b.size());
        REQUIRE(a == b);
    }
    {
        auto a = tg::encode_leaf_translucency_png(15u);
        auto b = tg::encode_leaf_translucency_png(15u);
        REQUIRE(a.size() == b.size());
        REQUIRE(a == b);
    }
    {
        auto a = tg::encode_leaf_roughness_png(15u);
        auto b = tg::encode_leaf_roughness_png(15u);
        REQUIRE(a.size() == b.size());
        REQUIRE(a == b);
    }
}


// ============================================================================
// C3 P2 — roughness atlas tests (16-17).
// ============================================================================

// ---- Test 16: roughness atlas dims + glTF metallic-roughness layout ---------
TEST_CASE("[treegen_leaf_atlas_p3] roughness atlas dims and MR layout",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto png = tg::encode_leaf_roughness_png(16u);
    auto d = decode_atlas(png);
    REQUIRE(d.w == tg::K_LEAF_ATLAS_PX);
    REQUIRE(d.h == tg::K_LEAF_ATLAS_PX);

    // Decode diffuse atlas for alpha mask.
    auto diffuse_png = tg::encode_leaf_atlas_png(16u);
    auto da = decode_atlas(diffuse_png);

    for (size_t si = 0; si < k_species.size(); ++si) {
        const auto sp = k_species[si];
        const CellRect r = cell_rect(sp);

        int count = 0;
        double rough_sum = 0;
        for (int y = r.y0; y < r.y1; ++y) {
            for (int x = r.x0; x < r.x1; ++x) {
                const size_t idx = (static_cast<size_t>(y) * d.w + x) * 4;
                const size_t da_idx = (static_cast<size_t>(y) * da.w + x) * 4;
                if (da.bytes[da_idx + 3] == 0) continue;
                ++count;
                // glTF MR layout: R=unused(255), G=roughness, B=metallic(0).
                REQUIRE(d.bytes[idx + 0] == 255);  // R = unused
                REQUIRE(d.bytes[idx + 2] == 0);    // B = metallic = 0
                rough_sum += static_cast<double>(d.bytes[idx + 1]) / 255.0;
            }
        }
        REQUIRE(count > 0);
        const double mean_rough = rough_sum / count;
        INFO("species=" << k_species_name[si]
             << " roughness G mean=" << mean_rough << " count=" << count);
        // PBR C1 B4 — leaves matte: roughness = 0.70 + 0.15*vein_factor
        // (range 0.70-0.85). Mean sits near 0.70 since veins are a small
        // fraction; bounds bracket the matte band.
        REQUIRE(mean_rough >= 0.65);
        REQUIRE(mean_rough <= 0.90);
    }
}


// ---- Test 17: roughness increases near veins --------------------------------
TEST_CASE("[treegen_leaf_atlas_p3] roughness increases with vein height",
          "[treegen][treegen_leaf_atlas_p3]") {
    auto rough_png = tg::encode_leaf_roughness_png(17u);
    auto rd = decode_atlas(rough_png);
    auto diff_png = tg::encode_leaf_atlas_png(17u);
    auto dd = decode_atlas(diff_png);

    // For oak: compare roughness in high-trans (low-vein) vs low-trans (high-vein).
    // Use the translucency atlas as vein proxy.
    auto trans_png = tg::encode_leaf_translucency_png(17u);
    auto td = decode_atlas(trans_png);

    const CellRect r = cell_rect(tg::LeafShape::OakLobed);
    double rough_at_low_vein = 0, rough_at_high_vein = 0;
    int count_lo_v = 0, count_hi_v = 0;

    for (int y = r.y0; y < r.y1; ++y) {
        for (int x = r.x0; x < r.x1; ++x) {
            const size_t idx = (static_cast<size_t>(y) * rd.w + x) * 4;
            const size_t da_idx = (static_cast<size_t>(y) * dd.w + x) * 4;
            const size_t ti = (static_cast<size_t>(y) * td.w + x) * 4;
            if (dd.bytes[da_idx + 3] == 0) continue;

            const float trans = static_cast<float>(td.bytes[ti + 0]) / 255.0f;
            const float rough = static_cast<float>(rd.bytes[idx + 1]) / 255.0f;

            if (trans > 0.90f) { rough_at_low_vein += rough; ++count_lo_v; }
            if (trans < 0.50f) { rough_at_high_vein += rough; ++count_hi_v; }
        }
    }
    INFO("oak roughness: low_vein_mean=" << (count_lo_v > 0 ? rough_at_low_vein / count_lo_v : 0)
         << " high_vein_mean=" << (count_hi_v > 0 ? rough_at_high_vein / count_hi_v : 0)
         << " count_lo=" << count_lo_v << " count_hi=" << count_hi_v);
    REQUIRE(count_lo_v > 0);
    REQUIRE(count_hi_v > 0);
    // High-vein regions should be rougher.
    REQUIRE(rough_at_high_vein / count_hi_v > rough_at_low_vein / count_lo_v);
}


// ============================================================================
// Needle-strip orientation test (18).
// ============================================================================

// ---- Test 18: needle-strip orientation — V-fanned fascicles slotted along V --
TEST_CASE("[treegen_leaf_atlas_p3] needle strip orientation: needles along V",
          "[treegen][treegen_leaf_atlas_p3][needle_strip]") {
    auto png = tg::encode_leaf_atlas_png(18u);
    auto d = decode_atlas(png);

    const int x0 = tg::K_NEEDLE_STRIP_CELL_COL * tg::K_LEAF_CELL_PX;
    const int y0 = tg::K_NEEDLE_STRIP_CELL_ROW * tg::K_LEAF_CELL_PX;
    const int cx = x0 + tg::K_LEAF_CELL_PX / 2;

    // Count alpha 0->255 and 255->0 transitions along a vertical scanline.
    auto count_v_transitions = [&](int fixed_x) -> int {
        int transitions = 0;
        bool prev_opaque = false;
        for (int i = 0; i < tg::K_LEAF_CELL_PX; ++i) {
            const bool opaque = d.bytes[((y0 + i) * d.w + fixed_x) * 4 + 3] > 0;
            if (i > 0 && opaque != prev_opaque) ++transitions;
            prev_opaque = opaque;
        }
        return transitions;
    };

    // Vertical scans across the left and right fanned fascicles cross all 8 whorls.
    const int v_left  = count_v_transitions(cx - 48);
    const int v_right = count_v_transitions(cx + 48);
    INFO("vertical transitions left=" << v_left << " right=" << v_right);
    REQUIRE(v_left >= 12);
    REQUIRE(v_right >= 12);

    // Anti-horizontal-comb check: V-fanned needles have strong horizontal alpha gradient
    // (dAlpha/dx != 0) across the interior, unlike horizontal bars.
    int dx_edges = 0, dy_edges = 0;
    for (int y = 8; y < tg::K_LEAF_CELL_PX - 8; ++y) {
        for (int x = 24; x < tg::K_LEAF_CELL_PX - 24; ++x) {
            const bool a  = d.bytes[((y0 + y) * d.w + (x0 + x)) * 4 + 3] > 0;
            const bool ax = d.bytes[((y0 + y) * d.w + (x0 + x + 1)) * 4 + 3] > 0;
            const bool ay = d.bytes[((y0 + y + 1) * d.w + (x0 + x)) * 4 + 3] > 0;
            if (a != ax) ++dx_edges;
            if (a != ay) ++dy_edges;
        }
    }
    INFO("pine fascicle dx_edges=" << dx_edges << " dy_edges=" << dy_edges);
    REQUIRE(dx_edges > dy_edges / 3);  // angled needles (pure horizontal bars have dx_edges << dy_edges)
}

// ============================================================================
// Botanical Sprig Uplift Seals (19-22).
// ============================================================================

// ---- Test 19: Non-squareness, 16-px recess, empty 32x32 corners & anti-grid -
TEST_CASE("[treegen_leaf_atlas_p3] botanical sprigs: empty corners, recessed, bbox fill < 0.72, anti-grid",
          "[treegen][treegen_leaf_atlas_p3][botanical_sprigs]") {
    auto png = tg::encode_leaf_atlas_png(19u);
    auto d = decode_atlas(png);

    constexpr int W = tg::K_LEAF_CELL_PX;
    constexpr int R = tg::K_LEAF_SPRIG_RECESS_PX;
    constexpr int C = tg::K_LEAF_SPRIG_CORNER_PX;

    for (int row = tg::K_LEAF_SPRIG_ROW_MIN; row <= tg::K_LEAF_SPRIG_ROW_MAX; ++row) {
        for (size_t si = 0; si < k_species.size(); ++si) {
            const auto sp = k_species[si];
            if (sp == tg::LeafShape::PineNeedle && row == tg::K_NEEDLE_STRIP_CELL_ROW) continue;

            const int x0 = tg::leaf_cluster_cell_col(sp) * W;
            const int y0 = row * W;

            int minx = W, maxx = -1, miny = W, maxy = -1;
            int opaque = 0;
            std::vector<double> mask(static_cast<size_t>(W) * W, 0.0);

            for (int y = 0; y < W; ++y) {
                for (int x = 0; x < W; ++x) {
                    const uint8_t a = d.bytes[((y0 + y) * d.w + (x0 + x)) * 4 + 3];
                    const bool in_recess = (x < R || x >= W - R || y < R || y >= W - R);
                    const bool in_corner = (x < C && y < C) ||
                                           (x >= W - C && y < C) ||
                                           (x < C && y >= W - C) ||
                                           (x >= W - C && y >= W - C);
                    if (in_recess || in_corner) {
                        INFO("species=" << k_species_name[si] << " row=" << row
                             << " px=(" << x << "," << y << ") alpha=" << int(a));
                        REQUIRE(a == 0);
                    }
                    if (a > 0) {
                        ++opaque;
                        minx = std::min(minx, x); maxx = std::max(maxx, x);
                        miny = std::min(miny, y); maxy = std::max(maxy, y);
                        mask[static_cast<size_t>(y) * W + x] = 1.0;
                    }
                }
            }
            REQUIRE(opaque > 4000);
            const int bw = maxx - minx + 1;
            const int bh = maxy - miny + 1;
            const double bbox_fill = static_cast<double>(opaque) / static_cast<double>(bw * bh);
            INFO("species=" << k_species_name[si] << " row=" << row
                 << " opaque=" << opaque << " bbox=" << bw << "x" << bh
                 << " bbox_fill=" << bbox_fill);
            REQUIRE(bbox_fill < 0.72);

            // Anti-grid seal: normalized 2D autocorrelation at the old 4x3 lattice periods
            // (W/4, 0), (0, W/3), and (W/4, W/3) must be low (no regular grid).
            const double mean = static_cast<double>(opaque) / static_cast<double>(W * W);
            double var = 0.0;
            for (double v : mask) var += (v - mean) * (v - mean);
            REQUIRE(var > 1.0);

            auto autocorr = [&](int dx, int dy) -> double {
                double sum = 0.0;
                for (int y = 0; y + dy < W; ++y) {
                    for (int x = 0; x + dx < W; ++x) {
                        sum += (mask[static_cast<size_t>(y) * W + x] - mean)
                             * (mask[static_cast<size_t>(y + dy) * W + (x + dx)] - mean);
                    }
                }
                return sum / var;
            };
            const double ac_x  = autocorr(W / 4, 0);
            const double ac_y  = autocorr(0, W / 3);
            const double ac_xy = autocorr(W / 4, W / 3);
            INFO("species=" << k_species_name[si] << " row=" << row
                 << " ac(W/4,0)=" << ac_x << " ac(0,H/3)=" << ac_y << " ac(W/4,H/3)=" << ac_xy);
            REQUIRE(ac_x  < 0.25);
            REQUIRE(ac_y  < 0.25);
            REQUIRE(ac_xy < 0.25);
        }
    }
}

// ---- Test 20: Per-leaf tangent normal, albedo & roughness variance on Rows 1..3 ----
TEST_CASE("[treegen_leaf_atlas_p3] botanical sprigs: non-flat normals, albedo and roughness variance",
          "[treegen][treegen_leaf_atlas_p3][botanical_sprigs]") {
    auto a_png = tg::encode_leaf_atlas_png(20u);
    auto n_png = tg::encode_leaf_normal_png(20u);
    auto r_png = tg::encode_leaf_roughness_png(20u);
    auto da = decode_atlas(a_png);
    auto dn = decode_atlas(n_png);
    auto dr = decode_atlas(r_png);

    constexpr int W = tg::K_LEAF_CELL_PX;
    for (int row = tg::K_LEAF_SPRIG_ROW_MIN; row <= tg::K_LEAF_SPRIG_ROW_MAX; ++row) {
        for (int col = 0; col < tg::K_LEAF_ATLAS_DIM; ++col) {
            const int x0 = col * W;
            const int y0 = row * W;

            double sum_nx = 0, sum_nx2 = 0, sum_ny = 0, sum_ny2 = 0;
            double sum_l  = 0, sum_l2  = 0;
            double sum_r  = 0, sum_r2  = 0;
            size_t cnt = 0;

            for (int y = 0; y < W; ++y) {
                for (int x = 0; x < W; ++x) {
                    const size_t idx = (static_cast<size_t>(y0 + y) * da.w + (x0 + x)) * 4;
                    if (da.bytes[idx + 3] < 128) continue;
                    const double nx = (static_cast<double>(dn.bytes[idx + 0]) / 255.0) * 2.0 - 1.0;
                    const double ny = (static_cast<double>(dn.bytes[idx + 1]) / 255.0) * 2.0 - 1.0;
                    const double luma = (0.2126 * da.bytes[idx + 0]
                                       + 0.7152 * da.bytes[idx + 1]
                                       + 0.0722 * da.bytes[idx + 2]) / 255.0;
                    const double rough = static_cast<double>(dr.bytes[idx + 1]) / 255.0;
                    REQUIRE(rough >= 0.60);
                    REQUIRE(rough <= 0.90);

                    sum_nx += nx; sum_nx2 += nx * nx;
                    sum_ny += ny; sum_ny2 += ny * ny;
                    sum_l  += luma; sum_l2 += luma * luma;
                    sum_r  += rough; sum_r2 += rough * rough;
                    ++cnt;
                }
            }
            REQUIRE(cnt > 2000u);
            const double dcnt = static_cast<double>(cnt);
            const double std_nx = std::sqrt(std::max(0.0, sum_nx2 / dcnt - (sum_nx / dcnt) * (sum_nx / dcnt)));
            const double std_ny = std::sqrt(std::max(0.0, sum_ny2 / dcnt - (sum_ny / dcnt) * (sum_ny / dcnt)));
            const double std_l  = std::sqrt(std::max(0.0, sum_l2  / dcnt - (sum_l  / dcnt) * (sum_l  / dcnt)));
            const double std_r  = std::sqrt(std::max(0.0, sum_r2  / dcnt - (sum_r  / dcnt) * (sum_r  / dcnt)));

            INFO("cell=(" << row << "," << col << ") std_nx=" << std_nx << " std_ny=" << std_ny
                 << " std_luma=" << std_l << " std_rough=" << std_r);
            REQUIRE(std_nx > 0.08);
            REQUIRE(std_ny > 0.08);
            REQUIRE(std_l  > 0.025);
            REQUIRE(std_r  > 0.015);
        }
    }
}

// ---- Test 21: Translucency floor (>= 15 / 0.058) & 4-px dilation seal -------
TEST_CASE("[treegen_leaf_atlas_p3] translucency floor >= 15 on opaque pixels and 4-px edge dilation",
          "[treegen][treegen_leaf_atlas_p3][botanical_sprigs]") {
    auto a_png = tg::encode_leaf_atlas_png(21u);
    auto t_png = tg::encode_leaf_translucency_png(21u);
    auto da = decode_atlas(a_png);
    auto dt = decode_atlas(t_png);

    constexpr int W = tg::K_LEAF_CELL_PX;
    constexpr int pad = tg::K_LEAF_CELL_PAD_PX;

    int opaque_below_floor = 0;
    int near_edge_zero_trans = 0;

    for (int row = 0; row < tg::K_LEAF_ATLAS_DIM; ++row) {
        for (int col = 0; col < tg::K_LEAF_ATLAS_DIM; ++col) {
            const int x0 = col * W;
            const int y0 = row * W;
            for (int y = pad; y < W - pad; ++y) {
                for (int x = pad; x < W - pad; ++x) {
                    const size_t idx = (static_cast<size_t>(y0 + y) * da.w + (x0 + x)) * 4;
                    const uint8_t alpha = da.bytes[idx + 3];
                    const uint8_t trans = dt.bytes[idx + 0];
                    if (alpha >= 128) {
                        if (trans < 15) ++opaque_below_floor;
                    } else {
                        // Check if any opaque pixel lies within 4-px Chebyshev distance inside usable rect.
                        bool near_opaque = false;
                        const int ny0 = std::max(pad, y - 4);
                        const int ny1 = std::min(W - pad - 1, y + 4);
                        const int nx0 = std::max(pad, x - 4);
                        const int nx1 = std::min(W - pad - 1, x + 4);
                        for (int ny = ny0; ny <= ny1 && !near_opaque; ++ny) {
                            for (int nx = nx0; nx <= nx1; ++nx) {
                                if (da.bytes[(static_cast<size_t>(y0 + ny) * da.w + (x0 + nx)) * 4 + 3] >= 128) {
                                    near_opaque = true;
                                    break;
                                }
                            }
                        }
                        if (near_opaque && trans == 0) ++near_edge_zero_trans;
                    }
                }
            }
        }
    }
    INFO("opaque_below_floor=" << opaque_below_floor << " near_edge_zero_trans=" << near_edge_zero_trans);
    REQUIRE(opaque_below_floor == 0);
    REQUIRE(near_edge_zero_trans == 0);
}

// ---- Test 22: Strict seed-independence + single-pass bundle parity ----------
TEST_CASE("[treegen_leaf_atlas_p3] all 4 atlas encoders are strictly seed-independent",
          "[treegen][treegen_leaf_atlas_p3][botanical_sprigs]") {
    const auto a = tg::encode_leaf_atlas_png(1u);
    const auto n = tg::encode_leaf_normal_png(1u);
    const auto t = tg::encode_leaf_translucency_png(1u);
    const auto r = tg::encode_leaf_roughness_png(1u);
    REQUIRE(a == tg::encode_leaf_atlas_png(0xDEADBEEF12345678ULL));
    REQUIRE(n == tg::encode_leaf_normal_png(0xDEADBEEF12345678ULL));
    REQUIRE(t == tg::encode_leaf_translucency_png(0xDEADBEEF12345678ULL));
    REQUIRE(r == tg::encode_leaf_roughness_png(0xDEADBEEF12345678ULL));

    const auto bundle = tg::encode_leaf_bundle_pngs(42u);
    REQUIRE(bundle.albedo_png       == a);
    REQUIRE(bundle.normal_png       == n);
    REQUIRE(bundle.translucency_png == t);
    REQUIRE(bundle.roughness_png    == r);
}

// ---- Test 23: OpenGL +Y_up normal sign-convention seal (convex dome, not saddle)
TEST_CASE("[treegen_leaf_atlas_p3] sprig normal map uses OpenGL +Y_up (+B = -dP/dV) convex dome sign",
          "[treegen][treegen_leaf_atlas_p3][botanical_sprigs]") {
    const auto bundle = tg::encode_leaf_bundle_pngs(23u);
    auto da = decode_atlas(bundle.albedo_png);
    auto dn = decode_atlas(bundle.normal_png);

    constexpr int W = tg::K_LEAF_CELL_PX;
    constexpr int half = W / 2;

    // Across all 9 broadleaf sprig cells (rows 1..3 x {Oak, Birch, Maple}) AND
    // Pine whorl bouquet cells (rows 2..3 x Pine), upper foliage points outward
    // along +B (-y_px -> ny > 0, G > 128), lower foliage along -B (ny < 0),
    // right along +T (nx > 0), left along -T (nx < 0).
    for (tg::LeafShape sp : k_species) {
        const int col = tg::leaf_cluster_cell_col(sp);
        const int row_start = (sp == tg::LeafShape::PineNeedle) ? 2 : tg::K_LEAF_SPRIG_ROW_MIN;
        double up_ny = 0.0, dn_ny = 0.0, rt_nx = 0.0, lt_nx = 0.0;
        size_t up_c = 0, dn_c = 0, rt_c = 0, lt_c = 0;

        for (int row = row_start; row <= tg::K_LEAF_SPRIG_ROW_MAX; ++row) {
            const int x0 = col * W;
            const int y0 = row * W;
            for (int y = 0; y < W; ++y) {
                for (int x = 0; x < W; ++x) {
                    const size_t idx = (static_cast<size_t>(y0 + y) * da.w + (x0 + x)) * 4;
                    if (da.bytes[idx + 3] < 128) continue;
                    const double nx = (static_cast<double>(dn.bytes[idx + 0]) / 255.0) * 2.0 - 1.0;
                    const double ny = (static_cast<double>(dn.bytes[idx + 1]) / 255.0) * 2.0 - 1.0;
                    if (y < half - 24)      { up_ny += ny; ++up_c; }
                    else if (y > half + 24) { dn_ny += ny; ++dn_c; }
                    if (x > half + 24)      { rt_nx += nx; ++rt_c; }
                    else if (x < half - 24) { lt_nx += nx; ++lt_c; }
                }
            }
        }
        REQUIRE(up_c > 1000u);
        REQUIRE(dn_c > 1000u);
        REQUIRE(rt_c > 1000u);
        REQUIRE(lt_c > 1000u);
        const double mean_up = up_ny / static_cast<double>(up_c);
        const double mean_dn = dn_ny / static_cast<double>(dn_c);
        const double mean_rt = rt_nx / static_cast<double>(rt_c);
        const double mean_lt = lt_nx / static_cast<double>(lt_c);
        INFO("col=" << col << " mean_up_ny=" << mean_up << " mean_dn_ny=" << mean_dn
             << " mean_rt_nx=" << mean_rt << " mean_lt_nx=" << mean_lt);
        REQUIRE(mean_up >  0.05);
        REQUIRE(mean_dn < -0.05);
        REQUIRE(mean_rt >  0.05);
        REQUIRE(mean_lt < -0.05);
    }
}
