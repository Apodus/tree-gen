// C6 P1 — PNG codec primitive. Pure decl; impl in png_encoder_api.cpp; the
// STB_IMAGE_WRITE_IMPLEMENTATION anchor is in png_encoder_anchor.cpp (linked
// only into rynx-treegen.exe). TestTech resolves stb_write_png symbols via
// Graphics → frame_capture.cpp anchor (transitive via Application).
#pragma once

#include <cstdint>
#include <vector>

namespace treegen {

    // Encode raw RGBA8 pixels to PNG bytes.
    // Pins stbi_write_png_compression_level=8 internally (defensive vs the
    // global mutable; intra-build determinism only).
    std::vector<uint8_t> encode_png_rgba8(int w, int h, int stride_bytes, const uint8_t* rgba);

}
