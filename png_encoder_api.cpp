// C6 P1 — PNG encode entry point. Decl in png_encoder.hpp; STB anchor in
// png_encoder_anchor.cpp. This TU never defines
// STB_IMAGE_WRITE_IMPLEMENTATION — its consumers (rynx-treegen.exe,
// TestTreegen) link the anchor.
#include "png_encoder.hpp"

#include <stb_image_write.h>

#include <stdexcept>

namespace treegen {

    namespace {
        // stbi_write_png_to_func sink — appends raw PNG bytes to a vector.
        // Mirrors rynx/src/rynx/graphics/capture/frame_capture.cpp:18-23.
        void png_write_cb(void* user, void* data, int size) {
            auto* out = static_cast<std::vector<uint8_t>*>(user);
            const auto* p = static_cast<const uint8_t*>(data);
            out->insert(out->end(), p, p + size);
        }
    }

    namespace {
        std::vector<uint8_t> encode_png_comp(int w, int h, int comp, int stride_bytes, const uint8_t* data) {
            // D7 defensive pin — global mutable in stb; latches level 8 every call so
            // the encoder is byte-deterministic regardless of other call sites.
            stbi_write_png_compression_level = 8;

            std::vector<uint8_t> out;
            out.reserve(static_cast<size_t>(w) * h);
            const int ok = stbi_write_png_to_func(
                &png_write_cb,
                &out,
                w,
                h,
                comp,
                data,
                stride_bytes);
            if (!ok || out.empty())
                throw std::runtime_error("treegen::encode_png: stbi_write_png_to_func failed");
            return out;
        }
    }

    std::vector<uint8_t> encode_png_rgba8(int w, int h, int stride_bytes, const uint8_t* rgba) {
        return encode_png_comp(w, h, 4, stride_bytes, rgba);
    }

    std::vector<uint8_t> encode_png_rgb8(int w, int h, int stride_bytes, const uint8_t* rgb) {
        return encode_png_comp(w, h, 3, stride_bytes, rgb);
    }

    std::vector<uint8_t> encode_png_gray8(int w, int h, int stride_bytes, const uint8_t* gray) {
        return encode_png_comp(w, h, 1, stride_bytes, gray);
    }

}
