// C9 P3 -- STB_IMAGE_IMPLEMENTATION anchor for rynx-treegen.exe.
// The Graphics library has its own anchor (image.cpp); treegen doesn't link
// Graphics, so it needs its own. Used by phash.hpp (decodes PNGs in the tests).
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
