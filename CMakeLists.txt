cmake_minimum_required(VERSION 3.15)
project(rynx-treegen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- library sources (everything except main.cpp and STB anchors) -----------
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS *.cpp)
list(FILTER LIB_SOURCES EXCLUDE REGEX "(main|stb_image_anchor|png_encoder_anchor)\\.cpp$")

# --- rynx-treegen exe -------------------------------------------------------
add_executable(rynx-treegen ${LIB_SOURCES} main.cpp stb_image_anchor.cpp png_encoder_anchor.cpp)
target_include_directories(rynx-treegen SYSTEM PRIVATE external/stb)
if(MSVC)
    target_compile_options(rynx-treegen PRIVATE /fp:precise)
endif()

# --- tests -------------------------------------------------------------------
enable_testing()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)

add_executable(treegen-tests ${LIB_SOURCES} stb_image_anchor.cpp png_encoder_anchor.cpp ${TEST_SOURCES})
target_include_directories(treegen-tests SYSTEM PRIVATE external/stb)
if(MSVC)
    target_compile_options(treegen-tests PRIVATE /fp:precise)
endif()

add_test(NAME treegen COMMAND treegen-tests)
