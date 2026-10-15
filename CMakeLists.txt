cmake_minimum_required(VERSION 3.24)  # $<PATH:RELATIVE_PATH,...>
project(rynx-treegen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- library sources (everything except main.cpp and STB anchors) -----------
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS *.cpp)
list(FILTER LIB_SOURCES EXCLUDE REGEX "(main|stb_image_anchor|png_encoder_anchor)\\.cpp$")

# Each exe finds the repository root from its own location (treegen_root.hpp): the
# relative path from its directory to this one, compiled in; never an absolute path.
function(treegen_root_from_exe target)
    target_compile_definitions(${target} PRIVATE
        TREEGEN_ROOT_FROM_EXE="$<PATH:RELATIVE_PATH,${CMAKE_CURRENT_SOURCE_DIR},$<TARGET_FILE_DIR:${target}>>")
endfunction()

# --- rynx-treegen exe -------------------------------------------------------
add_executable(rynx-treegen ${LIB_SOURCES} main.cpp stb_image_anchor.cpp png_encoder_anchor.cpp)
target_include_directories(rynx-treegen SYSTEM PRIVATE external/stb)
treegen_root_from_exe(rynx-treegen)
if(MSVC)
    target_compile_options(rynx-treegen PRIVATE /fp:precise)
endif()

# --- tests -------------------------------------------------------------------
enable_testing()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
# The GPU render gates spawn game_one's topdownshootergame.exe; they build only
# in game_one (TestTreegenRender).
list(FILTER TEST_SOURCES EXCLUDE REGEX "test_treegen_(render_phash|species_library|all_species_lod_render)\\.cpp$")

add_executable(treegen-tests ${LIB_SOURCES} stb_image_anchor.cpp png_encoder_anchor.cpp ${TEST_SOURCES})
target_include_directories(treegen-tests SYSTEM PRIVATE external/stb)
treegen_root_from_exe(treegen-tests)
if(MSVC)
    target_compile_options(treegen-tests PRIVATE /fp:precise)
endif()
# [treegen_species_registry] spawns the tool: this build's, by compile-time path.
add_dependencies(treegen-tests rynx-treegen)
target_compile_definitions(treegen-tests PRIVATE RYNX_TREEGEN_EXE="$<TARGET_FILE:rynx-treegen>")

add_test(NAME treegen COMMAND treegen-tests)
