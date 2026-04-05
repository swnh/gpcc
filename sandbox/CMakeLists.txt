### SECTION 1. Basic Setup ###
cmake_minimum_required(VERSION 3.0)

# Enable C language (needed for schroarith.c dependency)
enable_language(C)

# Platform-specific definitions
if(UNIX)
    add_definitions(-D_POSIX_C_SOURCE=200809L)
elseif(WIN32)
    add_definitions(-D_USE_MATH_DEFINES)
endif()

# Check for getrusage function (needed for pcc_chrono.cpp)
include(CheckSymbolExists)
check_symbol_exists(getrusage sys/resource.h HAVE_GETRUSAGE)


### SECTION 2. Original TMC3 .cpp files ###
set(ORIGINAL_SOURCES
    ${ORIGINAL_SRC_DIR}/AttributeCommon.cpp
    ${ORIGINAL_SRC_DIR}/AttributeDecoder.cpp
    ${ORIGINAL_SRC_DIR}/AttributeEncoder.cpp
    ${ORIGINAL_SRC_DIR}/DualLutCoder.cpp
    ${ORIGINAL_SRC_DIR}/FixedPoint.cpp
    ${ORIGINAL_SRC_DIR}/OctreeNeighMap.cpp
    ${ORIGINAL_SRC_DIR}/RAHT.cpp
    #${ORIGINAL_SRC_DIR}/TMC3.cpp
    ${ORIGINAL_SRC_DIR}/attribute_raw_decoder.cpp
    ${ORIGINAL_SRC_DIR}/attribute_raw_encoder.cpp
    ${ORIGINAL_SRC_DIR}/coordinate_conversion.cpp
    ${ORIGINAL_SRC_DIR}/decoder.cpp
    ${ORIGINAL_SRC_DIR}/encoder.cpp
    ${ORIGINAL_SRC_DIR}/entropydirac.cpp
    ${ORIGINAL_SRC_DIR}/frame.cpp
    ${ORIGINAL_SRC_DIR}/geometry_intra_pred.cpp
    ${ORIGINAL_SRC_DIR}/geometry_octree.cpp
    ${ORIGINAL_SRC_DIR}/geometry_octree_decoder.cpp
    ${ORIGINAL_SRC_DIR}/geometry_octree_encoder.cpp
    ${ORIGINAL_SRC_DIR}/geometry_predictive_decoder.cpp
    # ${ORIGINAL_SRC_DIR}/geometry_predictive_encoder.cpp # REPLACED BY CUSTOM
    ${ORIGINAL_SRC_DIR}/geometry_trisoup_decoder.cpp
    ${ORIGINAL_SRC_DIR}/geometry_trisoup_encoder.cpp
    ${ORIGINAL_SRC_DIR}/io_hls.cpp
    ${ORIGINAL_SRC_DIR}/io_tlv.cpp
    ${ORIGINAL_SRC_DIR}/misc.cpp
    ${ORIGINAL_SRC_DIR}/motionWip.cpp
    ${ORIGINAL_SRC_DIR}/osspecific.cpp
    ${ORIGINAL_SRC_DIR}/partitioning.cpp
    ${ORIGINAL_SRC_DIR}/pcc_chrono.cpp
    # ${ORIGINAL_SRC_DIR}/ply.cpp # REPLACED BY CUSTOM
    ${ORIGINAL_SRC_DIR}/pointset_processing.cpp
    ${ORIGINAL_SRC_DIR}/quantization.cpp
    ${ORIGINAL_SRC_DIR}/tables.cpp
)

# Custom
set(CUSTOM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/geometry_predictive_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ply.cpp
)

# Dependencies
set(DEPENDENCY_SOURCES
    ${DEPENDENCY_DIR}/program-options-lite/program_options_lite.cpp
    ${DEPENDENCY_DIR}/schroedinger/schroarith.c
)


### SECTION 3. Generate Required Files ###

# Generate TMC3Config.h (TMC.h needs this)
configure_file(
    ${ORIGINAL_SRC_DIR}/TMC3Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/TMC3Config.h
)

# Generate version.cpp (TMC3.cpp needs this)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
"#include \"version.h\"

namespace pcc {
const char version[] = \"pred\";
}
")

### SECTION 4. Include Directories ###
include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/source # Load modified header first if exists
    ${ORIGINAL_SRC_DIR}
    ${ORIGINAL_SRC_DIR}/..              # Parent dir for "dependencies/..." includes
    ${DEPENDENCY_DIR}/nanoflann
    ${DEPENDENCY_DIR}/program-options-lite
    ${DEPENDENCY_DIR}/schroedinger
)


### SECTION 5. Build Executable ###
add_executable(predgeom
    ${ORIGINAL_SOURCES}
    ${CUSTOM_SOURCES}
    ${DEPENDENCY_SOURCES}
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
)