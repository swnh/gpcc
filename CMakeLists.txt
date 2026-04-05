cmake_minimum_required(VERSION 3.0)
project(gpcc CXX)

# C++11 support check
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(ORIGINAL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mpeg-pcc-tmc13/tmc3)
set(DEPENDENCY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mpeg-pcc-tmc13/dependencies)

add_subdirectory(sandbox)
