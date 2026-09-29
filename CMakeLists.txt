cmake_minimum_required(VERSION 3.18)

project(canny VERSION 1.0.0)

# Without an explicit build type CMake adds no optimization flags at all and the
# compiler falls back to -O0, which makes the gaussian blur roughly 18x slower.
# Default to an optimized build so a plain `cmake -S . -B build` is usable.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# use c++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# use static libraries
set(BUILD_SHARED_LIBS OFF)

set(CPP_SOURCE canny.cpp)
set(HEADERS canny.h)

# add directory for find package to search
# list(APPEND CMAKE_PREFIX_PATH "$HOME/libtorch")

# find opencv
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add local files
add_library(
  canny.hpp
  canny.cpp
)


add_executable(canny main.cpp)

target_link_libraries(canny "${OpenCV_LIBS}" canny.hpp)

set_property(TARGET canny PROPERTY CXX_STANDARD 17)
