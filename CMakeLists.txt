cmake_minimum_required(VERSION 3.16.2)
project(collective_RL_Transport)

option(CLUSTER "whether we are running on the cluster" OFF)

# Leave for apple computers so that they can debug the code.
if(APPLE)
  # Add address sanitizer support for CLang
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-optimize-sibling-calls -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-optimize-sibling-calls -fno-omit-frame-pointer")
  set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} -fsanitize=address")
  set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} -fsanitize=address")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} -fsanitize=address")
endif(APPLE)

# Deactivate RPATH for MacOSX
set(CMAKE_MACOSX_RPATH 0)

# Find the ARGoS package, make sure to save the ARGoS prefix
find_package(PkgConfig)
pkg_check_modules(ARGOS REQUIRED argos3_simulator)
set(ARGOS_PREFIX ${ARGOS_PREFIX} CACHE INTERNAL "")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ARGOS_PREFIX}/share/argos3/cmake)

# Check whether all the necessary libs have been installed to compile the
# code that depends on Qt and OpenGL
include(ARGoSCheckQTOpenGL)

# Set ARGoS include dir
include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})

# Set ARGoS link dir
link_directories(${ARGOS_LIBRARY_DIRS})

if(NOT CLUSTER)
  # Find Lua
  find_package(Lua53)
  include_directories(${LUA_INCLUDE_DIR})
endif(NOT CLUSTER)

# Check for Buzz
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};/usr/local/share/buzz/cmake;/home/jdbloom/buzzbundle/share/buzz/cmake)
message("CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
find_package(Buzz REQUIRED)
include_directories(${BUZZ_C_INCLUDE_DIR})

# Check for ZeroMQ
pkg_check_modules(ZeroMQ REQUIRED libzmq)
include_directories(${ZeroMQ_INCLUDE_DIRS})

# Compile ARGoS code
add_subdirectory(argos)
