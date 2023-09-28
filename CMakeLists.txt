cmake_minimum_required(VERSION 3.16.2)
project(collective_RL_Transport)

option(CLUSTER "whether we are running on the cluster" OFF)
option(VICON "whether we are running Vicon" OFF)

find_package(ARGoS REQUIRED)
include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})
link_directories(${ARGOS_LIBRARY_DIR})
link_libraries(${ARGOS_LDFLAGS})

# Check for Buzz
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};/usr/local/share/buzz/cmake;/home/opletts/buzzbundle/share/buzz/cmake)
#message("CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
find_package(Buzz REQUIRED)
include_directories(${BUZZ_C_INCLUDE_DIR})
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${Buzz_DIR}")
include(UseBuzz)

# Check for ZeroMQ
find_package(PkgConfig)
pkg_check_modules(ZeroMQ REQUIRED libzmq)
include_directories(${ZeroMQ_INCLUDE_DIRS})

# Compile ARGoS code
add_subdirectory(argos)
