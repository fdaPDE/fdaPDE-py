cmake_minimum_required(VERSION 3.15)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(WARNING_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -march=x86-64 ${WARNING_FLAGS}")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/fdaPDE-cpp/
  ${CMAKE_CURRENT_SOURCE_DIR}/include/fdaPDE-cpp/fdaPDE/core/
  ${CMAKE_CURRENT_SOURCE_DIR}/src/
)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

nanobind_add_module(
  cpp
  NB_SUPPRESS_WARNINGS
  src/fdaPDE.cpp
  src/geometry.cpp
  src/geoframe.cpp
  src/fem.cpp
  src/models.cpp
)

install(TARGETS cpp DESTINATION fdaPDE)
install(FILES src/fdaPDE/__init__.py DESTINATION fdaPDE)
