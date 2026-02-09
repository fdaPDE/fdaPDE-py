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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -march=native ${WARNING_FLAGS}")

if(MSVC)
    add_compile_options(/Zc:lambda)
endif()

find_package(Eigen3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    set(LOCAL_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen3")
    if(EXISTS "${LOCAL_EIGEN_DIR}")
        set(EIGEN_INCLUDE_DIR "${LOCAL_EIGEN_DIR}")
    else()
        message(FATAL_ERROR "Eigen3 not found")
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fdaPDE-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fdaPDE-cpp/fdaPDE/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_INCLUDE_DIR}
)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

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
