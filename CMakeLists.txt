cmake_minimum_required(VERSION 3.15)

# -----------------------------------------------------------------------------
# Project metadata
# -----------------------------------------------------------------------------
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Compiler setup
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(WARNING_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -march=x86-64 ${WARNING_FLAGS}")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/fdaPDE-cpp/fdaPDE/core/
  ${CMAKE_CURRENT_SOURCE_DIR}/src/
)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# -----------------------------------------------------------------------------
# Helper function to create a pybind11 module
# -----------------------------------------------------------------------------
function(add_fdapde_module name filepath)
  message(STATUS "Adding Python module: ${name} (${filepath})")

  python_add_library(${name} MODULE ${filepath} WITH_SOABI)
  target_link_libraries(${name} PRIVATE pybind11::module Eigen3::Eigen)
  target_compile_definitions(${name} PRIVATE VERSION_INFO="${PROJECT_VERSION}")
  set_target_properties(${name} PROPERTIES OUTPUT_NAME "${name}")

  install(TARGETS ${name} DESTINATION fdapdepy)
endfunction()

# -----------------------------------------------------------------------------
# Automatically find all .cpp files in src/
# -----------------------------------------------------------------------------
file(GLOB MODULE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach(src_file IN LISTS MODULE_SOURCES)
 get_filename_component(module_name ${src_file} NAME_WE)
    pybind11_add_module(_${module_name} ${src_file})
    target_link_libraries(_${module_name} PRIVATE Eigen3::Eigen)
    install(TARGETS _${module_name} DESTINATION fdapdepy)
endforeach()

# -----------------------------------------------------------------------------
# Python package metadata (init file etc.)
# -----------------------------------------------------------------------------
install(FILES src/fdapdepy/__init__.py DESTINATION fdapdepy)
