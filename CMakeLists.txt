cmake_minimum_required(VERSION 3.15)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_WARNINGS_FLAGS "-Wall -Wpedantic -Wextra -Werror")
set(CMAKE_CXX_FLAGS "-g -O2 -march=x86-64 -std=c++20 ${CMAKE_WARNINGS_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include/fdaPDE-cpp/fdaPDE/core/ -I${CMAKE_CURRENT_SOURCE_DIR}/src/" )
  

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)

#add_subdirectory(pybind11)
#include_directories(include/fdaPDE-cpp/fdaPDE/)

#python_add_library(_domain MODULE src/mesh.cpp WITH_SOABI)
#target_link_libraries(_domain PRIVATE pybind11::headers Eigen3::Eigen)

#python_add_library(_functional_basis MODULE src/functional_basis.cpp WITH_SOABI)
#target_link_libraries(_functional_basis PRIVATE pybind11::headers Eigen3::Eigen)

#python_add_library(_pde MODULE src/pde.cpp WITH_SOABI)
#target_link_libraries(_pde PRIVATE pybind11::headers Eigen3::Eigen)

#python_add_library(_regression MODULE src/regression.cpp WITH_SOABI)
#target_link_libraries(_regression PRIVATE pybind11::headers Eigen3::Eigen)

#pybind11_add_module(fdapdepy src/mesh.cpp)
#target_link_libraries (fdapdepy PUBLIC Eigen3::Eigen)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
#target_compile_definitions(_domain PRIVATE VERSION_INFO=${PROJECT_VERSION})
#target_compile_definitions(_functional_basis PRIVATE VERSION_INFO=${PROJECT_VERSION})
#target_compile_definitions(_pde PRIVATE VERSION_INFO=${PROJECT_VERSION})
#target_compile_definitions(_regression PRIVATE VERSION_INFO=${PROJECT_VERSION})

#install(TARGETS _domain DESTINATION fdapdepy)
#install(TARGETS _functional_basis DESTINATION fdapdepy)
#install(TARGETS _pde DESTINATION fdapdepy)
#install(TARGETS _regression DESTINATION fdapdepy)
#install(FILES src/fdapdepy/__init__.py
#        DESTINATION fdapdepy)
