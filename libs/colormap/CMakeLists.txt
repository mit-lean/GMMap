cmake_minimum_required(VERSION 3.19)
project(colormap)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")
set(CMAKE_BUILD_TYPE "RelWithDebInfo")

add_library(colormap INTERFACE)
find_package(fmt)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_include_directories(colormap INTERFACE include)
target_link_libraries(colormap INTERFACE Eigen3::Eigen fmt::fmt)