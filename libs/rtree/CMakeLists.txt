cmake_minimum_required(VERSION 3.19)
option(TRACK_MEMORY_RTREE "Build programs for obtaining implementation dependent memory usage" OFF)
project(RTree)
set(CMAKE_CXX_STANDARD 17)
# Check optimization levels: https://www.linuxtopia.org/online_books/an_introduction_to_gcc/gccintro_49.html
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")
set(CMAKE_BUILD_TYPE "RELEASE")

add_library(RTree INTERFACE)
set_property (TARGET RTree PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

target_include_directories(RTree INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(RTree INTERFACE Eigen3::Eigen
										OpenMP::OpenMP_CXX
										${LIBS_MEM}
										)