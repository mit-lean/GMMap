cmake_minimum_required(VERSION 3.19)
project(nigh)
set(CMAKE_CXX_STANDARD 17)
# Check optimization levels: https://www.linuxtopia.org/online_books/an_introduction_to_gcc/gccintro_49.html
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")
set(CMAKE_BUILD_TYPE "RELEASE")

add_library(nigh INTERFACE)
set_property (TARGET nigh PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
target_include_directories(nigh INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nigh INTERFACE Eigen3::Eigen
										OpenMP::OpenMP_CXX)