# 3.10 introduces C++ 17 support
cmake_minimum_required (VERSION 3.10)
project (nigh_demo)

# Enable C++17 (required for Nigh)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable extensions to keep compatible with standards
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(../src)

add_executable(euclidean_demo euclidean_demo.cpp)
target_link_libraries(euclidean_demo Eigen3::Eigen)

add_executable(custom_vector_demo custom_vector_demo.cpp)
target_link_libraries(custom_vector_demo Eigen3::Eigen)
