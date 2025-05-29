cmake_minimum_required(VERSION 3.19)
project(DatasetUtils)
set(CMAKE_CXX_STANDARD 17)

# Build a library from the following command
# Add library or add executable
add_library(DatasetUtils include/dataset_utils/dataset_utils.h src/dataset_utils.cpp)
# set_property (TARGET DatasetUtils PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
# The following command will pass the location of the include directory with the package itself!
# Thus, there is no need to specify the location within the top-level cmake. Just add subdirectory & linking is required!

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/cmake ~/open3d_install/lib/cmake)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
target_include_directories(DatasetUtils PUBLIC include "${JSONCPP_INCLUDE_DIRS}")
target_link_libraries(DatasetUtils Eigen3::Eigen
                                   Open3D::Open3D
                                    ${JSONCPP_LINK_LIBRARIES})