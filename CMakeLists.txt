cmake_minimum_required(VERSION 3.19)
option(GMMAP_BUILD_CUDA "Build CUDA version" OFF)
option(GMMAP_BUILD_EXAMPLES "Build example programs" OFF)

if (GMMAP_BUILD_CUDA)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    project(gmm_map LANGUAGES CXX CUDA VERSION 2.3 DESCRIPTION "GMMap")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
else()
    project(gmm_map LANGUAGES CXX VERSION 2.3 DESCRIPTION "GMMap")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")
set(CMAKE_BUILD_TYPE "RelWithDebInfo")

if (GMMAP_BUILD_CUDA)
    message("${cxx_warning_flags}")
    foreach(flag IN LISTS cxx_warning_flags)
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
    endforeach()
endif()

# Add appropriate libraries
if(NOT TARGET RTree)
    add_subdirectory(libs/rtree/)
endif()

if (NOT TARGET nigh)
    add_subdirectory(libs/nigh)
endif ()

# Define source files (CPU version)
list(APPEND GMM_SOURCES
        include/gmm_map/cluster_ops.h
        include/gmm_map/commons.h
        include/gmm_map/cluster.h
        include/gmm_map/map.h
        include/gmm_map/map_param.h
        src/gmm_map/cluster_serialize.cpp
        src/gmm_map/map_serialize.cpp
        src/gmm_map/cluster_ops.cpp
        src/gmm_map/SPGFExtended.cpp
        src/gmm_map/map.cpp
        src/gmm_map/map_param.cpp
        src/path_planning/path_planner.cpp
        )

# Define source files (CUDA version)
if (GMMAP_BUILD_CUDA)
    list(APPEND GMM_SOURCES
            include/gmm_map_cuda/cuda_param.cuh
            include/gmm_map_cuda/map.h
            include/gmm_map_cuda/map.cuh
            include/gmm_map_cuda/cuda_common.cuh
            include/gmm_map_cuda/matrix.cuh
            include/gmm_map_cuda/buffer.cuh
            include/gmm_map_cuda/cluster.cuh
            include/gmm_map_cuda/map_param.cuh
            include/gmm_map_cuda/helper_math.h
            src/gmm_map_cuda/SPGFExtended.cpp
            src/gmm_map_cuda/matrix.cu
            src/gmm_map_cuda/map.cpp
            src/gmm_map_cuda/map.cu
            src/gmm_map_cuda/cluster.cu
            )
endif()

# Define visualization sources
list(APPEND GMM_VIZ_SOURCES
        include/gmm_map/Visualization.h
        include/gmm_map/evaluation.h
        include/gmm_map/map_param_init.h
        src/gmm_map/Visualization.cpp
        src/gmm_map/evaluation.cpp
        src/gmm_map/map_param_init.cpp
        src/path_planning/path_planning_param_init.cpp
        )

# Define source files (CUDA version)
if (GMMAP_BUILD_CUDA)
    list(APPEND GMM_VIZ_SOURCES
            include/gmm_map_cuda/map_param_init.h
            src/gmm_map_cuda/map_param_init.cpp
            )
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    list(APPEND CUDA_OPTS
            CUDA::cudart
            CUDA::cusolver
            CUDA::cublasLt
            )
endif(CUDAToolkit_FOUND)

if (APPLE)
    list(APPEND LIBS_OPT
            /opt/homebrew/Cellar/gcc/12.2.0/lib/gcc/current/libquadmath.a
            )
endif()

# Construct GMM library
add_library(gmm_map SHARED ${GMM_SOURCES})
set_property (TARGET gmm_map PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
target_include_directories(gmm_map PUBLIC include)

if (TRACK_MEMORY_GMM)
    message(STATUS "TRACK_MEMORY_GMM - ON")
    list(APPEND LIBS_MEM
            procps
            MemUtils
            )
    target_compile_options(gmm_map PUBLIC -DTRACK_MEM_USAGE_GMM)
endif()

target_link_libraries(gmm_map
                    PUBLIC
                    Eigen3::Eigen
                    RTree
                    nigh
                    PRIVATE
                    ${LIBS_MEM}
                    OpenMP::OpenMP_CXX
                    ${CUDA_OPTS}
                    fmt::fmt
                    ${LIBS_OPT}
                    )
if (GMMAP_BUILD_CUDA)
    set_target_properties(gmm_map PROPERTIES CUDA_ARCHITECTURES "61;62;72;89")
    target_compile_options(gmm_map PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --use_fast_math
            #--gpu-architecture=native
            >)
endif()

# Linking libraries for test / evaluation code
if (GMMAP_BUILD_EXAMPLES)
    # Find installed Open3D, which exports Open3D::Open3D
    # To fix linking errors, make sure to build Open3d with the cmake flag: -DGLIBCXX_USE_CXX11_ABI=ON
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/cmake ~/open3d_install/lib/cmake REQUIRED)

    if(NOT TARGET DatasetUtils)
        add_subdirectory(libs/dataset_utils/)
    endif()

    if(NOT TARGET colormap)
        add_subdirectory(libs/colormap/)
    endif()

    # Construct GMM Visualization library
    add_library(gmm_map_viz ${GMM_VIZ_SOURCES})
    set_property (TARGET gmm_map_viz PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
    target_include_directories(gmm_map_viz PUBLIC include)
    target_link_libraries(gmm_map_viz
            PUBLIC
            gmm_map
            Open3D::Open3D
            PRIVATE
            DatasetUtils
            colormap
            ${LIBS_MEM}
            OpenMP::OpenMP_CXX
            fmt::fmt
            ${CUDA_OPTS}
            ${LIBS_OPT}
            )

    add_executable(GMMMapVisualization ./example/gmm_map_viz.cpp) # GMM Mapping with visualization support
    add_executable(GMMMapNoViz ./example/gmm_map_no_viz.cpp) # GMM Mapping with no visualization (compute only)
    add_executable(GMMPathPlanningViz ./path_planning_example/gmm_path_planning_viz.cpp) # Efficient path planning example

    target_link_libraries(GMMMapVisualization Eigen3::Eigen
                        Open3D::Open3D
                        ${CUDA_OPTS}
                        DatasetUtils
                        colormap
                        gmm_map_viz
                        ${LIBS_OPT})

    target_link_libraries(GMMPathPlanningViz Eigen3::Eigen
                            Open3D::Open3D
                            ${CUDA_OPTS}
                            DatasetUtils
                            colormap
                            gmm_map_viz
                            ${LIBS_OPT})

    target_link_libraries(GMMMapNoViz Eigen3::Eigen
                        Open3D::Open3D
                        ${CUDA_OPTS}
                        DatasetUtils
                        colormap
                        gmm_map_viz
                        ${LIBS_OPT})

    if (GMMAP_BUILD_CUDA)
        add_executable(GMMMapCudaNoViz ./example/gmm_map_cuda_no_viz.cpp) # GMM Mapping with no visualization (compute only)
        target_link_libraries(GMMMapCudaNoViz Eigen3::Eigen
                            Open3D::Open3D
                            ${CUDA_OPTS}
                            DatasetUtils
                            colormap
                            gmm_map_viz)

        add_executable(GMMMapCudaVisualization ./example/gmm_map_cuda_viz.cpp) # GMM Mapping with visualization support & cuda SPGF acceleration
        target_link_libraries(GMMMapCudaVisualization Eigen3::Eigen
                Open3D::Open3D
                ${CUDA_OPTS}
                DatasetUtils
                colormap
                gmm_map_viz)
    endif()
endif()