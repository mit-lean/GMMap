// This file is obtained from a Udemy Course: CUDA programming Masterclass with C++
#ifndef CUDA_COMMON_CUH
#define CUDA_COMMON_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gmm_map_cuda/helper_math.h"
#include "cstdio"
#include <iostream>
// Define Error checking for all cuda functions
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline void query_cuda_device()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("No CUDA support device found");
    }

    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);

    printf("Device %d: %s\n", devNo, iProp.name);
    printf("  Number of multiprocessors:                     %d\n",
           iProp.multiProcessorCount);
    printf("  clock rate :                     %d\n",
           iProp.clockRate);
    printf("  Compute capability       :                     %d.%d\n",
           iProp.major, iProp.minor);
    printf("  Total amount of global memory:                 %4.2f KB\n",
           iProp.totalGlobalMem / 1024.0);
    printf("  Total amount of constant memory:               %4.2f KB\n",
           iProp.totalConstMem / 1024.0);
    printf("  Total amount of shared memory per block:       %4.2f KB\n",
           iProp.sharedMemPerBlock / 1024.0);
    printf("  Total amount of shared memory per MP:          %4.2f KB\n",
           iProp.sharedMemPerMultiprocessor / 1024.0);
    printf("  Total number of registers available per block: %d\n",
           iProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           iProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n",
           iProp.maxThreadsPerBlock);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           iProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of warps per multiprocessor:    %d\n",
           iProp.maxThreadsPerMultiProcessor / 32);
    printf("  Maximum Grid size                         :    (%d,%d,%d)\n",
           iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);
    printf("  Maximum block dimension                   :    (%d,%d,%d)\n",
           iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
}

#endif //CUDA_COMMON_CUH
