//
// Created by peter on 7/4/21.
//

#ifndef GMM_CUDA_PARAMETERS_H
#define GMM_CUDA_PARAMETERS_H
#include "cuda_common.cuh"

namespace gmm {
    struct map_cuda_param {
        // Struct used for cuda parameter initialization and management
        // https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
        int num_blocks = 4;
        int threads_per_block = 32;
        int num_streams = 1;
        // http://www.trevorsimonton.com/blog/2016/11/16/transfer-2d-array-memory-to-cuda.html
        int max_segments_per_line = 32; // Number of output segments for scanline segmentation (16 - 32 is good enough)

        int totalGPUThreads() const {
            return threads_per_block * num_blocks;
        }

        int maxConcurrentScanlineSegmentation(int num_cpu_threads, int num_scanlines) const {
            return std::min(num_cpu_threads, (int) std::ceil((float) num_scanlines / (float) totalGPUThreads()));
        }
    };
}

namespace map_cuda_const
{
    // Define constants used to initialize array size within CUDA
    constexpr int max_intermediate_clusters = 2; // Number of intermediate clusters used during scanline segmentation
}

#endif //GMM_CUDA_PARAMETERS_H
