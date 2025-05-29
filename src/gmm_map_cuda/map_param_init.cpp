//
// Created by peterli on 6/28/23.
//
#include "gmm_map_cuda/map_param_init.h"
#include "dataset_utils/dataset_utils.h"

namespace gmm {
    void initializeCudaMapParameters(map_cuda_param& param){
        auto gpu_param = dataset_param::dataset_info["gpu_compute_parameters"];
        if (gpu_param.empty()){
            throw std::invalid_argument("Unable to read parameters for CUDA from file. Check if the fields are correct!");
        }
        param.num_blocks = gpu_param["num_blocks"].asInt();
        param.threads_per_block = gpu_param["threads_per_block"].asInt();
        param.max_segments_per_line = gpu_param["max_segments_per_line"].asInt();
    }
}