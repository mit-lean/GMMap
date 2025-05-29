#ifndef GMM_CUDA_MAP_CUH
#define GMM_CUDA_MAP_CUH
#include "gmm_map_cuda/cuda_common.cuh"
#include "gmm_map_cuda/cuda_param.cuh"
#include "gmm_map_cuda/map_param.cuh"
#include "gmm_map_cuda/cluster.cuh"
#include "gmm_map_cuda/buffer.cuh"
#include "gmm_map_cuda/matrix.cuh"

// Allocate constant memory for device (Do not define, just declare these variables)
extern __constant__ frame_param_cuda cuda_frame_param_device;

// Kernel functions that should be only used in C++ wrapper functions
__global__ void lineSegmentationExtendedBlock(const float * depthmap, int row_idx_offset,
                                              Buffer2D<cuGMM::GMMmetadata_c>* cur_obs_line_segments);
__global__ void lineSegmentationExtendedOpt(const float * depthmap, float * x, float * y, int row_idx);
// Functions that can be called from both device and host
__host__ __device__ void lineSegmentationExtendedCuda(const float* depthmap, int row_idx, int thread_idx,
                                         Buffer2D<cuGMM::GMMmetadata_c>* cur_obs_line_segments, const frame_param_cuda* cuda_frame_param);
__host__ __device__ void constructSegmentsFromPointCuda(const cuGMM::V& point, const cuGMM::V& color, int row_idx, int col_idx,
                                           CircularBuffer<cuGMM::GMMmetadata_c, map_cuda_const::max_intermediate_clusters>& obs_imcomplete_queue,
                                           int thread_idx, Buffer2D<cuGMM::GMMmetadata_c>* cur_obs_line_segments, const frame_param_cuda* cuda_frame_param);
__host__ __device__ void addPointObsCuda(const cuGMM::V& point, const cuGMM::V& color, int v, int u, int depth_idx,
                            cuGMM::GMMmetadata_c& metadata, const frame_param_cuda* cuda_frame_param);

__host__ __device__ void forwardProjectScanline(const float* depth, float* x, float* y, int row_idx, int thread_idx, const frame_param_cuda* cuda_frame_param);

// Wrapper functions that can be used in other C++ files
void transferMapParamToDevice(frame_param_cuda* cuda_frame_param_host);
void freeMapParamOnDevice();

Buffer2D<cuGMM::GMMmetadata_c>* allocateScanlineSegmentsUnified(int total_threads, int max_segments_per_thread, cudaStreamBuffer* buffer, int cpu_thread_idx);
void freeScanlineSegmentsUnified(Buffer2D<cuGMM::GMMmetadata_c>* segments);

float* allocateScanlineGPUMemory(int total_threads, int img_width);
void transferScanlineToGPUMemory(float* scanlines, const float* depthmap_offset, int num_elements);
void freeScanlineGPUMemory(float* scanlines);

float* allocateImageGPUMemory(int img_width, int img_height);
void transferImageToGPUMemory(float* img_gpu, const float* img_cpu, int img_width, int img_height);
void freeImageGPUMemory(float* img_gpu);
void synchronizeDefaultStream();

void lineSegmentationExtendedWrapper(const float * depthmap, int row_idx_offset,
                                     Buffer2D<cuGMM::GMMmetadata_c>* cur_obs_line_segments,
                                     cudaStreamBuffer* buffer, int cpu_thread_idx,
                                     const gmm::map_cuda_param* cuda_config_param);

void lineSegmentationExtendedOptWrapper(const float * depthmap, int row_offset,
                                        int num_streams,
                                        cudaStreamBuffer* buffer,
                                        int img_width,
                                        float * x, float * y,
                                        const gmm::map_cuda_param* cuda_config_param);

#endif
