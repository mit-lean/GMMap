#include "gmm_map_cuda/map.cuh"
#include <cmath>
#include <iostream>
#include <chrono>
#include <fmt/format.h>

__constant__ frame_param_cuda cuda_frame_param_const_mem;

__global__ void lineSegmentationExtendedBlock(const float * depthmap, int row_idx_offset,
                                              Buffer2D<cuGMM::GMMmetadata_c>* cur_obs_line_segments){
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = row_idx_offset + global_thread_idx;
    if (row_idx < cuda_frame_param_const_mem.img_height) {
        lineSegmentationExtendedCuda(depthmap, row_idx, global_thread_idx,cur_obs_line_segments, &cuda_frame_param_const_mem);
    }
}

__global__ void lineSegmentationExtendedOpt(const float * depthmap, float * x, float * y, int row_idx){
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < cuda_frame_param_const_mem.img_width) {
        forwardProjectScanline(depthmap, x, y, row_idx, global_thread_idx, &cuda_frame_param_const_mem);
    }
}

__host__ __device__ void lineSegmentationExtendedCuda(const float* depthmap, int row_idx, int thread_idx,
                                                      Buffer2D<cuGMM::GMMmetadata_c>* cur_obs_line_segments, const frame_param_cuda* cuda_frame_param) {
    // Same as ICRA except that we also update some metadata for tracking free space
    // We merge the creation of scanline segments for free and obstacles!
    // Scanline Segmentation
    CircularBuffer<cuGMM::GMMmetadata_c, map_cuda_const::max_intermediate_clusters> obs_imcomplete_queue;
    //int row_offset = row_idx * cuda_frame_param->img_width;
    for (int col_idx = 0; col_idx < cuda_frame_param->img_width; ++col_idx){
        cuGMM::V point;
        cuGMM::V color;
        // Column major format!
        float depth = depthmap[col_idx * cuda_frame_param->img_height + row_idx];
        point = cuda_frame_param->forwardProject(row_idx, col_idx, depth);

        // Update obstacles and free space (metadata)
        constructSegmentsFromPointCuda(point, color, row_idx, col_idx, obs_imcomplete_queue, thread_idx, cur_obs_line_segments, cuda_frame_param);
    }

    // Completion detection for obstacles (insert incomplete clusters into the queue if necessary)
    int num_remaining_clusters = obs_imcomplete_queue.size();
    for (int i = 0; i < num_remaining_clusters; i++){
        if (obs_imcomplete_queue(i).N > cuda_frame_param->sparse_t){
            cur_obs_line_segments->push_back(thread_idx, obs_imcomplete_queue(i));
        }
    }
}
__host__ __device__ void constructSegmentsFromPointCuda(const cuGMM::V& point, const cuGMM::V& color, int row_idx, int col_idx,
                                                        CircularBuffer<cuGMM::GMMmetadata_c, map_cuda_const::max_intermediate_clusters>& obs_imcomplete_queue,
                                                        int thread_idx, Buffer2D<cuGMM::GMMmetadata_c>* cur_obs_line_segments, const frame_param_cuda* cuda_frame_param){
    // Compute for obstacles and free space segments
    // Same as ICRA, but with additional support for color and free space metadata
    if (point.z <= 0 || point.z > cuda_frame_param->max_depth){
        return;
    }

    // First, we compute the adaptive thresholds
    bool merged;
    float line_t_adapt, depth_t_adapt;
    cuda_frame_param->adaptThreshold(point, line_t_adapt, depth_t_adapt);
    merged = false; //Track whether the current point is merged into existing segments.
    int cur_depth_idx = cuda_frame_param->determineDepthIndex(point.z);
    float max_len = cuda_frame_param->determineDepthLength(cur_depth_idx);

#ifdef __CUDA_ARCH__
    float z_dist_thresh = fminf(fmaxf(depth_t_adapt, cuda_frame_param->noise_floor),cuda_frame_param->noise_thresh);
    float x_dist_thresh = fminf(fmaxf(line_t_adapt, cuda_frame_param->noise_floor),cuda_frame_param->line_t);
#else
    float z_dist_thresh = fmin(fmax(depth_t_adapt, cuda_frame_param->noise_floor),cuda_frame_param->noise_thresh);
    float x_dist_thresh = fmin(fmax(line_t_adapt, cuda_frame_param->noise_floor),cuda_frame_param->line_t);
#endif


    // Determine if the current point can be merged into existing clusters to the nearest cluster first
    for (int i = 0; i < obs_imcomplete_queue.size(); i++){
        auto& cur_cluster = obs_imcomplete_queue(i);
        // Note: need to insert new element at the beginning of the list
        // Unlike ICRA 2022's SPGF algorithm, we use the free space depth discretization
        // Better accuracy by restricting the max planewidth of the Gaussians
        //if (!merged && col_idx - it->RightPixel <= clustering_params.occ_x_t && it->PlaneWidth < pow(clustering_params.max_len, 2)){
        if (!merged && col_idx - cur_cluster.RightPixel <= cuda_frame_param->occ_x_t && cur_cluster.PlaneWidth < max_len * max_len){
            if (cur_cluster.N <= cuda_frame_param->ncheck_t){
                // Stage 1: Establish trend
                if (cur_cluster.distDepth(point) < z_dist_thresh &&
                    cur_cluster.distLine(point) < x_dist_thresh){
                    merged = true;
                    addPointObsCuda(point, color, row_idx, col_idx,cur_depth_idx, cur_cluster, cuda_frame_param);
                }
            } else {
                // Stage 2: See if trend continues
            #ifdef __CUDA_ARCH__
                float z_dist = fabsf(cur_cluster.distDepthEst(point) - point.z);
            #else
                float z_dist = std::abs((float) (cur_cluster.distDepthEst(point) - point.z));
            #endif
                if (z_dist < z_dist_thresh){
                    merged = true;
                    addPointObsCuda(point, color, row_idx, col_idx,cur_depth_idx, cur_cluster, cuda_frame_param);
                }
            }
        }
    }

    // Instantiate a new segment if the point is not merged into any existing clusters
    if (!merged){
        //cuGMM::GMMmetadata_c segment(this->mapParameters.track_color);
        cuGMM::GMMmetadata_c segment;
        addPointObsCuda(point, color, row_idx, col_idx, cur_depth_idx,segment, cuda_frame_param);
        // Remove if full (before add)
        if (obs_imcomplete_queue.full()){
            if (obs_imcomplete_queue.back().N > cuda_frame_param->sparse_t){
                cur_obs_line_segments->push_back(thread_idx, obs_imcomplete_queue.pop_back());
            } else {
                obs_imcomplete_queue.pop_back();
            }
        }
        obs_imcomplete_queue.push_front(segment); // Push new segments to the front of the queue
    }
}

__host__ __device__ void addPointObsCuda(const cuGMM::V& point, const cuGMM::V& color, int v, int u, int depth_idx,
                                         cuGMM::GMMmetadata_c& metadata, const frame_param_cuda* cuda_frame_param) {
    // Update free space: Compute the parameters of the Gaussian for each line segment in closed form!
    // Note that this function dominates the computation time in SPGF!
    float p_norm = cuEigen::norm(point);
    cuGMM::M ppT = cuEigen::dotRowVec(point, point);
    cuGMM::V S_free = point * (0.5f * p_norm);
    cuGMM::M J_free = ppT * (p_norm / 3.0f);

    if (cuda_frame_param->update_free_gmm){
        if (metadata.N == 0){
            metadata.freeBasis.depth_idx = depth_idx; // This is the index to the adaptive depth array
        } else {
        #ifdef __CUDA_ARCH__
            metadata.freeBasis.depth_idx = min(metadata.freeBasis.depth_idx, depth_idx); // This is the index to the adaptive depth array
        #else
            metadata.freeBasis.depth_idx = std::min(metadata.freeBasis.depth_idx, depth_idx); // This is the index to the adaptive depth array
        #endif
        }

        // The front accumulates all segments from zero
        metadata.freeBasis.W_ray_ends = metadata.freeBasis.W_ray_ends + p_norm;
        metadata.freeBasis.S_ray_ends = metadata.freeBasis.S_ray_ends + S_free;
        metadata.freeBasis.J_ray_ends = metadata.freeBasis.J_ray_ends + J_free;

        // The back accumulates the bases used to infer all other free space clusters
        metadata.freeBasis.W_basis = metadata.freeBasis.W_basis + p_norm / point.z;
        metadata.freeBasis.S_basis = metadata.freeBasis.S_basis + S_free / (point.z * point.z);
        metadata.freeBasis.J_basis = metadata.freeBasis.J_basis + J_free / (point.z * point.z * point.z); // Note: pow(x, 3) is very slow!
    }

    // Update obstacles
    metadata.RightPoint = point;

    metadata.N = metadata.N + 1;
    metadata.W = metadata.W + p_norm;
    if (cuda_frame_param->track_color){
        //V_c point_c;
        //point_c << point, color;
        //*metadata.S_c_eff += color;
        //*metadata.J_c_eff += (point_c * point_c.transpose()).bottomLeftCorner<3,6>();
    }

    metadata.S = metadata.S + point;
    metadata.J = metadata.J + ppT;

    if (metadata.N == 1) {
        metadata.LeftPoint = point;
        metadata.LeftPixel = u;
    } else {
        metadata.LineVec = metadata.LineVec + point - metadata.LeftPoint;
    }

    metadata.RightPixel = u;
    metadata.RightPoint = point;
    metadata.PlaneWidth = cuEigen::normSquared(metadata.LeftPoint-point);
}

__host__ __device__ void forwardProjectScanline(const float* depth, float* x, float* y, int row_idx, int thread_idx, const frame_param_cuda* cuda_frame_param){
    if (thread_idx < cuda_frame_param->img_width){
        // Assumes row-major storage here!
        //for (int i = 0; i < 4; i++){
            cuda_frame_param->forwardProject(row_idx, thread_idx, depth[row_idx * cuda_frame_param->img_width + thread_idx], x[thread_idx], y[thread_idx]);
        //}
    }
}

void transferMapParamToDevice(frame_param_cuda* cuda_frame_param_host) {
    // 1) Allocate device memory for internal array
    float* depth_dists_device;
    float* depth_dists_cum_device;
    gpuErrchk(cudaMalloc( (void **)&depth_dists_device, cuda_frame_param_host->depth_dist_array_size * sizeof(float) ));
    gpuErrchk(cudaMalloc( (void **)&depth_dists_cum_device, cuda_frame_param_host->depth_dist_array_size * sizeof(float) ));
    gpuErrchk(cudaMemcpy( depth_dists_device, cuda_frame_param_host->depth_dists,
                          cuda_frame_param_host->depth_dist_array_size * sizeof(float), cudaMemcpyHostToDevice ));
    gpuErrchk(cudaMemcpy( depth_dists_cum_device, cuda_frame_param_host->depth_dists_cum,
                          cuda_frame_param_host->depth_dist_array_size * sizeof(float), cudaMemcpyHostToDevice ));

    // 2) Transfer to constant memory
    frame_param_cuda* cuda_frame_param_device_addr;
    gpuErrchk(cudaMemcpyToSymbol(cuda_frame_param_const_mem, cuda_frame_param_host,
                                 sizeof(frame_param_cuda), 0, cudaMemcpyHostToDevice));
    // From https://stackoverflow.com/questions/15984913/cudamemcpytosymbol-vs-cudamemcpy-why-is-it-still-around-cudamemcpytosymbol
    // cudaMemcpyToSymbol = cudaGetSymbolAddress + cudaMemcpy
    gpuErrchk(cudaGetSymbolAddress((void **) &cuda_frame_param_device_addr, cuda_frame_param_const_mem));
    gpuErrchk(cudaMemcpy(&(cuda_frame_param_device_addr->depth_dists), &depth_dists_device, sizeof(float *), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(cuda_frame_param_device_addr->depth_dists_cum), &depth_dists_cum_device, sizeof(float *), cudaMemcpyHostToDevice));
    // Synchronize to ensure that the transfer is completed before function exits.
    gpuErrchk(cudaDeviceSynchronize());
}

void freeMapParamOnDevice() {
    frame_param_cuda* cuda_frame_param_device_addr;
    float* depth_dists_device;
    float* depth_dists_cum_device;
    gpuErrchk(cudaGetSymbolAddress((void **) &cuda_frame_param_device_addr, cuda_frame_param_const_mem));
    gpuErrchk(cudaMemcpy( &depth_dists_device, &(cuda_frame_param_device_addr->depth_dists),sizeof(float *), cudaMemcpyDeviceToHost ));
    gpuErrchk(cudaMemcpy( &depth_dists_cum_device,&(cuda_frame_param_device_addr->depth_dists_cum), sizeof(float *), cudaMemcpyDeviceToHost ));
    gpuErrchk(cudaFree(depth_dists_device));
    gpuErrchk(cudaFree(depth_dists_cum_device));
}

Buffer2D<cuGMM::GMMmetadata_c>* allocateScanlineSegmentsUnified(int total_threads, int max_segments_per_thread,
                                                                cudaStreamBuffer* buffer, int cpu_thread_idx){
    // Important: read documentation to find the rationale behind allocating unified memory:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
    Buffer2D<cuGMM::GMMmetadata_c>* segments;
    // Without this cudaMemAttachHost flag, a new allocation would be considered in-use on the GPU if a kernel launched by another thread happens to be running.
    // This might impact the thread’s ability to access the newly allocated data from the CPU
    gpuErrchk(cudaMallocManaged(&segments, sizeof(Buffer2D<cuGMM::GMMmetadata_c>), cudaMemAttachHost));
    gpuErrchk(cudaMallocManaged(&segments->buffer, total_threads * max_segments_per_thread * sizeof(cuGMM::GMMmetadata_c), cudaMemAttachHost));
    gpuErrchk(cudaMallocManaged(&segments->index, total_threads * sizeof(int), cudaMemAttachHost));
    segments->max_size_per_consumer = max_segments_per_thread;
    segments->num_consumers = total_threads;
    // Attach this unified memory to a specific stream so that it is not locked to all streams!
    gpuErrchk(cudaStreamAttachMemAsync(buffer->at(cpu_thread_idx), segments));
    gpuErrchk(cudaStreamAttachMemAsync(buffer->at(cpu_thread_idx), segments->buffer));
    gpuErrchk(cudaStreamAttachMemAsync(buffer->at(cpu_thread_idx), segments->index));
    // We must synchronize stream here to ensure that the attachment is completed!
    buffer->synchronizeStream(cpu_thread_idx);
    std::cout << fmt::format("Index: {}, Allocated unified memory buffer of {:.2}MB\n",
                             cpu_thread_idx, total_threads * max_segments_per_thread * sizeof(cuGMM::GMMmetadata_c) / 1024.0 / 1024.0) << std::endl;
    return segments;
}

void freeScanlineSegmentsUnified(Buffer2D<cuGMM::GMMmetadata_c>* segments){
    gpuErrchk(cudaFree(segments->buffer));
    gpuErrchk(cudaFree(segments->index));
    gpuErrchk(cudaFree(segments));
}

float* allocateScanlineGPUMemory(int total_threads, int img_width){
    float* scanlines;
    gpuErrchk(cudaMalloc( (void **)&scanlines, total_threads * img_width * sizeof(float)));
    return scanlines;
}

void transferScanlineToGPUMemory(float* scanlines, const float* depthmap_offset, int num_elements){
    gpuErrchk(cudaMemcpy( scanlines, depthmap_offset, num_elements * sizeof(float), cudaMemcpyHostToDevice ));
    gpuErrchk(cudaDeviceSynchronize());
}

void freeScanlineGPUMemory(float* scanlines){
    gpuErrchk(cudaFree(scanlines));
}

float* allocateImageGPUMemory(int img_width, int img_height){
    float* img_gpu;
    gpuErrchk(cudaMalloc( (void **)&img_gpu, img_width * img_height * sizeof(float)));
    return img_gpu;
}

void transferImageToGPUMemory(float* img_gpu, const float* img_cpu, int img_width, int img_height){
    gpuErrchk(cudaMemcpy( img_gpu, img_cpu, img_width * img_height * sizeof(float), cudaMemcpyHostToDevice ));
    // Need to synchronize because SPGF might not be launched on the default stream!
    gpuErrchk(cudaDeviceSynchronize());
}

void freeImageGPUMemory(float* img_gpu){
    gpuErrchk(cudaFree(img_gpu));
}

void synchronizeDefaultStream(){
    gpuErrchk(cudaDeviceSynchronize());
}

void lineSegmentationExtendedWrapper(const float * depthmap, int row_idx_offset,
                                     Buffer2D<cuGMM::GMMmetadata_c>* cur_obs_line_segments,
                                     cudaStreamBuffer* buffer, int cpu_thread_idx,
                                     const gmm::map_cuda_param* cuda_config_param){
    using namespace std::chrono;
    //auto start = steady_clock::now();
    // Prefetch data to GPU
    /*
    gpuErrchk(cudaMemPrefetchAsync(cur_obs_line_segments, sizeof(Buffer2D<cuGMM::GMMmetadata_c>), 0, strm));
    gpuErrchk(cudaMemPrefetchAsync(cur_obs_line_segments->buffer,
                         cuda_config_param->totalThreads() * cuda_config_param->max_segments_per_line * sizeof(cuGMM::GMMmetadata_c),
                         0, strm));
    gpuErrchk(cudaMemPrefetchAsync(cur_obs_line_segments->index, cuda_config_param->totalThreads() * sizeof(int), 0, strm));
     */
    // Launch CUDA kernel!
    cudaFuncSetCacheConfig(lineSegmentationExtendedBlock, cudaFuncCachePreferL1);
    lineSegmentationExtendedBlock<<<cuda_config_param->num_blocks,
    cuda_config_param->threads_per_block, 0, buffer->at(cpu_thread_idx)>>>
            (depthmap, row_idx_offset, cur_obs_line_segments);
    // Prefetch data to CPU
    /*
    gpuErrchk(cudaMemPrefetchAsync(cur_obs_line_segments, sizeof(Buffer2D<cuGMM::GMMmetadata_c>), cudaCpuDeviceId, strm));
    gpuErrchk(cudaMemPrefetchAsync(cur_obs_line_segments->buffer,
                                   cuda_config_param->totalThreads() * cuda_config_param->max_segments_per_line * sizeof(cuGMM::GMMmetadata_c),
                                   cudaCpuDeviceId, strm));
    gpuErrchk(cudaMemPrefetchAsync(cur_obs_line_segments->index, cuda_config_param->totalThreads() * sizeof(int), cudaCpuDeviceId, strm));
     */
    // Let the CPU wait for the said stream so that the unified memory is avaliable to access by CPU.
    buffer->synchronizeStream(cpu_thread_idx);
    //auto stop = steady_clock::now();
    //int num_threads = std::min(cuda_config_param->totalThreads(), 480 - row_idx_offset);
    //printf("%d scanlines are processed in %dus. Average: %.2fus / line\n", num_threads,
    //       (int) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count(),
    //       (float) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (float) num_threads);
}

void lineSegmentationExtendedOptWrapper(const float * depthmap, int row_offset,
                                        int num_streams,
                                        cudaStreamBuffer* buffer,
                                        int img_width,
                                        float * x, float * y,
                                        const gmm::map_cuda_param* cuda_config_param){
    //float *x, *y;
    //gpuErrchk(cudaMalloc(&x, img_width * num_streams * sizeof(float)));
    //gpuErrchk(cudaMalloc(&y, img_width * num_streams * sizeof(float)));
    //gpuErrchk(cudaDeviceSynchronize());
    for (int i = 0; i < num_streams; i++){
        lineSegmentationExtendedOpt<<<cuda_config_param->num_blocks,cuda_config_param->threads_per_block, 0, buffer->at(i)>>>
            (depthmap, &x[i*img_width], &y[i*img_width], row_offset + i);
    }
    // Synchronize at the end and destroy
    // In reality, we can actually synchronize just before the segments are needed in Segment Fusion
    for (int i = 0; i < num_streams; i++){
        buffer->synchronizeStream(i);
    }
    //gpuErrchk(cudaFree(x));
    //gpuErrchk(cudaFree(y));
}

