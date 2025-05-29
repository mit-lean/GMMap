//
// Created by peter on 3/5/22.
// Support colored GMM mapping
// Support of extended-SPGF algorithm
#include "gmm_map_cuda/map.h"
#include <chrono>

namespace gmm {

    std::list<GMMmetadata_c> GMMMapCuda::extendedSPGFCudaCPUTest(const float* depthmap){
        // Optimized implementation retains the original ICRA SPGF as much as possible
        // Additional metadata is tracked to support free cluster construction, which occurs near the end!
        // We use our own custom datatypes for implementation

        using namespace std::chrono;

        // Occupied Space
        std::list<cuGMM::GMMmetadata_c> obs_completed_clusters;
        std::list<cuGMM::GMMmetadata_c> obs_incomplete_clusters;
        int obs_numInactiveClusters;
        int max_current_streams_in_spgf = cuda_config_param.maxConcurrentScanlineSegmentation(cuda_frame_param_host.num_threads,
                                                                                              cuda_frame_param_host.img_height);

        // std::cout << fmt::format("Extended Clustering starts ..") << std::endl;
        // First, we pre-allocate result buffer
        std::vector<Buffer2D<cuGMM::GMMmetadata_c>> cur_obs_line_segments_block;
        cur_obs_line_segments_block.reserve(max_current_streams_in_spgf);
        for (int i = 0; i < max_current_streams_in_spgf; i++){
            // Perform clustering in blocks!
            cur_obs_line_segments_block.emplace_back(cuda_config_param.totalGPUThreads(),
                                                     cuda_config_param.max_segments_per_line);
        }

        #pragma omp parallel for ordered schedule(static, 1) default(shared) num_threads(max_current_streams_in_spgf)
        for (int row_idx_offset = 0; row_idx_offset < cuda_frame_param_host.img_height; row_idx_offset = row_idx_offset + cuda_config_param.totalGPUThreads()){
            //if (row_idx == 200){
            //    break;
            //}
            // Check number of threads
            /*
            if (row_idx == 0)
                std::cout << fmt::format("Currently running {} concurrent threads!", omp_get_num_threads()) << std::endl;
            */
            int tid = omp_get_thread_num();
            auto start = steady_clock::now();
            FP clusterMerge_mem_size = 0;
            // The following loop will be CUDA accelerated!
            for (int thread_idx = 0; thread_idx < cuda_config_param.totalGPUThreads(); thread_idx++){
                // The V2 version optimizes the process of constructing free and obstacle GMMs
                int row_idx = row_idx_offset + thread_idx;
                if (row_idx < cuda_frame_param_host.img_height) {
                    if (mapParameters.track_color) {
                        lineSegmentationExtendedCuda(depthmap, row_idx, thread_idx,&cur_obs_line_segments_block.at(tid), &cuda_frame_param_host);
                    } else {
                        lineSegmentationExtendedCuda(depthmap, row_idx, thread_idx,&cur_obs_line_segments_block.at(tid), &cuda_frame_param_host);
                    }
                }
                //std::cout << fmt::format("Processed scanline {} with {} segments", row_idx, cur_obs_line_segments_block.size(thread_idx)) << std::endl;
            }
            auto stop = steady_clock::now();
            // std::cout << fmt::format("Scanline segmentation duration: {}us\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
            // Print generated metadata for each scanline
            // std::cout << fmt::format("Processed scanline {} with {} segments", row_idx, cur_obs_line_segments.size()) << std::endl;
            // printCudaGMMMetadata_c(cur_obs_line_segments);

            #pragma omp ordered
            {
                for (int thread_idx = 0; thread_idx < cuda_config_param.totalGPUThreads(); thread_idx++) {
                    int row_idx = row_idx_offset + thread_idx;
                    if (row_idx < cuda_frame_param_host.img_height) {
                        if (row_idx == 0) {
                            for (int i = 0; i < cur_obs_line_segments_block.at(tid).size(0); i++) {
                                obs_incomplete_clusters.push_back(cur_obs_line_segments_block.at(tid)(0, i));
                            }
                            obs_numInactiveClusters = 0;
                        } else {
                            start = steady_clock::now();
                            this->clusterMergeExtendedCuda(obs_completed_clusters,
                                                           obs_incomplete_clusters,
                                                           cur_obs_line_segments_block.at(tid),
                                                           obs_numInactiveClusters,
                                                           row_idx,
                                                           thread_idx,
                                                           clusterMerge_mem_size,
                                                           (row_idx == cuda_frame_param_host.img_height - 1));
                            stop = steady_clock::now();
                            //std::cout << fmt::format("Segment fusion duration: {}us\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
                        }
                    } else {
                        break;
                    }
                    //std::cout << fmt::format("Completed obstacle cluster metadata (during scanline segmentation {}): {} obs_completed_clusters, {} obs_incomplete_clusters",
                    //                         row_idx, obs_completed_clusters.size(), obs_incomplete_clusters.size()) << std::endl;
                }
                cur_obs_line_segments_block.at(tid).clearAll();
            }
            // Print debug information
            // std::cout << fmt::format("Completed obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            // printCudaGMMMetadata_c(obs_completed_clusters);

            //std::cout << fmt::format("Incomplete obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            //printGMMMetadata_c(obs_incomplete_clusters);
        }

        // We free the memory used for the result buffer
        for (int i = 0; i < max_current_streams_in_spgf; i++){
            cur_obs_line_segments_block.at(i).freeBuffer();
        }
        return convert2EigenMetadata(obs_completed_clusters);
    }

    std::list<GMMmetadata_c> GMMMapCuda::extendedSPGFCudaCPUGPUTest(const float* depthmap_host, const float* depthmap_device){
        // Optimized implementation retains the original ICRA SPGF as much as possible
        // Additional metadata is tracked to support free cluster construction, which occurs near the end!
        // We use our own custom datatypes for implementation

        using namespace std::chrono;

        // Occupied Space
        std::list<cuGMM::GMMmetadata_c> obs_completed_clusters;
        std::list<cuGMM::GMMmetadata_c> obs_incomplete_clusters;
        int obs_numInactiveClusters;

        // std::cout << fmt::format("Extended Clustering starts ..") << std::endl;
        // First, we pre-allocate result buffer
        std::vector<Buffer2D<cuGMM::GMMmetadata_c>> cur_obs_line_segments_block;
        cudaStreamBuffer stream_block(cuda_config_param.num_streams);
        cur_obs_line_segments_block.reserve(1);
        for (int i = 0; i < 1; i++){
            // Perform clustering in blocks!
            cur_obs_line_segments_block.emplace_back(cuda_config_param.num_streams,
                                                     cuda_config_param.max_segments_per_line);
        }

        float* scanline_x = allocateScanlineGPUMemory(cuda_config_param.num_streams, cuda_frame_param_host.img_width);
        float* scanline_y = allocateScanlineGPUMemory(cuda_config_param.num_streams, cuda_frame_param_host.img_width);
        synchronizeDefaultStream();
        long cpu_duration = 0;
        long gpu_duration = 0;

        for (int row_idx_offset = 0; row_idx_offset < cuda_frame_param_host.img_height; row_idx_offset = row_idx_offset + cuda_config_param.num_streams){
            //if (row_idx == 200){
            //    break;
            //}
            // Check number of threads
            /*
            if (row_idx == 0)
                std::cout << fmt::format("Currently running {} concurrent threads!", omp_get_num_threads()) << std::endl;
            */
            int tid = 0;
            FP clusterMerge_mem_size = 0;
            int num_streams = std::min(cuda_config_param.num_streams, cuda_frame_param_host.img_height - row_idx_offset);
            // The following loop will be CUDA accelerated!
            auto start = steady_clock::now();
            #pragma omp parallel for default(shared) num_threads(cuda_frame_param_host.num_threads)
            for (int thread_idx = 0; thread_idx < num_streams; thread_idx++){
                // The V2 version optimizes the process of constructing free and obstacle GMMs
                int row_idx = row_idx_offset + thread_idx;
                if (mapParameters.track_color) {
                    lineSegmentationExtendedCuda(depthmap_host, row_idx, thread_idx,&cur_obs_line_segments_block.at(tid), &cuda_frame_param_host);
                } else {
                    lineSegmentationExtendedCuda(depthmap_host, row_idx, thread_idx,&cur_obs_line_segments_block.at(tid), &cuda_frame_param_host);
                }
                //std::cout << fmt::format("Processed scanline {} with {} segments", row_idx, cur_obs_line_segments_block.size(thread_idx)) << std::endl;
            }
            auto stop = steady_clock::now();
            std::cout << fmt::format("CPU -- Row offset {}: Scanline segmentation duration: {}us, {}us/scanline\n",
                                     row_idx_offset, std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count(),
                                     (float) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (float) num_streams);
            cpu_duration += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

            start = steady_clock::now();
            lineSegmentationExtendedOptWrapper(depthmap_device, row_idx_offset,
                                               num_streams,
                                               &stream_block,
                                               cuda_frame_param_host.img_width,
                                               scanline_x, scanline_y,
                                               &cuda_config_param);
            stop = steady_clock::now();
            std::cout << fmt::format("GPU -- Row offset {}: Scanline segmentation duration: {}us, {}us/scanline\n",
                                     row_idx_offset, std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count(),
                                     (float) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (float) num_streams);
            gpu_duration += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

            // Print generated metadata for each scanline
            // std::cout << fmt::format("Processed scanline {} with {} segments", row_idx, cur_obs_line_segments.size()) << std::endl;
            // printCudaGMMMetadata_c(cur_obs_line_segments);

            for (int thread_idx = 0; thread_idx < cuda_config_param.num_streams; thread_idx++) {
                int row_idx = row_idx_offset + thread_idx;
                if (row_idx < cuda_frame_param_host.img_height) {
                    if (row_idx == 0) {
                        for (int i = 0; i < cur_obs_line_segments_block.at(tid).size(0); i++) {
                            obs_incomplete_clusters.push_back(cur_obs_line_segments_block.at(tid)(0, i));
                        }
                        obs_numInactiveClusters = 0;
                    } else {
                        start = steady_clock::now();
                        this->clusterMergeExtendedCuda(obs_completed_clusters,
                                                       obs_incomplete_clusters,
                                                       cur_obs_line_segments_block.at(tid),
                                                       obs_numInactiveClusters,
                                                       row_idx,
                                                       thread_idx,
                                                       clusterMerge_mem_size,
                                                       (row_idx == cuda_frame_param_host.img_height - 1));
                        stop = steady_clock::now();
                        //std::cout << fmt::format("Segment fusion duration: {}us\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
                    }
                } else {
                    break;
                }
                //std::cout << fmt::format("Completed obstacle cluster metadata (during scanline segmentation {}): {} obs_completed_clusters, {} obs_incomplete_clusters",
                //                         row_idx, obs_completed_clusters.size(), obs_incomplete_clusters.size()) << std::endl;
            }
            cur_obs_line_segments_block.at(tid).clearAll();

            // Print debug information
            // std::cout << fmt::format("Completed obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            // printCudaGMMMetadata_c(obs_completed_clusters);

            //std::cout << fmt::format("Incomplete obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            //printGMMMetadata_c(obs_incomplete_clusters);
        }

        std::cout << fmt::format("CPU duration {:.2f}ms, GPU duration {:.2f}ms, GPU/CPU {:.2f}\n",
                                 (float) cpu_duration/1000.0,
                                 (float) gpu_duration/1000.0,
                                 (float) gpu_duration / (float) cpu_duration);

        // We free the memory used for the result buffer
        for (int i = 0; i < 1; i++){
            cur_obs_line_segments_block.at(i).freeBuffer();
        }
        freeScanlineGPUMemory(scanline_x);
        freeScanlineGPUMemory(scanline_y);
        stream_block.destroyAllStreams();
        return convert2EigenMetadata(obs_completed_clusters);
    }

    std::list<GMMmetadata_c> GMMMapCuda::extendedSPGFCudaGPU(const float* depthmap, int cpu_thread_idx){
        // Optimized implementation retains the original ICRA SPGF as much as possible
        // Additional metadata is tracked to support free cluster construction, which occurs near the end!
        // We use our own custom datatypes for implementation
        // Note: Careful usage of managed memory is required to avoid segfault.
        // See: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd

        using namespace std::chrono;

        // Occupied Space
        std::list<cuGMM::GMMmetadata_c> obs_completed_clusters;
        std::list<cuGMM::GMMmetadata_c> obs_incomplete_clusters;
        int obs_numInactiveClusters;
        int max_current_streams_in_spgf = cuda_config_param.maxConcurrentScanlineSegmentation(cuda_frame_param_host.num_threads,
                                                                                              cuda_frame_param_host.img_height);
        auto& cur_obs_line_segments_block = cur_obs_line_segments_blocks.at(cpu_thread_idx);
        auto& stream_block = stream_blocks.at(cpu_thread_idx);

        /*
        // std::cout << fmt::format("Extended Clustering starts ..") << std::endl;
        // 1) We pre-allocate result buffer (Unified memory). Note that we use dedicated memory for each OpenMP threads
        auto mem_alloc_start = steady_clock::now();
        std::vector<Buffer2D<cuGMM::GMMmetadata_c>*> cur_obs_line_segments_block;
        cudaStreamBuffer stream_block(cuda_frame_param_host.num_threads);
        cur_obs_line_segments_block.reserve(cuda_frame_param_host.num_threads);
        for (int i = 0; i < cuda_frame_param_host.num_threads; i++){
            // Perform clustering in blocks!
            cur_obs_line_segments_block.push_back(allocateScanlineSegmentsUnified(cuda_config_param.totalThreads(),
                                                                                  cuda_config_param.max_segments_per_line,
                                                                                  &stream_block,
                                                                                  i));
        }
        auto mem_alloc_stop = steady_clock::now();
        printf("CUDA intermediate memory allocated in %dus\n",
               (int) std::chrono::duration_cast<std::chrono::microseconds>(mem_alloc_stop - mem_alloc_start).count());
        */

        #pragma omp parallel for ordered schedule(static, 1) default(shared) num_threads(max_current_streams_in_spgf)
        for (int row_idx_offset = 0; row_idx_offset < cuda_frame_param_host.img_height; row_idx_offset = row_idx_offset + cuda_config_param.totalGPUThreads()){
            //if (row_idx == 200){
            //    break;
            //}
            // Check number of threads
            /*
            if (row_idx == 0)
                std::cout << fmt::format("Currently running {} concurrent threads!", omp_get_num_threads()) << std::endl;
            */
            // Getting tid from openmp is not recommended!
            // Note that tid from openmp is assigned after a thread is completed.
            // Due to workload imbalance, such thread might access unified memory from another thread that is still computing!
            int tid = (row_idx_offset / cuda_config_param.totalGPUThreads()) % max_current_streams_in_spgf;
            auto start = steady_clock::now();
            FP clusterMerge_mem_size = 0;
            cur_obs_line_segments_block.at(tid)->clearAll();

            // The following loop will be CUDA accelerated!
            lineSegmentationExtendedWrapper(depthmap, row_idx_offset,
                                            cur_obs_line_segments_block.at(tid),
                                            stream_block,
                                            tid,
                                            &cuda_config_param);
            auto stop = steady_clock::now();
            int num_threads = std::min(cuda_config_param.totalGPUThreads(), cuda_frame_param_host.img_height - row_idx_offset);
            printf("%d scanlines are processed in %dus. Average: %.2fus / line\n", num_threads,
                   (int) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count(),
                   (float) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (float) num_threads);
            // std::cout << fmt::format("Scanline segmentation duration: {}us\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
            // Print generated metadata for each scanline
            // std::cout << fmt::format("Processed scanline {} with {} segments", row_idx, cur_obs_line_segments.size()) << std::endl;
            // printCudaGMMMetadata_c(cur_obs_line_segments);

            #pragma omp ordered
            {
                start = steady_clock::now();
                for (int thread_idx = 0; thread_idx < cuda_config_param.totalGPUThreads(); thread_idx++) {
                    int row_idx = row_idx_offset + thread_idx;
                    if (row_idx < cuda_frame_param_host.img_height) {
                        if (row_idx == 0) {
                            for (int i = 0; i < cur_obs_line_segments_block.at(tid)->size(0); i++) {
                                //std::cout << fmt::format("Current size of scanline 0: {}, index: {}\n", cur_obs_line_segments_block.at(tid)->size(0), i);
                                obs_incomplete_clusters.push_back((*cur_obs_line_segments_block.at(tid))(0, i));
                            }
                            obs_numInactiveClusters = 0;
                        } else {
                            this->clusterMergeExtendedCuda(obs_completed_clusters,
                                                           obs_incomplete_clusters,
                                                           *cur_obs_line_segments_block.at(tid),
                                                           obs_numInactiveClusters,
                                                           row_idx,
                                                           thread_idx,
                                                           clusterMerge_mem_size,
                                                           (row_idx == cuda_frame_param_host.img_height - 1));
                        }
                    } else {
                        break;
                    }
                    //std::cout << fmt::format("Completed obstacle cluster metadata (during scanline segmentation {}): {} obs_completed_clusters, {} obs_incomplete_clusters",
                    //                         row_idx, obs_completed_clusters.size(), obs_incomplete_clusters.size()) << std::endl;
                }
                stop = steady_clock::now();
                std::cout << fmt::format("Segment fusion duration: {}us\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
                //cur_obs_line_segments_block.at(tid)->clearAll();
            }
            // Print debug information
            // std::cout << fmt::format("Completed obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            // printCudaGMMMetadata_c(obs_completed_clusters);

            //std::cout << fmt::format("Incomplete obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            //printGMMMetadata_c(obs_incomplete_clusters);
        }

        // We free the memory used for the result buffer
        /*
        for (int i = 0; i < cuda_frame_param_host.num_threads; i++){
            freeScanlineSegmentsUnified(cur_obs_line_segments_block.at(i));
        }
        stream_block.destroyAllStreams();
        */
        return convert2EigenMetadata(obs_completed_clusters);
    }

    void GMMMapCuda::clusterMergeExtendedCuda(std::list<cuGMM::GMMmetadata_c>& obs_completed_clusters,
                                                           std::list<cuGMM::GMMmetadata_c>& obs_incomplete_clusters,
                                                           Buffer2D<cuGMM::GMMmetadata_c>& cur_obs_line_segments,
                                                           int& obs_numInactiveClusters,
                                                           int row_idx,
                                                           int thread_idx,
                                                           FP& algorithm_size,
                                                           bool final_scanline, bool measure_memory){
        // The optimized version uses adaptive depth to bound the max size of the cluster.
        std::list<cuGMM::GMMmetadata_c>::iterator op_it;
        V_2 cur_LineVec, pre_LineVec;
        auto it_l_size = obs_incomplete_clusters.size();
        int j = 0;

        for (int i = 0; i < cur_obs_line_segments.size(thread_idx); i++){
            FP overlap_max = 0;
            auto& cur_cluster = cur_obs_line_segments(thread_idx, i);
            // For each scanline segment, determine the cluster that have the greatest overlap in image space
            auto it_l = obs_incomplete_clusters.begin();
            for (auto it_l_idx = 0; it_l_idx < it_l_size; it_l_idx++) {
                FP overlap_interval = (FP) (std::min<int>(cur_cluster.RightPixel, it_l->RightPixel) - std::max<int>(cur_cluster.LeftPixel, it_l->LeftPixel));
                if (overlap_interval > 0){
                    FP total_interval = (FP) (std::max<int>(cur_cluster.RightPixel, it_l->RightPixel) - std::min<int>(cur_cluster.LeftPixel, it_l->LeftPixel));
                    FP overlap_ratio = overlap_interval/total_interval;
                    if (overlap_ratio > overlap_max){
                        overlap_max = overlap_ratio;
                        op_it = it_l;
                    }
                }
                it_l++;
            }

            // If overlap exists, check for scanline parallelism and on-plane constraints
            if (overlap_max > 0){
                FP max_len = cuda_frame_param_host.determineDepthLength(op_it->freeBasis.depth_idx);
                cur_LineVec << cur_cluster.LineVec.x, cur_cluster.LineVec.z;
                pre_LineVec << op_it->LineVec.x, op_it->LineVec.z;
                // Check merging criteria
                //if (op_it->PlaneLen < pow(clustering_params.max_len, 2) &&
                if (op_it->PlaneLen < pow(max_len, 2.0f) &&
                    pow(cur_LineVec.dot(pre_LineVec),2.0f) >
                    cur_LineVec.dot(cur_LineVec)*pre_LineVec.dot(pre_LineVec)*pow(cuda_frame_param_host.angle_t,2.0f) &&
                    onPlane(cur_cluster.S/ (float) cur_cluster.N, *op_it)){
                    // Merge!
                    mergeMetadataObs(cur_cluster, *op_it);
                    op_it->Updated = true; // Tracks whether or not the previous cluster if fully utilized
                    j++;
                } else {
                    // Not merged. Transfer element to the incompleted cluster.
                    obs_incomplete_clusters.push_back(cur_cluster);
                    j++;
                }
            } else {
                // Not merged. Transfer element to the incompleted cluster.
                obs_incomplete_clusters.push_back(cur_cluster);
                j++;
            }
        }

        // Final clean up
        if (final_scanline){
            it_l_size = obs_incomplete_clusters.size();
        }

        auto it_l = obs_incomplete_clusters.begin();
        for (auto it_l_idx = 0; it_l_idx < it_l_size; it_l_idx++){
            if (!it_l->Updated || final_scanline){
                // Pruning Gaussian here (preserve_details_far_objects = false) SIGNIFICANTLY decreases memory overhead
                if (cuda_frame_param_host.preserve_details_far_objects ||
                    it_l->N >= cuda_frame_param_host.num_pixels_t || it_l->NumLines >= cuda_frame_param_host.num_line_t){
                    // Transfer to completed cluster list
                    it_l->Updated = false;
                    it_l++;
                    obs_completed_clusters.splice(obs_completed_clusters.end(),obs_incomplete_clusters,std::prev(it_l));
                } else {
                    it_l = obs_incomplete_clusters.erase(it_l);
                    obs_numInactiveClusters++;
                }
            } else {
                it_l->Updated = false;
                it_l++;
            }
        }
    }

    bool GMMMapCuda::onPlane(const cuGMM::V& point, const cuGMM::GMMmetadata_c &metadata) {
        // Check if a point lies on the same plane as cluster described by the metadata
        // Used for obstacle cluster only!
        float dist_sq = metadata.distPlaneSquared(point);
        if (metadata.NumLines == 1){
            float line_t_adapt, depth_t_adapt;
            cuda_frame_param_host.adaptThreshold(metadata.S/ (float) metadata.N, line_t_adapt, depth_t_adapt);
            return (dist_sq <= depth_t_adapt * depth_t_adapt);
        } else {
            return dist_sq <= cuda_frame_param_host.noise_floor * cuda_frame_param_host.noise_floor;
        }
    }

    void GMMMapCuda::mergeMetadataObs(cuGMM::GMMmetadata_c& source, cuGMM::GMMmetadata_c& destination) const {

        destination.N = destination.N + source.N;
        destination.W = destination.W + source.W;
        if (cuda_frame_param_host.track_color){
            //*destination.S_c_eff += *source.S_c_eff;
            //*destination.J_c_eff += *source.J_c_eff;
        }

        destination.S = destination.S + source.S;
        destination.J = destination.J + source.J;

        // Transfer the characteristics of merged cluster
        destination.LineVec = destination.LineVec + source.LineVec; // Averaging leads to much more accurate results

        // This will get rid of a lot of noise! Keep!
        destination.LeftPixel = int((source.LeftPixel + destination.LeftPixel)/2);
        destination.RightPixel = int((source.RightPixel + destination.RightPixel)/2);

        // Update vector connecting the means
        cuGMM::V src_mean = source.S/ (float) source.N;
        if (destination.NumLines == 1){
            destination.UpMean = destination.S / (float) destination.N;
            destination.PlaneVec = (src_mean-destination.UpMean);
        } else {
            destination.PlaneVec = destination.PlaneVec + (src_mean-destination.UpMean);
        }
        destination.PlaneLen = cuEigen::normSquared(src_mean - destination.UpMean);
        destination.NumLines = destination.NumLines + source.NumLines;

        // Merge metadata associated with free space
        if (cuda_frame_param_host.update_free_gmm){
            destination.freeBasis.merge(source.freeBasis);
        }
    }
}
