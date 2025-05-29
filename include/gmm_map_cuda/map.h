#ifndef GMM_CUDA_MAP_H
#define GMM_CUDA_MAP_H
#include "gmm_map/map.h"
#include "gmm_map_cuda/map.cuh"

namespace gmm {
    class GMMMapCuda : public GMMMap {
    public:
        map_cuda_param cuda_config_param;
        frame_param_cuda cuda_frame_param_host;
        bool cuda_frame_param_host_initialized;
        //frame_param_cuda* cuda_frame_param_device; // Pointer to device memory
        std::atomic<bool> *cuda_enable;

        // Pointer to CUDA allocated memory used in SPGF
        // This will support multiple current SPGF running!
        std::vector<std::vector<Buffer2D<cuGMM::GMMmetadata_c>*>> cur_obs_line_segments_blocks; // Stored intermediate segments after scanline segmentation
        std::vector<cudaStreamBuffer*> stream_blocks; // Stores streams used to launch kernels

        GMMMapCuda() = default;
        GMMMapCuda(const map_param& param, const map_cuda_param& cuda_param,
               std::atomic<bool>* update_obs, std::atomic<bool>* update_free, std::atomic<bool>* fuse_gmm_across_frames,
               std::atomic<bool> *cuda_enable);
        GMMMapCuda(GMMMapCuda&& map) noexcept; // Move constructor
        GMMMapCuda(const GMMMapCuda& map); // Copy constructor
        friend void swap( GMMMapCuda& first, GMMMapCuda& second); // Swap function
        GMMMapCuda& operator=(GMMMapCuda map); // Assignment operator
        ~GMMMapCuda();
        bool isCUDAEnabled();
        static std::string getGPUInputBufferName();
        static std::string getGPUOutputBufferName();

        // Internally, this function can call a CUDA version
        void insertFrameCuda(const Eigen::MatrixXf& depthmap, const Isometry3& pose);
        std::list<GMMmetadata_c> extendedSPGFCudaCPUTest(const float* depthmap);
        std::list<GMMmetadata_c> extendedSPGFCudaCPUGPUTest(const float* depthmap_host, const float* depthmap_device);
        std::list<GMMmetadata_c> extendedSPGFCudaGPU(const float* depthmap, int cpu_thread_idx = 0);

        void clusterMergeExtendedCuda(std::list<cuGMM::GMMmetadata_c>& obs_completed_clusters,
                                     std::list<cuGMM::GMMmetadata_c>& obs_incomplete_clusters,
                                     Buffer2D<cuGMM::GMMmetadata_c>& cur_obs_line_segments,
                                     int& obs_numInactiveClusters,
                                     int row_idx,
                                     int thread_idx,
                                     FP& algorithm_size,
                                     bool final_scanline, bool measure_memory = false);

        std::list<GMMmetadata_c> convert2EigenMetadata(std::list<cuGMM::GMMmetadata_c>& clusters) const;
        GMMmetadata_c convert2EigenMetadata(cuGMM::GMMmetadata_c& cluster) const;
        void mergeMetadataObs(cuGMM::GMMmetadata_c& source, cuGMM::GMMmetadata_c& destination) const;
        void updateCudaFrameParam();
        void allocatedSPGFCudaMemory();
        void freeSPGFCudaMemory();
        bool onPlane(const cuGMM::V& point, const cuGMM::GMMmetadata_c& metadata);
        void printCudaGMMMetadata_c(const std::list<cuGMM::GMMmetadata_c> &metadata) const;
        void printCudaGMMMetadata_o(const std::list<cuGMM::GMMmetadata_o> &metadata) const;
    };

    V convert2Eigen(const cuEigen::Vector3f& vec);
    M convert2Eigen(const cuEigen::Matrix3f& mat);
}
#endif
