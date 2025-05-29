#include "gmm_map_cuda/map.h"
#include <chrono>
#include <atomic>
#ifdef TRACK_MEM_USAGE_GMM
#include "mem_utils/mem_utils.h"
#endif


namespace gmm {
    GMMMapCuda::GMMMapCuda(const map_param &param, const map_cuda_param& cuda_param,
                           std::atomic<bool> *update_obs,
                           std::atomic<bool> *update_free,
                           std::atomic<bool> *fuse_gmm_across_frames,
                           std::atomic<bool> *cuda_enable) :
                            GMMMap(param, update_obs, update_free, fuse_gmm_across_frames), cuda_config_param(cuda_param) {
        this->cuda_enable = cuda_enable;
        //this->cuda_frame_param_device = nullptr;
        if (cuda_enable){
            std::cout << "CUDA Enabled!" << std::endl;
            this->mapParameters.track_color = false; // Not supported for now
        } else {
            std::cout << "CPU Enabled!" << std::endl;
        }
        this->cuda_frame_param_host_initialized = false;
        this->updateCudaFrameParam();
        this->allocatedSPGFCudaMemory();
    }

    GMMMapCuda::~GMMMapCuda(){
        // Clear device memory used to store mapping parameters
        if (cuda_frame_param_host_initialized){
            cuda_frame_param_host.deallocate_depth_array();
            freeMapParamOnDevice();
        }
        freeSPGFCudaMemory();
    }

    GMMMapCuda::GMMMapCuda(const GMMMapCuda& map) : GMMMap(map){
        cuda_config_param = map.cuda_config_param;
        cuda_enable = map.cuda_enable;
        cuda_frame_param_host_initialized = false;
        updateCudaFrameParam();
        allocatedSPGFCudaMemory();
    }

    void swap( GMMMapCuda& first, GMMMapCuda& second){
        using std::swap;
        swap(static_cast<GMMMap&>(first), static_cast<GMMMap&>(second));
        swap(first.cuda_config_param, second.cuda_config_param);
        swap(first.cuda_frame_param_host, second.cuda_frame_param_host);
        swap(first.cuda_frame_param_host_initialized, second.cuda_frame_param_host_initialized);
        swap(first.cuda_enable, second.cuda_enable);
        swap(first.cur_obs_line_segments_blocks, second.cur_obs_line_segments_blocks);
        swap(first.stream_blocks, second.stream_blocks);
    }

    // Assignment operator
    GMMMapCuda& GMMMapCuda::operator=(GMMMapCuda map) {
        swap(*this, map);
        return *this;
    }

    GMMMapCuda::GMMMapCuda(GMMMapCuda&& map) noexcept {
        swap(*this, map);
    }

    bool GMMMapCuda::isCUDAEnabled() {
        return true;
    }

    std::string GMMMapCuda::getGPUInputBufferName() {
        static std::string GPUInputBufferName = "GPUInputImageBuffer";
        return GPUInputBufferName;
    }

    std::string GMMMapCuda::getGPUOutputBufferName() {
        static std::string GPUOutputBufferName = "GPUOutputResultBuffer";
        return GPUOutputBufferName;
    }

    void GMMMapCuda::updateCudaFrameParam() {
        // Transfer parameters from regular frames into cuda compatible frame data structure
        // Define some constants within this program. These constants will be replaced by input args!
        frame_param& cur_frame_param = mapParameters.gmm_frame_param;
        cuda_frame_param_host.cx = mapParameters.gmm_frame_param.cx;
        cuda_frame_param_host.cy = mapParameters.gmm_frame_param.cy;
        cuda_frame_param_host.fx = mapParameters.gmm_frame_param.fx;
        cuda_frame_param_host.fy = mapParameters.gmm_frame_param.fy;
        cuda_frame_param_host.img_width = mapParameters.gmm_frame_param.img_width;
        cuda_frame_param_host.img_height = mapParameters.gmm_frame_param.img_height;
        cuda_frame_param_host.f = cur_frame_param.f;
        cuda_frame_param_host.num_threads = cur_frame_param.num_threads;
        cuda_frame_param_host.measure_memory = cur_frame_param.measure_memory;
        cuda_frame_param_host.occ_x_t = cur_frame_param.occ_x_t;
        cuda_frame_param_host.noise_thresh = cur_frame_param.noise_thresh;
        cuda_frame_param_host.sparse_t = cur_frame_param.sparse_t;
        cuda_frame_param_host.ncheck_t = cur_frame_param.ncheck_t;
        cuda_frame_param_host.gau_bd_scale = cur_frame_param.gau_bd_scale;
        cuda_frame_param_host.adaptive_thresh_scale = cur_frame_param.adaptive_thresh_scale;
        //cuda_frame_param_host.max_len = cur_frame_param.max_len;
        cuda_frame_param_host.max_depth = cur_frame_param.max_depth;
        cuda_frame_param_host.line_t = cur_frame_param.line_t;
        cuda_frame_param_host.angle_t = cur_frame_param.angle_t;
        cuda_frame_param_host.noise_floor = cur_frame_param.noise_floor;
        cuda_frame_param_host.num_line_t = cur_frame_param.num_line_t;
        cuda_frame_param_host.num_pixels_t = cur_frame_param.num_pixels_t;
        //cuda_frame_param_host.max_incomplete_clusters = cur_frame_param.max_incomplete_clusters;
        cuda_frame_param_host.free_space_start_len = cur_frame_param.free_space_start_len;
        cuda_frame_param_host.free_space_max_length = cur_frame_param.free_space_max_length;
        cuda_frame_param_host.free_space_dist_scale = cur_frame_param.free_space_dist_scale;
        cuda_frame_param_host.debug_row_idx = cur_frame_param.debug_row_idx;
        cuda_frame_param_host.update_free_gmm = *update_free_gmm;
        cuda_frame_param_host.track_color = mapParameters.track_color;
        cuda_frame_param_host.preserve_details_far_objects = mapParameters.gmm_frame_param.preserve_details_far_objects;

        if (cuda_frame_param_host_initialized){
            cuda_frame_param_host.deallocate_depth_array();
            std::cout << "Freeing map parameters on CUDA device" << std::endl;
            freeMapParamOnDevice();
        }
        cuda_frame_param_host.allocate_depth_array(cur_frame_param.depth_dists.size());
        for (int i = 0; i < cur_frame_param.depth_dists.size(); i++){
            cuda_frame_param_host.depth_dists[i] = cur_frame_param.depth_dists.at(i);
            cuda_frame_param_host.depth_dists_cum[i] = cur_frame_param.depth_dists_cum.at(i);
        }
        cuda_frame_param_host_initialized = true;
        std::cout << "Transferring map parameters to CUDA device" << std::endl;
        transferMapParamToDevice(&cuda_frame_param_host);
        std::cout << "Completed transferring map parameters to CUDA device" << std::endl;
    }

    void GMMMapCuda::allocatedSPGFCudaMemory() {
        using namespace std::chrono;
        auto mem_alloc_start = steady_clock::now();
        int max_current_streams_in_spgf = cuda_config_param.maxConcurrentScanlineSegmentation(cuda_frame_param_host.num_threads,
                                                                                              cuda_frame_param_host.img_height);
        printf("Allocating %d copies of CUDA memory that supports %d streams within each SPGF\n",
               cuda_frame_param_host.num_threads, max_current_streams_in_spgf);
        cur_obs_line_segments_blocks.reserve(cuda_frame_param_host.num_threads);
        stream_blocks.reserve(cuda_frame_param_host.num_threads);
        for (int tid = 0; tid < cuda_frame_param_host.num_threads; tid++){
            cur_obs_line_segments_blocks.emplace_back();
            stream_blocks.emplace_back();

            stream_blocks.back() = new cudaStreamBuffer(max_current_streams_in_spgf);
            cur_obs_line_segments_blocks.back().reserve(max_current_streams_in_spgf);
            for (int i = 0; i < max_current_streams_in_spgf; i++){
                // Perform clustering in blocks!
                cur_obs_line_segments_blocks.back().push_back(allocateScanlineSegmentsUnified(cuda_config_param.totalGPUThreads(),
                                                                                      cuda_config_param.max_segments_per_line,
                                                                                              stream_blocks.back(),
                                                                                      i));
#ifdef TRACK_MEM_USAGE_GMM
                auto& mem_tracker = mutil::memTracker::instance();
                // Obtain buffer size from allocateScanlineSegmentsUnified
                unsigned long long buffer_size = cuda_config_param.totalGPUThreads() * cuda_config_param.max_segments_per_line * sizeof(cuGMM::GMMmetadata_c) +
                        cuda_config_param.totalGPUThreads() * sizeof(int) + sizeof(Buffer2D<cuGMM::GMMmetadata_c>);
                mem_tracker.addMemUsage(getGPUOutputBufferName(),  buffer_size, 1);
#endif
            }
        }
        auto mem_alloc_stop = steady_clock::now();
        printf("CUDA intermediate memory allocated in %dus\n",
               (int) std::chrono::duration_cast<std::chrono::microseconds>(mem_alloc_stop - mem_alloc_start).count());
    }

    void GMMMapCuda::freeSPGFCudaMemory() {
        // We free the memory used for the result buffer
        for (auto& cur_obs_line_segments_block : cur_obs_line_segments_blocks){
            for (auto& unified_mem: cur_obs_line_segments_block){
                freeScanlineSegmentsUnified(unified_mem);
            }
        }

#ifdef TRACK_MEM_USAGE_GMM
        auto& mem_tracker = mutil::memTracker::instance();
        mem_tracker.clearMemUsage(getGPUOutputBufferName());
#endif

        for (auto& stream_block : stream_blocks){
            stream_block->destroyAllStreams();
            delete stream_block;
            stream_block = nullptr;
        }
    }

    void GMMMapCuda::insertFrameCuda(const Eigen::MatrixXf& depthmap, const Isometry3& pose){
        std::list<GMMcluster_o*> new_free_clusters, new_obs_clusters;
        std::list<GMMmetadata_c> new_obs_cluster_metadata;
        std::list<GMMmetadata_o> new_free_cluster_metadata;
        // Insert current frame into the map
        auto clustering_start = std::chrono::steady_clock::now();
        float * depthmap_gpu = nullptr;
        if (*cuda_enable){
            depthmap_gpu = allocateImageGPUMemory(cuda_frame_param_host.img_width, cuda_frame_param_host.img_height);
            transferImageToGPUMemory(depthmap_gpu, depthmap.data(), cuda_frame_param_host.img_width, cuda_frame_param_host.img_height);
            new_obs_cluster_metadata = extendedSPGFCudaGPU(depthmap_gpu);
        } else {
            new_obs_cluster_metadata = extendedSPGFCudaCPUTest(depthmap.data());
        }

        // Construct free space from obstacle metadata
        if (*update_free_gmm){
            new_free_cluster_metadata = constructFreeClustersFromObsClusters(new_obs_cluster_metadata);
        }
        // Construct final clusters and print statistics
        this->transferMetadata2ClusterExtended(new_obs_cluster_metadata, new_free_cluster_metadata,
                                               new_obs_clusters, new_free_clusters, mapParameters.gmm_frame_param);
        auto clustering_stop = std::chrono::steady_clock::now();
        // Print Statistics
        std::cout << fmt::format("Clusters - Obstacle: {}, Free: {}",
                                 new_obs_clusters.size(), new_free_clusters.size()) << std::endl;
        long clustering_latency = std::chrono::duration_cast<std::chrono::microseconds>(clustering_stop - clustering_start).count();

        insertGMMsIntoCurrentMap(pose, new_free_clusters, new_obs_clusters, clustering_latency);

        if (depthmap_gpu != nullptr){
            freeImageGPUMemory(depthmap_gpu);
        }
    }

    // Conversions
    GMMmetadata_c GMMMapCuda::convert2EigenMetadata(cuGMM::GMMmetadata_c& cluster) const {
        GMMmetadata_c resulting_cluster;
        // Convert custom datatypes to Eigen
        resulting_cluster.S = convert2Eigen(cluster.S);
        resulting_cluster.J = convert2Eigen(cluster.J);
        resulting_cluster.LeftPoint = convert2Eigen(cluster.LeftPoint);
        resulting_cluster.RightPoint = convert2Eigen(cluster.RightPoint);
        resulting_cluster.NumLines = cluster.NumLines;
        resulting_cluster.N = cluster.N;
        resulting_cluster.W = cluster.W;
        resulting_cluster.LeftPixel = cluster.LeftPixel;
        resulting_cluster.RightPixel = cluster.RightPixel;
        resulting_cluster.depth = cluster.depth;
        resulting_cluster.Updated = cluster.Updated;
        resulting_cluster.near_obs = cluster.near_obs;
        resulting_cluster.cur_pt_obs = cluster.cur_pt_obs;
        resulting_cluster.fused = cluster.fused;
        resulting_cluster.LineVec = convert2Eigen(cluster.LineVec);
        resulting_cluster.PlaneVec = convert2Eigen(cluster.PlaneVec);
        resulting_cluster.UpMean = convert2Eigen(cluster.UpMean);
        resulting_cluster.PlaneLen = cluster.PlaneLen;
        resulting_cluster.PlaneWidth = cluster.PlaneWidth;
        resulting_cluster.track_color = cluster.track_color;
        resulting_cluster.BBox = Rect(convert2Eigen(cluster.BBoxLow),
                                      convert2Eigen(cluster.BBoxHigh));

        // Convert free metadata
        if (cuda_frame_param_host.update_free_gmm){
            resulting_cluster.freeBasis.S_ray_ends = convert2Eigen(cluster.freeBasis.S_ray_ends);
            resulting_cluster.freeBasis.J_ray_ends = convert2Eigen(cluster.freeBasis.J_ray_ends);
            resulting_cluster.freeBasis.S_basis = convert2Eigen(cluster.freeBasis.S_basis);
            resulting_cluster.freeBasis.J_basis = convert2Eigen(cluster.freeBasis.J_basis);
            resulting_cluster.freeBasis.W_ray_ends = cluster.freeBasis.W_ray_ends;
            resulting_cluster.freeBasis.W_basis = cluster.freeBasis.W_basis;
            resulting_cluster.freeBasis.depth_idx = cluster.freeBasis.depth_idx;
            resulting_cluster.freeBasis.BBox = Rect(convert2Eigen(cluster.freeBasis.BBoxLow),
                                                   convert2Eigen(cluster.freeBasis.BBoxHigh));
            resulting_cluster.freeBasis.cluster_cross_depth_boundary = cluster.freeBasis.cluster_cross_depth_boundary;
        }

        //if (resulting_cluster.isTrackerInitialized()){
        //    std::cout << "Object tracker is initialized before output!" << std::endl;
        //}
        return resulting_cluster;
    }

    std::list<GMMmetadata_c> GMMMapCuda::convert2EigenMetadata(std::list<cuGMM::GMMmetadata_c>& clusters) const {
        //std::cout << "Converting GPU metadata to CPU metadata" << std::endl;
        using namespace std::chrono;
        auto start = steady_clock::now();
        std::list<GMMmetadata_c> resulting_clusters;
        for (auto& cluster : clusters){
            resulting_clusters.push_back(convert2EigenMetadata(cluster));
            //std::cout << "Output successfully pushed to the list" << std::endl;
        }
        auto stop = steady_clock::now();
        //std::cout << fmt::format("Conversion duration: {}us\n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
        return resulting_clusters;
    }

    // Print function for debugging
    void GMMMapCuda::printCudaGMMMetadata_o(const std::list<cuGMM::GMMmetadata_o> &metadata) const {
        int idx = 0;
        for (const auto& data : metadata) {
            std::cout << fmt::format("Printing info for free cluster metadata {}", idx) << std::endl;
            data.printInfo();
            idx++;
        }
    }

    void GMMMapCuda::printCudaGMMMetadata_c(const std::list<cuGMM::GMMmetadata_c> &metadata) const {
        int idx = 0;
        for (const auto& data : metadata) {
            std::cout << fmt::format("Printing info for obstacle cluster metadata {}", idx) << std::endl;
            data.printInfo();
            idx++;
        }
    }

    V convert2Eigen(const cuEigen::Vector3f& vec){
        V result;
        result << vec.x, vec.y, vec.z;
        return result;
    }

    M convert2Eigen(const cuEigen::Matrix3f& mat){
        M result;
        result << mat.data[0].x, mat.data[1].x, mat.data[2].x,
                  mat.data[0].y, mat.data[1].y, mat.data[2].y,
                  mat.data[0].z, mat.data[1].z, mat.data[2].z;
        return result;
    }

}