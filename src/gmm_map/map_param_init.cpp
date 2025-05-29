//
// Created by peterli on 6/28/23.
//
#include "gmm_map/map_param_init.h"
#include "dataset_utils/dataset_utils.h"

namespace gmm {
    void initializeMapParameters(map_param& mapParam, const std::string& name){
        // Initialize frame parameters
        initializeFrameParameters(mapParam.gmm_frame_param, name);

        // Define dataset
        mapParam.dataset = name;

        auto cpu_param = dataset_param::dataset_info["cpu_compute_parameters"];
        auto gmm_fusion_param = dataset_param::dataset_info["gmm_fusion_parameters"];
        auto gmm_clustering_param = dataset_param::dataset_info["gmm_clustering_parameters"];
        auto occupancy_inference_param = dataset_param::dataset_info["occupancy_inference_parameters"];

        if (cpu_param.empty() || gmm_fusion_param.empty() ||
            gmm_clustering_param.empty() || occupancy_inference_param.empty()){
            throw std::invalid_argument("Unable to read parameters for GMM mapping from file. Check if the fields are correct!");
        }

        // Define some parameters within this program. Some of these parameters will be replaced by input args!
        mapParam.num_threads = cpu_param["num_threads"].asInt(); // Hyper-threading not as good as physical cores within the CPU
        mapParam.measure_memory = cpu_param["measure_memory"].asBool();
        mapParam.frame_alg_name = gmm_clustering_param["algorithm"].asString();
        mapParam.max_depth = dataset_param::max_depth;
        mapParam.depth_scale = dataset_param::scale;

        // Set mapping parameters
        mapParam.hell_thresh_squard_free = gmm_fusion_param["hell_thresh_squard_free"].asFloat();
        mapParam.hell_thresh_squard_obs_scale = gmm_fusion_param["hell_thresh_squard_obs_scale"].asFloat();
        mapParam.hell_thresh_squard_oversized_gau = gmm_fusion_param["hell_thresh_squard_oversized_gau"].asFloat();
        mapParam.min_gau_len = gmm_fusion_param["min_gau_len"].asFloat();
        mapParam.frame_max_scale = gmm_fusion_param["frame_max_scale"].asFloat();
        mapParam.fusion_max_scale = gmm_fusion_param["fusion_max_scale"].asFloat();

        mapParam.hell_thresh_squard_min = gmm_fusion_param["hell_thresh_squard_min"].asFloat();
        mapParam.gau_fusion_bd = gmm_fusion_param["gau_fusion_bd"].asFloat(); // Used for fusion
        mapParam.gau_rtree_bd = gmm_fusion_param["gau_rtree_bd"].asFloat(); // Used for inference (Greater than 2 leads to free space "leaking" beyond the walls)

        mapParam.track_color = gmm_fusion_param["track_color"].asBool();
        mapParam.track_intensity = gmm_fusion_param["track_intensity"].asBool();
        mapParam.min_num_neighbor_clusters = occupancy_inference_param["min_num_neighbor_clusters"].asInt();
        mapParam.cur_debug_frame = cpu_param["debug"].asBool();

        mapParam.max_bbox_len = mapParam.min_gau_len * mapParam.fusion_max_scale;
        mapParam.hell_thresh_squard_obs = mapParam.hell_thresh_squard_free * mapParam.hell_thresh_squard_obs_scale;
        mapParam.updateFrameParameters();
    }

    void initializeFrameParameters(frame_param& frameParam, const std::string& name){
        // Define dataset
        frameParam.dataset = name;

        auto cpu_param = dataset_param::dataset_info["cpu_compute_parameters"];
        auto scanline_seg_param = dataset_param::dataset_info["gmm_clustering_parameters"]["scanline_segmentation"];
        auto seg_fusion_param = dataset_param::dataset_info["gmm_clustering_parameters"]["segment_fusion"];
        auto cluster_purge_param = dataset_param::dataset_info["gmm_clustering_parameters"]["spurious_gaussian_purge"];
        auto free_gau_generate_param = dataset_param::dataset_info["gmm_clustering_parameters"]["free_gaussian_generation"];

        if (cpu_param.empty() || scanline_seg_param.empty() ||
            seg_fusion_param.empty() || cluster_purge_param.empty() ||
            free_gau_generate_param.empty()){
            throw std::invalid_argument("Unable to read parameters for SPGF from file. Check if the fields are correct!");
        }

        // Define some constants within this program. These constants will be replaced by input args!
        frameParam.num_threads = cpu_param["num_threads"].asInt(); // Hyper-threading not as good as physical cores within the CPU

        // Define clustering parameters for proposed incremental clustering
        frameParam.gau_bd_scale = 2.0; // Standard deviation bounds for Gaussian (To be updated from map parameters)
        frameParam.occ_x_t = scanline_seg_param["occ_x_t"].asInt(); // Maximum number of pixels in the x direction for occlusion region
        frameParam.noise_thresh = scanline_seg_param["noise_thresh"].asFloat(); // Maximum z direction threshold of the sensor
        frameParam.sparse_t = cluster_purge_param["sparse_t"].asInt(); // Minimum number of points in a cluster
        frameParam.ncheck_t = scanline_seg_param["ncheck_t"].asInt(); // Number of points after which fitting is applied
        frameParam.max_depth = dataset_param::max_depth; // Maximum range of the sensor
        frameParam.line_t = scanline_seg_param["line_t"].asFloat(); // Minimum x direction threshold for consideration
        frameParam.max_incomplete_clusters = scanline_seg_param["max_incomplete_clusters"].asInt();
        frameParam.adaptive_thresh_scale = scanline_seg_param["adaptive_thresh_scale"].asFloat();
        frameParam.noise_floor = scanline_seg_param["noise_floor"].asFloat();
        frameParam.free_space_dist_scale = free_gau_generate_param["free_space_dist_scale"].asFloat();

        frameParam.f = fminf(dataset_param::fx, dataset_param::fy);
        frameParam.fx = dataset_param::fx;
        frameParam.fy = dataset_param::fy;
        frameParam.cx = dataset_param::cx;
        frameParam.cy = dataset_param::cy;

        frameParam.img_width = dataset_param::width;
        frameParam.img_height = dataset_param::height;

        frameParam.angle_t = seg_fusion_param["angle_t"].asFloat(); // Dot product between line vectors
        frameParam.num_line_t = cluster_purge_param["num_line_t"].asInt(); // Line threshold (Related to size of features to track)
        frameParam.num_pixels_t = cluster_purge_param["num_pixels_t"].asInt(); // Pixel threshold

        // Debug parameters
        frameParam.debug_row_idx = cpu_param["debug_row_idx"].asInt();
    }
}