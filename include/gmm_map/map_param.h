//
// Created by peter on 7/4/21.
//

#ifndef GMM_MAP_PARAMETERS_H
#define GMM_MAP_PARAMETERS_H
#include <iostream>
#include <string>
#include "gmm_map/commons.h"

namespace gmm {
    struct frame_param {
        // Struct used for parameter initialization and management
        frame_param() = default;

        // Define some constants within this program. These constants will be replaced by input args!
        std::string dataset;
        int img_width; // Width of the depthmap
        int img_height; // Height of the depthmap
        std::vector<gmm::FP> depth_dists;
        std::vector<gmm::FP> depth_dists_cum; // Cummulative depth distance

        int num_threads;
        bool measure_memory; // Remove intermediate representations for region growing to save memory usage.
        bool preserve_details_far_objects; // Preserve details for objects further away at significant expense to memory overhead (2x more)

        // Define clustering parameters for proposed incremental method
        int occ_x_t; // Maximum number of pixels in the x direction  for occlusion region
        gmm::FP noise_thresh; // Maximum noise threshold of the sensor
        int sparse_t; // Minimum number of points in a cluster
        int ncheck_t; // Number of points after which fitting is applied

        gmm::FP gau_bd_scale; // Bounds for computing boundary of Gaussians (BBox)
        gmm::FP adaptive_thresh_scale; // Adaptive threshold scale: 0.35 for Tartanair, 1.2 for TUM (more noisy)
        //gmm::FP max_len; // Maximum projected length of each line segment
        gmm::FP max_depth;

        gmm::FP f;
        gmm::FP fx;
        gmm::FP fy;
        gmm::FP cx;
        gmm::FP cy;

        gmm::FP line_t;
        gmm::FP angle_t; // Dot product between line vectors
        gmm::FP noise_floor; // Minimum noisy bound to avoid over segmentation (5cm)
        int num_line_t; // Line threshold (Related to size of features to track)
        int num_pixels_t; // Pixel threshold
        int max_incomplete_clusters; // Maximum number of previous segment to consider during line segmentation


        // Parameters for advanced free space modelling to ensure accuracy of unexplored space.
        gmm::FP free_space_start_len;
        gmm::FP free_space_max_length; // Maximum length for generating free-space GMMs
        gmm::FP free_space_dist_scale; // Multiplicative constant to increase free space merging within the frame

        // Debug parameters
        int debug_row_idx;

        void computeDepth(bool adaptive = false);
        int determineDepthIndex(const gmm::FP& depth) const;

        void determineDepthBounds(const int& index, gmm::FP& lowerBound, gmm::FP& upperBound) const;

        gmm::FP determineDepthLength(const int& index) const;

        gmm::FP determineDepthLength(const gmm::FP& depth) const;
    };

// Will "inherit" parameters for each frame
    struct map_param {
        // Struct used for parameter initialization and management
        // Parameters for each frame
        frame_param gmm_frame_param;

        // Define some constants within this program. These constants will be replaced by input args!
        std::string dataset;
        int num_threads;
        bool measure_memory; // Remove intermediate representations for region growing to save memory usage.
        std::string frame_alg_name;
        gmm::FP max_depth;

        // Mapping parameters (We only need to tune the following two parameters)
        gmm::FP hell_thresh_squard_oversized_gau; // Threshold for oversized Gaussians
        gmm::FP hell_thresh_squard_free; // Threshold for squared hellinger distance
        gmm::FP hell_thresh_squard_obs_scale; // Threshold for squared hellinger distance
        gmm::FP hell_thresh_squard_obs; // Threshold for squared hellinger distance
        gmm::FP min_gau_len; // Maximum resolution of the GMM (m)
        gmm::FP frame_max_scale; // Scale factor for maximum SPGF region length
        gmm::FP fusion_max_scale; // Scale factor for fusion region length

        // Following parameters are pre_set
        int min_num_neighbor_clusters; // Minimum number of neighboring clusters for computing occupancy and variance
        gmm::FP hell_thresh_squard_min; // Min threshold for squared hellinger distance
        gmm::FP gau_fusion_bd;
        gmm::FP gau_rtree_bd;
        gmm::FP depth_scale; // Scaling for values in the raw (png) depth image
        gmm::FP max_bbox_len;

        bool track_color;
        bool track_intensity;
        bool cur_debug_frame;

        map_param() = default;
        void updateParameters();
        void updateFrameParameters();
    };
}

#endif //GMM_MAP_PARAMETERS_H
