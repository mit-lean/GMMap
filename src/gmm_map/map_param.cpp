//
// Created by peterli on 9/18/22.
//
#include "gmm_map/map_param.h"

namespace gmm {
    void frame_param::computeDepth(bool adaptive){
        depth_dists.clear();
        depth_dists_cum.clear();
        // Use free_space_start_len as length of the first interval
        std::cout << fmt::format("Computing adaptive depth for {} dataset with starting length {:.2f} and max length {:.2f}",
                                 dataset, free_space_start_len, free_space_max_length) << std::endl;
        gmm::FP depth_interval_scale_factor = 0.5;
        gmm::FP cur_depth = 0;
        gmm::FP pre_depth = 0;
        bool initial_depth = true;
        bool final_depth = false;
        if (!adaptive){
            initial_depth = false;
            final_depth = true;
        }

        Eigen::Vector3d LeftP, RightP, UpP;
        forwardProject(0,0, max_depth, fx, fy, cx, cy, LeftP, dataset, max_depth);
        forwardProject(0,img_width,max_depth, fx, fy, cx, cy, RightP, dataset,max_depth);
        forwardProject(img_height,img_width,max_depth, fx, fy, cx, cy,UpP, dataset,max_depth);
        gmm::FP max_dist = std::fmin(std::abs(RightP(0) - LeftP(0)), std::abs(UpP(1) - LeftP(1)));
        gmm::FP slope = depth_interval_scale_factor * max_dist / max_depth;

        while (cur_depth < max_depth){
            if (initial_depth){
                pre_depth = cur_depth;
                gmm::FP dist = free_space_start_len;
                cur_depth += dist;
                initial_depth = false;
                depth_dists.push_back(dist);
            } else if (final_depth) {
                pre_depth = cur_depth;
                cur_depth += free_space_max_length;
                depth_dists.push_back(free_space_max_length);
            } else {
                gmm::FP dist = slope * cur_depth;
                if (dist > free_space_max_length){
                    pre_depth = cur_depth;
                    cur_depth += free_space_max_length;
                    depth_dists.push_back(free_space_max_length);
                    final_depth = true;
                } else if (dist < free_space_start_len) {
                    pre_depth = cur_depth;
                    cur_depth += free_space_start_len;
                    depth_dists.push_back(free_space_start_len);
                } else {
                    pre_depth = cur_depth;
                    cur_depth += dist;
                    depth_dists.push_back(dist);
                }
            }
            depth_dists_cum.push_back(cur_depth);
            std::cout << fmt::format("Depth [{:.2f}, {:.2f}]: Maximum cluster length {:.2f}", pre_depth, cur_depth, depth_dists.back()) << std::endl;
        }
    }

    int frame_param::determineDepthIndex(const gmm::FP& depth) const {
        int index = 0;
        for (auto& cum_depth : depth_dists_cum){
            if (cum_depth < depth){
                index++;
            } else {
                break;
            }
        }
        return index;
    }

    void frame_param::determineDepthBounds(const int& index, gmm::FP& lowerBound, gmm::FP& upperBound) const {
        upperBound = depth_dists_cum.at(index);
        lowerBound = upperBound - determineDepthLength(index);
    }

    gmm::FP frame_param::determineDepthLength(const int& index) const {
        return depth_dists.at(index);
    }

    gmm::FP frame_param::determineDepthLength(const gmm::FP& depth) const {
        gmm::FP total_depth = 0;
        for (auto& cur_depth : depth_dists) {
            total_depth += cur_depth;
            if (total_depth >= depth){
                return cur_depth;
            }
        }
        return free_space_max_length;
    }

    void map_param::updateParameters(){
        this->hell_thresh_squard_obs = hell_thresh_squard_free * hell_thresh_squard_obs_scale;
        this->max_bbox_len = min_gau_len * fusion_max_scale;
        updateFrameParameters();
    }

    void map_param::updateFrameParameters(){
        // Pass appropriate paramters to spgf_extended
        gmm_frame_param.num_threads = num_threads;
        gmm_frame_param.measure_memory = measure_memory;
        gmm_frame_param.max_depth = max_depth;
        gmm_frame_param.gau_bd_scale = gau_rtree_bd;
        gmm_frame_param.free_space_start_len = min_gau_len;
        gmm_frame_param.free_space_max_length = frame_max_scale * min_gau_len;
        gmm_frame_param.preserve_details_far_objects = false;
        // Compute virtual planes
        gmm_frame_param.computeDepth(true);
    }
}