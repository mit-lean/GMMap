//
// Created by peter on 3/12/22.
//

#ifndef GMM_MAP_EVALUATION_H
#define GMM_MAP_EVALUATION_H
#include "map.h"
#include <iostream>
#include <map>
#include <mutex>

namespace gmm {
    // Sample within a bounding box
    void evaluateOccupancyAndVarianceBBox(GMMMap const* map, const Rect& GlobalBBox, int num_points,
                                          FP unexplored_evidence, FP unexplored_variance,
                                          std::vector<Eigen::Vector3d>& obs_and_free_pts,
                                          std::vector<FP>& occ_value, std::vector<FP>& variance_value, FP& throughput);

    // Sample along a specific set of frame indices from accuracy_eval/sorted_frame_roc.csv
    void evaluateOccupancyAccuracyAndVarianceRay(GMMMap const* map, FP ray_sampling_dist, int num_frames,
                                                 FP unexplored_evidence, FP unexplored_variance,
                                                 std::vector<Eigen::Vector3d>& obs_and_free_pts,
                                                 std::vector<FP>& occ_value, std::vector<FP>& variance_value, std::vector<bool>& error, FP& throughput);
}
#endif //GMM_MAP_EVALUATION_H
