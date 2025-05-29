//
// Created by peterli on 5/6/24.
//
#include "path_planning//path_planner_param_init.h"
#include "dataset_utils/dataset_utils.h"

namespace gmm {
    void initializeSamplingBasedPlannerConfig(sampling_planner_config& config){
        auto planning_param = dataset_param::dataset_info["path_planning_parameters"];
        config.planner_name = planning_param["name"].asString();
        config.use_occupancy = planning_param["use_occupancy"].asBool();
        config.occ_free_threshold = planning_param["occ_free_threshold"].asFloat();
        config.validity_checking_resolution = planning_param["validity_checking_resolution"].asFloat();
        config.max_vertices = planning_param["num_of_samples"].asInt();
        config.free_space_evidence = dataset_param::dataset_info["occupancy_inference_parameters"]["unexplored_evidence"].asFloat();
    }
}