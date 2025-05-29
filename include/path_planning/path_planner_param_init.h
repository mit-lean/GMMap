//
// Created by peterli on 5/6/24.
//

#ifndef GMM_MAP_PATH_PLANNER_PARAM_INIT_H
#define GMM_MAP_PATH_PLANNER_PARAM_INIT_H
#include <iostream>
#include <string>
#include "path_planning/path_planner.h"

namespace gmm {
    // Initialize parameters
    void initializeSamplingBasedPlannerConfig(sampling_planner_config& config);
}

#endif //GMM_MAP_PATH_PLANNER_PARAM_INIT_H
