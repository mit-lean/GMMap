//
// Created by peterli on 6/28/23.
//

#ifndef GMM_MAP_PARAM_INIT_H
#define GMM_MAP_PARAM_INIT_H
#include "gmm_map/map_param.h"
namespace gmm {
    // Initialize parameters
    void initializeMapParameters(map_param& mapParam, const std::string& name);

    void initializeFrameParameters(frame_param& frameParam, const std::string& name);
}
#endif //GMM_MAP_PARAM_INIT_H
