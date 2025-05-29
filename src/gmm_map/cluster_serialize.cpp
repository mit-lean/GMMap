//
// Created by peter on 2/10/23.
//
// Defines the serialization function for different cluster types
#include "gmm_map/cluster.h"

namespace gmm {
    void GMMcluster_o::save(std::ostream &stream) const {
        stream.write((char *) &near_obs, sizeof(near_obs));
        stream.write((char *) &N, sizeof(N));
        stream.write((char *) &W, sizeof(W));
        stream.write((char *) mean.data(), (long) (3 * sizeof(FP)));
        M cov = cov_llt.reconstructedMatrix();
        FP cov_lower[6] = {cov(0,0), cov(0,1), cov(0,2),
                           cov(1,1), cov(1,2), cov(2,2)};
        stream.write((char *) cov_lower, 6 * sizeof(FP));
    }

    void GMMcluster_o::load(std::istream &stream) {
        stream.read((char *) &near_obs, sizeof(near_obs));
        stream.read((char *) &N, sizeof(N));
        stream.read((char *) &W, sizeof(W));
        stream.read((char *) mean.data(), (long) (3 * sizeof(FP)));

        FP cov_lower[6];
        stream.read((char *) cov_lower, 6 * sizeof(FP));
        M cov;
        cov << cov_lower[0], cov_lower[1], cov_lower[2],
                cov_lower[1], cov_lower[3], cov_lower[4],
                cov_lower[2], cov_lower[4], cov_lower[5];
        cov_llt.compute(cov);
    }

    void GMMcluster_c::save(std::ostream &stream) const {
        GMMcluster_o::save(stream);
        if (track_color){
            stream.write((char *) Mean_c_eff->data(), (long) (Mean_c_eff->size() * sizeof(FP)));
            stream.write((char *) Covariance_c_eff->data(), (long) (Covariance_c_eff->size() * sizeof(FP)));
        }
    }

    void GMMcluster_c::load(std::istream &stream) {
        GMMcluster_o::load(stream);
        if (track_color) {
            stream.read((char *) Mean_c_eff->data(), (long) (Mean_c_eff->size() * sizeof(FP)));
            stream.read((char *) Covariance_c_eff->data(), (long) (Covariance_c_eff->size() * sizeof(FP)));
        }
    }
}

