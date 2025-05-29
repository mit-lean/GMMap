//
// Created by peterli on 6/6/22.
//

#ifndef GMM_MAPPING_CLUSTER_OPS_H
#define GMM_MAPPING_CLUSTER_OPS_H
#include "gmm_map/cluster.h"

namespace gmm {
    V mean(const std::list<const V *> &l);

    M covariance(const std::list<const V *> &l, const V &Mean);

    void invertSymmPositive(M &Sigma_inverse, const M &Sigma);

    bool checkCovariance(const M &Cov);

    FP bhatDist(const V &mean0, const V &mean1, const M &cov0, const M &cov1);

    FP mahaDist(const V &mean0, const V &mean1, const M &cov_inv);

    FP mahaDist(const V &mean0, const V &mean1, const M &cov0, const M &cov1);

    FP hellingerSquared(const V_2 &mean0, const V_2 &mean1, const M_2 &cov0,
                            const M_2 &cov1);

    FP hellingerSquared(const V &mean0, const V &mean1, const M &cov0, const M &cov1);

    void eigenSymmP(const M &cov, M &evecs, V &evals);

    bool bboxIntersect(const V &lowerBound0, const V &upperBound0, const V &lowerBound1, const V &upperBound1);

    bool bboxEnclosed(const V &lowerBound0, const V &upperBound0, const V &lowerBound1, const V &upperBound1);

    bool intervalIntersect(const FP &lowerBound0, const FP &upperBound0, const FP &lowerBound1,
                           const FP &upperBound1);

    bool intervalEnclosed(const FP &lowerBound0, const FP &upperBound0, const FP &lowerBound1,
                          const FP &upperBound1);

// Evaluate the Gaussian
    FP eval3DGaussian(const V &pt, const V &mean, const M &cov);

// Compute the L2 distance between two GMMs
    FP l2Dist(const std::vector<FP> &w_vec0, const std::vector<V> &mean_vec0,
                  const std::vector<M> &cov_vec0,
                  const std::vector<FP> &w_vec1, const std::vector<V> &mean_vec1,
                  const std::vector<M> &cov_vec1);

    FP l2DistNotNormalized(const std::vector<FP> &w_vec0, const std::vector<V> &mean_vec0,
                               const std::vector<M> &cov_vec0,
                               const std::vector<FP> &w_vec1, const std::vector<V> &mean_vec1,
                               const std::vector<M> &cov_vec1);

    FP unscentedHellingerSquaredClusters(const std::vector<GMMcluster_o*>& a_clusters, const std::vector<GMMcluster_o*>& b_clusters);

    FP unscentedHellingerSquaredFreeBasis(const std::vector<freeSpaceBasis*>& a_data, const std::vector<freeSpaceBasis*>& b_data);

    FP unscentedHellingerSquared(const std::vector<V>& Means_a, const std::vector<Eigen::LLT<M>>& CovLLTs_a, const std::vector<FP>& Ws_a,
                                     const std::vector<V>& Means_b, const std::vector<Eigen::LLT<M>>& CovLLTs_b, const std::vector<FP>& Ws_b);

    void computeSigmaPointsAndWeights(Vector7& weights, Matrix3x7& points, const V& Mean, const Eigen::LLT<M>& CovLLT);

    void evaluatePointsUnderGMM(const Matrix3x7& points,
                                const std::vector<V>& Means_a, const std::vector<Eigen::LLT<M>>& CovLLTs_a, const std::vector<FP>& Ws_a,
                                const std::vector<V>& Means_b, const std::vector<Eigen::LLT<M>>& CovLLTs_b, const std::vector<FP>& Ws_b,
                                const FP& W_t_merged, const FP& W_t_a, const FP& W_t_b,
                                Array7& f_merged,  Array7& f_a, Array7& f_b);
}
#endif //GMM_MAPPING_CLUSTER_OPS_H
