#include "gmm_map/cluster_ops.h"
#include "gmm_map/cluster.h"
#include <iostream>
#include <Eigen/Cholesky>

namespace gmm {
	V mean(const std::list<const V*>& l) {
		if (l.empty()) throw std::invalid_argument("cannot compute mean of empty collection");
		auto it = l.begin();
		V Mean = *l.front();
		++it;
		std::for_each(it, l.end(), [&Mean](const V* pt) {
			Mean += *pt;
		});
		Mean = (1.0 / static_cast<FP>(l.size()) ) * Mean;
		return Mean;
	}

	M covariance(const std::list<const V*>& l, const V& Mean) {
		if (l.empty()) throw std::invalid_argument("cannot compute covariance of empty collection");
		M Cov(Mean.size(), Mean.size());
		Cov.setZero();
		std::for_each(l.begin(), l.end(), [&Cov, &Mean](const V* pt) {
			Cov += (*pt - Mean) * (*pt - Mean).transpose();
		});
		Cov = (1.0 / static_cast<FP>(l.size()))* Cov;
		return Cov;
	}

	void invertSymmPositive(M& Sigma_inverse, const M& Sigma) {
		//LLT<MatrixXf> lltOfCov(Sigma);
		//MatrixXf L(lltOfCov.matrixL());
		//*Sigma_inverse = L * L.transpose();

		Sigma_inverse = Sigma.inverse();
	}

	bool checkCovariance(const M& Cov) {
		std::size_t K = (std::size_t)Cov.cols();
		std::size_t c;
		for (std::size_t r = 0; r < K; ++r) {
			for (c = (r + 1); c < K; ++c) {
				if (abs(Cov(r, c) - Cov(c, r)) > 1e-5) return false;
			}
		}

		Eigen::EigenSolver<M> eig_solv(Cov);
		auto eigs = eig_solv.eigenvalues();

		//check all values are sufficient high
		for (std::size_t k = 0; k < K; ++k) {
			if (abs(eigs(k).real()) < 1e-5) return false;
		}
		return true;
	}

    FP bhatDist(const V& mean0, const V& mean1, const M& cov0, const M& cov1){
	    M cov = (cov0 + cov1)/2.0f;
	    return 0.125f * mahaDist(mean0, mean1, cov.inverse()) + 0.5f * logf(cov.determinant()/sqrtf(cov0.determinant()*cov1.determinant()));
	}

    FP mahaDist(const V& mean0, const V& mean1, const M& cov_inv){
	    V mean_diff = mean0 - mean1;
	    return sqrtf(mean_diff.transpose() * cov_inv * mean_diff);
	}

    FP mahaDist(const V& mean0, const V& mean1, const M& cov0, const M& cov1){
        M cov_avg = (cov0 + cov1) / 2.0f;
        return mahaDist(mean0, mean1, cov_avg.inverse());
    }

    FP hellingerSquared(const V_2& mean0, const V_2& mean1, const M_2& cov0, const M_2& cov1){
	    // Compute 2D Squared Hellinger distance
	    // Results are capped between 0 (same distribution) and 1 (different distribution)
	    // Resources: https://presentations.copernicus.org/EGU2020/EGU2020-3340_presentation.pdf
	    // https://www.tifr.res.in/~prahladh/teaching/2011-12/comm/lectures/l12.pdf
        // https://en.wikipedia.org/wiki/Hellinger_distance

        M_2 cov_avg = (cov0 + cov1)/2;
        V_2 mean_diff = mean0 - mean1;
        FP exp_term = expf(-0.125f * mean_diff.transpose() * cov_avg.inverse() * mean_diff);
        return 1.0f - powf(cov0.determinant() * cov1.determinant(), 0.25f) / sqrtf(cov_avg.determinant()) * exp_term;
	}

    FP hellingerSquared(const V& mean0, const V& mean1, const M& cov0, const M& cov1){
        // Compute 3D Squared Hellinger distance
        // Results are capped between 0 (same distribution) and 1 (different distribution)
        // Resources: https://presentations.copernicus.org/EGU2020/EGU2020-3340_presentation.pdf
        // https://www.tifr.res.in/~prahladh/teaching/2011-12/comm/lectures/l12.pdf
        // https://en.wikipedia.org/wiki/Hellinger_distance
        M cov_avg = (cov0 + cov1)/2.0f;
        V mean_diff = mean0 - mean1;
        FP exp_term = expf(-0.125f * mean_diff.transpose() * cov_avg.inverse() * mean_diff);
        return 1.0f - powf(cov0.determinant() * cov1.determinant(), 0.25f) / sqrtf(cov_avg.determinant()) * exp_term;
	}

    void eigenSymmP(const M& cov, M& evecs, V& evals){
        //Eigenvalues are ordered from smallest to largest
        Eigen::SelfAdjointEigenSolver<M> Sol (cov);
        // Column vectors containing eigenvectors
        evecs = Sol.eigenvectors();
        // Column vectors connecting eigenvalues
        evals = Sol.eigenvalues();
	}

    bool bboxIntersect(const V& lowerBound0, const V& upperBound0, const V& lowerBound1, const V& upperBound1){
        // Need to be in the same coordinate system
        // Need to be axis aligned!
        for (int i = 0; i < lowerBound0.size(); i++){
            if (!intervalIntersect(lowerBound0(i), upperBound0(i), lowerBound1(i), upperBound1(i))){
                return false;
            }
        }
        return true;
    }

    bool bboxEnclosed(const V& lowerBound0, const V& upperBound0, const V& lowerBound1, const V& upperBound1){
        // Determines if box0 is enclosed within box1
        // Need to be in the same coordinate system
        // Need to be axis aligned!
        for (int i = 0; i < lowerBound0.size(); i++){
            if (!intervalEnclosed(lowerBound0(i), upperBound0(i), lowerBound1(i), upperBound1(i))){
                return false;
            }
        }
        return true;

    }

    bool intervalIntersect(const FP& lowerBound0, const FP& upperBound0, const FP& lowerBound1, const FP& upperBound1){
        return (upperBound1 >= lowerBound0) & (upperBound0 >= lowerBound1);
    }

    bool intervalEnclosed(const FP& lowerBound0, const FP& upperBound0, const FP& lowerBound1, const FP& upperBound1){
        // Check if box0 is enclosed within box1
        return (upperBound1 >= upperBound0) & (lowerBound0 >= lowerBound1);
    }

    // APIs used to compute the L2 distance between two Gaussians
    FP eval3DGaussian(const V& pt, const V& mean, const M& cov){
        V mean_dist = pt-mean;
        FP num = -(mean_dist.transpose()*cov.inverse()*mean_dist)(0)/2;
        return expf(num) / (sqrtf(abs(cov.determinant())) * RT_2_PI_3);
    }

    FP l2Dist(const std::vector<FP>& w_vec0, const std::vector<V>& mean_vec0, const std::vector<M>& cov_vec0,
                  const std::vector<FP>& w_vec1, const std::vector<V>& mean_vec1, const std::vector<M>& cov_vec1){
        // L2 distance between GMMs are computed as shown in:
        // https://kisungyou.com/notes/note004/main004.pdf
        // Note that the weights do not have to be normalized

        FP dist0 = 0, dist1 = 0, dist2 = 0;
        FP w_total0 = 0, w_total1 = 0;

        for (int i = 0; i < w_vec0.size(); i++) {
            w_total0 += w_vec0.at(i);
            for (int j = 0; j < w_vec0.size(); j++){
                dist0 += w_vec0.at(i) * w_vec0.at(j) * eval3DGaussian(mean_vec0.at(i), mean_vec0.at(j), cov_vec0.at(i) + cov_vec0.at(j));
            }
        }
        dist0 = dist0 / powf(w_total0, 2.0f);

        for (int i = 0; i < w_vec1.size(); i++) {
            w_total1 += w_vec1.at(i);
            for (int j = 0; j < w_vec1.size(); j++){
                dist1 += w_vec1.at(i) * w_vec1.at(j) * eval3DGaussian(mean_vec1.at(i), mean_vec1.at(j), cov_vec1.at(i) + cov_vec1.at(j));
            }
        }
        dist1 = dist1 / powf(w_total1, 2.0f);

        for (int i = 0; i < w_vec0.size(); i++) {
            for (int j = 0; j < w_vec1.size(); j++){
                dist2 += w_vec0.at(i) * w_vec1.at(j) * eval3DGaussian(mean_vec0.at(i), mean_vec1.at(j), cov_vec0.at(i) + cov_vec1.at(j));
            }
        }
        dist2 = dist2 / (w_total0 * w_total1);
        return sqrtf(dist0 + dist1 - 2.0f * dist2);
    }

    FP l2DistNotNormalized(const std::vector<FP>& w_vec0, const std::vector<V>& mean_vec0, const std::vector<M>& cov_vec0,
                  const std::vector<FP>& w_vec1, const std::vector<V>& mean_vec1, const std::vector<M>& cov_vec1){
        // L2 distance between GMMs are computed as shown in:
        // https://kisungyou.com/notes/note004/main004.pdf
        // This computes the L2 distance for weights that are not normalized

        FP dist0 = 0, dist1 = 0, dist2 = 0;

        for (int i = 0; i < w_vec0.size(); i++) {
            for (int j = 0; j < w_vec0.size(); j++){
                dist0 += w_vec0.at(i) * w_vec0.at(j) * eval3DGaussian(mean_vec0.at(i), mean_vec0.at(j), cov_vec0.at(i) + cov_vec0.at(j));
            }
        }

        for (int i = 0; i < w_vec1.size(); i++) {
            for (int j = 0; j < w_vec1.size(); j++){
                dist1 += w_vec1.at(i) * w_vec1.at(j) * eval3DGaussian(mean_vec1.at(i), mean_vec1.at(j), cov_vec1.at(i) + cov_vec1.at(j));
            }
        }

        for (int i = 0; i < w_vec0.size(); i++) {
            for (int j = 0; j < w_vec1.size(); j++){
                dist2 += w_vec0.at(i) * w_vec1.at(j) * eval3DGaussian(mean_vec0.at(i), mean_vec1.at(j), cov_vec0.at(i) + cov_vec1.at(j));
            }
        }

        return sqrtf(dist0 + dist1 - 2 * dist2);
    }

    FP unscentedHellingerSquaredFreeBasis(const std::vector<freeSpaceBasis*>& a_data, const std::vector<freeSpaceBasis*>& b_data){
        std::vector<V> Means_a;
        std::vector<Eigen::LLT<M>> CovLLTs_a;
        std::vector<FP> Ws_a;
        Means_a.reserve(a_data.size());
        CovLLTs_a.reserve(a_data.size());
        Ws_a.reserve(a_data.size());

        for (const auto data : a_data){
            V mean = data->S_cluster / data->W_cluster;
            M cov = data->J_cluster / data->W_cluster  - mean * mean.transpose();
            Means_a.emplace_back(mean);
            CovLLTs_a.emplace_back(cov.llt());
            Ws_a.push_back(data->W_cluster);
        }

        std::vector<V> Means_b;
        std::vector<Eigen::LLT<M>> CovLLTs_b;
        std::vector<FP> Ws_b;
        Means_b.reserve(b_data.size());
        CovLLTs_b.reserve(b_data.size());
        Ws_b.reserve(b_data.size());
        for (const auto data : b_data){
            V mean = data->S_cluster / data->W_cluster;
            M cov = data->J_cluster / data->W_cluster  - mean * mean.transpose();
            Means_b.emplace_back(mean);
            CovLLTs_b.emplace_back(cov.llt());
            Ws_b.push_back(data->W_cluster);
        }
        return unscentedHellingerSquared(Means_a, CovLLTs_a, Ws_a, Means_b, CovLLTs_b, Ws_b);
    }

    FP unscentedHellingerSquaredClusters(const std::vector<GMMcluster_o*>& a_clusters, const std::vector<GMMcluster_o*>& b_clusters){
        std::vector<V> Means_a;
        std::vector<Eigen::LLT<M>> CovLLTs_a;
        std::vector<FP> Ws_a;
        Means_a.reserve(a_clusters.size());
        CovLLTs_a.reserve(a_clusters.size());
        Ws_a.reserve(a_clusters.size());

        for (const auto cluster : a_clusters){
            Means_a.push_back(cluster->mean);
            CovLLTs_a.push_back(cluster->cov_llt);
            if (cluster->is_free){
                Ws_a.push_back(cluster->W);
            } else {
                Ws_a.push_back(cluster->N);
            }
        }

        std::vector<V> Means_b;
        std::vector<Eigen::LLT<M>> CovLLTs_b;
        std::vector<FP> Ws_b;
        Means_b.reserve(b_clusters.size());
        CovLLTs_b.reserve(b_clusters.size());
        Ws_b.reserve(b_clusters.size());
        for (const auto cluster : b_clusters){
            Means_b.push_back(cluster->mean);
            CovLLTs_b.push_back(cluster->cov_llt);
            if (cluster->is_free){
                Ws_b.push_back(cluster->W);
            } else {
                Ws_b.push_back(cluster->N);
            }
        }
        return unscentedHellingerSquared(Means_a, CovLLTs_a, Ws_a, Means_b, CovLLTs_b, Ws_b);
    }

    FP unscentedHellingerSquared(const std::vector<V>& Means_a, const std::vector<Eigen::LLT<M>>& CovLLTs_a, const std::vector<FP>& Ws_a,
                                     const std::vector<V>& Means_b, const std::vector<Eigen::LLT<M>>& CovLLTs_b, const std::vector<FP>& Ws_b){
        // Compute unscented Hellinger squared distance between the fused cluster and a original set of clusters
        // We assume that the weights of the original set of clusters are not normalized
        // Merge clusters and compute normalization constants
        // See: https://towardsdatascience.com/the-unscented-kalman-filter-anything-ekf-can-do-i-can-do-it-better-ce7c773cf88d
        // See: http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=ECF106DE1F3E5E8DC6680F9CA7B37335?doi=10.1.1.46.6718&rep=rep1&type=pdf
        // See: https://www.researchgate.net/publication/228573393_Multivariate_online_kernel_density_estimation

        // Compute normalization constants
        FP W_t_merged = 0, W_t_a = 0, W_t_b = 0;
        for (const auto& W : Ws_a){
            W_t_a += W;
            W_t_merged += W;
        }
        for (const auto& W : Ws_b){
            W_t_b += W;
            W_t_merged += W;
        }

        FP dist = 0;
        for (int i = 0; i < Means_a.size(); i++){
            Vector7 Weights;
            Matrix3x7 SigmaPoints;
            // 2) Compute Sigma points and weights for the merged cluster
            computeSigmaPointsAndWeights(Weights, SigmaPoints, Means_a.at(i), CovLLTs_a.at(i));
            // 3) Perform unscented transform by evaluating the sigma points under each GMM
            Array7 f_merged, f_a, f_b;
            evaluatePointsUnderGMM(SigmaPoints,
                                   Means_a, CovLLTs_a, Ws_a,
                                   Means_b, CovLLTs_b, Ws_b,
                                   W_t_merged, W_t_a, W_t_b, f_merged, f_a, f_b);
            Vector7 g = (f_a.cwiseSqrt() - f_b.cwiseSqrt()).square() / f_merged;
            dist += Ws_a.at(i) / W_t_merged * (Weights.dot(g));
        }
        for (int i = 0; i < Means_b.size(); i++){
            Vector7 Weights;
            Matrix3x7 SigmaPoints;
            // 2) Compute Sigma points and weights for the merged cluster
            computeSigmaPointsAndWeights(Weights, SigmaPoints, Means_b.at(i), CovLLTs_b.at(i));
            // 3) Perform unscented transform by evaluating the sigma points under each GMM
            Array7 f_merged, f_a, f_b;
            evaluatePointsUnderGMM(SigmaPoints,
                                   Means_a, CovLLTs_a, Ws_a,
                                   Means_b, CovLLTs_b, Ws_b,
                                   W_t_merged, W_t_a, W_t_b, f_merged, f_a, f_b);
            Vector7 g = (f_a.cwiseSqrt() - f_b.cwiseSqrt()).square() / f_merged;
            dist += Ws_b.at(i) / W_t_merged * (Weights.dot(g));
        }
        return 0.5f * dist;
    }

    void computeSigmaPointsAndWeights(Vector7& weights, Matrix3x7& points, const V& Mean, const Eigen::LLT<M>& CovLLT){
        // Compute weights
        weights.setConstant(1.0f/6.0f);
        weights(0) = 0;

        // Compute sigma points
        M L = CovLLT.matrixL();
        L = sqrtf(3) * L;
        points.colwise() = Mean;
        points.block<3,3>(0,1) += L;
        points.block<3,3>(0,4) -= L;
    }

    void evaluatePointsUnderGMM(const Matrix3x7& points,
                                const std::vector<V>& Means_a, const std::vector<Eigen::LLT<M>>& CovLLTs_a, const std::vector<FP>& Ws_a,
                                const std::vector<V>& Means_b, const std::vector<Eigen::LLT<M>>& CovLLTs_b, const std::vector<FP>& Ws_b,
                                const FP& W_t_merged, const FP& W_t_a, const FP& W_t_b,
                                Array7& f_merged,  Array7& f_a, Array7& f_b){
        f_merged.setZero();
        f_a.setZero();
        f_b.setZero();

        for (int i = 0; i < Means_a.size(); i++){
            Matrix3x7 mean_dist = points.colwise() - Means_a.at(i);
            Array7 num = -0.5f * (mean_dist.array() * CovLLTs_a.at(i).solve(mean_dist).array()).colwise().sum().transpose();
            M L = CovLLTs_a.at(i).matrixL();
            FP CovDet = L(0,0) * L(1,1) * L(2,2);
            num = num.exp() / (std::abs(CovDet) * RT_2_PI_3);
            f_a += Ws_a.at(i) / W_t_a * num;
            f_merged += Ws_a.at(i) / W_t_merged * num;
        }

        for (int i = 0; i < Means_b.size(); i++){
            Matrix3x7 mean_dist = points.colwise() - Means_b.at(i);
            Array7 num = -0.5f * (mean_dist.array() * CovLLTs_b.at(i).solve(mean_dist).array()).colwise().sum().transpose();
            M L = CovLLTs_b.at(i).matrixL();
            FP CovDet = L(0,0) * L(1,1) * L(2,2);
            num = num.exp() / (std::abs(CovDet) * RT_2_PI_3);
            f_b += Ws_b.at(i) / W_t_b * num;
            f_merged += Ws_b.at(i) / W_t_merged * num;
        }
    }

    // Implementation of cluster specific APIs
    void GMMcluster_o::printInfo() const {
        // Print cluster information
        std::cout.precision(10);
        std::cout << fmt::format("Printing info for a cluster with N: {}, W: {:.10f} as follows:", N, W) << std::endl;
        Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[","]");
        std::cout << "Mean:" << std::endl;
        std::cout << Mean().format(CleanFmt) << std::endl;
        std::cout << "Cov:" << std::endl;
        std::cout << Cov().format(CleanFmt) << std::endl;
    }


    V GMMcluster_o::Mean() const {
        return mean;
    }

    M GMMcluster_o::Cov() const {
        return cov_llt.reconstructedMatrix();
    }

    M GMMcluster_o::CovL() const {
        return cov_llt.matrixL();
    }

    FP GMMcluster_o::CovDet() const {
        M L = CovL();
        return powf(L(0,0) * L (1,1) * L(2,2), 2.0f);
    }

    FP GMMcluster_o::CovDetL() const {
        M L = CovL();
        return L(0,0) * L (1,1) * L(2,2);
    }

    void GMMcluster_o::estimateLineAndNormal(V& line_vector, V& normal_vector){
        M evecs;
        V evals;
        // Vectors are ordered from the smallest to largest eigen-value
        eigenSymmP(Cov(),evecs,evals);
        line_vector = evecs.col(2);
        normal_vector = evecs.col(0);
    }

    /*
    FP GMMcluster_o::estOcc(const V& point) const {
        FP occ = Mean_o(3) + (Covariance_o.bottomLeftCorner<1,3>()*Inverse_Cov*(point - Mean_o.topLeftCorner<3,1>()))(0);
        //std::cout << fmt::format("Estimated occupancy for the current cluster is {}", occ) << std::endl;
        if (occ > 1){
            return 1;
        } else if (occ < 0){
            return 0;
        } else {
            return occ;
        }
    }

    FP GMMcluster_o::estOccVariance() const {
        FP var = Covariance_o(3,3) - (Covariance_o.bottomLeftCorner<1,3>()*Inverse_Cov*Covariance_o.topRightCorner<3,1>());
        //std::cout << fmt::format("Estimated variance for the current cluster is {}", var) << std::endl;
        return var;
    }
    */

    FP GMMcluster_o::estOcc(const V& point) const {
        // Since we do not mix measurements from both occupied and free regions, the occupancy is either 0 or 1
        if (is_free)
            return 0;
        else
            return 1;
    }

    FP GMMcluster_o::estOccVariance() const {
        // Since we do not mix measurements from both occupied and free regions, the variance is always 0
        return 0;
    }

    /*
    void GMMcluster_o::updateInvCov(){
        invertSymmPositive(Inverse_Cov, covariance);
        Abs_Deter_Cov = abs(covariance.determinant());
    }
    */

    void GMMcluster_o::updateMeanAndCovFromIntermediateParams(bool free, const V& S, const M& J, bool avoid_degeneracy) {
        // Note: It is very important to add the identity matrix to avoid degenerate Gaussian that leads to singular covariance matrix.
        // This is more evidence when we decrease the precision from double to float
        // Note: Eigen's Cholesky decomposition between intel and arm cpu might produce slightly different results.
        // Since our covariance matrices are computed from results from Cholesky decomposition, the resulting map between intel and arm cpu might be slightly different.
        if (free){
            // Use W instead of N so that gaussian fusion achieves desired results
            mean = S / W;
            M covariance = J / W  - mean * mean.transpose();
            cov_llt.compute(covariance);
            // Check and eliminate degeneracy!
            M small_I = (1e-5f)*M::Identity();
            while (cov_llt.info() == Eigen::NumericalIssue){
                covariance += small_I;
                cov_llt.compute(covariance);
                small_I = 10.0f * small_I;
            }
        } else {
            mean = S / (FP) N;
            M covariance = (J - (S * S.transpose() / (FP) N)) / ((FP) N - 1.0f);
            cov_llt.compute(covariance);
            // Check and eliminate degeneracy!
            M small_I = (1e-5f)*M::Identity();
            while (cov_llt.info() == Eigen::NumericalIssue){
                covariance += small_I;
                cov_llt.compute(covariance);
                small_I = 10.0f * small_I;
            }
        }
    }

    void GMMcluster_o::transformMeanAndCov(const Isometry3& pose) {
        //std::cout.precision(10);
        //Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[","]");
        //std::cout << "Before transformation: " << std::endl;
        //printInfo();
        mean = pose * mean;
        //std::cout << "Transformation matrix (during transform): " << std::endl;
        //std::cout << pose.matrix().format(CleanFmt) << std::endl;
        //std::cout << "Rotation matrix (during transform): " << std::endl;
        //std::cout << pose.rotation().format(CleanFmt) << std::endl;
        M covariance = pose.rotation() * Cov() * pose.rotation().transpose();
        //std::cout << "Covariance after transformation: " << std::endl;
        //std::cout << covariance.format(CleanFmt) << std::endl;
        cov_llt.compute(covariance);
        //std::cout << "After LLT Computation: " << std::endl;
        //printInfo();
    }

    void GMMcluster_o::computeIntermediateParams(bool free, V& S, M& J) const{
        // Compute intermediate parameters for fusion
        if (!free) {
            S = N * mean;
            J = (N - 1) * Cov() + N * mean * mean.transpose();
        } else {
            S = W * mean;
            J = W * Cov() + W * mean * mean.transpose();
        }
    }

    void GMMcluster_o::updateBBox(FP scale){
        computeBBox(BBox, scale);
    }

    void GMMcluster_o::computeBBox(Rect& NewBBox, FP scale) const {
        // Recompute BBox based mean and covariance (very easy!)
        // See: https://math.stackexchange.com/questions/3926884/smallest-axis-aligned-bounding-box-of-hyper-ellipsoid
        V d_pos;
        V cov_diag = Cov().diagonal(0);
        d_pos << sqrtf(cov_diag(0)), sqrtf(cov_diag(1)), sqrtf(cov_diag(2));
        d_pos = scale * d_pos;
        NewBBox = Rect(mean - d_pos, mean + d_pos);
    }

    FP GMMcluster_o::computeBBoxVolume(FP scale){
        V cov_diag = Cov().diagonal(0);
        return 8 * scale * scale * scale * sqrtf(cov_diag(0) * cov_diag(1) * cov_diag(2));
    }

    /*
    FP GMMcluster_o::evalGaussianExp(const V& x){
        // Note: We only need to evaluate the inverse here!
        V mean_dist = x - mean;
        FP num = -(mean_dist.transpose()*Inverse_Cov*mean_dist)(0)/2;
        //if (isnan(num)){
        //    std::cout << "NAN detected in computing evalGaussianExp" << std::endl;
        //}
        return exp(num)/sqrtf(Abs_Deter_Cov);
    }
    */

    FP GMMcluster_o::evalGaussianExp(const V& x) const{
        // Note: We only need to evaluate the inverse here!
        V mean_dist = x - mean;
        FP num = -0.5f*(mean_dist.transpose()*cov_llt.solve(mean_dist))(0);
        return expf(num)/CovDetL();
    }

    FP GMMcluster_o::evalGaussian(const V& x) const{
        return evalGaussianExp(x)/RT_2_PI_3;
    }

    void GMMcluster_o::transform(const Isometry3& pose, const FP& gau_bd_scale) {
        // Recompute mean and covariance
        transformMeanAndCov(pose);

        // Update bounding box
        updateBBox(gau_bd_scale);
    }

    void GMMcluster_o::fuseCluster(GMMcluster_o const* cluster, const FP& std){
        // Compute intermediate parameters for fusion
        V S_cur, S_new;
        M J_cur, J_new;
        cluster->computeIntermediateParams(cluster->is_free, S_new, J_new);
        computeIntermediateParams(is_free, S_cur, J_cur);

        // Update current cluster and recompute statistics
        N += cluster->N;
        W += cluster->W; // Actual weight in terms of ray length
        near_obs = near_obs | cluster->near_obs;
        updateMeanAndCovFromIntermediateParams(is_free, S_cur + S_new, J_cur + J_new, false);
        updateBBox(std);
    }

    unsigned long long GMMcluster_o::freeGaussianSizeInBytes() {
        auto FP_size = sizeof(FP);
        // For each cluster, we need number of points (N), weight (W), a list pointer, mean, covariance, and a bounding box
        return (sizeof(N) + sizeof(W) + sizeof(list_it) + 3 * FP_size + 9 * FP_size + 6 * FP_size);
    }

    std::string GMMcluster_o::getFreeGaussianName() {
        static std::string freeGaussianName = "FreeGaussian";
        return freeGaussianName;
    }

    // Cluster_c APIs
#ifdef TRACK_MEM_USAGE_GMM
    GMMcluster_c::GMMcluster_c(bool track_color) :
        mutil::objTracker(getObsGaussianName(), obsGaussianSizeInBytes(), 1){
#else
    GMMcluster_c::GMMcluster_c(bool track_color){
#endif
        if (track_color){
            Mean_c_eff = new V;
            Covariance_c_eff = new M_c_eff;
        }
        this->track_color = track_color;
    }

    GMMcluster_c::~GMMcluster_c(){
        delete Mean_c_eff;
        delete Covariance_c_eff;
    }

    void swap( GMMcluster_c& first, GMMcluster_c& second){
        using std::swap;
        swap(static_cast<GMMcluster_o&>(first), static_cast<GMMcluster_o&>(second));
        swap(first.Mean_c_eff, second.Mean_c_eff);
        swap(first.Covariance_c_eff, second.Covariance_c_eff);
        swap(first.track_color, second.track_color);
    }

#ifdef TRACK_MEM_USAGE_GMM
    GMMcluster_c::GMMcluster_c(const GMMcluster_c& cluster) :
        GMMcluster_o(cluster), mutil::objTracker(cluster){
#else
    GMMcluster_c::GMMcluster_c(const GMMcluster_c& cluster) : GMMcluster_o(cluster){
#endif
        track_color = cluster.track_color;
        if (track_color){
            Mean_c_eff = new V;
            *Mean_c_eff = *cluster.Mean_c_eff;
            Covariance_c_eff = new M_c_eff;
            *Covariance_c_eff = *cluster.Covariance_c_eff;
        }
    }

    GMMcluster_c& GMMcluster_c::operator=(GMMcluster_c cluster){
        swap(*this, cluster);
        return *this;
    }


    GMMcluster_c::GMMcluster_c(GMMcluster_c&& cluster) noexcept {
        // Note swapping trackers is already taken care off when swapping the base class GMMcluster_o
        swap(*this, cluster);
    }

    V GMMcluster_c::estColor(const V& point) const {
        if (track_color){
            return *Mean_c_eff + Covariance_c_eff->bottomLeftCorner<3,3>()*cov_llt.solve(point - mean);
            //return *Mean_c_eff;
        } else {
            return V::Zero();
        }
    }

    M GMMcluster_c::estColorCov(const V& point) const {
        if (track_color){
            return Covariance_c_eff->bottomRightCorner<3,3>() -
                    Covariance_c_eff->bottomLeftCorner<3,3>() * cov_llt.solve(Covariance_c_eff->bottomLeftCorner<3,3>().transpose());
        } else {
            return M::Zero();
        }
    }

    FP GMMcluster_c::estIntensity(const V& point) const {
        // See conversion here: https://www.baeldung.com/cs/convert-rgb-to-grayscale
        V Weights = {0.299f, 0.587f, 0.114f};
        return estColor(point).dot(Weights);
    }

    FP GMMcluster_c::estIntensityVariance(const V& point) const {
        V Weights = {0.299f, 0.587f, 0.114f};
        return Weights.transpose() * estColorCov(point) * Weights;
    }

    V GMMcluster_c::estIntensityInRGB(const V& point) const {
        float intensity = estIntensity(point);
        return {intensity, intensity, intensity};
    }

    void GMMcluster_c::updateMeanAndCovFromIntermediateParamsC(bool free, const V& S_c_eff, const M_c_eff& J_c_eff){
        if (track_color){
            if (free){
                *Mean_c_eff = S_c_eff / W;
                V_c Mean_c;
                Mean_c << mean, *Mean_c_eff;
                *Covariance_c_eff = J_c_eff / W - (Mean_c * Mean_c.transpose()).bottomLeftCorner<3,6>();
            } else {
                *Mean_c_eff = S_c_eff / (FP) N;
                V_c S_c;
                S_c << N * mean, S_c_eff;
                *Covariance_c_eff = (J_c_eff - (S_c * S_c.transpose()).bottomLeftCorner<3,6>() / (FP) N) / ((FP) N - 1.0f);
            }
        }
    }

    void GMMcluster_c::computeIntermediateParamsC(bool free, V& S_c_eff, M_c_eff& J_c_eff) const{
        // Update J and S given mean and covariance
        if (track_color){
            if (!free) {
                S_c_eff = N * *Mean_c_eff;
                J_c_eff.bottomLeftCorner<3,3>() = (N - 1) * Covariance_c_eff->bottomLeftCorner<3,3>() + N * *Mean_c_eff * mean.transpose();
                J_c_eff.bottomRightCorner<3,3>() = (N - 1) * Covariance_c_eff->bottomRightCorner<3,3>() + N * *Mean_c_eff * Mean_c_eff->transpose();
            } else {
                S_c_eff = W * *Mean_c_eff;
                J_c_eff.bottomLeftCorner<3,3>() = W * Covariance_c_eff->bottomLeftCorner<3,3>() + W * *Mean_c_eff * mean.transpose();
                J_c_eff.bottomRightCorner<3,3>() = W * Covariance_c_eff->bottomRightCorner<3,3>() + W * *Mean_c_eff * Mean_c_eff->transpose();
            }
        }
    }

    void GMMcluster_c::transformMeanAndCov(const Isometry3& pose) {
        GMMcluster_o::transformMeanAndCov(pose);
        // Implemented for color transformations
        if (track_color){
            Covariance_c_eff->bottomLeftCorner<3,3>() = Covariance_c_eff->bottomLeftCorner<3,3>() * pose.rotation().transpose();
        }
    }

    void GMMcluster_c::fuseCluster(GMMcluster_c const* cluster, const FP& std){
        // Fuse occupancy and color information
        if (track_color && cluster->track_color){
            //std::cout << "Color fusion is performed!" << std::endl;
            // Note: Computation of the intermediate params must happen before N and W are incremented!
            V S_cur, S_new;
            M_c_eff J_cur, J_new;
            cluster->computeIntermediateParamsC(is_free, S_new, J_new);
            computeIntermediateParamsC(is_free, S_cur, J_cur);

            GMMcluster_o::fuseCluster(cluster, std);
            updateMeanAndCovFromIntermediateParamsC(is_free, S_cur + S_new, J_cur + J_new);
        } else {
            //std::cout << "color not updated!" << std::endl;
            GMMcluster_o::fuseCluster(cluster, std);
        }
    }

    void GMMcluster_c::transform(const Isometry3& pose, const FP& gau_bd_scale) {
        // Recompute mean and covariance
        transformMeanAndCov(pose);

        // Update bounding box
        updateBBox(gau_bd_scale);
    }

    unsigned long long GMMcluster_c::obsGaussianSizeInBytes() {
        return GMMcluster_c::freeGaussianSizeInBytes();
    }

    std::string GMMcluster_c::getObsGaussianName() {
        static std::string obsGaussianName = "ObstacleGaussian";
        return obsGaussianName;
    }

    // metacluster_o
    void GMMmetadata_o::computeMeanAndCovXZ(V_2& Mean, M_2& Covariance, bool free) const {
        // Compute mean and covariance in zx direction only
        V_2 S_2D;
        M_2 J_2D;
        S_2D << S(0), S(2);
        J_2D << J(0,0), J(0, 2),
                J(2,0), J(2, 2);

        if (free){
            Mean = S_2D / W;
            Covariance = J_2D / W - Mean * Mean.transpose();
            Covariance += (1e-5)*M_2::Identity();
        } else {
            Mean = S_2D / N;
            Covariance = 1.0 / (N - 1.0) * (J_2D - (S_2D * S_2D.transpose() / N));
            Covariance += (1e-5)*M_2::Identity();
        }
    }

    void GMMmetadata_o::printInfo() const {
        // Print cluster metadata information
        std::cout.precision(8);
        std::cout << fmt::format("Printing info for a cluster metadata with N: {}, W: {:.4f} as follows:", N, W) << std::endl;
        /*
        Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[","]");
        std::cout << "S:" << std::endl;
        std::cout << S.format(CleanFmt) << std::endl;
        std::cout << "J:" << std::endl;
        std::cout << J.format(CleanFmt) << std::endl;
         */
    }

    unsigned long long GMMmetadata_o::freeMetadataSizeInBytes() {
        auto FP_size = sizeof(FP);
        // For each metadata, we need number of points (N), weight (W), left/right pixel, depth, num_lines, mean, covariance, left/right point
        return (sizeof(N) + sizeof(W) + 2 * sizeof(LeftPixel) + sizeof(depth) + sizeof(NumLines) +
                3 * FP_size + 9 * FP_size + 6 * FP_size);
    }

    std::string GMMmetadata_o::getFreeMetadataName() {
        static std::string freeMetadataName = "FreeMetadata";
        return freeMetadataName;
    }

#ifdef TRACK_MEM_USAGE_GMM
    GMMmetadata_c::GMMmetadata_c(bool track_color) :
        mutil::objTracker(getObsMetadataName(), obsMetadataSizeInBytes(), 1){
#else
    GMMmetadata_c::GMMmetadata_c(bool track_color){
#endif
        this->track_color = track_color;
        if (this->track_color){
            S_c_eff = new V;
            S_c_eff->setZero();
            J_c_eff = new M_c_eff;
            J_c_eff->setZero();
        }
    }

#ifdef TRACK_MEM_USAGE_GMM
    GMMmetadata_c::GMMmetadata_c(const GMMmetadata_c& metadata) : GMMmetadata_o(metadata), mutil::objTracker(metadata){
#else
    GMMmetadata_c::GMMmetadata_c(const GMMmetadata_c& metadata) : GMMmetadata_o(metadata){
#endif
        track_color = metadata.track_color;
        if (track_color){
            S_c_eff = new V;
            *S_c_eff = *metadata.S_c_eff;
            J_c_eff = new M_c_eff;
            *J_c_eff = *metadata.J_c_eff;
        }

        freeBasis = metadata.freeBasis;
        LineVec = metadata.LineVec;
        PlaneVec = metadata.PlaneVec;
        UpMean = metadata.UpMean;
        PlaneLen = metadata.PlaneLen;
        PlaneWidth = metadata.PlaneWidth;
    }

    void swap( GMMmetadata_c& first, GMMmetadata_c& second){
        using std::swap;
        // Note, swapping the base class already takes care of the tracker!
        //std::cout << "Before swapping GMMmetadata_o" << std::endl;
        //std::cout << fmt::format("State before swap GMMmetadata_o - first: {}, second: {}",
        //                         first.isTrackerInitialized(), second.isTrackerInitialized()) << std::endl;
        // Note: It seems that this swap is implemented with one move and two copy assignment constructors (involves uninitialized tracker).
        swap(static_cast<GMMmetadata_o&>(first), static_cast<GMMmetadata_o&>(second));
        //std::cout << "Swapped GMMmetadata_o" << std::endl;
        //std::cout << fmt::format("State after swap GMMmetadata_o - first: {}, second: {}",
        //                         first.isTrackerInitialized(), second.isTrackerInitialized()) << std::endl;
        // Note the free basis is implemented with three move constructors and two assignment constructors to list.
        swap(first.freeBasis, second.freeBasis);
        //std::cout << "Swapped free basis" << std::endl;
        swap(first.S_c_eff, second.S_c_eff);
        swap(first.J_c_eff, second.J_c_eff);

        swap(first.LineVec, second.LineVec);
        swap(first.PlaneVec, second.PlaneVec);
        swap(first.UpMean, second.UpMean);
        swap(first.PlaneLen, second.PlaneLen);
        swap(first.PlaneWidth, second.PlaneWidth);

        // Index of the depth array used for storing depth bounds
        swap(first.track_color, second.track_color);

        //std::cout << "Swapped GMMmetadata_c" << std::endl;
        //std::cout << fmt::format("State after swap GMMmetadata_c - first: {}, second: {}",
        //                         first.isTrackerInitialized(), second.isTrackerInitialized()) << std::endl;
    }

    GMMmetadata_c& GMMmetadata_c::operator=(GMMmetadata_c metadata){
        // See: https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
        swap(*this, metadata);
        return *this;
    }


    GMMmetadata_c::GMMmetadata_c(GMMmetadata_c&& metadata) noexcept{
        //std::cout << fmt::format("State before swap - this: {}, other: {}",
        //                         this->isTrackerInitialized(), metadata.isTrackerInitialized()) << std::endl;
        swap(*this, metadata);
        //std::cout << fmt::format("State after swap - this: {}, other: {}",
        //                         this->isTrackerInitialized(), metadata.isTrackerInitialized()) << std::endl;
    }

    GMMmetadata_c::~GMMmetadata_c(){
        delete S_c_eff;
        delete J_c_eff;
    }

    unsigned long long GMMmetadata_c::obsMetadataSizeInBytes() {
        // Size of free space metadata plus LineVec, PlaneVec, UpMean, PlaneLen, PlaneWidth
        return GMMmetadata_o::freeMetadataSizeInBytes() + (3 * 3) * sizeof(FP) + sizeof(PlaneLen) + sizeof(PlaneWidth);
    }

    std::string GMMmetadata_c::getObsMetadataName() {
        static std::string obsMetadataName = "ObstacleMetadata";
        return obsMetadataName;
    }

    // Free space metadata updates
    V freeSpaceBasis::Mean() const {
        return S_cluster / W_cluster;
    }

    M freeSpaceBasis::Cov() const {
        V mean = Mean();
        return J_cluster / W_cluster  - mean * mean.transpose();
    }

    void freeSpaceBasis::updateBBox(FP scale){
        // Recompute BBox based mean and covariance (very easy!)
        // See: https://math.stackexchange.com/questions/3926884/smallest-axis-aligned-bounding-box-of-hyper-ellipsoid
        V d_pos;
        V cov_diag = Cov().diagonal(0);
        d_pos << sqrtf(cov_diag(0)), sqrtf(cov_diag(1)), sqrtf(cov_diag(2));
        d_pos = scale * d_pos;
        V mean = Mean();
        BBox = Rect(mean - d_pos, mean + d_pos);
    }

    bool freeSpaceBasis::isBBoxIntersect(const freeSpaceBasis &new_data) {
        return BBox.intersects(new_data.BBox);
    }

    void freeSpaceBasis::merge(const freeSpaceBasis &new_data) {
        // Merge metadata associated with free space
        W_ray_ends += new_data.W_ray_ends;
        S_ray_ends += new_data.S_ray_ends;
        J_ray_ends += new_data.J_ray_ends;

        // The back accumulates the bases used to infer all other free space clusters
        W_basis += new_data.W_basis;
        S_basis += new_data.S_basis;
        J_basis += new_data.J_basis;
        depth_idx = std::min<int>(depth_idx, new_data.depth_idx);
        cluster_valid = false;
    }

    void freeSpaceBasis::mergeWithInternalParam(const freeSpaceBasis& new_data, const FP& bbox_scale){
        // Merge at current depth index
        merge(new_data);

        // Fuse PBox and basis
        S_cluster += new_data.S_cluster;
        J_cluster += new_data.J_cluster;
        W_cluster += new_data.W_cluster;
        updateBBox(bbox_scale);
        cluster_valid = true;
    }

    void freeSpaceBasis::updateFreeCluster(const FP &d_near, const FP &d_far, const int &d_index,
                                              GMMmetadata_o &free_cluster) {
        updateFreeClusterParam(d_near, d_far, d_index, free_cluster.W, free_cluster.S, free_cluster.J);
    }

    void freeSpaceBasis::transferFreeClusterParam(GMMmetadata_o &free_cluster) {
        free_cluster.W = W_cluster;
        free_cluster.S = S_cluster;
        free_cluster.J = J_cluster;
    }

    void freeSpaceBasis::updateInternalFreeClusterParam(const FP &d_near, const FP &d_far, const int &d_index, const FP& bbox_scale){
        if (!cluster_valid){
            updateFreeClusterParam(d_near, d_far, d_index, W_cluster, S_cluster, J_cluster);
            updateBBox(bbox_scale);
            //cross_boundary_depth_idx_ub = d_index;

            // Print some debug information
            /*
            std::cout.precision(12);
            std::cout << fmt::format("Internal Free Cluster Param is updated with W_cluster: {:.8f}", W_cluster) << std::endl;
            Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[","]");
            std::cout << "S_cluster:" << std::endl;
            std::cout << S_cluster.format(CleanFmt) << std::endl;
            std::cout << "J_cluster" << std::endl;
            std::cout << J_cluster.format(CleanFmt) << std::endl;
            */
        }
        cluster_valid = true;
    }

    void freeSpaceBasis::updateFreeClusterParam(const FP& d_near, const FP& d_far, const int& d_index,
                                                   FP& W, V& S, M& J){
        // Note that FP results might be different across compilers or CPU architectures (Intel vs. ARM)
        // See: https://randomascii.wordpress.com/2013/07/16/floating-point-determinism/
        // Debug information
        /*
        std::cout.precision(12);
        std::cout << fmt::format("Internal Free Cluster Param is updating with d_near {:.8f}, d_far {:.8f}, W_basis {:.8f}, W_ray_ends {:.8f}:",
                                 d_near, d_far, W_basis, W_ray_ends) << std::endl;
        Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[","]");
        std::cout << "S_ray_ends:" << std::endl;
        std::cout << S_ray_ends.format(CleanFmt) << std::endl;
        std::cout << "J_ray_ends" << std::endl;
        std::cout << J_ray_ends.format(CleanFmt) << std::endl;
        std::cout << "S_basis:" << std::endl;
        std::cout << S_basis.format(CleanFmt) << std::endl;
        std::cout << "J_basis" << std::endl;
        std::cout << J_basis.format(CleanFmt) << std::endl;
        */
        if (d_index == depth_idx) {
            W = W_ray_ends - W_basis * d_near;
            S = S_ray_ends - S_basis * (d_near * d_near);
            J = J_ray_ends - J_basis * (d_near * d_near * d_near);
        } else if (d_index < depth_idx) {
            FP cur_d_pow, pre_d_pow;
            W = W_basis * (d_far - d_near);

            cur_d_pow = d_far * d_far;
            pre_d_pow = d_near * d_near;
            S = S_basis * (cur_d_pow - pre_d_pow);

            cur_d_pow = cur_d_pow * d_far;
            pre_d_pow = pre_d_pow * d_near;
            J = J_basis * (cur_d_pow - pre_d_pow);
        } else {
            throw std::invalid_argument("Current depth index cannot be larger than stored index!");
        }
    }

    void freeSpaceBasis::fuseWithLowerDepth(const FP& d_near, const FP& d_far, const int& d_index, const FP& bbox_scale, const FP& dist_thresh){
        // Use Hellinger's distance to determine we can fuse across boundary
        // Assumes that the current cluster is valid!
        freeSpaceBasis cur_lower_depth_data, fusion_candidate;
        updateFreeClusterParam(d_near, d_far, d_index, cur_lower_depth_data.W_cluster, cur_lower_depth_data.S_cluster, cur_lower_depth_data.J_cluster);
        fusion_candidate.W_cluster = W_cluster + cur_lower_depth_data.W_cluster;
        fusion_candidate.S_cluster = S_cluster + cur_lower_depth_data.S_cluster;
        fusion_candidate.J_cluster = J_cluster + cur_lower_depth_data.J_cluster;
        FP dist = unscentedHellingerSquaredFreeBasis({&fusion_candidate}, {this, &cur_lower_depth_data});
        if (dist <= dist_thresh){
            W_cluster = fusion_candidate.W_cluster;
            S_cluster = fusion_candidate.S_cluster;
            J_cluster = fusion_candidate.J_cluster;
            updateBBox(bbox_scale);
            cluster_cross_depth_boundary = true;
        } else {
            cluster_cross_depth_boundary = false;
        }
    }

    void freeSpaceBasis::computeMeanAndCovXYAndZ(V_2& MeanXZ, M_2& CovXZ, V_2& MeanYZ, M_2& CovYZ) const {
        // Compute projected covariance and mean on the XZ and YZ plane.
        // Compute mean and covariance in zx direction only
        V mean = Mean();
        M cov = Cov();

        MeanXZ << mean(0), mean(2);
        CovXZ << cov(0,0), cov(0, 2),
                cov(2,0), cov(2, 2);

        MeanYZ << mean(1), mean(2);
        CovYZ << cov(1,1), cov(1, 2),
                cov(2,1), cov(2, 2);
    }

    void freeSpaceBasis::printInfo() const {
        // Print cluster information
        std::cout.precision(10);
        std::cout << fmt::format("Printing info for a freeSpaceBasis with W_cluster: {:.8f} as follows:", W_cluster) << std::endl;
        Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[","]");
        std::cout << "Mean:" << std::endl;
        std::cout << Mean().format(CleanFmt) << std::endl;
        std::cout << "Cov:" << std::endl;
        std::cout << Cov().format(CleanFmt) << std::endl;
    }

    unsigned long long freeSpaceBasis::freeBasisSizeInBytes() {
        auto FP_size = sizeof(FP);
        // For each free basis, we have depth index, list iterator, 3 Gaussians, a bounding box
        return (sizeof(depth_idx) + sizeof(list_it) + 3 * (sizeof(W_cluster) + 3 * FP_size + 9 * FP_size) + 6 * FP_size);
    }

    std::string freeSpaceBasis::getFreeBasisName() {
        static std::string freeBasisName = "FreeBasis";
        return freeBasisName;
    }
}