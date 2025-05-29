//
// Created by peter on 3/17/22.
//

#ifndef GMM_MAPPING_GMMCLUSTERS_H
#define GMM_MAPPING_GMMCLUSTERS_H
#include "commons.h"

#ifdef TRACK_MEM_USAGE_GMM
#include "mem_utils/mem_utils.h"
#endif

namespace gmm {
    // GMM Cluster information
#ifdef TRACK_MEM_USAGE_GMM
    struct GMMcluster_o : public virtual mutil::objTracker {
#else
    struct GMMcluster_o {
#endif
        // Clusters for representing free space
        // Marginalization is easy (just read off appropriate entries from Mean_c and Covariance_c)
        int     N = 0; // Used to track the number of measurements used to construct the Gaussians
        FP  W = 0;

        // Bounding Box Information
        Rect BBox;

        // Joint distribution with occupancy information
        V mean = V::Zero();
        //M covariance = M::Zero();
        Eigen::LLT<M> cov_llt;

        // We pre-compute the two quantities below so that inference becomes fast!
        //M		Inverse_Cov;
        //FP	Abs_Deter_Cov;

        std::list<GMMcluster_o*>::iterator list_it; // Iterator to a list of cluster objects

        std::list<GMMcluster_o*> neighbor_ptrs; // Pointer to the neighbors clusters

        // Visualization label (used for visualization only)
        std::string label;
        std::string bbox_label;
        std::string color_label;
        std::string debug_color_label;

        bool has_label = false; // Indicate whether the label is valid
        bool near_obs = false; // Indicate whether the Gaussian is placed near an obstacle
        bool is_free = false; // Indicate whether the Gaussian represent free or obstacle
        bool has_debug_color = false; // Indicate whether a debug color is assigned
        Eigen::Matrix<uint8_t, 3, 1> debug_color; // Debug color

#ifdef TRACK_MEM_USAGE_GMM
        GMMcluster_o() : mutil::objTracker(getFreeGaussianName(), freeGaussianSizeInBytes(), 1) {};
#endif

        virtual ~GMMcluster_o() = default;

        void printInfo() const;

        V Mean() const;

        M Cov() const;

        FP CovDet() const;

        FP CovDetL() const;

        M CovL() const;

        void estimateLineAndNormal(V& line_vector, V& normal_vector);

        FP estOcc(const V& point) const ;

        FP estOccVariance() const ;

        //void updateInvCov();

        void updateMeanAndCovFromIntermediateParams(bool free, const V& S, const M& J, bool avoid_degeneracy = false);

        void computeIntermediateParams(bool free, V& S, M& J) const;

        void updateBBox(FP scale = 2);

        void computeBBox(Rect& NewBBox, FP scale = 2) const;

        FP computeBBoxVolume(FP scale = 2);

        FP evalGaussianExp(const V& x) const;

        FP evalGaussian(const V& x) const;

        void fuseCluster(GMMcluster_o const* cluster, const FP& std);

        // Static functions for memory tracking
        static unsigned long long freeGaussianSizeInBytes();

        static std::string getFreeGaussianName();

        // Get the number of neighbors
        inline unsigned long numOfNeighbors() const {
            return neighbor_ptrs.size();
        }

        // Get pointers to all neighbors
        inline std::list<GMMcluster_o*> getNeighbors() const {
            return neighbor_ptrs;
        }

        // Get neighboring Gaussians that represent either free or occupied region
        inline std::list<GMMcluster_o*> getNeighbors(bool is_free) const {
            std::list<GMMcluster_o*> result;
            for (auto& cluster : neighbor_ptrs){
                if (cluster->is_free == is_free){
                    result.push_back(cluster);
                }
            }
            return result;
        }

        // Add and remove neighbors
        inline void addNeighbor(GMMcluster_o* cluster) {
            neighbor_ptrs.push_back(cluster);
        }

        inline void removeNeighbor(GMMcluster_o* cluster) {
            neighbor_ptrs.remove(cluster);
        }

        inline void clearNeighbors() {
            neighbor_ptrs.clear();
        }

        // Virtual functions
        virtual void transformMeanAndCov(const Isometry3& pose);

        virtual void transform(const Isometry3& pose, const FP& gau_bd_scale);

        virtual void save(std::ostream& stream) const;

        virtual void load(std::istream& stream);
    };

#ifdef TRACK_MEM_USAGE_GMM
    struct GMMcluster_c : public GMMcluster_o, public virtual mutil::objTracker {
#else
    struct GMMcluster_c : public GMMcluster_o {
#endif
        // Joint distribution with color information (Stored for ease of inference)
        V*		Mean_c_eff = nullptr;
        M_c_eff*		Covariance_c_eff = nullptr;

        bool    track_color = false;

        GMMcluster_c(bool track_color = false);

        GMMcluster_c(GMMcluster_c&& cluster) noexcept;

        GMMcluster_c(const GMMcluster_c& cluster);

        friend void swap( GMMcluster_c& first, GMMcluster_c& second);

        GMMcluster_c& operator=(GMMcluster_c cluster);

        virtual ~GMMcluster_c();

        void updateMeanAndCovFromIntermediateParamsC(bool free, const V& S_c_eff, const M_c_eff& J_c_eff);

        void computeIntermediateParamsC(bool free, V& S_c_eff, M_c_eff& J_c_eff) const;

        void transformMeanAndCov(const Isometry3& pose);

        void transform(const Isometry3& pose, const FP& gau_bd_scale);

        void fuseCluster(GMMcluster_c const* cluster, const FP& std);

        // Color and intensity computation
        V estColor(const V& point) const;

        M estColorCov(const V& point) const;

        FP estIntensity(const V& point) const;

        FP estIntensityVariance(const V& point) const;

        V estIntensityInRGB(const V& point) const;

        void save(std::ostream& stream) const;

        void load(std::istream& stream);

        // Static functions for memory tracking
        static unsigned long long obsGaussianSizeInBytes();

        static std::string getObsGaussianName();

    };

    // Metadata required during clustering
#ifdef TRACK_MEM_USAGE_GMM
    struct GMMmetadata_o : public virtual mutil::objTracker {
#else
    struct GMMmetadata_o {
#endif
        V     S = V::Zero();
        M     J = M::Zero();

        // Note that these only tracks the center of the segments
        V       LeftPoint;
        V       RightPoint;

        int     NumLines = 1;
        int     N = 0;
        FP  W = 0;
        int     LeftPixel = 0;
        int     RightPixel = 0;
        FP  depth = 0; // Current z-distance from the camera

        bool    Updated = false; // Used during segment fusion to track if it is fused with another segment from the following scanline
        bool    near_obs = false; // Flag to track if the free cluster appears near an obstacle cluster
        bool    cur_pt_obs = false; // Flag to check if the current cluster is near an obstacle
        bool    fused = false; // Flag to check if the segment is fused wth other segments

        // Bounding Box Information
        Rect BBox; // Not really used

#ifdef TRACK_MEM_USAGE_GMM
        GMMmetadata_o() : mutil::objTracker(getFreeMetadataName(), freeMetadataSizeInBytes(), 1) {};
#endif
        virtual ~GMMmetadata_o() = default;

        void computeMeanAndCovXZ(V_2& Mean, M_2& Covariance, bool free) const;

        void printInfo() const;

        // Static methods for memory tracking
        static unsigned long long  freeMetadataSizeInBytes();

        static std::string getFreeMetadataName();
    };

    // Metadata needed to track free space
#ifdef TRACK_MEM_USAGE_GMM
    struct freeSpaceBasis : public virtual mutil::objTracker {
#else
    struct freeSpaceBasis {
#endif
        V S_ray_ends = V::Zero();
        M J_ray_ends = M::Zero();
        V S_basis = V::Zero();
        M J_basis = M::Zero();
        FP W_ray_ends = 0;
        FP W_basis = 0;
        int depth_idx = 0;
        int cross_boundary_depth_idx_ub;
        Rect BBox;

        V S_cluster;
        M J_cluster;
        FP W_cluster;
        bool cluster_valid = false;
        bool cluster_cross_depth_boundary = false;
        std::list<freeSpaceBasis*>::iterator list_it;

#ifdef TRACK_MEM_USAGE_GMM
        freeSpaceBasis() : mutil::objTracker(getFreeBasisName(), freeBasisSizeInBytes(), 1) {};
#endif
        V Mean() const;
        M Cov() const;
        bool isBBoxIntersect(const freeSpaceBasis& new_data);
        void updateBBox(FP scale);
        void merge(const freeSpaceBasis& new_data);
        void mergeWithInternalParam(const freeSpaceBasis& new_data, const FP& bbox_scale);
        void transferFreeClusterParam(GMMmetadata_o& free_cluster);
        void updateFreeCluster(const FP& d_near, const FP& d_far, const int& d_index, GMMmetadata_o& free_cluster);
        void updateFreeClusterParam(const FP& d_near, const FP& d_far, const int& d_index, FP& W, V& S, M& J);
        void updateInternalFreeClusterParam(const FP& d_near, const FP& d_far, const int& d_index, const FP& bbox_scale);
        void fuseWithLowerDepth(const FP& d_near, const FP& d_far, const int& d_index, const FP& bbox_scale, const FP& dist_thresh);
        void computeMeanAndCovXYAndZ(V_2& MeanXZ, M_2& CovXZ, V_2& MeanYZ, M_2& CovYZ) const;
        void printInfo() const;

        static unsigned long long freeBasisSizeInBytes();
        static std::string getFreeBasisName();
    };

    // Metadata required during clustering
#ifdef TRACK_MEM_USAGE_GMM
    struct GMMmetadata_c : public GMMmetadata_o, public virtual mutil::objTracker {
#else
    struct GMMmetadata_c : public GMMmetadata_o {
#endif
        V*		S_c_eff = nullptr;
        M_c_eff*		J_c_eff = nullptr;

        V       LineVec = V::Zero();
        V       PlaneVec = V::Zero();
        V       UpMean = V::Zero();
        FP  PlaneLen = 0;
        FP  PlaneWidth = 0;

        // Index of the depth array used for storing depth bounds
        bool    track_color = false;

        // Free space metadata
        freeSpaceBasis freeBasis;

        GMMmetadata_c(bool track_color = false);

        GMMmetadata_c(GMMmetadata_c&& metadata) noexcept;

        GMMmetadata_c(const GMMmetadata_c& metadata);

        friend void swap( GMMmetadata_c& first, GMMmetadata_c& second);

        GMMmetadata_c& operator=(GMMmetadata_c metadata);

        virtual ~GMMmetadata_c();

        // Static methods for memory tracking
        static unsigned long long  obsMetadataSizeInBytes();

        static std::string getObsMetadataName();
    };
}
#endif //GMM_MAPPING_GMMCLUSTERS_H
