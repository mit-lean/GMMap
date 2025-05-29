#ifndef GMM_MAP_H
#define GMM_MAP_H
#include "map_param.h"
#include "cluster_ops.h"
#include "cluster.h"
#include "RTree/RTree.h"
#include <mutex>

namespace gmm {

    using GMMRtree = RTree<GMMcluster_o*, FP, 3, FP>;

    // Objects used for visualization
    struct map_visualization {
        // Number of frames of updates already incorporated
        int num_frames = 0;

        // Average clustering, fusion and total throughput
        FP gmm_clustering_tp = 0;
        FP gmm_fusion_tp = 0;
        FP gmm_mapping_tp = 0;
        int num_free_clusters = 0;
        int num_obs_clusters = 0;

        // Clusters to be updated (label, cluster pair)
        std::unordered_map<std::string, GMMcluster_o> update_free_cluster;
        std::unordered_map<std::string, GMMcluster_c> update_obs_cluster;

        // Clusters to be added (label, cluster pair)
        std::unordered_map<std::string, GMMcluster_o> add_free_cluster;
        std::unordered_map<std::string, GMMcluster_c> add_obs_cluster;

        // Clusters to be deleted (just labels)
        std::list<std::string> delete_free_cluster;
        std::list<std::string> delete_obs_cluster;

        // Clear everything expect throughput and bbox
        void clear(){
            num_frames = 0;
            update_free_cluster.clear();
            update_obs_cluster.clear();
            add_free_cluster.clear();
            add_obs_cluster.clear();
            delete_free_cluster.clear();
            delete_obs_cluster.clear();
        }
    };

    // Objects used to store and update 2D intervals
    struct interval2D {
        std::list<std::pair<int, int>> interval;

        bool isContained(const int& lowerBound, const int& upperBound){
            for (const auto& seg : interval){
                if (lowerBound >= seg.first && upperBound <= seg.second){
                    return true;
                }
            }
            return false;
        }

        void appendInterval(const int& lowerBound, const int& upperBound){
            int cur_lb = lowerBound;
            int cur_ub = upperBound;
            auto it = interval.begin();
            while (it != interval.end()){
                int overlap_interval = std::min(it->second, cur_ub) - std::max(it->first, cur_lb);
                if (overlap_interval >= 0){
                    cur_lb = std::min(it->first, cur_lb);
                    cur_ub = std::max(it->second, cur_ub);
                    it = interval.erase(it);
                } else {
                    it++;
                }
            }
            interval.emplace_back(std::pair(cur_lb, cur_ub));
        }

        void clear(){
            interval.clear();
        }
    };

	class GMMMap {
	public:
        map_param mapParameters;
        GMMRtree rtree;

        // String prefixes for visualization
        std::string gau_obs_base; // Prefix for occupied Gaussian
        std::string gau_free_base; // Prefix for free Gaussian
        std::string gau_free_near_obs_base; // Prefix for free Gaussian near obstacles
        std::string gau_debug_color_base; // Prefix for colored obstacle Gaussians for debugging
        std::string gau_color_base; // Prefix for colored GMM
        std::string bbox_obs_base; // Prefix for bbox associated with occupied Gaussian
        std::string bbox_free_base; // Prefix for bbox associated with free Gaussian
        int gau_obs_idx; // Index for occupied Gaussian
        int gau_free_idx; // Index for occupied Gaussian
        std::mutex geometry_lock;

        // Tracking throughput
        int total_processed_frames;
        FP total_gmm_clustering_latency;
        FP total_gmm_fusion_latency;
        map_visualization cur_visualization_update;

        // Specify dataset and algorithm, and whether or not the map should be visualized
        GMMMap() = default;
        GMMMap(const map_param& param, std::atomic<bool>* update_obs, std::atomic<bool>* update_free,
               std::atomic<bool>* fuse_gmm_across_frames);
        GMMMap(GMMMap&& map) noexcept; // Move constructor
        GMMMap(const GMMMap& map); // Copy constructor
        friend void swap( GMMMap& first, GMMMap& second); // Swap function
        GMMMap& operator=(GMMMap map); // Assignment operator
        virtual ~GMMMap(); // Needs to be virtual to ensure that its operation is correct when inherited
        static std::string getInputVariableName();

        virtual bool isCUDAEnabled(); // Check if the instance is the CUDA version
        // Insert new RGB-D images into the map
        void insertFrame(const RowMatrixXi& r, const RowMatrixXi& g, const RowMatrixXi& b, const RowMatrixXf& depthmap, const Isometry3& pose);
        void insertFrame(const RowMatrixXi& r, const RowMatrixXi& g, const RowMatrixXi& b,
                         const RowMatrixXf& depthmap, const RowMatrixXf& depth_variance, const Isometry3& pose);
        void insertFrame(const RowMatrixXf& depthmap, const Isometry3& pose);
        void insertFrame(const RowMatrixXf& depthmap, const RowMatrixXf& depth_variance, const Isometry3& pose);

        // Fuse newly created Gaussians into the map
        void insertGMMsIntoCurrentMap(const Isometry3& pose,
                                      std::list<GMMcluster_o*>& new_free_clusters,
                                      std::list<GMMcluster_o*>& new_obs_clusters,
                                      const long& clustering_latency);

        // Create rtrees for the current frame
        void transformAndCreateRtrees(const Isometry3& pose, GMMRtree& rtree_free, GMMRtree& rtree_obs,
                                      std::list<GMMcluster_o*>& new_free_clusters, std::list<GMMcluster_o*>& new_obs_clusters);
        // Determine a set of clusters that could be merged with the current frame
        void extractFusionCandidates(const Rect& BBox,
                                     std::list<GMMcluster_o*>& existing_free_clusters, std::list<GMMcluster_o*>& existing_obs_clusters);
        // Cluster fusion (optimized)
        long fuseClusters(const Isometry3& pose, GMMRtree& new_rtree,
                          std::list<GMMcluster_o*>& new_clusters, std::list<GMMcluster_o*>& previous_existing_clusters,
                          std::list<GMMcluster_o*>& clusters_remove, std::list<GMMcluster_o*>& clusters_add, bool is_free);
        // Update the global Rtree at the end of the computation
        long updateGlobalRtree(std::list<GMMcluster_o*>& free_clusters_remove, std::list<GMMcluster_o*>& free_clusters_add,
                               std::list<GMMcluster_o*>& obs_clusters_remove, std::list<GMMcluster_o*>& obs_clusters_add);
        // Decision to fuse clusters
        bool clusterFusionDecision(GMMcluster_o* new_cluster, GMMcluster_o* existing_cluster);
        bool clusterFusionDecision(GMMcluster_o* new_cluster, GMMcluster_o* existing_cluster, GMMcluster_o* fused_cluster);
        bool clusterFusionDecisionOpt(const Isometry3& pose, GMMcluster_o* new_cluster, GMMcluster_o* existing_cluster, GMMcluster_o* fused_cluster);
        // Compute occupancy
        FP computeOccupancy(const V& pt, FP unexplored_evidence) const;
        void computeOccupancyAndVariance(const V& pt, FP& occupancy, FP& variance,
                                         FP unexplored_evidence, FP unexplored_variance) const;
        void estimateMaxOccupancyAndVariance(const V& bbox_min, const V& bbox_max, FP& occupancy, FP& variance,
                                         FP unexplored_evidence, FP unexplored_variance) const;

        // Unlike occupancy, predicted color can be valid or invalid
        bool computeColorAndVariance(const V& pt, V& color, M& variance) const;
        bool computeIntensityAndVariance(const V& pt, FP& intensity, FP& variance) const;
        void computeOccupancyAndVarianceKNN(const V& pt, const int& Nfree, const int& Nobs,
                                            FP& occupancy, FP& variance,
                                            FP unexplored_evidence, FP unexplored_variance) const;
        //void computeOccupancyAndVariance2Pass(const V& pt, FP& occupancy, FP& variance,
        //                                      FP unexplored_evidence, FP unexplored_variance) const;

        // Set visualization queue and number of frames to update the state of the map
        void configureVisualization();
        // Clear all visualization update (called by the visualizer)
        void clearCurVisualizationUpdates();
        // Check for remaining visualization
        bool remainingVisualizationExists();
        // Estimate map size
        void estimateMapSize(FP& cluster_size, FP& rtree_size, int& num_rtree_nodes);
        void estimateMapSize(FP& obs_cluster_size, FP& free_cluster_size, FP& rtree_size, int& num_rtree_nodes);
        FP estimateClusteringTp();
        FP estimateFusionTp();
        FP estimateMappingTp();
        int numOfObsGMMs();
        int numOfFreeGMMs();

        // Clear the entire map
        void clear();
        // Optimized SPGF Extended for more robustness, less parameters, and future SLAM compatibility
        std::list<GMMmetadata_c> extendedSPGFOpt(const RowMatrixXi& r, const RowMatrixXi& g, const RowMatrixXi& b,
                                                 const RowMatrixXf& depthmap);
        std::list<GMMmetadata_c> extendedSPGFOpt(const RowMatrixXi& r, const RowMatrixXi& g, const RowMatrixXi& b,
                                                 const RowMatrixXf& depthmap, const RowMatrixXf& depth_variance);
        std::list<GMMmetadata_c> extendedSPGFOpt(const RowMatrixXf& depthmap);
        std::list<GMMmetadata_c> extendedSPGFOpt(const RowMatrixXf& depthmap, const RowMatrixXf& depth_variance);


        void lineSegmentationExtendedOpt(const RowDepthScanlineXf& scanline_depth,
                                      const RowColorChannelScanlineXi& scanline_r,
                                      const RowColorChannelScanlineXi& scanline_g,
                                      const RowColorChannelScanlineXi& scanline_b,
                                      int row_idx, const frame_param& clustering_params,
                                      std::list<GMMmetadata_c>& cur_obs_line_segments);

        void lineSegmentationExtendedOpt(const RowDepthScanlineXf& scanline_depth,
                                         const RowDepthScanlineXf& scanline_depth_variance,
                                         const RowColorChannelScanlineXi& scanline_r,
                                         const RowColorChannelScanlineXi& scanline_g,
                                         const RowColorChannelScanlineXi& scanline_b,
                                         int row_idx, const frame_param& clustering_params,
                                         std::list<GMMmetadata_c>& cur_obs_line_segments);

        void clusterMergeExtendedOpt(std::list<GMMmetadata_c>& obs_completed_clusters,
                                  std::list<GMMmetadata_c>& obs_incomplete_clusters,
                                  std::list<GMMmetadata_c>& cur_obs_line_segments,
                                  int& obs_numInactiveClusters,
                                  const frame_param& clustering_params,
                                  int row_idx,
                                  bool final_scanline);

        void constructSegmentsFromPointOpt(const V& point, const V& color, int row_idx, int col_idx,
                                        std::list<GMMmetadata_c>& obs_imcomplete_queue, std::list<GMMmetadata_c>& cur_obs_line_segments,
                                        const frame_param& clustering_params);

        void constructSegmentsFromPointOpt(const V& point, const M& covariance, const V& color, int row_idx, int col_idx,
                                           std::list<GMMmetadata_c>& obs_imcomplete_queue, std::list<GMMmetadata_c>& cur_obs_line_segments,
                                           const frame_param& clustering_params);

        std::list<GMMmetadata_o> constructFreeClustersFromObsClusters(std::list<GMMmetadata_c>& obs_completed_clusters);

        void fuseFreeSpaceBasis(std::list<freeSpaceBasis*>& original_freeBasis, std::list<freeSpaceBasis*>& fused_freeBasis,
                                   const int& cur_depth_idx, const int& min_depth_idx);
        void addPointObs(const V& point, const V& color, int v, int u, int depth_idx, GMMmetadata_c& metadata, const frame_param& clustering_params);
        void addPointObs(const V& point, const M& covariance, const V& color, int v, int u, int depth_idx, GMMmetadata_c& metadata, const frame_param& clustering_params);
        void mergeMetadataObs(GMMmetadata_c& source, GMMmetadata_c& destination, const frame_param& clustering_params);
        void transferMetadata2ClusterExtended(std::list<GMMmetadata_c>& obs_completed_metadata, std::list<GMMmetadata_o>& free_completed_metadata,
                                              std::list<GMMcluster_o*>& obs_gaussians, std::list<GMMcluster_o*>& free_gaussians,
                                              const frame_param& clustering_params);

        FP estFreeSegmentOverlapDist(const GMMmetadata_o& pre_seg, const GMMmetadata_o& cur_seg);

        void adaptThreshold(const V& point, FP& line_t_adapt, FP& depth_t_adapt, const frame_param& clustering_params);
        FP distDepth(const V& point, const GMMmetadata_c& metadata);
        FP distLine(const V& point, const GMMmetadata_c& metadata);
        bool onPlane(const V& point, const GMMmetadata_c& metadata, const frame_param& clustering_params);
        FP distDepthEst(const V& point, const GMMmetadata_c& metadata);

        std::list<GMMcluster_o*> kNNBruteForce(const V& point, bool free, int N) const;
        void printGMMs(const std::list<GMMcluster_o *>& clusters) const;
        void printGMMMetadata_o(const std::list<GMMmetadata_o>& metadata) const;
        void printGMMMetadata_c(const std::list<GMMmetadata_c>& metadata) const;

        std::string printStatistics(const int& cur_frame_idx, const int& max_frame_idx, bool print_throughput = true);
        std::string printFrameStatisticsCSV(const int& cur_frame_idx);
        std::string printFrameStatisticsCSV(const int& cur_frame_idx, const double& overall_fps);

        // Save and load
        void save(std::ostream& stream) const;
        void load(std::istream& stream);

        // Incremental visualization update
        void firstClusters(std::list<GMMcluster_o*>::iterator& free_cluster_it,
                           std::list<GMMcluster_o*>::iterator& obstacle_cluster_it);
        bool pushClustersToVisualizationUpdateQueue(std::list<GMMcluster_o*>::iterator& free_cluster_it,
                                                    std::list<GMMcluster_o*>::iterator& obstacle_cluster_it,
                                                    const int& max_num_clusters);

        // Path planning supports
        void computeEdgesToNeighbors();
        void removeEdgesToNeighbors();
        std::vector<GMMcluster_o*> getObsGaussians() const;
        std::vector<GMMcluster_o*> getFreeGaussians() const;
        V relativeToAbsoluteCoordinate(FP x_scale, FP y_scale, FP z_scale) const;
        void getEnvBounds(Rect& bound) const;

	protected:
        // Main data structure for storing the map
        bool enable_visualization;
        int num_free_clusters;
        int num_obstacle_clusters;
        std::atomic<bool>* update_obs_gmm;
        std::atomic<bool>* update_free_gmm;
        std::atomic<bool>* fuse_gmm_across_frames;

        // All objects will be heap allocated
        std::list<GMMcluster_o*> free_clusters;
        std::list<GMMcluster_o*> obstacle_clusters;
	};
}

#endif
