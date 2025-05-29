//
// Created by peter on 4/19/22.
// Contains all visualization APIs related to GMM Maps
//
#ifndef GMM_MAP_VISUALIZATION_H
#define GMM_MAP_VISUALIZATION_H
#include "open3d/Open3D.h"
#include "gmm_map/map.h"
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include "path_planning/path_planner.h"

namespace gmm {

    // Managing point clouds associated with occupancy and its variance
    struct OccVarWithMetadata {
        // Building blocks
        std::vector<Eigen::Vector3d> obs_and_free_pts;
        std::vector<Eigen::Vector3d> occ_colors;
        std::vector<Eigen::Vector3d> var_colors;
        std::vector<bool> error_value;
        std::vector<FP> occ_value;
        std::vector<FP> variance_value;
        FP accuracy = 0;
        FP throughput = 0;
        int total_num_pts = 0;
        bool building_blocks_computed = false; // Track if the building blocks are updated!
        Eigen::Vector3d prob_error_color; // Color for error points
        Eigen::MatrixXd OccVarCMap;

        // Geometries
        std::shared_ptr<open3d::geometry::PointCloud> occ_pcd_original = nullptr; // Point cloud for storing occupancy of samples
        std::shared_ptr<open3d::geometry::PointCloud> occ_var_pcd_original = nullptr; // Point cloud for storing occupancy variance of samples
        std::shared_ptr<open3d::geometry::PointCloud> occ_pcd_cropped = nullptr; // Point cloud for storing occupancy of samples (cropped)
        std::shared_ptr<open3d::geometry::PointCloud> occ_var_pcd_cropped = nullptr; // Point cloud for storing occupancy variance of samples (cropped)

        OccVarWithMetadata(const Eigen::Vector3d& prob_error_color, const Eigen::MatrixXd& OccVarCMap);
        OccVarWithMetadata(const Eigen::MatrixXd& OccVarCMap);
        void clear();
        void cropGeometries(bool crop_existing, const std::shared_ptr<open3d::geometry::OrientedBoundingBox>& env_bbox);
        void generateGeometriesWithErrors(const FP& occ_var_pt_high, const FP& occ_var_pt_low, const Isometry3& pose,
                                          FP& cur_max_variance, FP& cur_min_variance);
        void generateGeometries(const FP& occ_var_pt_high, const FP& occ_var_pt_low, const Isometry3& pose,
                                FP& cur_max_variance, FP& cur_min_variance);
    };

    std::shared_ptr<open3d::geometry::LineSet> GenerateRayLinesetFromDepthmap(const RowMatrixXf& depthmap,
                                                                             int row_idx,
                                                                             const Eigen::Vector3d& color,
                                                                             const std::string& dataset_name);

    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMLinesetColorSingle(const GMMcluster_c& Cluster,
                                                                     FP scale = 1.73, M_o pose = M_o::Identity(),
                                                                     bool colorizeIntensity = false);

    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMLinesetOccSingle(const GMMcluster_o& Cluster,
                                                                   const Eigen::Vector3d& OccupancyColor,
                                                                   bool& completeGeometry,
                                                                   FP scale = 1.73, M_o pose = M_o::Identity());

    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMLinesetColorSingle(const GMMcluster_c& Cluster, GMMMap* map,
                                                                   FP scale = 1.73, M_o pose = M_o::Identity());

    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMLinesetDebugColorSingle(const GMMcluster_c& Cluster,
                                                                     FP scale = 1.73, M_o pose = M_o::Identity());

    // Create a set of bounding boxes for the GMMs
    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMBBoxLinesetSingle(const GMMcluster_c& Cluster, const Eigen::Vector3d& color,
                                                                    FP scale = 1.73, M_o pose = M_o::Identity());
    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMBBoxLinesetSingle(const GMMcluster_o& Cluster, const Eigen::Vector3d& color,
                                                                    FP scale = 1.73, M_o pose = M_o::Identity());

    // Generating Gaussian graph
    std::shared_ptr<open3d::geometry::LineSet> GenerateGaussianGraph(const GMMMap& map, const Eigen::Vector3d& obs_color,
                                                                     const Eigen::Vector3d& free_color, bool free_region_only = false);

    std::shared_ptr<open3d::geometry::LineSet> GenerateNavigationGraph(const GMMapSamplingBasedPlanner& planner, const Eigen::Vector3d& color);

    std::shared_ptr<open3d::geometry::LineSet> GenerateSolutionPath(const GMMapSamplingBasedPlanner& planner, const Eigen::Vector3d& color);

    // Visualization flags to track (used by the slider)
    struct vizMapAtomicFlags {
        // For GMMs (redraw compute geometry)
        std::atomic<FP> occ_low = 0; // Lower bound for occupancy
        std::atomic<FP> occ_high = 1; // Upper bound for occupancy
        std::atomic<FP> std = 2; // Standard deviation for displaying Gaussians
        std::atomic<FP> voxel_resolution = 0.1;

        // For GMMs (update visibility)
        std::atomic<bool> show_gmm_obs = false;
        std::atomic<bool> show_gmm_free = false;
        std::atomic<bool> show_gmm_free_near_obs = false;
        std::atomic<bool> show_gmm_color = false;
        std::atomic<bool> show_gmm_debug_color = false; // Debug color
        std::atomic<bool> show_gmm_bbox_obs = false;
        std::atomic<bool> show_gmm_bbox_free = false;
        std::atomic<bool> show_env_bbox = false;

        // For Occupancy and Variance (recompute geometry)
        std::atomic<FP> occ_var_pt_low = 0; // Controls occupancy and variance pt bounds
        std::atomic<FP> occ_var_pt_high = 1; // Controls occupancy and variance pt bounds

        // For Occupancy and Variance (update visibility)
        std::atomic<bool> show_occupancy_pts = false;
        std::atomic<bool> show_variance_pts = false;
        std::atomic<bool> show_occupancy_voxels = false;
        std::atomic<bool> from_ray = true; // Compute occupancy and variance from ray of bounding box
        std::atomic<int> num_of_points = 100000; // Number of points to compute occupancy and variance (from_ray == false only!)
        std::atomic<int> num_inaccurate_roc_frames; // Number of frames with the lowest accuracy for visualization

        // Bounding Box (Update visibility)
        // Represent scales from 0 to 1
        std::atomic<FP> env_bbox_low_x = 0;
        std::atomic<FP> env_bbox_low_y = 0;
        std::atomic<FP> env_bbox_low_z = 0;
        std::atomic<FP> env_bbox_high_x = 0;
        std::atomic<FP> env_bbox_high_y = 0;
        std::atomic<FP> env_bbox_high_z = 0;
    };

    // Used internally within the visualization object
    struct vizMapFlags {
        // For GMMs (redraw compute geometry)
        FP occ_low = 0; // Lower bound for occupancy
        FP occ_high = 1; // Upper bound for occupancy
        FP std = 2; // Standard deviation for displaying Gaussians
        FP voxel_resolution = 0.1;

        // For GMMs (update visibility)
        bool show_gmm_obs = false;
        bool show_gmm_free = false;
        bool show_gmm_free_near_obs = false;
        bool show_gmm_color = false;
        bool show_gmm_debug_color = false;
        bool show_gmm_bbox_obs = false;
        bool show_gmm_bbox_free = false;
        bool show_env_bbox = false;

        // For Occupancy and Variance (recompute geometry)
        FP occ_var_pt_low = 0; // Controls occupancy and variance pt bounds
        FP occ_var_pt_high = 1; // Controls occupancy and variance pt bounds

        // For Occupancy and Variance (update visibility)
        bool show_occupancy_pts = false;
        bool show_variance_pts = false;
        bool show_occupancy_voxels = false;
        bool from_ray = true; // Compute occupancy and variance from ray of bounding box
        int num_of_points = 100000; // Number of points to compute occupancy and variance (from_ray == false only!)
        int num_inaccurate_roc_frames = 10; // Number of inaccurate frames to visualize (from_ray == true only!)

        // Bounding Box (Update visibility)
        FP env_bbox_low_x = 0;
        FP env_bbox_low_y = 0;
        FP env_bbox_low_z = 0;
        FP env_bbox_high_x = 0;
        FP env_bbox_high_y = 0;
        FP env_bbox_high_z = 0;

        // Update indicator flags
        std::atomic<bool> update_env_geometry = false; // Determines if the bounding box is changed
        std::atomic<bool> update_env_visibility = false; // Determines if the bounding box is visible or not
        std::atomic<bool> update_gmm_visibility = false; // Determines if the GMM should be upgraded
        std::atomic<bool> update_gmm_geometry = false; // Determines if the GMM should be upgraded
        std::atomic<bool> update_occ_var_visibility = false; // Determines if the occupancy or variance points should be recomputed
        std::atomic<bool> update_occ_var_geometry = false; // Determines if the occupancy or variance points should be recomputed
        std::atomic<bool> update_occ_var_building_block = false; // Determines if the building block need to be recomputed
        std::atomic<bool> update_occ_var_source = false; // Determines if the source has changed
        std::atomic<bool> update_occ_voxels_visibility = false; // Determine if the visibility is changed
        std::atomic<bool> update_occ_voxels_geometry = false; // Determines if the source has changed
        std::atomic<bool> update_occ_voxels_source = false; // Determines if the source has changed

        // Update flags given the ones from visualizer
        void updateGMMGeometryFlags(const vizMapAtomicFlags& curFlags);
        void updatePCDGeometryFlags(const vizMapAtomicFlags& curFlags);
        void updateEnvBBoxGeometryFlags(const vizMapAtomicFlags& curFlags);
        void updateVisibilityFlags(const vizMapAtomicFlags& curFlags);
        bool isEnvBBoxGeometryUpdated(const vizMapAtomicFlags& curFlags);

        // Clear flags after updates
        void clearGeometryUpdateFlags();
        void clearVisibilityUpdateFlags();
        // Track visibility
        bool gmmVisible();
        bool occVarVisible();
        bool occVoxelVisible();
    };

    // TODO Add variables associated with the GUI
    struct pathVizAtomicFlags{
        // Selection of start and end location
        std::atomic<gmm::FP> start_scale_x;
        std::atomic<gmm::FP> start_scale_y;
        std::atomic<gmm::FP> start_scale_z;
        std::atomic<gmm::FP> dest_scale_x;
        std::atomic<gmm::FP> dest_scale_y;
        std::atomic<gmm::FP> dest_scale_z;

        // Visibility of geometries
        std::atomic<bool> show_start_and_destinations;
        std::atomic<bool> show_path_from_opt;
        std::atomic<bool> show_path_from_sampling;
        std::atomic<bool> show_edges_to_neighbors;
        std::atomic<bool> show_navigation_graph;

        // Number of samples for sampling-based planning
        std::atomic<int> num_of_samples;

        // Optimization-based planner flags
        std::atomic<bool> show_fpp_offline_graph;
        std::atomic<bool> show_fpp_online_graph;
    };

    // TODO Add flags associated with the GUI
    struct pathVizFlags{
        // Selection of start and end location
        gmm::FP start_scale_x = -1;
        gmm::FP start_scale_y = -1;
        gmm::FP start_scale_z = -1;
        gmm::FP dest_scale_x = -1;
        gmm::FP dest_scale_y = -1;
        gmm::FP dest_scale_z = -1;


        // Flags that should be "enabled" when geometry is changed
        // They are disabled after the geometry is updated in the visualizer
        std::atomic<bool> locations_updated = false;
        std::atomic<bool> gaussians_updated = false;
        std::atomic<bool> navigation_graph_updated = false;

        // Optimization-based planner flags
        std::atomic<bool> optimization_path_updated = false;
        std::atomic<bool> fpp_offline_graph_updated = false;
        std::atomic<bool> fpp_online_graph_updated = false;

        bool updateStartOrDestination(const pathVizAtomicFlags& curFlags);
        void enableLocationUpdateFlag();
        bool isLocationUpdated() const;
        void disableLocationUpdateFlag();

        void enableGaussianUpdateFlag();
        bool areGaussiansUpdated() const;
        void disableGaussianUpdateFlag();

        void enableNavigationGraphUpdateFlag();
        bool isNavigationGraphUpdated() const;
        void disableNavigationUpdateFlag();
    };

    struct ellipsoidWithMetadata {
        // The entire visualizer is build to ensure that the most updated map is rendered before parameters are treaked.
        std::shared_ptr<open3d::geometry::LineSet> color_geometry = nullptr;
        std::shared_ptr<open3d::geometry::LineSet> color_debug_geometry = nullptr;
        std::shared_ptr<open3d::geometry::LineSet> occ_geometry = nullptr;
        std::shared_ptr<open3d::geometry::LineSet> bbox = nullptr;

        // Need to create a new cluster object everytime
        std::shared_ptr<GMMcluster_o> cluster = nullptr;
        FP min_occ = -1;
        FP max_occ = 10;
        FP std_occ_scale = -1;
        FP std_color_scale = -1;
        FP std_debug_color_scale = -1;
        bool completeOccGMM = false; // Dictates whether or not the GMM is visualized in full
        bool updateVisualizer = false; // Dictates whether or not the visualizer should be updated

        void setGeometry(const GMMcluster_o* gaussian_cluster);
        // Returns whether or not the geometry is actually updated
        bool updateOccGeometry(const Eigen::Vector3d& OccupancyColor,const FP& std);
        bool updateColorGeometry(const FP& std, const bool& track_intensity = false);
        bool updateColorGeometry(const FP& std, GMMMap* map);
        bool updateDebugColorGeometry(const FP& std);
    };

    struct PartitionLinesets {
        struct PartitionLinesetMetadata {
            open3d::geometry::LineSet occ_gaussians;
            open3d::geometry::LineSet color_gaussians;
            open3d::geometry::LineSet color_debug_gaussians;
            open3d::geometry::LineSet bboxes;
            std::set<ellipsoidWithMetadata*> cur_lineset_src;
            std::set<ellipsoidWithMetadata*> next_lineset_src;
            bool updateVisualizer;
            void appendLineset(ellipsoidWithMetadata* gau_metadata);
        };

        const int num_gaussians_per_partition = 100; // Number of Gaussians per partition
        std::string occ_gau_partition_base;
        std::string color_gau_partition_base;
        std::string color_debug_gau_partition_base;
        std::string bbox_partition_base;
        int partition_idx = 0;

        std::unordered_map<int, std::set<ellipsoidWithMetadata*>> partition_to_gaussian_map;
        std::unordered_map<ellipsoidWithMetadata*, int> gaussian_to_partition_map;
        std::unordered_map<int, PartitionLinesetMetadata> active_gaussian_linesets;

        void clear();
        void add(ellipsoidWithMetadata* gau_metadata);
        void remove(ellipsoidWithMetadata* gau_metadata);
        void addActive(ellipsoidWithMetadata* gau_metadata, bool updateRequired);
        void updateAllActiveGeometries();
        std::string getOccGaussianPartitionLabel(int idx) const;
        std::string getColorGaussianPartitionLabel(int idx) const;
        std::string getColorDebugGaussianPartitionLabel(int idx) const;
        std::string getBBoxPartitionLabel(int idx) const;
        int findAvailablePartition(int start_idx) const;
        void printInfo() const;
        void checkActiveValidity() const;
    };

    struct GaussianLinesets {
        // Stores all Gaussians and their linesets
        using LinesetRtree = RTree<std::string, FP, 3, FP>;
        // Stores the current set of geometries that could be visualized
        std::unordered_map<std::string, ellipsoidWithMetadata> gaussians;
        // Instantiate an RTree for visibility query
        LinesetRtree linesetRtree;

        // Indicates which geometry falls within the bounding box
        std::unordered_set<std::string> active_obs_gaussians;
        std::unordered_set<std::string> active_free_gaussians;
        std::unordered_set<std::string> active_free_near_obs_gaussians;

        // Gaussian partition information (Visualize groups of Gausssians at a time)
        PartitionLinesets obs_gaussian_partition;
        PartitionLinesets free_gaussian_partition;
        PartitionLinesets free_near_obs_gaussian_partition;

        void updateGaussians(const map_visualization& gmm_changes);

        void clear();
        std::string changePrefix(const std::string& source_label, const std::string& target_prefix);
    };

    class GMMMapViz {
    public:
        std::shared_ptr<GMMMap> gmm_map = nullptr;

        int num_processed_frames;
        std::atomic<int>* update_interval;

        FP cur_max_variance;
        FP cur_min_variance;
        FP ray_sampling_dist;

        // This will be set by the map itself
        std::string gau_obs_base; // Prefix for occupied Gaussian
        std::string gau_color_base; // Prefix for colored occupied Gaussian
        std::string gau_debug_color_base; // Prefic for colored debug Gaussian
        std::string gau_free_base; // Prefix for free Gaussian
        std::string gau_free_near_obs_base; // Prefix for free Gaussian near obstacles
        std::string bbox_obs_base; // Prefix for bbox associated with occupied Gaussian
        std::string bbox_free_base; // Prefix for bbox associated with free Gaussian

        // Following with be set within the visualizer
        std::string env_bbox_base; // Bounding box for the bounding box of the environment
        std::string occ_pcd_base; // Base string for storing occupancy pcd
        std::string occ_var_pcd_base; // Base string for storing variance pcd
        std::string occ_voxel_base; // Base string for storing occupancy voxels

        Eigen::Vector3d prob_zero_color; // Base color for GMMs with occupancy probability of zero
        Eigen::Vector3d prob_one_color; // Base color for GMMs with occupancy probability of one
        Eigen::Vector3d prob_error_color; // Color for error points

        GaussianLinesets gmm_linesets; // Stores all Gaussian geometries

        map_visualization cur_map_updates; // Stores the current updates to the map itself

        std::set<std::string> hidden_clusters; // We use a set to store a list of cluster names that be hidden regardless of other visualization options

        //std::shared_ptr<open3d::geometry::PointCloud> gnd_truth_pcd; // Point cloud that is associated with the current depthmap
        std::shared_ptr<open3d::geometry::OrientedBoundingBox> env_bbox = nullptr; // Bounding box for the current frame

        // Storing BBox information
        Rect map_BBox;
        Rect adj_BBox;
        Rect pre_adj_BBox;

        OccVarWithMetadata* occ_var_from_rays = nullptr;
        OccVarWithMetadata* occ_var_from_bbox = nullptr;

        // Occupancy voxels for visualization
        std::shared_ptr<open3d::geometry::LineSet> occ_voxels = nullptr;
        RTree<std::pair<float, open3d::geometry::LineSet>, FP, 3, FP> occ_voxels_with_prob_rtree;

        Eigen::MatrixXd OccVarCMap;
        std::atomic<FP> unexplored_evidence;
        std::atomic<FP> unexplored_variance;

        // Store the statistics for the currently visible point cloud
        std::atomic<FP> query_accuracy = 0;
        std::atomic<FP> query_throughput = 0;
        std::atomic<int> total_num_query_pts;

        std::shared_ptr<open3d::visualization::gui::SceneWidget> env_widget; // Pointer to visualization object
        open3d::visualization::rendering::MaterialRecord mat;
        open3d::visualization::rendering::MaterialRecord adaptive_pt_mat;
        std::mutex* geometry_lock_ptr;

        // Tracking previous state
        vizMapFlags curMapVizFlags;

        // Flags to determine if a new sets of geometry is avaliable
        std::atomic<bool> map_update_avaliable; // Indicates if the underlying map is updated
        std::atomic<bool> gmm_update_avaliable; // Indicates if the gmm geometry is updated
        std::atomic<bool> occ_var_update_avaliable;
        std::atomic<bool> occ_voxels_update_avaliable;
        std::atomic<bool> env_visibility_update_avaliable;

        GMMMapViz(const std::shared_ptr<open3d::visualization::gui::SceneWidget>& env_widget,
                  const std::shared_ptr<GMMMap>& map,
                  std::atomic<int>* update_interval,
                   const Eigen::Vector3d& prob_error_color,
                   const std::string& colorClass,
                   const std::string& colorName,
                   FP unexplored_evidence = 100000,
                   FP unexplored_variance = 0.25,
                   FP ray_sampling_dist = 0.1);

        // Compute GMM geometries without updating the visualizer!
        void updateGeometrySources();
        void updateGeometry(const vizMapAtomicFlags& sliderFlags, bool final_frame);
        void clearOccVarGeometry();
        void clearAllGeometry();
        void clearGMMGeometry();
        void updateGMM(); // Update GMM geometries based on changes in env bounding boxes
        void redrawOccVarPointCloud(); // Compute occupancy and construct primitive elements
        void redrawOccVarPointCloudGeometry(); // Compute and actual geometric objects
        void redrawEnvBBox();

        // Handle a custom list of hidden GMM objects
        void appendHiddenGMMList(const std::list<GMMcluster_o*>& cluster_list);
        void replaceHiddenGMMList(const std::list<GMMcluster_o*>& cluster_list);
        void clearHiddenGMMList();

        // Update Gaussians and bounding boxes
        void updateVisualizer(const vizMapAtomicFlags& sliderFlags, bool force_gmm_visibility_update = false);
        void updateVisibility(bool update_gmm, bool update_occvar, bool update_env);
        void updatePartitionVisibility(bool update_gmm, bool update_occvar, bool update_occ_voxels, bool update_env);
        void clearVisualizer(bool clear_gmm, bool clear_occvar, bool clear_occ_voxel, bool clear_env);
        void updateGMMVisibility();
        void updateGMMPartitionVisibility();
        void updateOccVarPointCloudVisibility();
        void updateOccVoxelVisibility();
        void updateEnvBBoxVisibility();

        // Check if view bounding box is updated
        bool isEnvBBoxUpdated(const vizMapAtomicFlags& sliderFlags);
        void adjustGlobalBBox(const vizMapAtomicFlags& sliderFlags, Eigen::Vector3d& bbox_min, Eigen::Vector3d& bbox_max);
        bool isMapCropped(const vizMapAtomicFlags& sliderFlags);

        // generate pictures
        void generateCrossSectionOccupancyMap(const int& axis, const FP& value, const std::string& filename,
                                              const int& max_resolution = 500, const FP& border_percentage = 0);

        // Generate occupancy voxels
        void generateOccupancyVoxels(float resolution, bool update_source);
    private:
        std::string changePrefix(const std::string& source_label, const std::string& target_prefix);
    };
}

#endif //GMM_MAP_VISUALIZATION_H
