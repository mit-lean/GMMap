#include "gmm_map/map.h"
//#include "dataset_utils/dataset_utils.h"
#include <chrono>
#include <mutex>
#include <atomic>
#include <queue>
#ifdef TRACK_MEM_USAGE
#include "mem_utils/mem_utils.h"
#endif

using namespace std;

namespace gmm {

    GMMMap::GMMMap(const map_param& param,
                   std::atomic<bool>* update_obs, std::atomic<bool>* update_free, std::atomic<bool>* fuse_gmm_across_frames) :
                   mapParameters(param) {
        // Flag for whether visualization of the map is enabled (i.e. whether the queue should be updated)
        enable_visualization = false;

        // Visualization prefixes
        gau_obs_base = "gmm_obs_"; // Prefix for occupied Gaussian
        gau_free_base = "gmm_free_away_obs_"; // Prefix for free Gaussian
        gau_free_near_obs_base = "gmm_free_near_obs_"; // Prefix for free Gaussian near obstacles
        gau_color_base = "gmm_color_"; // Prefix for colored GMM
        gau_debug_color_base = "gmm_debug_color_"; // Prefix for debug colored GMMs
        bbox_obs_base = "bbox_obs_"; // Prefix for bbox associated with occupied Gaussian
        bbox_free_base = "bbox_free_"; // Prefix for bbox associated with free Gaussian

        total_processed_frames = 0;
        total_gmm_clustering_latency = 0;
        total_gmm_fusion_latency = 0;

        num_free_clusters = 0;
        num_obstacle_clusters = 0;

        update_obs_gmm = update_obs;
        update_free_gmm = update_free;
        this->fuse_gmm_across_frames = fuse_gmm_across_frames;
    }

    // Deconstructor
    GMMMap::~GMMMap() {
        // Clears memory used to store clusters on the heap to prevent memory leakage
        this->clear();
    }

    // Swap function
    void swap(GMMMap& first, GMMMap& second) {
        using std::swap;
        swap(first.mapParameters, second.mapParameters);
        swap(first.rtree, second.rtree);

        // String prefixes for visualization
        swap(first.gau_obs_base, second.gau_obs_base); // Prefix for occupied Gaussian
        swap(first.gau_free_base, second.gau_free_base); // Prefix for free Gaussian
        swap(first.gau_free_near_obs_base, second.gau_free_near_obs_base); // Prefix for free Gaussian near obstacles
        swap(first.gau_debug_color_base, second.gau_debug_color_base); // Prefix for colored obstacle Gaussians for debugging
        swap(first.gau_color_base, second.gau_color_base); // Prefix for colored GMM
        swap(first.bbox_obs_base, second.bbox_obs_base); // Prefix for bbox associated with occupied Gaussian
        swap(first.bbox_free_base, second.bbox_free_base); // Prefix for bbox associated with free Gaussian
        swap(first.gau_obs_idx, second.gau_obs_idx); // Index for occupied Gaussian
        swap(first.gau_free_idx, second.gau_free_idx); // Index for occupied Gaussian
        //swap(first.geometry_lock, second.geometry_lock);

        // Tracking throughput
        swap(first.total_processed_frames, second.total_processed_frames);
        swap(first.total_gmm_clustering_latency, second.total_gmm_clustering_latency);
        swap(first.total_gmm_fusion_latency, second.total_gmm_fusion_latency);
        swap(first.cur_visualization_update, second.cur_visualization_update);

        // Main data structure for storing the map
        swap(first.enable_visualization, second.enable_visualization);
        swap(first.num_free_clusters, second.num_free_clusters);
        swap(first.num_obstacle_clusters, second.num_obstacle_clusters);
        swap(first.update_obs_gmm, second.update_obs_gmm);
        swap(first.update_free_gmm, second.update_free_gmm);
        swap(first.fuse_gmm_across_frames, second.fuse_gmm_across_frames);

        // All objects will be heap allocated
        swap(first.free_clusters, second.free_clusters);
        swap(first.obstacle_clusters, second.obstacle_clusters);
    }

    // Deep copy constructor
    GMMMap::GMMMap(const GMMMap& map) {
        mapParameters = map.mapParameters;

        // String prefixes for visualization
        gau_obs_base = map.gau_obs_base; // Prefix for occupied Gaussian
        gau_free_base = map.gau_free_base; // Prefix for free Gaussian
        gau_free_near_obs_base = map.gau_free_near_obs_base; // Prefix for free Gaussian near obstacles
        gau_debug_color_base = map.gau_debug_color_base; // Prefix for colored obstacle Gaussians for debugging
        gau_color_base = map.gau_color_base; // Prefix for colored GMM
        bbox_obs_base = map.bbox_obs_base; // Prefix for bbox associated with occupied Gaussian
        bbox_free_base = map.bbox_free_base; // Prefix for bbox associated with free Gaussian
        gau_obs_idx = map.gau_obs_idx; // Index for occupied Gaussian
        gau_free_idx = map.gau_free_idx; // Index for occupied Gaussian
        //geometry_lock, second.geometry_lock;

        // Tracking throughput
        total_processed_frames = map.total_processed_frames;
        total_gmm_clustering_latency = map.total_gmm_clustering_latency;
        total_gmm_fusion_latency = map.total_gmm_fusion_latency;
        cur_visualization_update = map.cur_visualization_update;

        // Main data structure for storing the map
        enable_visualization = map.enable_visualization;
        num_free_clusters = map.num_free_clusters;
        num_obstacle_clusters = map.num_obstacle_clusters;
        update_obs_gmm = map.update_obs_gmm;
        update_free_gmm = map.update_free_gmm;
        fuse_gmm_across_frames = map.fuse_gmm_across_frames;

        // Copy clusters and construct the rtree
        // TODO: Properly copy the RTree so that the tree is exactly the same
        for (auto& cluster : map.free_clusters) {
            auto new_cluster = new GMMcluster_o(*cluster);
            free_clusters.push_back(new_cluster);
            new_cluster->list_it = std::prev(free_clusters.end(), 1);
            rtree.Insert(new_cluster->BBox, new_cluster);
        }

        for (auto& cluster : map.obstacle_clusters){
            GMMcluster_o* new_cluster = new GMMcluster_c(* dynamic_cast<GMMcluster_c*>(cluster));
            obstacle_clusters.push_back(new_cluster);
            new_cluster->list_it = std::prev(obstacle_clusters.end(), 1);
            rtree.Insert(new_cluster->BBox, new_cluster);
        }
    }

    // Assignment operator
    GMMMap& GMMMap::operator=(GMMMap map) {
        swap(*this, map);
        return *this;
    }

    GMMMap::GMMMap(GMMMap&& map) noexcept {
        swap(*this, map);
    }

    bool GMMMap::isCUDAEnabled() {
        return false;
    }

    void GMMMap::insertFrame(const RowMatrixXi& r, const RowMatrixXi& g, const RowMatrixXi& b, const RowMatrixXf& depthmap, const Isometry3& pose){
        std::list<GMMcluster_o*> new_free_clusters, new_obs_clusters;
        std::list<GMMmetadata_c> new_obs_cluster_metadata;
        std::list<GMMmetadata_o> new_free_cluster_metadata;
        // Insert current frame into the map
        auto clustering_start = std::chrono::steady_clock::now();
        new_obs_cluster_metadata = extendedSPGFOpt(r, g, b, depthmap);
        // Construct free space from obstacle metadata
        if (*update_free_gmm){
            new_free_cluster_metadata = constructFreeClustersFromObsClusters(new_obs_cluster_metadata);
        }
        // Construct final clusters and print statistics
        this->transferMetadata2ClusterExtended(new_obs_cluster_metadata, new_free_cluster_metadata,
                                               new_obs_clusters, new_free_clusters, mapParameters.gmm_frame_param);
        auto clustering_stop = std::chrono::steady_clock::now();
        // Print Statistics
        std::cout << fmt::format("Clusters - Obstacle: {}, Free: {}",
                                 new_obs_clusters.size(), new_free_clusters.size()) << std::endl;
        long clustering_latency = std::chrono::duration_cast<std::chrono::microseconds>(clustering_stop - clustering_start).count();

        insertGMMsIntoCurrentMap(pose, new_free_clusters, new_obs_clusters, clustering_latency);
    }

    void GMMMap::insertFrame(const RowMatrixXi& r, const RowMatrixXi& g, const RowMatrixXi& b,
                             const RowMatrixXf& depthmap, const RowMatrixXf& depth_variance, const Isometry3& pose){
        std::list<GMMcluster_o*> new_free_clusters, new_obs_clusters;
        std::list<GMMmetadata_c> new_obs_cluster_metadata;
        std::list<GMMmetadata_o> new_free_cluster_metadata;
        // Insert current frame into the map
        auto clustering_start = std::chrono::steady_clock::now();
        new_obs_cluster_metadata = extendedSPGFOpt(r, g, b, depthmap, depth_variance);
        // Construct free space from obstacle metadata
        if (*update_free_gmm){
            new_free_cluster_metadata = constructFreeClustersFromObsClusters(new_obs_cluster_metadata);
        }
        // Construct final clusters and print statistics
        this->transferMetadata2ClusterExtended(new_obs_cluster_metadata, new_free_cluster_metadata,
                                               new_obs_clusters, new_free_clusters, mapParameters.gmm_frame_param);
        auto clustering_stop = std::chrono::steady_clock::now();
        // Print Statistics
        std::cout << fmt::format("Clusters - Obstacle: {}, Free: {}",
                                 new_obs_clusters.size(), new_free_clusters.size()) << std::endl;
        long clustering_latency = std::chrono::duration_cast<std::chrono::microseconds>(clustering_stop - clustering_start).count();

        insertGMMsIntoCurrentMap(pose, new_free_clusters, new_obs_clusters, clustering_latency);
    }

    void GMMMap::insertFrame(const RowMatrixXf& depthmap, const Isometry3& pose){
        RowMatrixXi temp;
        insertFrame(temp, temp, temp, depthmap, pose);
    }

    void GMMMap::insertFrame(const RowMatrixXf& depthmap, const RowMatrixXf& depth_variance, const Isometry3& pose){
        RowMatrixXi temp;
        insertFrame(temp, temp, temp, depthmap, depth_variance, pose);
    }

    void GMMMap::insertGMMsIntoCurrentMap(const Isometry3& pose,
                                               std::list<GMMcluster_o*>& new_free_clusters,
                                               std::list<GMMcluster_o*>& new_obs_clusters,
                                               const long& clustering_latency){
        using namespace std::chrono;
        // 1) Create rTrees for newly created obstacle and free clusters within the current frame (transform them into the global frame)
        const std::lock_guard<std::mutex> g_lock(geometry_lock);
        //std::cout << "Before transformation:" << std::endl;
        //printGMMs(new_free_clusters);
        //printGMMs(new_obs_clusters);

        total_gmm_clustering_latency += (FP) clustering_latency;

        auto cluster_fusion_start = std::chrono::steady_clock::now();
        GMMRtree rtree_free, rtree_obs;
        transformAndCreateRtrees(pose, rtree_free, rtree_obs, new_free_clusters, new_obs_clusters);

        // Print cluster info for debug
        //std::cout << "After transformation:" << std::endl;
        //printGMMs(new_free_clusters);
        //printGMMs(new_obs_clusters);

        // 2) Query the global rtree to obtain a set of existing obstacle and free clusters in the map
        std::list<GMMcluster_o*> existing_free_clusters, existing_obs_clusters;
        std::list<GMMcluster_o*> free_clusters_remove, obs_clusters_remove;
        std::list<GMMcluster_o*> free_clusters_add, obs_clusters_add;

        if (*fuse_gmm_across_frames){
            V BBoxLowFree, BBoxHighFree;
            V BBoxLowObs, BBoxHighObs;
            V BBoxLow, BBoxHigh;
            bool BBoxValidFree = rtree_free.GetBounds(BBoxLowFree, BBoxHighFree);
            bool BBoxValidObs = rtree_obs.GetBounds(BBoxLowObs, BBoxHighObs);

            if (BBoxValidFree && BBoxValidObs){
                BBoxLow = BBoxLowFree.cwiseMin(BBoxLowObs);
                BBoxHigh = BBoxHighFree.cwiseMax(BBoxHighObs);
            } else if (BBoxValidFree) {
                BBoxLow = BBoxLowFree;
                BBoxHigh = BBoxHighFree;
            } else if (BBoxValidObs) {
                BBoxLow = BBoxLowObs;
                BBoxHigh = BBoxHighObs;
            } else {
                std::cout << "Bounding box for the current frame is invalid. Abort!" << std::endl;
                return;
            }
            //std::cout << fmt::format("Current Frame Bounding Box: [{:.7f}, {:.7f}, {:.7f}] to [{:.7f}, {:.7f}, {:.7f}]",
            //                         BBoxLow(0), BBoxLow(1), BBoxLow(2), BBoxHigh(0), BBoxHigh(1), BBoxHigh(2)) << std::endl;

            extractFusionCandidates(Rect(BBoxLow, BBoxHigh), existing_free_clusters, existing_obs_clusters);
        } else {
            // Clear existing map
            free_clusters_remove = free_clusters;
            obs_clusters_remove = obstacle_clusters;
        }

        //std::cout << fmt::format("Existing clusters used for fusion - {} obstacles, {} free.", existing_obs_clusters.size(), existing_free_clusters.size()) << std::endl;

        // 3) Determine if the existing clusters can be merged with the newly created clusters. Update the global map accordingly
        auto start = steady_clock::now();
        #pragma omp parallel sections num_threads(mapParameters.num_threads)
        {
            #pragma omp section
            {
                auto free_start = steady_clock::now();
                if (*update_free_gmm) {
                    fuseClusters(pose, rtree_free, new_free_clusters, existing_free_clusters,free_clusters_remove, free_clusters_add, true);
                }
                auto free_end = steady_clock::now();
                //std::cout << fmt::format("Fuse free cluster - id = {}, duration: {}us",
                //                         omp_get_thread_num(), duration_cast<std::chrono::microseconds>(free_end - free_start).count()) << std::endl;
            }

            #pragma omp section
            {
                auto obs_start = steady_clock::now();
                if (*update_obs_gmm){
                    fuseClusters(pose, rtree_obs, new_obs_clusters, existing_obs_clusters, obs_clusters_remove, obs_clusters_add, false);
                }
                auto obs_end = steady_clock::now();
                //std::cout << fmt::format("Fuse obstacle cluster - id = {}, duration: {}us",
                //                         omp_get_thread_num(), duration_cast<std::chrono::microseconds>(obs_end - obs_start).count()) << std::endl;
            }
        }
        auto end = steady_clock::now();
        //std::cout << fmt::format("Fuse obstacle and free cluster - duration: {}us",
        //                         duration_cast<std::chrono::microseconds>(end - start).count()) << std::endl;

        updateGlobalRtree(free_clusters_remove, free_clusters_add, obs_clusters_remove, obs_clusters_add);
        auto cluster_fusion_stop = std::chrono::steady_clock::now();
        total_gmm_fusion_latency += std::chrono::duration_cast<std::chrono::microseconds>(cluster_fusion_stop - cluster_fusion_start).count();

        //std::cout << fmt::format("Number of free clusters removed: {}, Number of free clusters added: {}", free_clusters_remove.size(), free_clusters_add.size()) << std::endl;
        //std::cout << fmt::format("Number of obs clusters removed: {}, Number of obs clusters added: {}", obs_clusters_remove.size(), obs_clusters_add.size()) << std::endl;

        // Update throughput statistics
        total_processed_frames++;

        // Push map changes to the visualization queue
        if (enable_visualization){
            cur_visualization_update.num_frames++;
            // Update throughput statistics
            cur_visualization_update.gmm_clustering_tp = total_processed_frames * 1000000.0 / total_gmm_clustering_latency;
            cur_visualization_update.gmm_fusion_tp = total_processed_frames * 1000000.0 / total_gmm_fusion_latency;
            cur_visualization_update.gmm_mapping_tp = total_processed_frames * 1000000.0 / (total_gmm_clustering_latency + total_gmm_fusion_latency);
            cur_visualization_update.num_free_clusters = num_free_clusters;
            cur_visualization_update.num_obs_clusters = num_obstacle_clusters;
        }
    }

    void GMMMap::transformAndCreateRtrees(const Isometry3& pose,
                                          GMMRtree &rtree_free, GMMRtree &rtree_obs,
                                          std::list<GMMcluster_o *> &new_free_clusters,
                                          std::list<GMMcluster_o *> &new_obs_clusters) {
        /*
        std::cout.precision(10);
        Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[","]");
        std::cout << "Transformation Matrix: " << std::endl;
        std::cout << pose.matrix().format(CleanFmt) << std::endl;
        std::cout << "Rotation Matrix: " << std::endl;
        std::cout << pose.rotation().format(CleanFmt) << std::endl;
         */
        // Transform obstacles and free clusters into the global frame and create rtrees
        #pragma omp parallel sections num_threads(mapParameters.num_threads)
        {
            #pragma omp section
            {
                for (auto &new_cluster: new_free_clusters) {
                    // Transform and insert into the map
                    new_cluster->transform(pose, mapParameters.gau_rtree_bd);
                    if (mapParameters.gau_rtree_bd != mapParameters.gau_fusion_bd) {
                        Rect BBox;
                        new_cluster->computeBBox(BBox, mapParameters.gau_fusion_bd);
                        rtree_free.Insert(BBox, new_cluster);
                    } else {
                        rtree_free.Insert(new_cluster->BBox, new_cluster);
                    }
                }
            }

            #pragma omp section
            {
                for (auto &new_cluster: new_obs_clusters) {
                    // Transform and insert into the map
                    new_cluster->transform(pose, mapParameters.gau_rtree_bd);
                    if (mapParameters.gau_rtree_bd != mapParameters.gau_fusion_bd) {
                        Rect BBox;
                        new_cluster->computeBBox(BBox, mapParameters.gau_fusion_bd);
                        rtree_obs.Insert(BBox, new_cluster);
                    } else {
                        rtree_obs.Insert(new_cluster->BBox, new_cluster);
                    }
                }
            }
        }
    }

    void GMMMap::extractFusionCandidates(const Rect& BBox,
                                         std::list<GMMcluster_o *> &existing_free_clusters,
                                         std::list<GMMcluster_o *> &existing_obs_clusters) {
        // Given bounding box (BBoxLow, BBoxHigh), determine the sets of free and obstacle clusters that fall within
        auto existing_clusters = rtree.Search(BBox);
        for (auto cluster : existing_clusters){
            if (cluster->is_free){
                existing_free_clusters.push_back(cluster);
            } else {
                existing_obs_clusters.push_back(cluster);
            }
        }
    }

    long GMMMap::fuseClusters(const Isometry3& pose, GMMRtree& new_rtree, std::list<GMMcluster_o*>& new_clusters,
                              std::list<GMMcluster_o*>& previous_existing_clusters,
                                std::list<GMMcluster_o*>& clusters_remove, std::list<GMMcluster_o*>& clusters_add, bool is_free){
        // Determine if each existing cluster could be merged with some clusters in the new rtree
        // The global rtree will be updated as well
        // Variables (new_rtree, new_clusters) are newly generated clusters from the current frame.
        // Variables (previous_existing_clusters) contains existing clusters that could be merged with the new clusters
        // Variables (clusters_remove, clusters_add) are used to update the global rtree

        /*
        if (is_free && mapParameters.cur_debug_frame){
            std::cout << "A list of previous existing free clusters:" << std::endl;
            int idx = 0;
            for (auto cluster : previous_existing_clusters){
                std::cout << fmt::format("Previous cluster: {} ", idx) << std::endl;
                cluster->printInfo();
                idx++;
            }
        }
         */

        long duration = 0;
        chrono::steady_clock::time_point cluster_fusion_start, cluster_fusion_stop;
        cluster_fusion_start = std::chrono::steady_clock::now();
        // See if the existing clusters could be fused
        for (auto existing_cluster : previous_existing_clusters){
            // Determines a set of cluster candidate that could be fused
            Rect BBox;
            if (mapParameters.gau_fusion_bd != mapParameters.gau_rtree_bd){
                existing_cluster->computeBBox(BBox, mapParameters.gau_fusion_bd);
            } else {
                BBox = existing_cluster->BBox;
            }
            auto new_cluster_candidates = new_rtree.Search(BBox);

            GMMcluster_o* fused_cluster = nullptr;
            std::list<GMMcluster_o*>::iterator orig_new_cluster_it;
            bool mergeCluster;

            for (auto new_cluster_candidate : new_cluster_candidates){
                // Allocated temporary memory for storing the fused cluster candidate
                GMMcluster_o* fused_cluster_candidate;
                if (is_free){
                    fused_cluster_candidate = new GMMcluster_o;
                } else {
                    //std::cout << "Temp obs cluster is constructed" << std::endl;
                    fused_cluster_candidate = new GMMcluster_c(mapParameters.track_color);
                }

                // First check if the existing cluster is similar to (unfused) new cluster.
                if (fused_cluster == nullptr){
                    // Check if the existing cluster can be fused with the candidate
                    mergeCluster = clusterFusionDecisionOpt(pose, new_cluster_candidate, existing_cluster, fused_cluster_candidate);
                } else {
                    // Check if the existing cluster can be fused with the fused_cluster
                    mergeCluster = clusterFusionDecisionOpt(pose, fused_cluster, new_cluster_candidate,fused_cluster_candidate);
                }

                if (mergeCluster){
                    // Remove from the local rtree
                    if (mapParameters.gau_fusion_bd != mapParameters.gau_rtree_bd){
                        new_cluster_candidate->computeBBox(BBox, mapParameters.gau_fusion_bd);
                    } else {
                        BBox = new_cluster_candidate->BBox;
                    }
                    new_rtree.Remove(BBox, new_cluster_candidate);
                    if (fused_cluster != nullptr){
                        new_clusters.erase(new_cluster_candidate->list_it);
                        delete fused_cluster;
                    }
                    delete new_cluster_candidate;
                    // Accept the candidate
                    fused_cluster = fused_cluster_candidate;
                    *fused_cluster->list_it = fused_cluster;
                } else {
                    // Reject the candidate
                    delete fused_cluster_candidate;
                }
            }

            // Existing cluster is merged with a new cluster. Thus, it will be removed from the global rtree.
            // In addition, the fused cluster will be re-inserted back into the rtree
            if (fused_cluster != nullptr){
                clusters_remove.push_back(existing_cluster);

                // Insert fused_cluster back into the local rtree
                if (mapParameters.gau_fusion_bd != mapParameters.gau_rtree_bd){
                    fused_cluster->computeBBox(BBox, mapParameters.gau_fusion_bd);
                } else {
                    BBox = fused_cluster->BBox;
                }
                new_rtree.Insert(BBox, fused_cluster);
            }
        }

        // Insert new (fused) clusters into the global rtree
        clusters_add.splice(clusters_add.end(), new_clusters);

        /*
        if (is_free && mapParameters.cur_debug_frame){
            std::cout << "A list of clusters to be removed." << std::endl;
            int idx = 0;
            for (auto cluster : clusters_remove){
                std::cout << fmt::format("Remove cluster: {} ", idx) << std::endl;
                cluster->printInfo();
                idx++;
            }
            std::cout << "A list of clusters to be added." << std::endl;
            idx = 0;
            for (auto cluster : clusters_add){
                std::cout << fmt::format("Add cluster: {} ", idx) << std::endl;
                cluster->printInfo();
                idx++;
            }
        }
         */

        cluster_fusion_stop = std::chrono::steady_clock::now();
        duration += std::chrono::duration_cast<std::chrono::microseconds>(cluster_fusion_stop - cluster_fusion_start).count();
        return duration;
    }

    long GMMMap::updateGlobalRtree(std::list<GMMcluster_o*>& free_clusters_remove, std::list<GMMcluster_o*>& free_clusters_add,
                           std::list<GMMcluster_o*>& obs_clusters_remove, std::list<GMMcluster_o*>& obs_clusters_add){
        // Update the RTree and create visualization objects if enabled
        auto cluster_fusion_start = std::chrono::steady_clock::now();
        // Remove free Gaussians
        for (auto existing_cluster : free_clusters_remove){
            rtree.Remove(existing_cluster->BBox, existing_cluster);
            free_clusters.erase(existing_cluster->list_it);
            if (enable_visualization){
                cur_visualization_update.delete_free_cluster.push_back(existing_cluster->label);
            }
            delete existing_cluster;
            num_free_clusters--;
        }

        // Add free Gaussians
        for (auto new_cluster : free_clusters_add){
            rtree.Insert(new_cluster->BBox, new_cluster);
            free_clusters.push_back(new_cluster);
            new_cluster->list_it = std::prev(free_clusters.end());
            num_free_clusters++;

            if (enable_visualization){
                // Assign visualization label
                if (!new_cluster->has_label) {
                    new_cluster->has_label = true;
                    if (new_cluster->near_obs){
                        new_cluster->label = gau_free_near_obs_base + std::to_string(gau_free_idx);
                        new_cluster->bbox_label = bbox_free_base + std::to_string(gau_free_idx);
                        gau_free_idx++;
                    } else {
                        new_cluster->label = gau_free_base + std::to_string(gau_free_idx);
                        new_cluster->bbox_label = bbox_free_base + std::to_string(gau_free_idx);
                        gau_free_idx++;
                    }
                }
                // Insert into the visualization queue
                cur_visualization_update.add_free_cluster.insert(std::pair<std::string, GMMcluster_o>(new_cluster->label, *new_cluster));
            }
        }

        // Remove obstacle Gaussians
        for (auto existing_cluster : obs_clusters_remove){
            rtree.Remove(existing_cluster->BBox, existing_cluster);
            obstacle_clusters.erase(existing_cluster->list_it);
            if (enable_visualization){
                cur_visualization_update.delete_obs_cluster.push_back(existing_cluster->label);
            }
            delete existing_cluster;
            num_obstacle_clusters--;
        }

        // Add obstacle clusters
        for (auto new_cluster : obs_clusters_add){
            rtree.Insert(new_cluster->BBox, new_cluster);
            obstacle_clusters.push_back(new_cluster);
            new_cluster->list_it = std::prev(obstacle_clusters.end());
            num_obstacle_clusters++;

            if (enable_visualization){
                // Assign visualization label
                if (!new_cluster->has_label) {
                    new_cluster->has_label = true;
                    new_cluster->label = gau_obs_base + std::to_string(gau_obs_idx);
                    new_cluster->bbox_label = bbox_obs_base + std::to_string(gau_obs_idx);
                    new_cluster->color_label = gau_color_base + std::to_string(gau_obs_idx);
                    gau_obs_idx++;
                }

                // Insert into the visualization queue
                cur_visualization_update.add_obs_cluster.insert(std::pair<std::string, GMMcluster_c>(new_cluster->label,
                                                                                                     *reinterpret_cast<GMMcluster_c*>(new_cluster)));
            }
        }
        auto cluster_fusion_stop = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(cluster_fusion_stop - cluster_fusion_start).count();
    }

    bool GMMMap::clusterFusionDecision(GMMcluster_o *new_cluster, GMMcluster_o *existing_cluster) {
        // Fuse the clusters and see if the fused cluster can well represent both clusters!
        GMMcluster_o fused_cluster = *new_cluster;
        fused_cluster.fuseCluster(existing_cluster, mapParameters.gmm_frame_param.gau_bd_scale);

        // See if the fused cluster can well represent its components
        std::vector<GMMcluster_o*> a_clusters{&fused_cluster};
        std::vector<GMMcluster_o*> b_clusters{new_cluster, existing_cluster};

        //FP dist_hell_sq = clusterHellDistanceSquared(&fused_cluster, new_cluster, existing_cluster);
        FP dist_hell_sq = unscentedHellingerSquaredClusters(a_clusters, b_clusters);

        Rect new_bbox, existing_bbox;
        Rect intersection_bbox, union_bbox;

        new_cluster->computeBBox(new_bbox, mapParameters.gau_fusion_bd);
        existing_cluster->computeBBox(existing_bbox, mapParameters.gau_fusion_bd);
        intersection_bbox = new_bbox.intersection(existing_bbox);
        union_bbox = new_bbox.merged(existing_bbox);
        //std::cout << fmt::format("Threshold weight scale: {:.4f}, Threshold volume scale: {:.4f}, Threshold value: {:.4f}", thresh_scale_weight, thresh_scale_volume,
        //                         final_thresh_scale) << std::endl;

        if (fused_cluster.is_free){
            FP final_thresh_scale = intersection_bbox.volume() / union_bbox.volume();
            return dist_hell_sq <= fmax(pow(final_thresh_scale, 2.0f) * mapParameters.hell_thresh_squard_free, mapParameters.hell_thresh_squard_min);
        } else {
            V intersection_bbox_extent = intersection_bbox.sizes().cast<FP>();
            V union_bbox_extent = union_bbox.sizes().cast<FP>();
            std::vector<FP> bbox_intersection_vec = {intersection_bbox_extent(0), intersection_bbox_extent(1), intersection_bbox_extent(2)};
            std::vector<FP> bbox_union_vec = {union_bbox_extent (0) , union_bbox_extent(1), union_bbox_extent(2)};

            sort(bbox_intersection_vec.begin(), bbox_intersection_vec.end());
            sort(bbox_union_vec.begin(), bbox_union_vec.end());
            FP final_thresh_scale = (bbox_intersection_vec.at(1) * bbox_intersection_vec.at(2)) /
                                        (bbox_union_vec.at(1) * bbox_union_vec.at(2));
            return dist_hell_sq <= fmax(fmin(pow(final_thresh_scale, 2.0f) * mapParameters.hell_thresh_squard_obs, 0.9f), mapParameters.hell_thresh_squard_min);
        }
    }

    bool GMMMap::clusterFusionDecision(GMMcluster_o *new_cluster, GMMcluster_o *existing_cluster, GMMcluster_o* fused_cluster) {
        // Fuse the clusters and see if the fused cluster can well represent both clusters!
        if (existing_cluster->is_free || new_cluster->is_free){
            *fused_cluster = *new_cluster;
            fused_cluster->fuseCluster(existing_cluster, mapParameters.gmm_frame_param.gau_bd_scale);
            /*
            if (mapParameters.cur_debug_frame){
                std::cout << "Attempt to fuse the following two clusters:" << std::endl;
                new_cluster->printInfo();
                existing_cluster->printInfo();
            }
             */
        } else {
            *reinterpret_cast<GMMcluster_c *>(fused_cluster) = *reinterpret_cast<GMMcluster_c *>(new_cluster);
            reinterpret_cast<GMMcluster_c *>(fused_cluster)->fuseCluster(reinterpret_cast<GMMcluster_c *>(existing_cluster), mapParameters.gmm_frame_param.gau_bd_scale);
        }

        Rect new_bbox, existing_bbox, fused_bbox;
        Rect intersection_bbox, new_fused_intersection_bbox, existing_fused_intersection_bbox, union_bbox;
        if (mapParameters.gau_fusion_bd != mapParameters.gau_rtree_bd){
            new_cluster->computeBBox(new_bbox, mapParameters.gau_fusion_bd);
            existing_cluster->computeBBox(existing_bbox, mapParameters.gau_fusion_bd);
            fused_cluster->computeBBox(fused_bbox, mapParameters.gau_fusion_bd);
        } else {
            new_bbox = new_cluster->BBox;
            existing_bbox = existing_cluster->BBox;
            fused_bbox = fused_cluster->BBox;
        }
        intersection_bbox = new_bbox.intersection(existing_bbox);
        union_bbox = new_bbox.merged(existing_bbox);

        if (fused_bbox.sizes().maxCoeff() > mapParameters.max_bbox_len){
            // We will run into some cases where there are many overlapping clusters that are off by a little bit.
            // In this case, the free space will grow drastically.
            // Thus, we still allow clusters to grow even if it reaches the maximum desirsed size!
            FP hell_overlap_squared = hellingerSquared(new_cluster->Mean(), existing_cluster->Mean(), new_cluster->Cov(), existing_cluster->Cov());
            if (fused_cluster->is_free){
                return hell_overlap_squared <= fmin(mapParameters.hell_thresh_squard_free, mapParameters.hell_thresh_squard_oversized_gau);
            } else {
                return hell_overlap_squared <= fmin(fmin(mapParameters.hell_thresh_squard_obs, 0.9f), mapParameters.hell_thresh_squard_oversized_gau);
            }
        } else {
            // See if the fused cluster can well represent its components
            std::vector<GMMcluster_o*> a_clusters{fused_cluster};
            std::vector<GMMcluster_o*> b_clusters{new_cluster, existing_cluster};

            //FP dist_hell_sq = clusterHellDistanceSquared(&fused_cluster, new_cluster, existing_cluster);
            FP dist_hell_sq = unscentedHellingerSquaredClusters(a_clusters, b_clusters);

            //std::cout << fmt::format("Threshold weight scale: {:.4f}, Threshold volume scale: {:.4f}, Threshold value: {:.4f}", thresh_scale_weight, thresh_scale_volume,
            //                         final_thresh_scale) << std::endl;

            if (fused_cluster->is_free){
                FP final_thresh_scale =intersection_bbox.volume() / union_bbox.volume();
                /*
                if (mapParameters.cur_debug_frame){
                    std::cout << fmt::format("Free cluster fusion info - Hellinger: {:.4f}, thresh: {:.4f}",
                                             dist_hell_sq, fmax(pow(final_thresh_scale, 2) * mapParameters.hell_thresh_squard_free, mapParameters.hell_thresh_squard_min)) << std::endl;
                }
                 */
                return dist_hell_sq <= fmax(pow(final_thresh_scale, 2.0f) * mapParameters.hell_thresh_squard_free, mapParameters.hell_thresh_squard_min);
            } else {
                V intersection_bbox_extent = intersection_bbox.sizes().cast<FP>();
                V union_bbox_extent = union_bbox.sizes().cast<FP>();
                std::vector<FP> bbox_intersection_vec = {intersection_bbox_extent(0), intersection_bbox_extent(1), intersection_bbox_extent(2)};
                std::vector<FP> bbox_union_vec = {union_bbox_extent (0) , union_bbox_extent(1), union_bbox_extent(2)};

                sort(bbox_intersection_vec.begin(), bbox_intersection_vec.end());
                sort(bbox_union_vec.begin(), bbox_union_vec.end());
                FP final_thresh_scale = (bbox_intersection_vec.at(1) * bbox_intersection_vec.at(2)) /
                                            (bbox_union_vec.at(1) * bbox_union_vec.at(2));
                return dist_hell_sq <= fmax(fmin(pow(final_thresh_scale, 2.0f) * mapParameters.hell_thresh_squard_obs, 0.9f), mapParameters.hell_thresh_squard_min);
            }
        }
    }

    bool GMMMap::clusterFusionDecisionOpt(const Isometry3& pose, GMMcluster_o *new_cluster, GMMcluster_o *existing_cluster, GMMcluster_o* fused_cluster) {
        // Fuse the clusters and see if the fused cluster can well represent both clusters!
        if (existing_cluster->is_free || new_cluster->is_free){
            *fused_cluster = *new_cluster;
            fused_cluster->fuseCluster(existing_cluster, mapParameters.gmm_frame_param.gau_bd_scale);
            /*
            if (mapParameters.cur_debug_frame){
                std::cout << "Attempt to fuse the following two clusters:" << std::endl;
                new_cluster->printInfo();
                existing_cluster->printInfo();
            }
             */
        } else {
            *reinterpret_cast<GMMcluster_c *>(fused_cluster) = *reinterpret_cast<GMMcluster_c *>(new_cluster);
            reinterpret_cast<GMMcluster_c *>(fused_cluster)->fuseCluster(reinterpret_cast<GMMcluster_c *>(existing_cluster), mapParameters.gmm_frame_param.gau_bd_scale);
        }

        Rect new_bbox, existing_bbox, fused_bbox;
        Rect intersection_bbox, new_fused_intersection_bbox, existing_fused_intersection_bbox, union_bbox;
        if (mapParameters.gau_fusion_bd != mapParameters.gau_rtree_bd){
            new_cluster->computeBBox(new_bbox, mapParameters.gau_fusion_bd);
            existing_cluster->computeBBox(existing_bbox, mapParameters.gau_fusion_bd);
            fused_cluster->computeBBox(fused_bbox, mapParameters.gau_fusion_bd);
        } else {
            new_bbox = new_cluster->BBox;
            existing_bbox = existing_cluster->BBox;
            fused_bbox = fused_cluster->BBox;
        }
        intersection_bbox = new_bbox.intersection(existing_bbox);
        union_bbox = new_bbox.merged(existing_bbox);

        if (fused_bbox.sizes().maxCoeff() > mapParameters.max_bbox_len){
            // We will run into some cases where there are many overlapping clusters that are off by a little bit.
            // In this case, the free space will grow drastically.
            // Thus, we still allow clusters to grow even if it reaches the maximum desirsed size!
            FP hell_overlap_squared = hellingerSquared(new_cluster->Mean(), existing_cluster->Mean(), new_cluster->Cov(), existing_cluster->Cov());
            if (fused_cluster->is_free){
                return hell_overlap_squared <= fmin(mapParameters.hell_thresh_squard_free, mapParameters.hell_thresh_squard_oversized_gau);
            } else {
                return hell_overlap_squared <= fmin(fmin(mapParameters.hell_thresh_squard_obs, 0.9f), mapParameters.hell_thresh_squard_oversized_gau);
            }
        } else {
            // See if the fused cluster can well represent its components
            std::vector<GMMcluster_o*> a_clusters{fused_cluster};
            std::vector<GMMcluster_o*> b_clusters{new_cluster, existing_cluster};

            //FP dist_hell_sq = clusterHellDistanceSquared(&fused_cluster, new_cluster, existing_cluster);
            FP dist_hell_sq = unscentedHellingerSquaredClusters(a_clusters, b_clusters);

            //std::cout << fmt::format("Threshold weight scale: {:.4f}, Threshold volume scale: {:.4f}, Threshold value: {:.4f}", thresh_scale_weight, thresh_scale_volume,
            //                         final_thresh_scale) << std::endl;

            if (fused_cluster->is_free){
                FP final_thresh_scale =intersection_bbox.volume() / union_bbox.volume();
                /*
                if (mapParameters.cur_debug_frame){
                    std::cout << fmt::format("Free cluster fusion info - Hellinger: {:.4f}, thresh: {:.4f}",
                                             dist_hell_sq, fmax(pow(final_thresh_scale, 2) * mapParameters.hell_thresh_squard_free, mapParameters.hell_thresh_squard_min)) << std::endl;
                }
                 */
                return dist_hell_sq <= fmax(pow(final_thresh_scale, 2.0f) * mapParameters.hell_thresh_squard_free, mapParameters.hell_thresh_squard_min);
            } else {
                FP final_thresh_scale =intersection_bbox.volume() / union_bbox.volume();
                if (dist_hell_sq <=
                    fmax(fmin(pow(final_thresh_scale, 2.0f) * mapParameters.hell_thresh_squard_obs, 0.9f),
                         mapParameters.hell_thresh_squard_min)){
                    return true;
                } else {
                    // We try to use a more robust approach to determine fusion of obstacle clusters via eigen-decomposition
                    // Eigenvalues are ordered from smallest to largest
                    M new_evecs, exisiting_evecs;
                    V new_evals, exisiting_evals;
                    eigenSymmP(new_cluster->Cov(), new_evecs, new_evals);
                    eigenSymmP(existing_cluster->Cov(), exisiting_evecs, exisiting_evals);
                    // We square this quantity to obtain a better scaling curve (instead of acos!)
                    FP normal_similarity = std::abs(new_evecs.col(0).dot(exisiting_evecs.col(0)));

                    V intersection_bbox_extent = intersection_bbox.sizes().cast<FP>();
                    V union_bbox_extent = union_bbox.sizes().cast<FP>();
                    std::vector<FP> bbox_intersection_vec = {intersection_bbox_extent(0), intersection_bbox_extent(1), intersection_bbox_extent(2)};
                    std::vector<FP> bbox_union_vec = {union_bbox_extent (0) , union_bbox_extent(1), union_bbox_extent(2)};
                    sort(bbox_intersection_vec.begin(), bbox_intersection_vec.end());
                    sort(bbox_union_vec.begin(), bbox_union_vec.end());
                    final_thresh_scale = (bbox_intersection_vec.at(1) * bbox_intersection_vec.at(2)) /
                                            (bbox_union_vec.at(1) * bbox_union_vec.at(2));
                    return dist_hell_sq <= fmax(fmin(pow( final_thresh_scale * normal_similarity, 2.0f) *
                                                     mapParameters.hell_thresh_squard_obs, 0.9f),mapParameters.hell_thresh_squard_min);
                }
            }
        }
    }

    void GMMMap::configureVisualization() {
        // Initialize visualization parameters
        enable_visualization = true;
        gau_obs_idx = 0; // Index for occupied Gaussian
        gau_free_idx = 0; // Index for occupied Gaussian
    }

    std::string GMMMap::getInputVariableName() {
        static std::string GMMapInputVariableName = "InputDepthData";
        return GMMapInputVariableName;
    }

    void GMMMap::clear() {
        const std::lock_guard<std::mutex> g_lock(geometry_lock);
        // Clear the map and visualization geometries
        total_processed_frames = 0;
        total_gmm_clustering_latency = 0;
        total_gmm_fusion_latency = 0;

        num_free_clusters = 0;
        num_obstacle_clusters = 0;
        gau_obs_idx = 0;
        gau_free_idx = 0;
        rtree.RemoveAll();

        // Clear heap memory
        for (auto cluster_ptr : free_clusters) {
            delete cluster_ptr;
        }
        free_clusters.clear();

        for (auto cluster_ptr : obstacle_clusters) {
            delete cluster_ptr;
        }
        obstacle_clusters.clear();

        clearCurVisualizationUpdates();
    }

    bool GMMMap::remainingVisualizationExists(){
        if (!enable_visualization){
            return false;
        } else {
            if (cur_visualization_update.num_frames > 0){
                // More map updates available
                return true;
            } else {
                return false;
            }
        }
    }

    void GMMMap::clearCurVisualizationUpdates() {
        // Visualizer should call this function after update so that the next batch can begin
        cur_visualization_update.clear();
    }

    FP GMMMap::computeOccupancy(const gmm::V &pt, gmm::FP unexplored_evidence) const {
        // Compute occupancy
        // Initialized with unexplored occupancy prob
        FP occupancy = unexplored_evidence * 0.5f;
        FP total_weight = unexplored_evidence;

        // Treat a point as BBox
        Rect BBox(pt, pt);
        // TODO: Better R-tree implementation required! https://github.com/madmann91/bvh
        auto visible_clusters = this->rtree.Search(BBox);
        for (const auto cluster : visible_clusters){
            FP weight = cluster->W * cluster->evalGaussianExp(pt);
            FP cluster_occ = cluster->estOcc(pt);
            occupancy += weight * cluster_occ;
            total_weight += weight;
        }
        occupancy = occupancy / total_weight;

        if (std::isnan(occupancy)){
            throw std::invalid_argument(fmt::format("Invalid value for occupancy ({}) at point [{:.2f}, {:.2f}, {:.2f}]!",
                                                    occupancy, pt(0), pt(1), pt(2)));
        }
        return occupancy;
    }

    void GMMMap::computeOccupancyAndVariance(const V &pt, FP &occupancy, FP &variance,
                                             FP unexplored_evidence, FP unexplored_variance) const {
        // Compute occupancy and variance
        // Initialized with unexplored occupancy prob and variance
        occupancy = unexplored_evidence * 0.5f;
        variance = unexplored_evidence * (0.25f + unexplored_variance);
        FP total_weight = unexplored_evidence;

        // Treat a point as BBox
        Rect BBox(pt, pt);
        // TODO: Better R-tree implementation required! https://github.com/madmann91/bvh
        auto visible_clusters = this->rtree.Search(BBox);
        for (const auto cluster : visible_clusters){
            FP weight = cluster->W * cluster->evalGaussianExp(pt);
            FP cluster_occ = cluster->estOcc(pt);
            FP cluster_variance = cluster->estOccVariance();
            if (cluster_occ == 0) {
                if (cluster_variance != 0){
                    variance += weight * cluster_variance;
                }
            } else if (cluster_occ == 1) {
                occupancy += weight;
                if (cluster_variance == 0){
                    variance += weight;
                } else {
                    variance += weight * (1.0f + cluster_variance);
                }
            } else {
                occupancy += weight * cluster_occ;
                if (cluster_variance == 0){
                    variance += weight * cluster_occ * cluster_occ;
                } else {
                    variance += weight * (cluster_occ * cluster_occ + cluster_variance);
                }
            }

            //if (isnan(weight) || isnan(cluster->evalGaussianExp(pt)) || isnan(cluster->estOcc(pt))){
            //    std::cout << fmt::format("Nan detected: weight ({:.3f}), evalGaussianExp ({:.3f}), estOcc ({:.3f})",
            //                             weight, cluster->evalGaussianExp(pt), cluster->estOcc(pt)) << std::endl;
            //}
            total_weight += weight;
        }
        occupancy = occupancy / total_weight;
        variance = variance / total_weight - pow(occupancy, 2.0f);

        if (std::isnan(occupancy) || std::isnan(variance)){
            throw std::invalid_argument(fmt::format("Invalid value for occupancy ({}) or variance ({}) at point [{:.2f}, {:.2f}, {:.2f}]!",
                                              occupancy, variance, pt(0), pt(1), pt(2)));
        }
    }

    void GMMMap::estimateMaxOccupancyAndVariance(const V& bbox_min, const V& bbox_max, FP &occupancy, FP &variance,
                                             FP unexplored_evidence, FP unexplored_variance) const {
        // Determine the maximum occupancy by monte-carlo sampling within the bounding box
        Rect BBox(bbox_min, bbox_max);
        // TODO: Better R-tree implementation required! https://github.com/madmann91/bvh
        auto visible_clusters = this->rtree.Search(BBox);
        for (int i = 0; i < 100; i++){
            V pt = BBox.sample().cast<FP>();

            // Compute occupancy and variance
            // Initialized with unexplored occupancy prob and variance
            FP cur_occupancy = unexplored_evidence * 0.5f;
            FP cur_variance = unexplored_evidence * (0.25f + unexplored_variance);
            FP total_weight = unexplored_evidence;
            for (const auto cluster : visible_clusters){
                FP weight = cluster->W * cluster->evalGaussianExp(pt);
                FP cluster_occ = cluster->estOcc(pt);
                FP cluster_variance = cluster->estOccVariance();
                if (cluster_occ == 0) {
                    if (cluster_variance != 0){
                        cur_variance += weight * cluster_variance;
                    }
                } else if (cluster_occ == 1) {
                    cur_occupancy += weight;
                    if (cluster_variance == 0){
                        cur_variance += weight;
                    } else {
                        cur_variance += weight * (1.0f + cluster_variance);
                    }
                } else {
                    cur_occupancy += weight * cluster_occ;
                    if (cluster_variance == 0){
                        cur_variance += weight * cluster_occ * cluster_occ;
                    } else {
                        cur_variance += weight * (cluster_occ * cluster_occ + cluster_variance);
                    }
                }

                //if (isnan(weight) || isnan(cluster->evalGaussianExp(pt)) || isnan(cluster->estOcc(pt))){
                //    std::cout << fmt::format("Nan detected: weight ({:.3f}), evalGaussianExp ({:.3f}), estOcc ({:.3f})",
                //                             weight, cluster->evalGaussianExp(pt), cluster->estOcc(pt)) << std::endl;
                //}
                total_weight += weight;
            }
            cur_occupancy = cur_occupancy / total_weight;
            cur_variance = cur_variance / total_weight - pow(cur_occupancy, 2.0f);

            if (std::isnan(cur_occupancy) || std::isnan(cur_variance)){
                throw std::invalid_argument(fmt::format("Invalid value for occupancy ({}) or variance ({}) at point [{:.2f}, {:.2f}, {:.2f}]!",
                                                        cur_occupancy, cur_variance, pt(0), pt(1), pt(2)));
            }

            if (i == 0){
                occupancy = cur_occupancy;
                variance = cur_variance;
            } else {
                if (cur_occupancy > occupancy){
                    occupancy = cur_occupancy;
                    variance = cur_variance;
                }
            }
        }
    }

    bool GMMMap::computeColorAndVariance(const V& pt, V& color, M& variance) const {
        // Compute occupancy and variance
        color = V::Zero();
        variance =  M::Zero();
        FP total_weight = 0;

        // Treat a point as BBox
        Rect BBox(pt, pt);
        // TODO: Better R-tree implementation required! https://github.com/madmann91/bvh
        auto visible_clusters = this->rtree.Search(BBox);
        for (const auto cluster : visible_clusters){
            if (!cluster->is_free){
                // Note: We use N instead of W for inference here! W is only used for occupancy!
                FP weight = (FP) cluster->N * cluster->evalGaussianExp(pt);
                V cluster_color = reinterpret_cast<GMMcluster_c *>(cluster)->estColor(pt);
                M cluster_variance = reinterpret_cast<GMMcluster_c *>(cluster)->estColorCov(pt);
                color += weight * cluster_color;
                variance += weight * ( cluster_color * cluster_color.transpose() + cluster_variance);
                total_weight += weight;
            }
        }

        if (total_weight == 0){
            // Invalid color
            return false;
        } else {
            color = color / total_weight;
            variance = variance / total_weight - color * color.transpose();
            if (color.hasNaN() || variance.hasNaN()){
                throw std::invalid_argument(fmt::format("Invalid value for color or variance at point [{:.2f}, {:.2f}, {:.2f}]!",
                                                        pt(0), pt(1), pt(2)));
            }
            // Valid color
            return true;
        }
    }

    bool GMMMap::computeIntensityAndVariance(const V& pt, FP& intensity, FP& variance) const {
        V color;
        M color_variance;
        if (computeColorAndVariance(pt, color, color_variance)){
            V Weights = {0.299f, 0.587f, 0.114f};
            intensity = Weights.dot(color);
            variance = Weights.transpose() * color_variance * Weights;
            return true;
        } else {
            return false;
        }
    }

    void GMMMap::computeOccupancyAndVarianceKNN(const V& pt, const int& Nfree, const int& Nobs,
                                                FP& occupancy, FP& variance,
                                                FP unexplored_evidence, FP unexplored_variance) const {
        // Compute occupancy and variance
        // Initialized with unexplored occupancy prob and variance
        occupancy = unexplored_evidence * 0.5f;
        variance = unexplored_evidence * (0.25f + unexplored_variance);
        FP total_weight = unexplored_evidence;

        // Query nearest clusters
        auto visible_clusters = this->kNNBruteForce(pt, true, Nfree);
        visible_clusters.splice(visible_clusters.end(), this->kNNBruteForce(pt, false, Nobs));

        for (const auto cluster : visible_clusters){
            FP weight = cluster->W * cluster->evalGaussianExp(pt);
            FP cluster_occ = cluster->estOcc(pt);
            FP cluster_variance = cluster->estOccVariance();
            occupancy += weight * cluster_occ;
            variance += weight * (pow(cluster_occ, 2.0f) + cluster_variance);
            total_weight += weight;
        }
        occupancy = occupancy / total_weight;
        variance = variance / total_weight - pow(occupancy, 2.0f);
    }

    void GMMMap::estimateMapSize(FP& cluster_size, FP& rtree_size, int& num_rtree_nodes) {
        FP obs_cluster_size, free_cluster_size;
        estimateMapSize(obs_cluster_size, free_cluster_size, rtree_size, num_rtree_nodes);
        cluster_size = obs_cluster_size + free_cluster_size;
    }

    void GMMMap::estimateMapSize(FP& obs_cluster_size, FP& free_cluster_size, FP& rtree_size, int& num_rtree_nodes){
        // Estimate the size of the map in kB
        if (!obstacle_clusters.empty()){
            obs_cluster_size = (FP) (obstacle_clusters.size() * GMMcluster_c::obsGaussianSizeInBytes()) / 1024;
        } else {
            obs_cluster_size = 0;
        }

        if (!free_clusters.empty()) {
            free_cluster_size = (FP) (free_clusters.size() * GMMcluster_o::freeGaussianSizeInBytes()) / 1024;
        } else {
            free_cluster_size = 0;
        }

        num_rtree_nodes = rtree.CountAllNodes();
        rtree_size = (FP) num_rtree_nodes * (FP) GMMRtree::Node::sizeInBytes() / (FP) 1024.0;
    }

    FP GMMMap::estimateClusteringTp(){
        return total_processed_frames * 1000000.0 / total_gmm_clustering_latency;
    }

    FP GMMMap::estimateFusionTp(){
        return total_processed_frames * 1000000.0 / total_gmm_fusion_latency;
    }

    FP GMMMap::estimateMappingTp(){
        return total_processed_frames * 1000000.0 / (total_gmm_clustering_latency + total_gmm_fusion_latency);
    }

    int GMMMap::numOfObsGMMs(){
        return num_obstacle_clusters;
    }

    int GMMMap::numOfFreeGMMs(){
        return num_free_clusters;
    }

    std::list<GMMcluster_o*> GMMMap::kNNBruteForce(const V& point, bool free, int N) const {
        // Obtain kNN via brute force approach.
        // The number of elements returned have at least N elements. It will include all elements that encloses the query point

        // Using lambda to compare elements.
        auto compare = [&](GMMcluster_o* cluster_a, GMMcluster_o* cluster_b)
        {
            // Indicate conditions for a to go before (at the end of the queue) b
            FP dist_a = cluster_a->BBox.squaredExteriorDistance(point);
            FP dist_b = cluster_b->BBox.squaredExteriorDistance(point);
            return dist_a > dist_b;
        };

        const std::list<GMMcluster_o*>* clusters_list;
        if (free){
            clusters_list = &free_clusters;
        } else {
            clusters_list = &obstacle_clusters;
        }

        std::priority_queue<GMMcluster_o*, std::vector<GMMcluster_o*>, decltype(compare)> queue(clusters_list->begin(), clusters_list->end(), compare);

        std::list<GMMcluster_o*> results;

        while (!queue.empty()){
            auto p = queue.top();
            if (p->BBox.squaredExteriorDistance(point) == 0 || results.size() < N){
                queue.pop();
                results.push_back(p);
            } else {
                break;
            }
        }
        return results;
    }

    std::string GMMMap::printStatistics(const int& cur_frame_idx, const int& max_frame_idx, bool print_throughput){
        std::stringstream output_stream;

        gmm::FP cluster_size, rtree_size;
        int num_rtree_nodes;
        estimateMapSize(cluster_size, rtree_size, num_rtree_nodes);

        output_stream << fmt::format("Frame {}/{}\n", cur_frame_idx, max_frame_idx);
        {
            const std::lock_guard<std::mutex> g_lock(geometry_lock);
            output_stream << fmt::format("Obstacle clusters: {}\nFree clusters: {}\n",
                                         numOfObsGMMs(), numOfFreeGMMs());
            output_stream << fmt::format("Total cluster size: {:.2f}KB\nRtree size: {:.2f}KB,\nTotal Rtree nodes: {}\n",
                                         cluster_size, rtree_size, num_rtree_nodes);
            if (print_throughput){
                output_stream << fmt::format("GMM Clustering FPS: {:.3f}\nGMM Fusion FPS: {:.3f}\nGMM Mapping FPS: {:.3f}\nMap Size: {:.2f}KB",
                                             estimateClusteringTp(), estimateFusionTp(),
                                             estimateMappingTp(), cluster_size + rtree_size);
            }
        }
        return output_stream.str();
    }

    std::string GMMMap::printFrameStatisticsCSV(const int& cur_frame_idx){
        // Print in a single line
        // frame index, number of obstacle GMMs, size of obstacle GMMs (KBs), number of free space GMMs, size of free space GMMs (KBs),
        // RTree Nodes, RTree size (KBs), total map size (KBs), Clustering (FPS), Fusion (FPS), Total mapping (FPS)
        const std::lock_guard<std::mutex> g_lock(geometry_lock);
        gmm::FP obs_cluster_size, free_cluster_size, rtree_size;
        int num_rtree_nodes;
        estimateMapSize(obs_cluster_size, free_cluster_size, rtree_size, num_rtree_nodes);
        return fmt::format("{} {} {:.2f} {} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}",
                           cur_frame_idx, numOfObsGMMs(), obs_cluster_size, numOfFreeGMMs(), free_cluster_size,
                           num_rtree_nodes, rtree_size, obs_cluster_size + free_cluster_size + rtree_size,
                           estimateClusteringTp(), estimateFusionTp(), estimateMappingTp());
    }

    std::string GMMMap::printFrameStatisticsCSV(const int& cur_frame_idx, const double& overall_fps){
        return fmt::format("{} {:.2f}", printFrameStatisticsCSV(cur_frame_idx), overall_fps);
    }

    void GMMMap::computeEdgesToNeighbors() {
        // Clear existing clusters
        removeEdgesToNeighbors();

        // Use the RTree to find neighbors
        for (auto& cluster : obstacle_clusters){
            auto visible_clusters = this->rtree.Search(cluster->BBox);
            for (const auto neighbor : visible_clusters){
                if (neighbor != cluster){
                    cluster->addNeighbor(neighbor);
                }
            }
        }

        for (auto& cluster : free_clusters) {
            auto visible_clusters = this->rtree.Search(cluster->BBox);
            for (const auto neighbor : visible_clusters){
                if (neighbor != cluster){
                    cluster->addNeighbor(neighbor);
                }
            }
        }
    }

    void GMMMap::removeEdgesToNeighbors() {
        for (auto& cluster : obstacle_clusters){
            cluster->clearNeighbors();
        }

        for (auto& cluster : free_clusters) {
            cluster->clearNeighbors();
        }
    }

    V GMMMap::relativeToAbsoluteCoordinate(FP x_scale, FP y_scale, FP z_scale) const {
        Rect global_bbox;
        rtree.GetBounds(global_bbox);
        V scale = {x_scale, y_scale, z_scale};
        return scale.array() * (global_bbox.max() - global_bbox.min()).array() + global_bbox.min().array();
    }

    std::vector<GMMcluster_o*> GMMMap::getObsGaussians() const {
        std::vector<GMMcluster_o*> result;
        result.reserve(obstacle_clusters.size());
        result.insert(result.end(), obstacle_clusters.begin(), obstacle_clusters.end());
        return result;
    }

    std::vector<GMMcluster_o*> GMMMap::getFreeGaussians() const {
        std::vector<GMMcluster_o*> result;
        result.reserve(free_clusters.size());
        result.insert(result.end(), free_clusters.begin(), free_clusters.end());
        return result;
    }

    void GMMMap::getEnvBounds(gmm::Rect &bound) const {
        rtree.GetBounds(bound);
    }

}
