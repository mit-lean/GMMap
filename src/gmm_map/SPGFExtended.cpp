//
// Created by peter on 3/5/22.
// Support colored GMM mapping
// Support of extended-SPGF algorithm
#include "gmm_map/map.h"
#include <iostream>
#include <chrono>
#include <Eigen/Eigenvalues>
//#include "dataset_utils/dataset_utils.h"
#ifdef TRACK_MEM_USAGE
#include "mem_utils/mem_utils.h"
#endif

namespace gmm {

    std::list<GMMmetadata_c> GMMMap::extendedSPGFOpt(const RowMatrixXf& depthmap){
        RowMatrixXi temp;
        return extendedSPGFOpt(temp, temp, temp, depthmap);
    }

    // With depth variance support
    std::list<GMMmetadata_c> GMMMap::extendedSPGFOpt(const RowMatrixXf& depthmap, const RowMatrixXf& depth_variance){
        RowMatrixXi temp;
        return extendedSPGFOpt(temp, temp, temp, depthmap, depth_variance);
    }

    std::list<GMMmetadata_c> GMMMap::extendedSPGFOpt(const RowMatrixXi& r, const RowMatrixXi& g, const RowMatrixXi& b, const RowMatrixXf& depthmap){
        // Optimized implementation retains the original ICRA SPGF as much as possible
        // Additional metadata is tracked to support free cluster construction, which occurs near the end!
#ifdef TRACK_MEM_USAGE_GMM
        auto& mem_tracker = mutil::memTracker::instance();
        mem_tracker.addMemUsage(getInputVariableName(), mapParameters.num_threads * sizeof(float), mapParameters.num_threads);
#endif

        using namespace std::chrono;
        auto& clustering_params = this->mapParameters.gmm_frame_param;

        // Occupied Space
        std::list<GMMmetadata_c> obs_completed_clusters;
        std::list<GMMmetadata_c> obs_incomplete_clusters;
        int obs_numInactiveClusters;

        //std::cout << fmt::format("Extended Clustering starts ..") << std::endl;
        // Perform clustering
        #pragma omp parallel for ordered schedule(static, 1) default(shared) num_threads(clustering_params.num_threads)
        for (int row_idx = 0; row_idx < depthmap.rows(); ++row_idx){
            //if (row_idx == 200){
            //    break;
            //}

            // Check number of threads
            /*
            if (row_idx == 0)
                std::cout << fmt::format("Currently running {} concurrent threads!", omp_get_num_threads()) << std::endl;
            */

            //auto start = steady_clock::now();

            // Data storage for occupied and free line segments
            std::list<GMMmetadata_c> cur_obs_line_segments;
            std::list<std::list<GMMmetadata_o>> cur_free_line_segments;
            // The V2 version optimizes the process of constructing free and obstacle GMMs
            if (mapParameters.track_color){
                this->lineSegmentationExtendedOpt(depthmap.row(row_idx),
                                               r.row(row_idx), g.row(row_idx), b.row(row_idx),
                                               row_idx, clustering_params,cur_obs_line_segments);
            } else {
                RowColorChannelScanlineXi scanline_temp = Eigen::Matrix<uint8_t, 1, 0, Eigen::RowMajor>();
                this->lineSegmentationExtendedOpt(depthmap.row(row_idx),
                                               scanline_temp, scanline_temp, scanline_temp,
                                               row_idx, clustering_params,cur_obs_line_segments);
            }

            // Print generated metadata for each scanline
            // std::cout << fmt::format("Processed scanline {} with {} segments", row_idx, cur_obs_line_segments.size()) << std::endl;
            // printGMMMetadata_c(cur_obs_line_segments);

            #pragma omp ordered
            {
                if (row_idx == 0) {
                    obs_incomplete_clusters = cur_obs_line_segments;
                    obs_numInactiveClusters = 0;
                } else {
                    //start = steady_clock::now();
                    this->clusterMergeExtendedOpt(obs_completed_clusters,
                                               obs_incomplete_clusters,
                                               cur_obs_line_segments,
                                               obs_numInactiveClusters,
                                               clustering_params,
                                               row_idx,
                                               (row_idx == depthmap.rows() - 1));
                }
            }
            // Print debug information
            // std::cout << fmt::format("Completed obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            // printGMMMetadata_c(obs_completed_clusters);

            //std::cout << fmt::format("Incomplete obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            //printGMMMetadata_c(obs_incomplete_clusters);
        }
#ifdef TRACK_MEM_USAGE_GMM
        mem_tracker.subtractMemUsage(getInputVariableName(), mapParameters.num_threads * sizeof(float), mapParameters.num_threads);
#endif
        return obs_completed_clusters;
    }

    std::list<GMMmetadata_c> GMMMap::extendedSPGFOpt(const RowMatrixXi& r, const RowMatrixXi& g, const RowMatrixXi& b,
                                                     const RowMatrixXf& depthmap, const RowMatrixXf& depth_variance){
        // Optimized implementation retains the original ICRA SPGF as much as possible
        // Additional metadata is tracked to support free cluster construction, which occurs near the end!
#ifdef TRACK_MEM_USAGE_GMM
        auto& mem_tracker = mutil::memTracker::instance();
        mem_tracker.addMemUsage(getInputVariableName(), 2 * mapParameters.num_threads * sizeof(float), mapParameters.num_threads);
#endif
        using namespace std::chrono;
        auto& clustering_params = this->mapParameters.gmm_frame_param;

        // Occupied Space
        std::list<GMMmetadata_c> obs_completed_clusters;
        std::list<GMMmetadata_c> obs_incomplete_clusters;
        int obs_numInactiveClusters;

        //std::cout << fmt::format("Extended Clustering starts ..") << std::endl;
        // Perform clustering
        #pragma omp parallel for ordered schedule(static, 1) default(shared) num_threads(clustering_params.num_threads)
        for (int row_idx = 0; row_idx < depthmap.rows(); ++row_idx){
            //if (row_idx == 200){
            //    break;
            //}

            // Check number of threads
            /*
            if (row_idx == 0)
                std::cout << fmt::format("Currently running {} concurrent threads!", omp_get_num_threads()) << std::endl;
            */

            //auto start = steady_clock::now();

            // Data storage for occupied and free line segments
            std::list<GMMmetadata_c> cur_obs_line_segments;
            std::list<std::list<GMMmetadata_o>> cur_free_line_segments;
            // The V2 version optimizes the process of constructing free and obstacle GMMs
            if (mapParameters.track_color){
                this->lineSegmentationExtendedOpt(depthmap.row(row_idx), depth_variance.row(row_idx),
                                                  r.row(row_idx), g.row(row_idx), b.row(row_idx),
                                                  row_idx, clustering_params,cur_obs_line_segments);
            } else {
                RowColorChannelScanlineXi scanline_temp = Eigen::Matrix<uint8_t, 1, 0, Eigen::RowMajor>();
                this->lineSegmentationExtendedOpt(depthmap.row(row_idx), depth_variance.row(row_idx),
                                                  scanline_temp, scanline_temp, scanline_temp,
                                                  row_idx, clustering_params,cur_obs_line_segments);
            }

            // Print generated metadata for each scanline
            // std::cout << fmt::format("Processed scanline {} with {} segments", row_idx, cur_obs_line_segments.size()) << std::endl;
            // printGMMMetadata_c(cur_obs_line_segments);

            #pragma omp ordered
            {
                if (row_idx == 0) {
                    obs_incomplete_clusters = cur_obs_line_segments;
                    obs_numInactiveClusters = 0;
                } else {
                    //start = steady_clock::now();
                    this->clusterMergeExtendedOpt(obs_completed_clusters,
                                                  obs_incomplete_clusters,
                                                  cur_obs_line_segments,
                                                  obs_numInactiveClusters,
                                                  clustering_params,
                                                  row_idx,
                                                  (row_idx == depthmap.rows() - 1));
                }
            }
            // Print debug information
            // std::cout << fmt::format("Completed obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            // printGMMMetadata_c(obs_completed_clusters);

            //std::cout << fmt::format("Incomplete obstacle cluster metadata (during scanline segmentation {}): ", row_idx) << std::endl;
            //printGMMMetadata_c(obs_incomplete_clusters);
        }
#ifdef TRACK_MEM_USAGE_GMM
        mem_tracker.subtractMemUsage(getInputVariableName(), 2 * mapParameters.num_threads * sizeof(float), mapParameters.num_threads);
#endif
        return obs_completed_clusters;
    }

    void GMMMap::lineSegmentationExtendedOpt(const RowDepthScanlineXf& scanline_depth,
                                     const RowColorChannelScanlineXi& scanline_r,
                                     const RowColorChannelScanlineXi& scanline_g,
                                     const RowColorChannelScanlineXi& scanline_b,
                                     int row_idx, const frame_param& clustering_params,
                                     std::list<GMMmetadata_c>& cur_obs_line_segments){
        // Same as ICRA except that we also update some metadata for tracking free space
        // We merge the creation of scanline segments for free and obstacles!
        // Scanline Segmentation
        std::list<GMMmetadata_c> obs_imcomplete_queue; // queue for storing incomplete obstacle segments

        for (int col_idx = 0; col_idx < scanline_depth.cols(); ++col_idx){
            using namespace std::chrono;
            V point;
            V color;
            float depth = scanline_depth(col_idx);
            if (mapParameters.track_color){
                color << scanline_r(col_idx), scanline_g(col_idx), scanline_b(col_idx);
                //color << 1, 2, 3;
            }
            forwardProject(row_idx, col_idx, depth, clustering_params.fx, clustering_params.fy,
                           clustering_params.cx, clustering_params.cy,point, clustering_params.dataset, clustering_params.max_depth);

            // Update obstacles and free space (metadata)
            constructSegmentsFromPointOpt(point, color, row_idx, col_idx, obs_imcomplete_queue, cur_obs_line_segments,
                                       clustering_params);
        }

        // Completion detection for obstacles (insert incomplete clusters into the queue if necessary)
        for (auto it = obs_imcomplete_queue.begin(); it != obs_imcomplete_queue.end(); ++it){
            if (it->N > clustering_params.sparse_t){
                cur_obs_line_segments.push_back(*it);
            }
        }
    }

    void GMMMap::lineSegmentationExtendedOpt(const RowDepthScanlineXf& scanline_depth,
                                             const RowDepthScanlineXf& scanline_variance,
                                             const RowColorChannelScanlineXi& scanline_r,
                                             const RowColorChannelScanlineXi& scanline_g,
                                             const RowColorChannelScanlineXi& scanline_b,
                                             int row_idx, const frame_param& clustering_params,
                                             std::list<GMMmetadata_c>& cur_obs_line_segments){
        // Same as ICRA except that we also update some metadata for tracking free space
        // We merge the creation of scanline segments for free and obstacles!
        // Scanline Segmentation
        std::list<GMMmetadata_c> obs_imcomplete_queue; // queue for storing incomplete obstacle segments

        for (int col_idx = 0; col_idx < scanline_depth.cols(); ++col_idx){
            using namespace std::chrono;
            V point;
            V color;
            float depth = scanline_depth(col_idx);
            float variance = scanline_variance(col_idx);
            if (mapParameters.track_color){
                color << scanline_r(col_idx), scanline_g(col_idx), scanline_b(col_idx);
                //color << 1, 2, 3;
            }
            forwardProject(row_idx, col_idx, depth, clustering_params.fx, clustering_params.fy,
                           clustering_params.cx, clustering_params.cy, point, clustering_params.dataset, clustering_params.max_depth);

            if (variance == 0){
                constructSegmentsFromPointOpt(point, color, row_idx, col_idx, obs_imcomplete_queue, cur_obs_line_segments,
                                              clustering_params);
            } else {
                M covariance;
                forwardProjectVariance(row_idx, col_idx, clustering_params.fx, clustering_params.fy,
                                       clustering_params.cx, clustering_params.cy, variance, covariance, clustering_params.dataset);
                // Update obstacles and free space (metadata)
                constructSegmentsFromPointOpt(point, covariance, color, row_idx, col_idx, obs_imcomplete_queue, cur_obs_line_segments,
                                              clustering_params);
            }
        }

        // Completion detection for obstacles (insert incomplete clusters into the queue if necessary)
        for (auto it = obs_imcomplete_queue.begin(); it != obs_imcomplete_queue.end(); ++it){
            if (it->N > clustering_params.sparse_t){
                cur_obs_line_segments.push_back(*it);
            }
        }
    }


    void GMMMap::constructSegmentsFromPointOpt(const V& point, const V& color, int row_idx, int col_idx,
                                            std::list<GMMmetadata_c>& obs_imcomplete_queue, std::list<GMMmetadata_c>& cur_obs_line_segments,
                                            const frame_param& clustering_params){
        // Compute for obstacles and free space segments
        // Same as ICRA, but with additional support for color and free space metadata
        if (point(2) <= 0 || std::isnan(point(2))){
            return;
        }

        // First, we compute the adaptive thresholds
        bool merged;
        FP line_t_adapt, depth_t_adapt;
        adaptThreshold(point, line_t_adapt, depth_t_adapt, clustering_params);
        merged = false; //Track whether the current point is merged into existing segments.
        int cur_depth_idx = clustering_params.determineDepthIndex(point(2));
        FP max_len = clustering_params.determineDepthLength(cur_depth_idx);
        // Determine if the current point can be merged into existing clusters
        auto it = obs_imcomplete_queue.begin();
        while(it != obs_imcomplete_queue.end()){
            // Note: need to insert new element at the beginning of the list
            // Unlike ICRA 2022's SPGF algorithm, we use the free space depth discretization
            // Better accuracy by restricting the max planewidth of the Gaussians
            //if (!merged && col_idx - it->RightPixel <= clustering_params.occ_x_t && it->PlaneWidth < pow(clustering_params.max_len, 2)){
            if (!merged && col_idx - it->RightPixel <= clustering_params.occ_x_t && it->PlaneWidth < pow(max_len, 2.0f)){
                if (it->N <= clustering_params.ncheck_t){
                    // Stage 1: Establish trend
                    if (distDepth(point, *it) < fmin(fmax(depth_t_adapt, clustering_params.noise_floor),clustering_params.noise_thresh) &&
                        distLine(point, *it) < fmin(fmax(line_t_adapt, clustering_params.noise_floor),clustering_params.line_t)){
                        merged = true;
                        addPointObs(point, color, row_idx, col_idx,cur_depth_idx, *it, clustering_params);
                    }
                } else {
                    // Stage 2: See if trend continues
                    if (std::abs((FP) (distDepthEst(point, *it) - point(2))) < fmin(fmax(depth_t_adapt, clustering_params.noise_floor),clustering_params.noise_thresh)){
                        merged = true;
                        addPointObs(point, color, row_idx, col_idx,cur_depth_idx, *it, clustering_params);
                    }
                }
            }
            it++;
        }

        // Instantiate a new segment if the point is not merged into any existing clusters
        if (!merged){
            GMMmetadata_c segment(this->mapParameters.track_color);
            addPointObs(point, color, row_idx, col_idx, cur_depth_idx,segment, clustering_params);
            obs_imcomplete_queue.push_front(segment); // Push new segments to the front of the queue

            // Check the size of the imcomplete_queue
            if (obs_imcomplete_queue.size() > clustering_params.max_incomplete_clusters){
                if (obs_imcomplete_queue.back().N > clustering_params.sparse_t){
                    cur_obs_line_segments.splice(cur_obs_line_segments.end(),obs_imcomplete_queue,std::prev(obs_imcomplete_queue.end()));
                } else {
                    obs_imcomplete_queue.pop_back();
                }
            }
        }
    }

    void GMMMap::constructSegmentsFromPointOpt(const V& point, const M& covariance, const V& color, int row_idx, int col_idx,
                                               std::list<GMMmetadata_c>& obs_imcomplete_queue, std::list<GMMmetadata_c>& cur_obs_line_segments,
                                               const frame_param& clustering_params){
        // Compute for obstacles and free space segments
        // Same as ICRA, but with additional support for color and free space metadata
        if (point(2) <= 0 || std::isnan(point(2))){
            return;
        }

        // First, we compute the adaptive thresholds
        bool merged;
        FP line_t_adapt, depth_t_adapt;
        adaptThreshold(point, line_t_adapt, depth_t_adapt, clustering_params);
        merged = false; //Track whether the current point is merged into existing segments.
        int cur_depth_idx = clustering_params.determineDepthIndex(point(2));
        FP max_len = clustering_params.determineDepthLength(cur_depth_idx);
        // Determine if the current point can be merged into existing clusters
        auto it = obs_imcomplete_queue.begin();
        while(it != obs_imcomplete_queue.end()){
            // Note: need to insert new element at the beginning of the list
            // Unlike ICRA 2022's SPGF algorithm, we use the free space depth discretization
            // Better accuracy by restricting the max planewidth of the Gaussians
            //if (!merged && col_idx - it->RightPixel <= clustering_params.occ_x_t && it->PlaneWidth < pow(clustering_params.max_len, 2)){
            if (!merged && col_idx - it->RightPixel <= clustering_params.occ_x_t && it->PlaneWidth < pow(max_len, 2.0f)){
                if (it->N <= clustering_params.ncheck_t){
                    // Stage 1: Establish trend
                    if (distDepth(point, *it) < fmin(fmax(depth_t_adapt, clustering_params.noise_floor),clustering_params.noise_thresh) &&
                        distLine(point, *it) < fmin(fmax(line_t_adapt, clustering_params.noise_floor),clustering_params.line_t)){
                        merged = true;
                        addPointObs(point, covariance, color, row_idx, col_idx,cur_depth_idx, *it, clustering_params);
                    }
                } else {
                    // Stage 2: See if trend continues
                    if (std::abs((FP) (distDepthEst(point, *it) - point(2))) < fmin(fmax(depth_t_adapt, clustering_params.noise_floor),clustering_params.noise_thresh)){
                        merged = true;
                        addPointObs(point, covariance, color, row_idx, col_idx,cur_depth_idx, *it, clustering_params);
                    }
                }
            }
            it++;
        }

        // Instantiate a new segment if the point is not merged into any existing clusters
        if (!merged){
            GMMmetadata_c segment(this->mapParameters.track_color);
            addPointObs(point, covariance, color, row_idx, col_idx, cur_depth_idx,segment, clustering_params);
            obs_imcomplete_queue.push_front(segment); // Push new segments to the front of the queue

            // Check the size of the imcomplete_queue
            if (obs_imcomplete_queue.size() > clustering_params.max_incomplete_clusters){
                if (obs_imcomplete_queue.back().N > clustering_params.sparse_t){
                    cur_obs_line_segments.splice(cur_obs_line_segments.end(),obs_imcomplete_queue,std::prev(obs_imcomplete_queue.end()));
                } else {
                    obs_imcomplete_queue.pop_back();
                }
            }
        }
    }

    void GMMMap::clusterMergeExtendedOpt(std::list<GMMmetadata_c>& obs_completed_clusters,
                                     std::list<GMMmetadata_c>& obs_incomplete_clusters,
                                     std::list<GMMmetadata_c>& cur_obs_line_segments,
                                     int& obs_numInactiveClusters,
                                     const frame_param& clustering_params,
                                     int row_idx,
                                     bool final_scanline){
        // The optimized version uses adaptive depth to bound the max size of the cluster.
        std::list<GMMmetadata_c>::iterator op_it;
        V_2 cur_LineVec, pre_LineVec;
        auto it_l_size = obs_incomplete_clusters.size();
        auto it_h = cur_obs_line_segments.begin();
        int j = 0;

        while (it_h != cur_obs_line_segments.end()) {
            FP overlap_max = 0;
            // For each scanline segment, determine the cluster that have the greatest overlap in image space
            auto it_l = obs_incomplete_clusters.begin();
            for (auto it_l_idx = 0; it_l_idx < it_l_size; it_l_idx++) {
                FP overlap_interval = (FP) (std::min<int>(it_h->RightPixel, it_l->RightPixel) - std::max<int>(it_h->LeftPixel, it_l->LeftPixel));
                if (overlap_interval > 0){
                    FP total_interval = (FP) (std::max<int>(it_h->RightPixel, it_l->RightPixel) - std::min<int>(it_h->LeftPixel, it_l->LeftPixel));
                    FP overlap_ratio = overlap_interval/total_interval;
                    if (overlap_ratio > overlap_max){
                        overlap_max = overlap_ratio;
                        op_it = it_l;
                    }
                }
                it_l++;
            }

            // If overlap exists, check for scanline parallelism and on-plane constraints
            if (overlap_max > 0){
                FP max_len = clustering_params.determineDepthLength(op_it->freeBasis.depth_idx);
                cur_LineVec << it_h->LineVec(0), it_h->LineVec(2);
                pre_LineVec << op_it->LineVec(0), op_it->LineVec(2);
                // Check merging criteria
                //if (op_it->PlaneLen < pow(clustering_params.max_len, 2) &&
                if (op_it->PlaneLen < pow(max_len, 2.0f) &&
                    pow(cur_LineVec.dot(pre_LineVec),2.0f) >
                    cur_LineVec.dot(cur_LineVec)*pre_LineVec.dot(pre_LineVec)*pow(clustering_params.angle_t,2.0f) &&
                    onPlane(it_h->S/it_h->N, *op_it, clustering_params)){
                    // Merge!
                    mergeMetadataObs(*it_h, *op_it, clustering_params);
                    op_it->Updated = true; // Tracks whether or not the previous cluster if fully utilized
                    it_h++;
                    j++;
                } else {
                    // Not merged. Transfer element to the incompleted cluster.
                    it_h++;
                    obs_incomplete_clusters.splice(obs_incomplete_clusters.end(),cur_obs_line_segments,std::prev(it_h));
                    j++;
                }
            } else {
                // Not merged. Transfer element to the incompleted cluster.
                it_h++;
                obs_incomplete_clusters.splice(obs_incomplete_clusters.end(),cur_obs_line_segments,std::prev(it_h));
                j++;
            }
        }

        // Final clean up
        if (final_scanline){
            it_l_size = obs_incomplete_clusters.size();
        }

        auto it_l = obs_incomplete_clusters.begin();
        for (auto it_l_idx = 0; it_l_idx < it_l_size; it_l_idx++){
            if (!it_l->Updated || final_scanline){
                // Pruning Gaussian here (preserve_details_far_objects = false) SIGNIFICANTLY decreases memory overhead
                if (clustering_params.preserve_details_far_objects ||
                    it_l->N >= clustering_params.num_pixels_t || it_l->NumLines >= clustering_params.num_line_t){
                    // Transfer to completed cluster list
                    it_l->Updated = false;
                    it_l++;
                    obs_completed_clusters.splice(obs_completed_clusters.end(),obs_incomplete_clusters,std::prev(it_l));
                } else {
                    it_l = obs_incomplete_clusters.erase(it_l);
                    obs_numInactiveClusters++;
                }
            } else {
                it_l->Updated = false;
                it_l++;
            }
        }
    }

    std::list<GMMmetadata_o> GMMMap::constructFreeClustersFromObsClusters(std::list<GMMmetadata_c>& obs_completed_clusters){
        // Construct free clusters using metadata from obstacle clusters
        std::list<GMMmetadata_o> free_completed_clusters;

        // Process obstacle clusters first using depth
        int max_depth_idx = 0;
        int min_depth_idx = mapParameters.gmm_frame_param.depth_dists_cum.size();
        std::vector<std::list<freeSpaceBasis*>> freeBasis_organized_by_depth(mapParameters.gmm_frame_param.depth_dists_cum.size());
        for (auto& cluster : obs_completed_clusters){
            freeBasis_organized_by_depth.at(cluster.freeBasis.depth_idx).push_back(&cluster.freeBasis);
            cluster.freeBasis.list_it = std::prev(freeBasis_organized_by_depth.at(cluster.freeBasis.depth_idx).end());
            max_depth_idx = std::max<int>(max_depth_idx, cluster.freeBasis.depth_idx);
            min_depth_idx = std::min<int>(min_depth_idx, cluster.freeBasis.depth_idx);
        }

        // Process from the max depth towards the min depth (camera)
        for (int depth_idx = max_depth_idx; depth_idx >= 0; depth_idx--){
            // 1) Merge free clusters at the current depth level
            std::list<freeSpaceBasis*> fused_freeBasis;
            fuseFreeSpaceBasis(freeBasis_organized_by_depth.at(depth_idx), fused_freeBasis, depth_idx, min_depth_idx);

            // 2) Create free clusters at the current depth level
            for (const auto& freeBasis: fused_freeBasis){
                free_completed_clusters.emplace_back();
                auto& free_cluster = free_completed_clusters.back();
                free_cluster.near_obs = (depth_idx == freeBasis->depth_idx);
                freeBasis->transferFreeClusterParam(free_cluster);
                freeBasis->cluster_valid = false;
            }

            // 3) Move to the next depth level
            if (depth_idx > 0){
                freeBasis_organized_by_depth.at(depth_idx-1).splice(freeBasis_organized_by_depth.at(depth_idx-1).end(),
                                                                   fused_freeBasis);
            }
        }

        // Some observations
        // 1) Creating one free cluster per obstacle does not work well during fusion!
        // 2) Creating free clusters along the depth of each obstacle and NOT merging gives good accuracy, but degrades compactness & fusion throughput
        // 3) Fuse across depth boundary gives really bad result! Also degrades query performance!
        return free_completed_clusters;
    }

    void GMMMap::fuseFreeSpaceBasis(std::list<freeSpaceBasis*>& original_freeBasis,
                                       std::list<freeSpaceBasis*>& fused_freeBasis,
                                       const int& cur_depth_idx, const int& min_depth_idx){
        // Fuse free clusters at the current depth level. Generate the valid cluster data.
        // std::cout << fmt::format("Merging free clusters at depth index {}: ", cur_depth_idx) << std::endl;
        FP dist_near, dist_far;
        mapParameters.gmm_frame_param.determineDepthBounds(cur_depth_idx, dist_near, dist_far);
        while (!original_freeBasis.empty()){
            // 1) Take the first item in the list as a seed and find its neighbors
            auto seed_data = original_freeBasis.front();
            std::list<freeSpaceBasis*> seed_neighbors;
            for (auto data : original_freeBasis){
                data->updateInternalFreeClusterParam(dist_near, dist_far, cur_depth_idx, mapParameters.gau_fusion_bd);
                //data->printInfo();
                // Skip the seed
                if (data != seed_data){
                    if (seed_data->isBBoxIntersect(*data)){
                        seed_neighbors.push_back(data);
                    }
                }
            }

            // 2) Compute Hellinger distance and do recursive merge with neighbors!
            bool merged = false;
            for (auto neighbor : seed_neighbors) {
                freeSpaceBasis fused_candidate = *seed_data;
                fused_candidate.mergeWithInternalParam(*neighbor, mapParameters.gau_fusion_bd);
                if (fused_candidate.BBox.sizes().maxCoeff() > mapParameters.max_bbox_len){
                    continue;
                }

                if (cur_depth_idx < min_depth_idx) {
                    *seed_data = fused_candidate;
                    original_freeBasis.erase(neighbor->list_it);
                    merged = true;
                } else {
                    FP depth_diff_seed = seed_data->BBox.sizes()(2);
                    FP depth_diff_neighbor = neighbor->BBox.sizes()(2);
                    FP thresh_scale;
                    if (depth_diff_seed > depth_diff_neighbor){
                        thresh_scale = depth_diff_neighbor / depth_diff_seed;
                    } else {
                        thresh_scale = depth_diff_seed / depth_diff_neighbor;
                    }

                    FP thresh = fmax(mapParameters.gmm_frame_param.free_space_dist_scale * pow(thresh_scale, 2.0f) * mapParameters.hell_thresh_squard_free,
                                        mapParameters.hell_thresh_squard_min);
                    FP hell_dist_sq = unscentedHellingerSquaredFreeBasis({seed_data, neighbor}, {&fused_candidate});
                    if (hell_dist_sq <= thresh){
                        //std::cout << fmt::format ("Free cluster fused with dist: {:.8f} <= threshold: {:.8f}", hell_dist_sq, thresh) << std::endl;
                        *seed_data = fused_candidate;
                        original_freeBasis.erase(neighbor->list_it);
                        merged = true;
                    }
                }
            }

            if (!merged){
                // We found all regions connected to the seed!
                original_freeBasis.erase(seed_data->list_it);
                fused_freeBasis.push_back(seed_data);
                seed_data->list_it = std::prev(fused_freeBasis.end());
            }
        }
    }


    void GMMMap::addPointObs(const V& point, const V& color, int v, int u, int depth_idx,
                             GMMmetadata_c& metadata, const frame_param& clustering_params){
        // Update free space: Compute the parameters of the Gaussian for each line segment in closed form!
        // Note that this function dominates the computation time in SPGF!
        FP p_norm = point.norm();
        M ppT = point * point.transpose();
        V S_free = 0.5f * p_norm * point;
        M J_free = (p_norm / 3.0f) * ppT;

        // Pixel coordinate tracking
        Eigen::Vector2i pixel_coord;
        pixel_coord << v,u;

        //if (*update_free_gmm){
            if (metadata.N == 0){
                //metadata.clusters_free.emplace_back();
                //metadata.clusters_free.emplace_back();
                //metadata.freeBasis.PBox = PRect(pixel_coord - Eigen::Vector2i::Constant(mapParameters.gmm_frame_param.occ_x_t),
                //                               pixel_coord + Eigen::Vector2i::Constant(mapParameters.gmm_frame_param.occ_x_t));
                metadata.freeBasis.depth_idx = depth_idx; // This is the index to the adaptive depth array
            } else {
                // Note: we enlarge the PBox by one pixel on each side to enforce the correct overlap during free space creation.
                //metadata.freeBasis.PBox.extend(pixel_coord + Eigen::Vector2i::Constant(mapParameters.gmm_frame_param.occ_x_t));
                // We track the minimum depth index to support free cluster construction
                metadata.freeBasis.depth_idx = std::min(metadata.freeBasis.depth_idx, depth_idx); // This is the index to the adaptive depth array
            }

            // The front accumulates all segments from zero
            metadata.freeBasis.W_ray_ends += p_norm;
            metadata.freeBasis.S_ray_ends += S_free;
            metadata.freeBasis.J_ray_ends += J_free;

            // The back accumulates the bases used to infer all other free space clusters
            metadata.freeBasis.W_basis += p_norm / point(2);
            metadata.freeBasis.S_basis += S_free / (point(2) * point(2));
            metadata.freeBasis.J_basis += J_free / (point(2) * point(2) * point(2)); // Note: pow(x, 3) is very slow!
        //}

        // Update obstacles
        metadata.RightPoint = point;

        metadata.N = metadata.N + 1;
        metadata.W = metadata.W + p_norm;
        if (this->mapParameters.track_color){
            V_c point_c;
            point_c << point, color;
            *metadata.S_c_eff += color;
            *metadata.J_c_eff += (point_c * point_c.transpose()).bottomLeftCorner<3,6>();
        }
        metadata.S += point;
        metadata.J += ppT;

        if (metadata.N == 1) {
            metadata.LeftPoint = point;
            metadata.LeftPixel = u;
        } else {
            metadata.LineVec += point - metadata.LeftPoint;
        }

        metadata.RightPixel = u;
        metadata.RightPoint = point;
        metadata.PlaneWidth = (metadata.LeftPoint-point).dot(metadata.LeftPoint-point);
    }

    void GMMMap::addPointObs(const V& point, const M& covariance, const V& color, int v, int u, int depth_idx,
                             GMMmetadata_c& metadata, const frame_param& clustering_params){
        addPointObs(point,color, v, u, depth_idx, metadata, clustering_params);
        metadata.J = metadata.J + covariance;
    }

    void GMMMap::mergeMetadataObs(GMMmetadata_c& source, GMMmetadata_c& destination, const frame_param& clustering_params){

        destination.N = destination.N + source.N;
        destination.W = destination.W + source.W;
        if (this->mapParameters.track_color){
            *destination.S_c_eff += *source.S_c_eff;
            *destination.J_c_eff += *source.J_c_eff;
        }
        destination.S = destination.S + source.S;
        destination.J = destination.J + source.J;

        // Transfer the characteristics of merged cluster
        destination.LineVec = destination.LineVec + source.LineVec; // Averaging leads to much more accurate results

        // This will get rid of a lot of noise! Keep!
        destination.LeftPixel = int((source.LeftPixel + destination.LeftPixel)/2);
        destination.RightPixel = int((source.RightPixel + destination.RightPixel)/2);

        // Update vector connecting the means
        V src_mean = source.S/source.N;
        if (destination.NumLines == 1){
            destination.UpMean = destination.S/destination.N;
            destination.PlaneVec = (src_mean-destination.UpMean);
        } else {
            destination.PlaneVec = destination.PlaneVec + (src_mean-destination.UpMean);
        }
        destination.PlaneLen = (src_mean - destination.UpMean).dot(src_mean - destination.UpMean);
        destination.NumLines = destination.NumLines + source.NumLines;

        // Merge metadata associated with free space
        //if (*update_free_gmm){
            destination.freeBasis.merge(source.freeBasis);
        //}
    }

    void GMMMap::transferMetadata2ClusterExtended(std::list<GMMmetadata_c>& obs_completed_metadata, std::list<GMMmetadata_o>& free_completed_metadata,
                                                  std::list<GMMcluster_o*>& obs_gaussians, std::list<GMMcluster_o*>& free_gaussians,
                                                    const frame_param& clustering_params){

        // To greatly reduce memory consumption (at the expense of accuracy), move the Gaussian prune earlier in the algorithm.
        auto obs_cluster_it = obs_completed_metadata.begin();
        while (obs_cluster_it != obs_completed_metadata.end()){
            //std::cout << fmt::format("Converting obstacle metadata {} to cluster: ", idx) << std::endl;
            //idx++;
            //if (obs_cluster_it->N >= clustering_params.num_pixels_t || obs_cluster_it->NumLines >= clustering_params.num_line_t){
                auto new_cluster = new GMMcluster_c(mapParameters.track_color);
                new_cluster->is_free = false;
                new_cluster->track_color = this->mapParameters.track_color;
                new_cluster->N = obs_cluster_it->N;
                new_cluster->W = obs_cluster_it->W;
                new_cluster->updateMeanAndCovFromIntermediateParams(false, obs_cluster_it->S, obs_cluster_it->J, true);
                if (mapParameters.track_color){
                    new_cluster->updateMeanAndCovFromIntermediateParamsC(false, *obs_cluster_it->S_c_eff, *obs_cluster_it->J_c_eff);
                }
                //new_cluster->updateInvCov();
                // Transfer BBox (Will be re-computed when transformed)
                new_cluster->updateBBox(clustering_params.gau_bd_scale);
                // Compute Normal
                obs_gaussians.push_back(new_cluster);
                new_cluster->list_it = std::prev(obs_gaussians.end());
            //}
            obs_cluster_it++;

            // Remove metadata for memory efficiency
            obs_completed_metadata.erase(std::prev(obs_cluster_it));
        }

        // Transfer
        auto free_cluster_it = free_completed_metadata.begin();
        while (free_cluster_it != free_completed_metadata.end()){
            if (free_cluster_it->W >= (float) clustering_params.num_pixels_t * clustering_params.free_space_start_len){
                auto new_cluster = new GMMcluster_o;
                new_cluster->is_free = true;
                new_cluster->N = free_cluster_it->N;
                new_cluster->W = free_cluster_it->W;
                new_cluster->updateMeanAndCovFromIntermediateParams(true, free_cluster_it->S, free_cluster_it->J, true);
                //new_cluster->updateInvCov();
                // Transfer BBox (Will be re-computed when transformed)
                new_cluster->updateBBox(clustering_params.gau_bd_scale);
                // Track clusters that are near obstacles
                new_cluster->near_obs = free_cluster_it->near_obs;
                free_gaussians.push_back(new_cluster);
                new_cluster->list_it = std::prev(free_gaussians.end());
            }
            free_cluster_it++;

            // Remove metadata for memory efficiency
            free_completed_metadata.erase(std::prev(free_cluster_it));
        }
    }

    FP GMMMap::estFreeSegmentOverlapDist(const GMMmetadata_o& pre_seg, const GMMmetadata_o& cur_seg){
        // Compute the Hellinger distance between the projection of the segment and cluster on the xz plane
        V_2 pre_seg_zx_mean, cur_seg_zx_mean;
        M_2 pre_seg_zx_cov, cur_seg_zx_cov;
        pre_seg.computeMeanAndCovXZ(pre_seg_zx_mean, pre_seg_zx_cov, true);
        cur_seg.computeMeanAndCovXZ(cur_seg_zx_mean, cur_seg_zx_cov, true);
        return hellingerSquared(pre_seg_zx_mean, cur_seg_zx_mean, pre_seg_zx_cov, cur_seg_zx_cov);
    }

    void GMMMap::adaptThreshold(const V& point, FP& line_t_adapt, FP& depth_t_adapt, const frame_param& clustering_params){
        // Determine the adaptive threshold for RGBD sensors
        line_t_adapt = clustering_params.adaptive_thresh_scale*(pow(point(2),2.0f)/clustering_params.f);
        depth_t_adapt = 6.0f * line_t_adapt;
    }

    FP GMMMap::distDepth(const V& point, const GMMmetadata_c& metadata){
        // Distance alone the depth
        if (metadata.N == 0){
            return 0;
        } else {
            return std::abs((FP) (point(2) - metadata.RightPoint(2)));
        }
    }

    FP GMMMap::distLine(const V& point, const GMMmetadata_c& metadata){
        // Distance alone the scanline
        if (metadata.N == 0){
            return 0;
        } else {
            return std::abs((FP) (point(0) - metadata.RightPoint(0)));
        }
    }

    bool GMMMap::onPlane(const V& point, const GMMmetadata_c& metadata, const frame_param& clustering_params){
        // Check if a point lies on the same plane as cluster described by the metadata
        // Used for obstacle cluster only!
        if (metadata.NumLines == 1){
            FP line_t_adapt, depth_t_adapt;
            adaptThreshold(metadata.S/metadata.N, line_t_adapt, depth_t_adapt, clustering_params);
            if (std::abs((FP) (metadata.S(2)/ (FP) metadata.N - point(2))) < depth_t_adapt){
                return true;
            } else {
                return false;
            }
        } else {
            V normal = metadata.LineVec.head<3>().cross(metadata.PlaneVec.head<3>());
            FP normal_scale = normal.dot(normal);
            if (pow(normal.dot(metadata.S/metadata.N)-normal.dot(point),2.0f) <
                normal_scale*pow(clustering_params.noise_floor,2.0f)){
                return true;
            } else {
                return false;
            }
        }
    }

    FP GMMMap::distDepthEst(const V& point, const GMMmetadata_c& metadata){
        return metadata.LeftPoint(2) + metadata.LineVec(2)*(point(0)-metadata.LeftPoint(0))/metadata.LineVec(0);
    }

    void GMMMap::printGMMs(const std::list<GMMcluster_o *>& clusters) const {
        int idx = 0;
        for (auto cluster : clusters) {
            if (cluster->is_free){
                std::cout << fmt::format("Printing info for free cluster {}", idx) << std::endl;
            } else {
                std::cout << fmt::format("Printing info for obstacle cluster {}", idx) << std::endl;
            }
            cluster->printInfo();
            idx++;
        }
    }

    void GMMMap::printGMMMetadata_o(const std::list<GMMmetadata_o> &metadata) const {
        int idx = 0;
        for (const auto& data : metadata) {
            std::cout << fmt::format("Printing info for free cluster metadata {}", idx) << std::endl;
            data.printInfo();
            idx++;
        }
    }

    void GMMMap::printGMMMetadata_c(const std::list<GMMmetadata_c> &metadata) const {
        int idx = 0;
        for (const auto& data : metadata) {
            std::cout << fmt::format("Printing info for obstacle cluster metadata {}", idx) << std::endl;
            data.printInfo();
            idx++;
        }
    }

}
