//
//
#include "gmm_map/Visualization.h"
#include "gmm_map/evaluation.h"
#include "dataset_utils/dataset_utils.h"
#include "colormap/colormap.h"

namespace gmm {
    // OccVarWithMetadata
    OccVarWithMetadata::OccVarWithMetadata(const Eigen::MatrixXd& OccVarCMap){
        this->OccVarCMap = OccVarCMap;
    }

    OccVarWithMetadata::OccVarWithMetadata(const Eigen::Vector3d& prob_error_color, const Eigen::MatrixXd& OccVarCMap){
        this->OccVarCMap = OccVarCMap;
        this->prob_error_color = prob_error_color;
    }

    void OccVarWithMetadata::clear() {
        obs_and_free_pts.clear();
        occ_colors.clear();
        var_colors.clear();
        error_value.clear();
        occ_value.clear();
        variance_value.clear();
        accuracy = 0;
        throughput = 0;
        total_num_pts = 0;
        building_blocks_computed = false;
        occ_pcd_original = nullptr;
        occ_var_pcd_original = nullptr;
        occ_var_pcd_cropped = nullptr;
        occ_pcd_cropped = nullptr;
    }

    void OccVarWithMetadata::cropGeometries(bool crop_existing,
                                            const std::shared_ptr<open3d::geometry::OrientedBoundingBox>& env_bbox){
        // Introduce cropping so that only the points within the bbox is visible
        if (crop_existing){
            //std::cout << "Enclosed optimization!" << std::endl;
            if (occ_pcd_cropped != nullptr && occ_var_pcd_cropped != nullptr){
                occ_pcd_cropped = occ_pcd_cropped->Crop(*env_bbox);
                occ_var_pcd_cropped = occ_var_pcd_cropped->Crop(*env_bbox);
            }
        } else {
            if (occ_pcd_original != nullptr && occ_var_pcd_original != nullptr){
                occ_pcd_cropped = occ_pcd_original->Crop(*env_bbox);
                occ_var_pcd_cropped = occ_var_pcd_original->Crop(*env_bbox);
            }
        }
    }

    void OccVarWithMetadata::generateGeometriesWithErrors(const FP& occ_var_pt_high, const FP& occ_var_pt_low, const Isometry3& pose,
                                                          FP& cur_max_variance, FP& cur_min_variance){
        total_num_pts = obs_and_free_pts.size();
        if (total_num_pts == 0){
            return;
        }

        occ_pcd_original = std::make_shared<open3d::geometry::PointCloud>();
        occ_var_pcd_original = std::make_shared<open3d::geometry::PointCloud>();

        // Move this to another function outside!
        cur_max_variance = *std::max_element(variance_value.begin(), variance_value.end());
        cur_min_variance = *std::min_element(variance_value.begin(), variance_value.end());

        // Check if the number of colors is consistent with the number of points
        bool has_color = true;
        if (occ_colors.size() != total_num_pts || var_colors.size() != total_num_pts){
            occ_colors.clear();
            var_colors.clear();
            has_color = false;
        }

        FP variance_diff = cur_max_variance - cur_min_variance;
        FP std_diff = sqrt(cur_max_variance) - sqrt(cur_min_variance);
        for (int i = 0; i < total_num_pts; i++){
            // Update occ_pcd color
            Eigen::Vector3d color;
            if (!has_color){
                if (error_value[i]){
                    color = prob_error_color;
                } else {
                    color = colormap::interpolateNearestNeighbor(OccVarCMap, occ_value[i]);
                }
                occ_colors.push_back(color);
            } else {
                color = occ_colors.at(i);
            }

            if (occ_value[i] <= occ_var_pt_high && occ_value[i] >= occ_var_pt_low){
                occ_pcd_original->points_.push_back(obs_and_free_pts[i]);
                occ_pcd_original->colors_.push_back(color);
            }

            // Update variance pcd
            //FP rel_var = variance_value[i] / variance_diff;
            FP rel_var = (sqrt(variance_value[i]) - sqrt(cur_min_variance)) / std_diff; // visual standard deviation
            if (!has_color){
                color = colormap::interpolateNearestNeighbor(OccVarCMap, rel_var);
                var_colors.push_back(color);
            } else {
                color = var_colors.at(i);
            }

            if (rel_var <= occ_var_pt_high && rel_var >= occ_var_pt_low){
                occ_var_pcd_original->points_.push_back(obs_and_free_pts[i]);
                occ_var_pcd_original->colors_.push_back(color);
            } else {
                //std::cout << fmt::format("Standard deviation with value {:.2f} ({:.2f}), relative {:.2f} (diff: {:.2f}) is rejected!",
                //                         sqrt(variance_value[i]), variance_value[i], rel_var, std_diff) << std::endl;
            }
        }

        if (!pose.matrix().isIdentity()){
            occ_pcd_original->Transform(pose.matrix().cast<double>());
            occ_var_pcd_original->Transform(pose.matrix().cast<double>());
        }
    }

    void OccVarWithMetadata::generateGeometries(const FP& occ_var_pt_high, const FP& occ_var_pt_low, const Isometry3& pose,
                                                FP& cur_max_variance, FP& cur_min_variance) {
        total_num_pts = obs_and_free_pts.size();
        if (total_num_pts == 0){
            return;
        }

        occ_pcd_original = std::make_shared<open3d::geometry::PointCloud>();
        occ_var_pcd_original = std::make_shared<open3d::geometry::PointCloud>();

        // Move this to another function outside!
        cur_max_variance = *std::max_element(variance_value.begin(), variance_value.end());
        cur_min_variance = *std::min_element(variance_value.begin(), variance_value.end());

        FP variance_diff = cur_max_variance - cur_min_variance;
        FP std_diff = sqrt(cur_max_variance) - sqrt(cur_min_variance);

        // Check if the number of colors is consistent with the number of points
        bool has_color = true;
        if (occ_colors.size() != total_num_pts || var_colors.size() != total_num_pts){
            occ_colors.clear();
            var_colors.clear();
            has_color = false;
        }

        for (int i = 0; i < total_num_pts; i++){
            // Update occ_pcd color
            Eigen::Vector3d color;
            if (!has_color){
                color = colormap::interpolateNearestNeighbor(OccVarCMap, occ_value[i]);
                occ_colors.push_back(color);
            } else {
                color = occ_colors.at(i);
            }

            if (occ_value[i] <= occ_var_pt_high && occ_value[i] >= occ_var_pt_low){
                occ_pcd_original->points_.push_back(obs_and_free_pts[i]);
                occ_pcd_original->colors_.push_back(color);
            }

            // Update variance pcd
            //FP rel_var = variance_value[i] / variance_diff;
            FP rel_var = (sqrt(variance_value[i]) - sqrt(cur_min_variance)) / std_diff; // visual standard deviation
            if (!has_color){
                color = colormap::interpolateNearestNeighbor(OccVarCMap, rel_var);
                var_colors.push_back(color);
            } else {
                color = var_colors.at(i);
            }

            if (rel_var <= occ_var_pt_high && rel_var >= occ_var_pt_low){
                occ_var_pcd_original->points_.push_back(obs_and_free_pts[i]);
                occ_var_pcd_original->colors_.push_back(color);
            } else {
                //std::cout << fmt::format("Standard deviation with value {:.2f} ({:.2f}), relative {:.2f} (diff: {:.2f}) is rejected!",
                //                         sqrt(variance_value[i]), variance_value[i], rel_var, std_diff) << std::endl;
            }
        }

        if (!pose.matrix().isIdentity()) {
            occ_pcd_original->Transform(pose.matrix().cast<double>());
            occ_var_pcd_original->Transform(pose.matrix().cast<double>());
        }
    }

    std::shared_ptr<open3d::geometry::LineSet> GenerateRayLinesetFromDepthmap(const RowMatrixXf& depthmap,
                                                                             int row_idx,
                                                                             const Eigen::Vector3d& color,
                                                                             const std::string& dataset_name) {
        std::shared_ptr<open3d::geometry::LineSet> ray_lineset = std::make_shared<open3d::geometry::LineSet>();
        int pt_idx = 0;
        for (int col_idx = 0; col_idx < depthmap.cols(); col_idx++){
            float depth = depthmap(row_idx, col_idx);
            if (depth <= 0 || std::isnan(depth)){
                continue;
            }

            V point;
            forwardProject(row_idx, col_idx, depth, dataset_param::fx, dataset_param::fy,
                           dataset_param::cx, dataset_param::cy,point, dataset_name, dataset_param::max_depth);

            ray_lineset->points_.push_back({0,0,0});
            ray_lineset->points_.push_back(point.cast<double>());
            ray_lineset->lines_.push_back({pt_idx, pt_idx+1});
            ray_lineset->colors_.push_back(color);
            pt_idx += 2;
        }

        return ray_lineset;
    }

    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMLinesetOccSingle(const GMMcluster_o& Cluster,
                                                                   const Eigen::Vector3d& OccupancyColor,
                                                                   bool& completeGeometry,
                                                                   FP scale, M_o pose){
        auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0, 8);
        auto sphere_lineset = open3d::geometry::LineSet::CreateFromTriangleMesh(*sphere);
        sphere_lineset->colors_.clear();
        std::shared_ptr<open3d::geometry::LineSet> gmm_lineset = std::make_shared<open3d::geometry::LineSet>();
        *gmm_lineset = *sphere_lineset;
        //std::cout << "Computing Cholesky" << std::endl;
        M L = Cluster.CovL();
        M transform = L * scale;
        //std::cout << Clusters.at(idx).Covariance <<std::endl;
        //std::cout << L * L.transpose() << std::endl;

        //std::cout << "Transforming points" << std::endl;
        for (int j = 0; j < gmm_lineset->points_.size(); j++){
            gmm_lineset->points_.at(j) = (pose.topLeftCorner(3,3) * (transform * gmm_lineset->points_.at(j).cast<FP>() +
                                            Cluster.Mean().topLeftCorner(3,1))
                                         + pose.topRightCorner(3,1)).cast<double>();
        }

        completeGeometry = true;
        gmm_lineset->colors_.reserve(gmm_lineset->lines_.size());
        std::vector<Vector2i> visible_lines;
        for (int j = 0; j < gmm_lineset->lines_.size(); j++){
            visible_lines.push_back(gmm_lineset->lines_.at(j));
            gmm_lineset->colors_.emplace_back(OccupancyColor);
        }
        gmm_lineset->lines_ = visible_lines;
        return gmm_lineset;
    }

    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMLinesetColorSingle(const GMMcluster_c& Cluster, GMMMap* map,
                                                                     FP scale, M_o pose){
        auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0, 8);
        auto sphere_lineset = open3d::geometry::LineSet::CreateFromTriangleMesh(*sphere);
        sphere_lineset->colors_.clear();
        std::shared_ptr<open3d::geometry::LineSet> gmm_lineset = std::make_shared<open3d::geometry::LineSet>();
        *gmm_lineset = *sphere_lineset;
        //std::cout << "Computing Cholesky" << std::endl;
        M L = Cluster.CovL();
        M transform = L * scale;
        //std::cout << Clusters.at(idx).Covariance <<std::endl;
        //std::cout << L * L.transpose() << std::endl;
        gmm_lineset->colors_.reserve(gmm_lineset->lines_.size());
        std::vector<Vector2i> visible_lines;
        //std::cout << "Transforming points" << std::endl;
        for (int j = 0; j < gmm_lineset->points_.size(); j++){
            gmm_lineset->points_.at(j) = (pose.topLeftCorner(3,3) * (transform * gmm_lineset->points_.at(j).cast<FP>() +
                                                                     Cluster.Mean().topLeftCorner(3,1))
                                          + pose.topRightCorner(3,1)).cast<double>();
        }

        for (int j = 0; j < gmm_lineset->lines_.size(); j++){
            V pt0 = gmm_lineset->points_.at(gmm_lineset->lines_.at(j)(0)).cast<FP>();
            V pt1 = gmm_lineset->points_.at(gmm_lineset->lines_.at(j)(1)).cast<FP>();
            pt0 = pose.topLeftCorner(3,3).transpose()*(pt0 - pose.topRightCorner(3,1));
            pt1 = pose.topLeftCorner(3,3).transpose()*(pt1 - pose.topRightCorner(3,1));
            V avg_point = (pt0 + pt1) / 2;
            V color;
            if (map->mapParameters.track_color){
                if (map->mapParameters.track_intensity){
                    FP intensity, intensity_var;
                    if (!map->computeIntensityAndVariance(avg_point, intensity, intensity_var)){
                        // Skip invalid color
                        continue;
                    }
                    color << intensity, intensity, intensity;
                } else {
                    M var;
                    if (!map->computeColorAndVariance(avg_point, color, var)){
                        // Skip invalid color
                        continue;
                    }

                }
            } else {
                color = V::Zero();
            }
            visible_lines.push_back(gmm_lineset->lines_.at(j));
            gmm_lineset->colors_.emplace_back(color.cast<double>() / 255);
        }
        gmm_lineset->lines_ = visible_lines;
        return gmm_lineset;
    }


    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMLinesetColorSingle(const GMMcluster_c& Cluster,
                                                                     FP scale, M_o pose, bool colorizeIntensity){
        auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0, 8);
        auto sphere_lineset = open3d::geometry::LineSet::CreateFromTriangleMesh(*sphere);
        sphere_lineset->colors_.clear();
        std::shared_ptr<open3d::geometry::LineSet> gmm_lineset = std::make_shared<open3d::geometry::LineSet>();
        *gmm_lineset = *sphere_lineset;
        //std::cout << "Computing Cholesky" << std::endl;
        M L = Cluster.CovL();
        M transform = L * scale;
        //std::cout << Clusters.at(idx).Covariance <<std::endl;
        //std::cout << L * L.transpose() << std::endl;
        gmm_lineset->colors_.reserve(gmm_lineset->lines_.size());
        //std::cout << "Transforming points" << std::endl;
        for (int j = 0; j < gmm_lineset->points_.size(); j++){
            gmm_lineset->points_.at(j) = (pose.topLeftCorner(3,3) * (transform * gmm_lineset->points_.at(j).cast<FP>() + Cluster.Mean())
                                         + pose.topRightCorner(3,1)).cast<double>();
        }
        for (int j = 0; j < gmm_lineset->lines_.size(); j++){
            Eigen::Vector3d avg_point = (gmm_lineset->points_.at(gmm_lineset->lines_.at(j)(0)) + gmm_lineset->points_.at(gmm_lineset->lines_.at(j)(1))) / 2;
            avg_point = (pose.topLeftCorner(3,3).transpose()*(avg_point.cast<FP>() - pose.topRightCorner(3,1))).cast<double>();
            Eigen::Vector3d predicted_color;
            if (colorizeIntensity){
                predicted_color = Cluster.estIntensityInRGB(avg_point.cast<FP>()).cast<double>()/255;
            } else {
                predicted_color = Cluster.estColor(avg_point.cast<FP>()).cast<double>()/255;
            }
            gmm_lineset->colors_.emplace_back(predicted_color);
            //if (j == gmm_lineset->lines_.size()-1)
            //    std::cout << fmt::format("Color: [{:.2f}, {:.2f}, {:.2f}]\n", predicted_color[0], predicted_color[1], predicted_color[2]);
        }
        return gmm_lineset;
    }

    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMLinesetDebugColorSingle(const GMMcluster_c& Cluster,
                                                                     FP scale, M_o pose){
        auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0, 8);
        auto sphere_lineset = open3d::geometry::LineSet::CreateFromTriangleMesh(*sphere);
        sphere_lineset->colors_.clear();
        std::shared_ptr<open3d::geometry::LineSet> gmm_lineset = std::make_shared<open3d::geometry::LineSet>();
        *gmm_lineset = *sphere_lineset;
        //std::cout << "Computing Cholesky" << std::endl;
        M L = Cluster.CovL();
        M transform = L * scale;
        //std::cout << Clusters.at(idx).Covariance <<std::endl;
        //std::cout << L * L.transpose() << std::endl;
        gmm_lineset->colors_.reserve(gmm_lineset->lines_.size());
        //std::cout << "Transforming points" << std::endl;
        for (int j = 0; j < gmm_lineset->points_.size(); j++){
            gmm_lineset->points_.at(j) = (pose.topLeftCorner(3,3) * (transform * gmm_lineset->points_.at(j).cast<FP>() + Cluster.Mean())
                                          + pose.topRightCorner(3,1)).cast<double>();
        }

        // Note that debug color needs to be set elsewhere
        Eigen::Vector3d debug_color;
        if (Cluster.has_debug_color){
            debug_color = Cluster.debug_color.cast<double>() / 255;
        } else {
            debug_color.setOnes();
        }

        for (int j = 0; j < gmm_lineset->lines_.size(); j++){
            Eigen::Vector3d avg_point = (gmm_lineset->points_.at(gmm_lineset->lines_.at(j)(0)) + gmm_lineset->points_.at(gmm_lineset->lines_.at(j)(1))) / 2;
            avg_point = (pose.topLeftCorner(3,3).transpose()*(avg_point.cast<FP>() - pose.topRightCorner(3,1))).cast<double>();
            gmm_lineset->colors_.emplace_back(debug_color);
            //if (j == gmm_lineset->lines_.size()-1)
            //    std::cout << fmt::format("Color: [{:.2f}, {:.2f}, {:.2f}]\n", predicted_color[0], predicted_color[1], predicted_color[2]);
        }
        return gmm_lineset;
    }

    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMBBoxLinesetSingle(const GMMcluster_c& Cluster, const Eigen::Vector3d& color,
                                                                    FP scale, M_o pose){
        auto bbox = open3d::geometry::AxisAlignedBoundingBox(Cluster.BBox.min().cast<double>(), Cluster.BBox.max().cast<double>());
        auto bbox_lineset = open3d::geometry::LineSet::CreateFromAxisAlignedBoundingBox(bbox);
        bbox_lineset->colors_.clear();
        for (int j = 0; j < bbox_lineset->points_.size(); j++){
            bbox_lineset->points_.at(j) = pose.topLeftCorner(3,3).cast<double>() * bbox_lineset->points_.at(j)
                    + pose.topRightCorner(3,1).cast<double>();
        }
        for (int j = 0; j < bbox_lineset->lines_.size(); j++){
            bbox_lineset->colors_.push_back(color);
        }
        return bbox_lineset;
    }

    std::shared_ptr<open3d::geometry::LineSet> GenerateGMMBBoxLinesetSingle(const GMMcluster_o& Cluster, const Eigen::Vector3d& color,
                                                                    FP scale, M_o pose){
        Rect BBox;
        Cluster.computeBBox(BBox, scale);
        auto bbox = open3d::geometry::AxisAlignedBoundingBox(BBox.min().cast<double>(), BBox.max().cast<double>());
        auto bbox_lineset = open3d::geometry::LineSet::CreateFromAxisAlignedBoundingBox(bbox);
        bbox_lineset->colors_.clear();
        for (int j = 0; j < bbox_lineset->points_.size(); j++){
            bbox_lineset->points_.at(j) = pose.topLeftCorner(3,3).cast<double>() * bbox_lineset->points_.at(j)
                    + pose.topRightCorner(3,1).cast<double>();
        }
        for (int j = 0; j < bbox_lineset->lines_.size(); j++){
            bbox_lineset->colors_.push_back(color);
        }
        return bbox_lineset;
    }

    std::shared_ptr<open3d::geometry::LineSet> GenerateGaussianGraph(const GMMMap& map, const Eigen::Vector3d& obs_color,
                                                                     const Eigen::Vector3d& free_color, bool free_region_only) {
        auto result = std::make_shared<open3d::geometry::LineSet>();

        auto free_gaussians = map.getFreeGaussians();

        // Determine total number of elements that we need to pre-allocate
        unsigned num_points = free_gaussians.size();
        unsigned number_edges = 0;
        for (auto& cluster : free_gaussians){
            number_edges += cluster->numOfNeighbors();
        }
        result->points_.reserve(num_points);
        result->lines_.reserve(number_edges);
        result->colors_.reserve(number_edges);

        // Populate the lineset
        unsigned cur_point_idx = 0;
        for (auto& cluster : free_gaussians){
            result->points_.emplace_back(cluster->Mean().cast<double>());
            int num_of_neighbors = 0;
            for (auto& neighbor : cluster->getNeighbors()){
                if (neighbor->is_free){
                    result->points_.emplace_back(neighbor->Mean().cast<double>());
                    result->lines_.emplace_back(cur_point_idx, result->points_.size() - 1);
                    result->colors_.emplace_back(free_color);
                    num_of_neighbors++;
                } else if (!free_region_only) {
                    result->points_.emplace_back(neighbor->Mean().cast<double>());
                    result->lines_.emplace_back(cur_point_idx, result->points_.size() - 1);
                    result->colors_.emplace_back(obs_color);
                    num_of_neighbors++;
                }
            }
            cur_point_idx += num_of_neighbors + 1;
        }

        return result;
    }

    std::shared_ptr<open3d::geometry::LineSet> GenerateNavigationGraph(
            const GMMapSamplingBasedPlanner& planner, const Eigen::Vector3d& color){
        auto result = std::make_shared<open3d::geometry::LineSet>();
        planner.exportGraph(result->points_, result->lines_);
        result->PaintUniformColor(color);
        return result;
    }

    std::shared_ptr<open3d::geometry::LineSet> GenerateSolutionPath(
            const GMMapSamplingBasedPlanner& planner, const Eigen::Vector3d& color){
        auto result = std::make_shared<open3d::geometry::LineSet>();
        planner.exportSolutionPath(result->points_, result->lines_);
        result->PaintUniformColor(color);
        return result;
    }

    bool vizMapFlags::gmmVisible() {
        return show_gmm_obs | show_gmm_free | show_gmm_free_near_obs | show_gmm_color | show_gmm_debug_color | show_gmm_bbox_obs | show_gmm_bbox_free;
    }

    bool vizMapFlags::occVarVisible() {
        return show_occupancy_pts | show_variance_pts;
    }

    bool vizMapFlags::occVoxelVisible() {
        return show_occupancy_voxels;
    }

    void vizMapFlags::updateGMMGeometryFlags(const vizMapAtomicFlags& curFlags){
        // occ_low and occ_high are not used for now. To use, check of changes in the conditional block below
        occ_low = curFlags.occ_low;
        occ_high = curFlags.occ_high;

        if (std != curFlags.std) {
            update_gmm_geometry = true;
            std = curFlags.std;
        }
    }

    void vizMapFlags::updatePCDGeometryFlags(const vizMapAtomicFlags& curFlags){
        if (occ_var_pt_low != curFlags.occ_var_pt_low || occ_var_pt_high != curFlags.occ_var_pt_high) {
            // Update the geometry for points and voxels
            update_occ_var_geometry = true;
            update_occ_voxels_geometry = true;

            occ_var_pt_low = curFlags.occ_var_pt_low;
            occ_var_pt_high = curFlags.occ_var_pt_high;
        }

        if (voxel_resolution != curFlags.voxel_resolution){
            update_occ_voxels_source = true;
            voxel_resolution = curFlags.voxel_resolution;
        }

        // Check if the number of samples or frames has changed
        if (!curFlags.from_ray) {
            if (curFlags.num_of_points != num_of_points) {
                //std::cout << "Number of samples changed! Pending reset for occ_var_from_bbox" << std::endl;
                update_occ_var_building_block = true;
                num_of_points = curFlags.num_of_points;
            }
        } else {
            if (curFlags.num_inaccurate_roc_frames != num_inaccurate_roc_frames) {
                update_occ_var_building_block = true;
                num_inaccurate_roc_frames = curFlags.num_inaccurate_roc_frames;
            }
        }
    }

    void vizMapFlags::updateEnvBBoxGeometryFlags(const vizMapAtomicFlags& curFlags){
        if (env_bbox_low_x != curFlags.env_bbox_low_x || env_bbox_low_y != curFlags.env_bbox_low_y ||
            env_bbox_low_z != curFlags.env_bbox_low_z ||
            env_bbox_high_x != curFlags.env_bbox_high_x || env_bbox_high_y != curFlags.env_bbox_high_y ||
            env_bbox_high_z != curFlags.env_bbox_high_z) {
            update_env_geometry = true;

            env_bbox_low_x = curFlags.env_bbox_low_x;
            env_bbox_low_y = curFlags.env_bbox_low_y;
            env_bbox_low_z = curFlags.env_bbox_low_z;
            env_bbox_high_x = curFlags.env_bbox_high_x;
            env_bbox_high_y = curFlags.env_bbox_high_y;
            env_bbox_high_z = curFlags.env_bbox_high_z;
        }
    }

    bool vizMapFlags::isEnvBBoxGeometryUpdated(const vizMapAtomicFlags &curFlags){
        if (env_bbox_low_x != curFlags.env_bbox_low_x || env_bbox_low_y != curFlags.env_bbox_low_y ||
            env_bbox_low_z != curFlags.env_bbox_low_z ||
            env_bbox_high_x != curFlags.env_bbox_high_x || env_bbox_high_y != curFlags.env_bbox_high_y ||
            env_bbox_high_z != curFlags.env_bbox_high_z) {
            return true;
        } else {
            return false;
        }
    }

    void vizMapFlags::updateVisibilityFlags(const vizMapAtomicFlags& curFlags){
        if (show_env_bbox != curFlags.show_env_bbox){
            update_env_visibility = true;
            show_env_bbox = curFlags.show_env_bbox;
        }

        if (show_gmm_obs != curFlags.show_gmm_obs || show_gmm_free != curFlags.show_gmm_free ||
            show_gmm_free_near_obs != curFlags.show_gmm_free_near_obs ||
            show_gmm_color != curFlags.show_gmm_color ||
            show_gmm_debug_color != curFlags.show_gmm_debug_color ||
            show_gmm_bbox_obs != curFlags.show_gmm_bbox_obs ||
            show_gmm_bbox_free != curFlags.show_gmm_bbox_free) {

            update_gmm_visibility = true;

            show_gmm_obs = curFlags.show_gmm_obs;
            show_gmm_free = curFlags.show_gmm_free;
            show_gmm_free_near_obs = curFlags.show_gmm_free_near_obs;
            show_gmm_color = curFlags.show_gmm_color;
            show_gmm_debug_color = curFlags.show_gmm_debug_color;
            show_gmm_bbox_obs = curFlags.show_gmm_bbox_obs;
            show_gmm_bbox_free = curFlags.show_gmm_bbox_free;
        }

        if (show_occupancy_pts != curFlags.show_occupancy_pts || show_variance_pts != curFlags.show_variance_pts) {
            update_occ_var_visibility = true;

            show_occupancy_pts = curFlags.show_occupancy_pts;
            show_variance_pts = curFlags.show_variance_pts;
        }

        if (show_occupancy_voxels != curFlags.show_occupancy_voxels){
            update_occ_voxels_visibility = true;
            show_occupancy_voxels = curFlags.show_occupancy_voxels;
        }

        if (from_ray != curFlags.from_ray){
            update_occ_var_source = true;
            from_ray = curFlags.from_ray;
        }
    }

    void vizMapFlags::clearGeometryUpdateFlags(){
        update_env_geometry = false;
        update_gmm_geometry = false; // Determines if the GMM should be upgraded
        update_occ_var_geometry = false; // Determines if the occupancy or variance points should be recomputed
        update_occ_var_building_block = false;
    }

    void vizMapFlags::clearVisibilityUpdateFlags(){
        update_env_visibility = false; // Determines if the bounding box is changed
        update_gmm_visibility = false; // Determines if the GMM should be upgraded
        update_occ_var_visibility = false; // Determines if the occupancy or variance points should be recomputed
        update_occ_var_source = false; // Determines if the source of the occupancy and variance has changed.
        update_occ_voxels_visibility = false;
    }

    bool pathVizFlags::updateStartOrDestination(const gmm::pathVizAtomicFlags &curFlags) {
        bool cur_locations_updated = (curFlags.start_scale_x != start_scale_x) ||
                (curFlags.start_scale_y != start_scale_y) ||
                (curFlags.start_scale_z != start_scale_z) ||
                (curFlags.dest_scale_x != dest_scale_x) ||
                (curFlags.dest_scale_y != dest_scale_y) ||
                (curFlags.dest_scale_z != dest_scale_z);

        if (cur_locations_updated){
            start_scale_x = curFlags.start_scale_x;
            start_scale_y = curFlags.start_scale_y;
            start_scale_z = curFlags.start_scale_z;
            dest_scale_x = curFlags.dest_scale_x;
            dest_scale_y = curFlags.dest_scale_y;
            dest_scale_z = curFlags.dest_scale_z;
        }

        return cur_locations_updated;
    }

    bool pathVizFlags::isLocationUpdated() const {
        return locations_updated;
    }

    void pathVizFlags::enableLocationUpdateFlag() {
        locations_updated = true;
    }

    void pathVizFlags::disableLocationUpdateFlag() {
        locations_updated = false;
    }

    void pathVizFlags::enableGaussianUpdateFlag(){
        gaussians_updated = true;
    }

    bool pathVizFlags::areGaussiansUpdated() const {
        return gaussians_updated;
    }

    void pathVizFlags::disableGaussianUpdateFlag() {
        gaussians_updated = false;
    }

    void pathVizFlags::enableNavigationGraphUpdateFlag() {
        navigation_graph_updated = true;
    }

    bool pathVizFlags::isNavigationGraphUpdated() const {
        return navigation_graph_updated;
    }

    void pathVizFlags::disableNavigationUpdateFlag() {
        navigation_graph_updated = false;
    }

    void ellipsoidWithMetadata::setGeometry(const GMMcluster_o *gaussian_cluster) {
        // Initializes the association between Gaussian geometry to its object
        if (cluster == nullptr){
            if (gaussian_cluster->is_free){
                cluster = std::make_shared<GMMcluster_o>(*gaussian_cluster);
            } else {
                cluster = std::make_shared<GMMcluster_c>(*dynamic_cast<const GMMcluster_c*>(gaussian_cluster));
            }
        } else {
            if (gaussian_cluster->is_free){
                *cluster = *gaussian_cluster;
            } else {
                *dynamic_cast<GMMcluster_c*>(cluster.get()) = *dynamic_cast<const GMMcluster_c*>(gaussian_cluster);
            }
        }

        // Clear geometries
        min_occ = -1;
        max_occ = 10;
        std_occ_scale = -1;
        std_color_scale = -1;
        std_debug_color_scale = -1;
        completeOccGMM = false;
        color_geometry = nullptr;
        color_debug_geometry = nullptr;
        occ_geometry = nullptr;
        bbox = nullptr;
    }

    bool ellipsoidWithMetadata::updateOccGeometry(const Eigen::Vector3d& OccupancyColor,const FP& std){
        //if (occ_geometry == nullptr || bbox == nullptr || occ_interval_overlaps){
        if (occ_geometry == nullptr || bbox == nullptr || (!completeOccGMM || std_occ_scale != std)){
            // Generate geometries for occupancy GMM and its bbox
            occ_geometry = GenerateGMMLinesetOccSingle(*cluster,  OccupancyColor,
                                                       completeOccGMM, std);
            bbox = GenerateGMMBBoxLinesetSingle(*cluster, OccupancyColor, std);
            std_occ_scale = std;
            updateVisualizer = true;
            return true;
        }
        return false;
    }

    bool ellipsoidWithMetadata::updateColorGeometry(const FP &std, const bool& track_intensity) {
        if (color_geometry == nullptr || std != std_color_scale){
            // Generate geometries for color GMM
            color_geometry = GenerateGMMLinesetColorSingle(*dynamic_cast<GMMcluster_c*>(cluster.get()), std,
                                                           M_o::Identity(), track_intensity);
            std_color_scale = std;
            return true;
        }
        return false;
    }

    bool ellipsoidWithMetadata::updateColorGeometry(const FP& std, GMMMap* map){
        if (color_geometry == nullptr || std != std_color_scale){
            // Generate geometries for color GMM
            color_geometry = GenerateGMMLinesetColorSingle(*dynamic_cast<GMMcluster_c*>(cluster.get()), map,
                                                           std,M_o::Identity());
            std_color_scale = std;
            return true;
        }
        return false;
    }

    bool ellipsoidWithMetadata::updateDebugColorGeometry(const FP& std){
        if (color_debug_geometry == nullptr || std != std_debug_color_scale){
            // Generate geometries for color GMM
            color_debug_geometry = GenerateGMMLinesetDebugColorSingle(*dynamic_cast<GMMcluster_c*>(cluster.get()),
                                                           std,M_o::Identity());
            std_debug_color_scale = std;
            return true;
        }
        return false;
    }

    void GaussianLinesets::updateGaussians(const map_visualization& gmm_changes){
        // Update geometry sources for the Gaussians
        // Assign Gaussians geometries to the appropriate visualization partition
        // Note: Visualization geometry is not constructed in this function

        // Take away clusters that we don't need
        for (const auto& cluster_name : gmm_changes.delete_obs_cluster){
            auto it = gaussians.find(cluster_name);
            linesetRtree.Remove(it->second.cluster->BBox, cluster_name);
            obs_gaussian_partition.remove(&it->second);
            gaussians.erase(it);
        }

        for (const auto& cluster_name : gmm_changes.delete_free_cluster){
            auto it = gaussians.find(cluster_name);
            linesetRtree.Remove(it->second.cluster->BBox, cluster_name);
            if (it->second.cluster->near_obs){
                free_near_obs_gaussian_partition.remove(&it->second);
            } else {
                free_gaussian_partition.remove(&it->second);
            }
            gaussians.erase(it);
        }

        // Update selected clusters
        for (const auto& cluster : gmm_changes.update_obs_cluster){
            auto it = gaussians.find(cluster.first);
            linesetRtree.Remove(it->second.cluster->BBox, cluster.first);
            it->second.setGeometry(&cluster.second);
            linesetRtree.Insert(it->second.cluster->BBox, cluster.first);
        }

        for (const auto& cluster : gmm_changes.update_free_cluster){
            auto it = gaussians.find(cluster.first);
            linesetRtree.Remove(it->second.cluster->BBox, cluster.first);
            it->second.setGeometry(&cluster.second);
            linesetRtree.Insert(it->second.cluster->BBox, cluster.first);
        }

        // Add clusters
        for (const auto& cluster : gmm_changes.add_obs_cluster){
            ellipsoidWithMetadata new_cluster;
            new_cluster.setGeometry(&cluster.second);
            auto it = gaussians.insert(std::make_pair(cluster.first, new_cluster));
            linesetRtree.Insert(new_cluster.cluster->BBox, cluster.first);
            obs_gaussian_partition.add(&it.first->second);
        }

        for (const auto& cluster : gmm_changes.add_free_cluster){
            ellipsoidWithMetadata new_cluster;
            new_cluster.setGeometry(&cluster.second);
            auto it = gaussians.insert(std::make_pair(cluster.first, new_cluster));
            linesetRtree.Insert(new_cluster.cluster->BBox, cluster.first);
            if (cluster.second.near_obs){
                free_near_obs_gaussian_partition.add(&it.first->second);
            } else {
                free_gaussian_partition.add(&it.first->second);
            }
        }
    }

    std::string GaussianLinesets::changePrefix(const std::string &source_label, const std::string &target_prefix) {
        return target_prefix + source_label.substr(source_label.find_last_of('_')+1);
    }

    void GaussianLinesets::clear() {
        gaussians.clear();
        active_obs_gaussians.clear();
        active_free_gaussians.clear();
        active_free_near_obs_gaussians.clear();
        linesetRtree.RemoveAll();

        obs_gaussian_partition.clear();
        free_gaussian_partition.clear();
        free_near_obs_gaussian_partition.clear();
    }

    void PartitionLinesets::clear() {
        partition_idx = 0;
        partition_to_gaussian_map.clear();
        gaussian_to_partition_map.clear();
        active_gaussian_linesets.clear();
    }

    int PartitionLinesets::findAvailablePartition(int start_idx) const {
        int available_partition_idx;
        // When searching for available partition, we need to ensure spatial locality (i.e., update to 2 partition index from start)
        for (available_partition_idx = start_idx; available_partition_idx >= std::max(0, start_idx - 2); available_partition_idx--){
            if (partition_to_gaussian_map.at(available_partition_idx).size() < num_gaussians_per_partition){
                return available_partition_idx;
            }
        }
        return -1;
    }

    void PartitionLinesets::add(ellipsoidWithMetadata *gau_metadata) {
        // Check if the metadata is not in a partition
        if (gaussian_to_partition_map.find(gau_metadata) == gaussian_to_partition_map.end()){
            // Create new partition if needed
            if (partition_to_gaussian_map.empty()){
                // Make a new partition
                partition_to_gaussian_map[partition_idx];
                // Insert into the partition
                gaussian_to_partition_map.insert(std::pair(gau_metadata, partition_idx));
                partition_to_gaussian_map.at(partition_idx).insert(gau_metadata);
            } else {
                auto available_partition_idx = findAvailablePartition(partition_idx);
                if (available_partition_idx < 0){
                    // Make a new partition
                    partition_idx++;
                    partition_to_gaussian_map[partition_idx];
                    // Insert into the partition
                    gaussian_to_partition_map.insert(std::pair(gau_metadata, partition_idx));
                    partition_to_gaussian_map.at(partition_idx).insert(gau_metadata);
                } else {
                    // Insert into an existing partition
                    gaussian_to_partition_map.insert(std::pair(gau_metadata, available_partition_idx));
                    partition_to_gaussian_map.at(available_partition_idx).insert(gau_metadata);
                }
            }
        } else {
            std::cout << fmt::format("Warning: Adding a Gaussian geometry that is already in the partition {}!", occ_gau_partition_base) << std::endl;
        }
    }

    void PartitionLinesets::remove(ellipsoidWithMetadata *gau_metadata) {
        // Remove from gaussian-to-partition, and partition-to-gaussian maps
        auto gaussian_to_partition_map_it = gaussian_to_partition_map.find(gau_metadata);
        if (gaussian_to_partition_map_it != gaussian_to_partition_map.end()){
            partition_to_gaussian_map.at(gaussian_to_partition_map_it->second).erase(gau_metadata);
            gaussian_to_partition_map.erase(gaussian_to_partition_map_it);
        } else {
            std::cout << fmt::format("Warning: Removing a Gaussian geometry that does not exist in the partition {}!", occ_gau_partition_base) << std::endl;
        }
    }

    void PartitionLinesets::addActive(ellipsoidWithMetadata *gau_metadata, bool updateRequired) {
        // Add elements into to the active sets
        auto gaussian_to_partition_map_it = gaussian_to_partition_map.find(gau_metadata);
        if (gaussian_to_partition_map_it != gaussian_to_partition_map.end()){
            int gau_partition_idx = gaussian_to_partition_map_it->second;
            auto active_gaussian_linesets_it = active_gaussian_linesets.find(gau_partition_idx);
            if (active_gaussian_linesets_it != active_gaussian_linesets.end()){
                active_gaussian_linesets_it->second.next_lineset_src.insert(gau_metadata);
                active_gaussian_linesets_it->second.updateVisualizer |= updateRequired;
            } else {
                auto& new_active_partition = active_gaussian_linesets[gau_partition_idx];
                new_active_partition.next_lineset_src.insert(gau_metadata);
                new_active_partition.updateVisualizer = true;
            }
        } else {
            throw std::invalid_argument("Cannot add an active Gaussian that is not mapped to a partition!");
        }
    }

    void PartitionLinesets::updateAllActiveGeometries() {
        // Update geometries from sources
        for (auto& active_gaussian : active_gaussian_linesets){
            auto& LinesetMetadata = active_gaussian.second;
            // Only update if the 1) some Gaussians changed or 2) number of Gaussians changed
            if (LinesetMetadata.updateVisualizer || LinesetMetadata.next_lineset_src != LinesetMetadata.cur_lineset_src){
                // Update with newest information
                LinesetMetadata.cur_lineset_src = LinesetMetadata.next_lineset_src;
                LinesetMetadata.updateVisualizer = true;
                LinesetMetadata.next_lineset_src.clear();

                // Update lineset information
                LinesetMetadata.occ_gaussians.Clear();
                LinesetMetadata.color_gaussians.Clear();
                LinesetMetadata.color_debug_gaussians.Clear();
                LinesetMetadata.bboxes.Clear();
                for (auto& ellipsoid : LinesetMetadata.cur_lineset_src){
                    LinesetMetadata.appendLineset(ellipsoid);
                }
            } else {
                LinesetMetadata.next_lineset_src.clear();
            }
        }
    }

    void PartitionLinesets::PartitionLinesetMetadata::appendLineset(ellipsoidWithMetadata *gau_metadata) {
        if (gau_metadata->color_geometry != nullptr){
            auto offset = color_gaussians.points_.size();
            for (auto& pt : gau_metadata->color_geometry->points_){
                color_gaussians.points_.push_back(pt);
            }
            for (auto& line : gau_metadata->color_geometry->lines_){
                color_gaussians.lines_.emplace_back(line(0) + offset, line(1) + offset);
            }
            for (auto& color : gau_metadata->color_geometry->colors_){
                color_gaussians.colors_.push_back(color);
            }
        }

        if (gau_metadata->color_debug_geometry != nullptr){
            auto offset = color_debug_gaussians.points_.size();
            for (auto& pt : gau_metadata->color_debug_geometry->points_){
                color_debug_gaussians.points_.push_back(pt);
            }
            for (auto& line : gau_metadata->color_debug_geometry->lines_){
                color_debug_gaussians.lines_.emplace_back(line(0) + offset, line(1) + offset);
            }
            for (auto& color : gau_metadata->color_debug_geometry->colors_){
                color_debug_gaussians.colors_.push_back(color);
            }
        }

        if (gau_metadata->occ_geometry != nullptr){
            auto offset = occ_gaussians.points_.size();
            for (auto& pt : gau_metadata->occ_geometry->points_){
                occ_gaussians.points_.push_back(pt);
            }
            for (auto& line : gau_metadata->occ_geometry->lines_){
                occ_gaussians.lines_.emplace_back(line(0) + offset, line(1) + offset);
            }
            for (auto& color : gau_metadata->occ_geometry->colors_){
                occ_gaussians.colors_.push_back(color);
            }
        }

        if (gau_metadata->bbox != nullptr){
            auto offset = bboxes.points_.size();
            for (auto& pt : gau_metadata->bbox->points_){
                bboxes.points_.push_back(pt);
            }
            for (auto& line : gau_metadata->bbox->lines_){
                bboxes.lines_.emplace_back(line(0) + offset, line(1) + offset);
            }
            for (auto& color : gau_metadata->bbox->colors_){
                bboxes.colors_.push_back(color);
            }
        }
    }

    void PartitionLinesets::printInfo() const {
        std::cout << fmt::format("Printing information about a partition with bases: {}, {}, {}, {}",
                                 occ_gau_partition_base, color_gau_partition_base, color_debug_gau_partition_base, bbox_partition_base) << std::endl;
        std::cout << fmt::format("There are currently {} partitions containing {} Gaussians. partition_idx = {}",
                                 active_gaussian_linesets.size(), gaussian_to_partition_map.size(), partition_idx) << std::endl;
        std::cout << fmt::format("Print partition information below:") << std::endl;
        for (const auto& partition : active_gaussian_linesets) {
            std::cout << fmt::format("\tPartition {}: Number of active Gaussians = {}/{}, updateVisualizer = {}",
                                     partition.first, partition.second.cur_lineset_src.size(),
                                     partition_to_gaussian_map.at(partition.first).size(),
                                     partition.second.updateVisualizer) << std::endl;
        }
    }

    void PartitionLinesets::checkActiveValidity() const {
        // Check if active Gaussians are currently mapped correctly
        for (const auto& partition : active_gaussian_linesets){
            for (const auto& ellipsoid : partition.second.cur_lineset_src){
                const auto& it = gaussian_to_partition_map.find(ellipsoid);
                if (it == gaussian_to_partition_map.end() || it->second != partition.first){
                    throw std::invalid_argument(fmt::format("Invalid active Gaussian geometry in partition {}! Not found ({}) or Incorrect partition (expected {} != {})",
                                                            occ_gau_partition_base, it == gaussian_to_partition_map.end(), it->second, partition.first));
                }
            }
        }
    }

    std::string PartitionLinesets::getOccGaussianPartitionLabel(int idx) const {
        return occ_gau_partition_base + std::to_string(idx);
    }

    std::string PartitionLinesets::getColorGaussianPartitionLabel(int idx) const {
        return color_gau_partition_base + std::to_string(idx);
    }

    std::string PartitionLinesets::getColorDebugGaussianPartitionLabel(int idx) const {
        return color_debug_gau_partition_base + std::to_string(idx);
    }

    std::string PartitionLinesets::getBBoxPartitionLabel(int idx) const {
        return bbox_partition_base + std::to_string(idx);
    }

    // Definition for member functions in GMMMapViz
    GMMMapViz::GMMMapViz(const std::shared_ptr<open3d::visualization::gui::SceneWidget>& env_widget,
                         const std::shared_ptr<GMMMap>& map,
                         std::atomic<int>* update_interval,
                       const Eigen::Vector3d& prob_error_color,
                       const std::string& colorClass,
                       const std::string& colorName,
                       FP unexplored_evidence,
                       FP unexplored_variance,
                       FP ray_sampling_dist){

        this->prob_error_color = prob_error_color;
        this->unexplored_evidence = unexplored_evidence;
        this->unexplored_variance = unexplored_variance;
        this->ray_sampling_dist = ray_sampling_dist;

        // Visualization setup
        this->env_widget = std::move(env_widget);

        // Initialize visualization on the map side
        this->num_processed_frames = 0;
        this->update_interval = update_interval;

        // Need to avoid overlap of string prefixes!
        this->gmm_map = map;
        this->gmm_map->configureVisualization();
        this->gau_obs_base = map->gau_obs_base;
        this->gau_free_base = map->gau_free_base;
        this->gau_free_near_obs_base = map->gau_free_near_obs_base;
        this->gau_color_base = map->gau_color_base;
        this->gau_debug_color_base = map->gau_debug_color_base;
        this->bbox_obs_base = map->bbox_obs_base;
        this->bbox_free_base = map->bbox_free_base;
        this->geometry_lock_ptr = &map->geometry_lock;

        this->occ_pcd_base = "occ_pt_pcd_";
        this->occ_var_pcd_base = "occ_var_pcd_";
        this->env_bbox_base = "env_bbox";
        this->occ_voxel_base = "occ_voxel";

        // Track previous visualization states
        this->occ_var_update_avaliable = false;
        this->gmm_update_avaliable = false; // indicates whether or not the gmm is updated
        this->env_visibility_update_avaliable = false; // Indicates whether or not environment bbox is changed

        // Material - For each Gaussian in the frame, create GMM and assign objects
        this->mat = open3d::visualization::rendering::MaterialRecord();
        this->mat.shader = "unlitLine";
        this->mat.line_width = 3.0f;

        this->adaptive_pt_mat = open3d::visualization::rendering::MaterialRecord();
        this->adaptive_pt_mat.shader = "defaultUnlit";
        this->adaptive_pt_mat.point_size = 5.0f;

        // ColorMap
        this->OccVarCMap = colormap::createColorMap(colorClass, colorName, 1000);
        this->prob_zero_color = OccVarCMap.row(0).transpose();
        this->prob_one_color = OccVarCMap.bottomLeftCorner<1,3>().transpose();

        // BBox
        this->pre_adj_BBox.setEmpty();

        // OccVariance PCDs (from rays or random samples within the bounding box)
        occ_var_from_rays = new OccVarWithMetadata(this->prob_error_color, this->OccVarCMap);
        occ_var_from_bbox = new OccVarWithMetadata(this->OccVarCMap);

        // Partition name
        gmm_linesets.obs_gaussian_partition.occ_gau_partition_base = "obs_occ_gau_partition_";
        gmm_linesets.obs_gaussian_partition.color_gau_partition_base = "obs_color_gau_partition_";
        gmm_linesets.obs_gaussian_partition.color_debug_gau_partition_base = "obs_color_debug_gau_partition_";
        gmm_linesets.obs_gaussian_partition.bbox_partition_base = "obs_bbox_partition_";

        gmm_linesets.free_gaussian_partition.occ_gau_partition_base = "free_occ_gau_partition_";
        gmm_linesets.free_gaussian_partition.color_gau_partition_base = "free_color_gau_partition_";
        gmm_linesets.free_gaussian_partition.color_debug_gau_partition_base = "free_color_debug_gau_partition_";
        gmm_linesets.free_gaussian_partition.bbox_partition_base = "free_bbox_partition_";

        gmm_linesets.free_near_obs_gaussian_partition.occ_gau_partition_base = "free_near_obs_occ_gau_partition_";
        gmm_linesets.free_near_obs_gaussian_partition.color_gau_partition_base = "free_near_obs_color_gau_partition_";
        gmm_linesets.free_near_obs_gaussian_partition.color_debug_gau_partition_base = "free_near_obs_color_debug_gau_partition_";
        gmm_linesets.free_near_obs_gaussian_partition.bbox_partition_base = "free_near_obs_bbox_partition_";
    }

    void GMMMapViz::updateGeometry(const vizMapAtomicFlags& sliderFlags, bool final_frame){
        // Just updates geometries and map, not the visualizer itself ("write")
        // In this function, we update the pointer to gaussians (all Gaussians), Gaussian geometries (visible ones), and geometries themselves (env_bbox)
        {
            const std::lock_guard<std::mutex> g_lock(*geometry_lock_ptr);

            // Update geometry flags. These flags are cleared in the current function call
            curMapVizFlags.updateGMMGeometryFlags(sliderFlags);
            curMapVizFlags.updatePCDGeometryFlags(sliderFlags);
            curMapVizFlags.updateEnvBBoxGeometryFlags(sliderFlags);

            // We update visibility flags here to keep them in sync with geometry flags
            // Visibility flags are cleared in the visualizer thread
            curMapVizFlags.updateVisibilityFlags(sliderFlags);
            bool update_interval_met;
            if (gmm_map != nullptr) {
                // Check if the map can be updated. If so, update the sources of geometries
                if (gmm_map->remainingVisualizationExists()) {
                    // Transfer updates here!
                    map_update_avaliable = true;
                    // Transfer updates
                    cur_map_updates = gmm_map->cur_visualization_update;
                    // Update the number of frames
                    num_processed_frames += cur_map_updates.num_frames;
                    // Clear updates on the mapping side
                    gmm_map->cur_visualization_update.clear();
                    // Update geometry source
                    updateGeometrySources();
                    // Clear current updates after it is processed by updateGeometrySources
                    cur_map_updates.clear();
                }

                update_interval_met = final_frame | (((num_processed_frames - 1) % *update_interval == 0) & map_update_avaliable);

                // This enables the delayed computation of occupancy and variance points
                bool building_block_available;
                if (curMapVizFlags.from_ray) {
                    building_block_available = occ_var_from_rays->building_blocks_computed;
                } else {
                    building_block_available = occ_var_from_bbox->building_blocks_computed;
                }

                bool computing_occ_var_pts = curMapVizFlags.occVarVisible() && (curMapVizFlags.update_occ_var_building_block || !building_block_available);

                // Update geometries if the above conditions do not hold
                if (curMapVizFlags.update_env_geometry ||
                    computing_occ_var_pts ||
                    update_interval_met) {
                    redrawEnvBBox();
                    curMapVizFlags.update_env_geometry = false;
                    env_visibility_update_avaliable = true;
                }

                // Then, we use the updated bounding box to update other geometric objects (intersects or falls within the bounding box)
                if (curMapVizFlags.gmmVisible() || update_interval_met) {
                    if (curMapVizFlags.update_gmm_geometry ||
                        env_visibility_update_avaliable ||
                        computing_occ_var_pts ||
                        update_interval_met) {
                        // Update the geometries within the bounding box for all gmm geometries
                        updateGMM();
                        curMapVizFlags.update_gmm_geometry = false;
                        gmm_update_avaliable = true;
                    }
                }

                // Check if the number of samples or number of frames changed
                if (curMapVizFlags.update_occ_var_building_block) {
                    if (!curMapVizFlags.from_ray){
                        occ_var_from_bbox->clear();
                        curMapVizFlags.update_occ_var_building_block = false;
                    } else {
                        occ_var_from_rays->clear();
                        curMapVizFlags.update_occ_var_building_block = false;
                    }
                }

                // Build geometry here if it was not built previously
                if (curMapVizFlags.occVarVisible()) {
                    if (curMapVizFlags.update_occ_var_geometry || env_visibility_update_avaliable ||
                        !building_block_available || update_interval_met) {
                        redrawOccVarPointCloud();
                        curMapVizFlags.update_occ_var_geometry = false;
                        occ_var_update_avaliable = true;
                    }
                }

                curMapVizFlags.update_occ_voxels_source = curMapVizFlags.update_occ_voxels_source | map_update_avaliable;
                if (curMapVizFlags.occVoxelVisible()){
                    if (curMapVizFlags.update_occ_voxels_geometry || curMapVizFlags.update_occ_voxels_source || env_visibility_update_avaliable || update_interval_met) {
                        // Regenerate voxels
                        generateOccupancyVoxels(curMapVizFlags.voxel_resolution, curMapVizFlags.update_occ_voxels_source);
                        curMapVizFlags.update_occ_voxels_source = false;
                        curMapVizFlags.update_occ_voxels_geometry = false;
                        occ_voxels_update_avaliable = true;
                    }
                }
            }
            map_update_avaliable = false;
        }
    }

    void GMMMapViz::clearOccVarGeometry(){
        if (curMapVizFlags.from_ray){
            occ_var_from_rays->clear();
        } else {
            occ_var_from_bbox->clear();
        }
    }

    void GMMMapViz::clearAllGeometry(){
        // Clear all visualization geometry
        occ_voxels = nullptr;
        occ_voxels_with_prob_rtree.RemoveAll();
        occ_var_from_rays->clear();
        occ_var_from_bbox->clear();
        clearGMMGeometry();
    }

    void GMMMapViz::clearGMMGeometry(){
        // Clear GMM visualization geometry
        gmm_linesets.clear();
    }

    void GMMMapViz::updateGMM() {
        // Update geometry linesets of the visible Gaussians
        gmm_linesets.active_free_gaussians.clear();
        gmm_linesets.active_obs_gaussians.clear();
        gmm_linesets.active_free_near_obs_gaussians.clear();
        //std::cout << "Active list cleared!" << std::endl;

        // Directly search the lineset RTree
        auto visible_clusters_labels = gmm_linesets.linesetRtree.Search(adj_BBox);
        for (const auto& cluster_label : visible_clusters_labels){
            // Note: Vertices of the lineset is colored by occupancy from regression
            // Check if the cluster exists because we could be updating the map one part at a time.
            auto gmm_linesets_it = gmm_linesets.gaussians.find(cluster_label);
            if (gmm_linesets_it != gmm_linesets.gaussians.end()) {
                bool updateRequired = false;
                Eigen::Vector3d OccupancyColor;
                if (gmm_linesets_it->second.cluster->is_free){
                    OccupancyColor = prob_zero_color;
                } else {
                    OccupancyColor = prob_one_color;
                }
                updateRequired = updateRequired | gmm_linesets_it->second.updateOccGeometry( OccupancyColor, curMapVizFlags.std);
                if (gmm_linesets_it->second.cluster->is_free){
                    if (gmm_linesets_it->second.cluster->near_obs){
                        gmm_linesets.active_free_near_obs_gaussians.insert(cluster_label);
                    } else {
                        gmm_linesets.active_free_gaussians.insert(cluster_label);
                    }
                } else {
                    // Note: Vertices of the lineset is colored from regression
                    updateRequired = updateRequired | gmm_linesets_it->second.updateColorGeometry(curMapVizFlags.std, gmm_map.get());
                    updateRequired = updateRequired | gmm_linesets_it->second.updateDebugColorGeometry(curMapVizFlags.std);
                    gmm_linesets.active_obs_gaussians.insert(cluster_label);
                }

                // Update partition information
                if (gmm_linesets_it->second.cluster->is_free){
                    if (gmm_linesets_it->second.cluster->near_obs){
                        gmm_linesets.free_near_obs_gaussian_partition.addActive(&gmm_linesets_it->second, updateRequired);
                    } else {
                        gmm_linesets.free_gaussian_partition.addActive(&gmm_linesets_it->second, updateRequired);
                    }
                } else {
                    gmm_linesets.obs_gaussian_partition.addActive(&gmm_linesets_it->second, updateRequired);
                }
            }
        }

        // Update partitions geometries
        gmm_linesets.obs_gaussian_partition.updateAllActiveGeometries();
        gmm_linesets.free_gaussian_partition.updateAllActiveGeometries();
        gmm_linesets.free_near_obs_gaussian_partition.updateAllActiveGeometries();

        // Print partition information
        //std::cout << std::endl;
        //gmm_linesets.obs_gaussian_partition.printInfo();
        //gmm_linesets.free_gaussian_partition.printInfo();
        //gmm_linesets.free_near_obs_gaussian_partition.printInfo();
        //std::cout << std::endl;

        //gmm_linesets.obs_gaussian_partition.checkActiveValidity();
        //gmm_linesets.free_gaussian_partition.checkActiveValidity();
        //gmm_linesets.free_near_obs_gaussian_partition.checkActiveValidity();
    }

    void GMMMapViz::redrawOccVarPointCloudGeometry(){
        if (curMapVizFlags.from_ray){
            occ_var_from_rays->clear();
            evaluateOccupancyAccuracyAndVarianceRay(gmm_map.get(), ray_sampling_dist, curMapVizFlags.num_inaccurate_roc_frames,
                                                    unexplored_evidence, unexplored_variance,
                                                    occ_var_from_rays->obs_and_free_pts,occ_var_from_rays->occ_value,
                                                    occ_var_from_rays->variance_value,occ_var_from_rays->error_value, occ_var_from_rays->throughput);
            occ_var_from_rays->building_blocks_computed = true;
            occ_var_from_rays->occ_colors.reserve(occ_var_from_rays->obs_and_free_pts.size());
            occ_var_from_rays->var_colors.reserve(occ_var_from_rays->obs_and_free_pts.size());
            query_throughput = occ_var_from_rays->throughput;
            total_num_query_pts = occ_var_from_rays->total_num_pts;
        } else {
            occ_var_from_bbox->clear();
            evaluateOccupancyAndVarianceBBox(gmm_map.get(), map_BBox, curMapVizFlags.num_of_points, unexplored_evidence, unexplored_variance,
                                             occ_var_from_bbox->obs_and_free_pts,occ_var_from_bbox->occ_value,
                                             occ_var_from_bbox->variance_value,occ_var_from_bbox->throughput);
            occ_var_from_bbox->total_num_pts = occ_var_from_bbox->obs_and_free_pts.size();
            occ_var_from_bbox->building_blocks_computed = true;
            occ_var_from_bbox->occ_colors.reserve(occ_var_from_bbox->obs_and_free_pts.size());
            occ_var_from_bbox->var_colors.reserve(occ_var_from_bbox->obs_and_free_pts.size());
            query_throughput = occ_var_from_bbox->throughput;
            total_num_query_pts = occ_var_from_bbox->total_num_pts;
        }
    }

    void GMMMapViz::redrawOccVarPointCloud() {
        // Check if an update to the gmm is available
        // Avoid recomputing the geometry if only the environment bbox is changed
        // Check if the building block is available first!
        bool building_block_available;
        if (curMapVizFlags.from_ray){
            building_block_available = occ_var_from_rays->building_blocks_computed;
        } else {
            building_block_available = occ_var_from_bbox->building_blocks_computed;
        }

        if (map_update_avaliable || !building_block_available){
            redrawOccVarPointCloudGeometry();
            // Recompute geometry
            if (curMapVizFlags.from_ray){
                occ_var_from_rays->generateGeometriesWithErrors(curMapVizFlags.occ_var_pt_high, curMapVizFlags.occ_var_pt_low,
                                                                Isometry3::Identity(), cur_max_variance, cur_min_variance);
                occ_var_from_rays->cropGeometries(false, env_bbox);
            } else {
                occ_var_from_bbox->generateGeometries(curMapVizFlags.occ_var_pt_high, curMapVizFlags.occ_var_pt_low,
                                                      Isometry3::Identity(), cur_max_variance, cur_min_variance);
                occ_var_from_bbox->cropGeometries(false, env_bbox);
            }
        } else if (curMapVizFlags.update_occ_var_geometry){
            // Recompute if the occupancy bounds changed.
            if (curMapVizFlags.from_ray){
                occ_var_from_rays->generateGeometriesWithErrors(curMapVizFlags.occ_var_pt_high, curMapVizFlags.occ_var_pt_low,
                                                                Isometry3::Identity(), cur_max_variance, cur_min_variance);
                occ_var_from_rays->cropGeometries(false, env_bbox);
            } else {
                occ_var_from_bbox->generateGeometries(curMapVizFlags.occ_var_pt_high, curMapVizFlags.occ_var_pt_low,
                                                      Isometry3::Identity(), cur_max_variance, cur_min_variance);
                occ_var_from_bbox->cropGeometries(false, env_bbox);
            }

        } else if (env_visibility_update_avaliable){
            // Check if we could just crop the existing point cloud
            bool is_enclosed = pre_adj_BBox.contains(adj_BBox);
            if (curMapVizFlags.from_ray){
                occ_var_from_rays->cropGeometries(is_enclosed, env_bbox);
            } else {
                occ_var_from_bbox->cropGeometries(is_enclosed, env_bbox);
            }

        } else {
            if (curMapVizFlags.from_ray){
                occ_var_from_rays->cropGeometries(false, env_bbox);
            } else {
                occ_var_from_bbox->cropGeometries(false, env_bbox);
            }

        }
    }

    void GMMMapViz::redrawEnvBBox() {
        // Redraws the bounding box based on the current map
        // Update previous values
        this->pre_adj_BBox = this->adj_BBox;

        // Update GMM BBox
        Eigen::Vector3d extent, ratio_lowerbound, ratio_upperbound;

        /*
        std::cout << fmt::format("Rtree - lowerBound: [{:.2f}, {:.2f}, {:.2f}], upperBound: [{:.2f}, {:.2f}, {:.2f}]",
                                 lowerBound(0), lowerBound(1), lowerBound(2), upperBound(0), upperBound(1), upperBound(2)) << std::endl;
        Eigen::Vector3d frame_lb, frame_ub;
        frame->computeBBox(frame_lb, frame_ub);
        std::cout << fmt::format("Frame - lowerBound: [{:.2f}, {:.2f}, {:.2f}], upperBound: [{:.2f}, {:.2f}, {:.2f}]",
                                 frame_lb(0), frame_lb(1), frame_lb(2), frame_ub(0), frame_ub(1), frame_ub(2)) << std::endl;
        */
        extent = map_BBox.sizes().cast<double>();
        ratio_lowerbound << curMapVizFlags.env_bbox_low_x, curMapVizFlags.env_bbox_low_y, curMapVizFlags.env_bbox_low_z;
        ratio_upperbound << curMapVizFlags.env_bbox_high_x, curMapVizFlags.env_bbox_high_y, curMapVizFlags.env_bbox_high_z;

        Eigen::Vector3d adj_lowerBound, adj_upperBound;
        adj_lowerBound = map_BBox.min().cast<double>() + ratio_lowerbound.cwiseProduct(extent);
        adj_upperBound = map_BBox.min().cast<double>() + ratio_upperbound.cwiseProduct(extent);
        for (int dim = 0; dim < 3; dim++){
            // Resolve conflicts
            if (adj_lowerBound(dim) >= adj_upperBound(dim)){
                // Prevent zero-size runtime error in Open3D
                adj_upperBound(dim) = adj_lowerBound(dim) + 0.001;
            }
        }
        this->adj_BBox = Rect(adj_lowerBound.cast<FP>(), adj_upperBound.cast<FP>());

        Eigen::Vector3d adj_center, adj_extent;
        adj_center = (adj_lowerBound + adj_upperBound) / 2;
        //adj_center = pose.topLeftCorner<3,3>() * (adj_lowerBound + adj_upperBound) / 2  +  pose.topRightCorner<3,1>();
        adj_extent = adj_upperBound - adj_lowerBound;

        //std::cout << fmt::format("Env_bbox constructed with center: [{:.2f}, {:.2f}, {:.2f}], extent: [{:.2f}, {:.2f}, {:.2f}]",
        //                         adj_center(0), adj_center(1), adj_center(2), adj_extent(0), adj_extent(1), adj_extent(2)) << std::endl;

        if (!isnan(adj_center(0)) && !isnan(adj_center(1)) && !isnan(adj_center(2))){
            env_bbox = std::make_shared<open3d::geometry::OrientedBoundingBox>(open3d::geometry::OrientedBoundingBox(adj_center, Eigen::Matrix3d::Identity(), adj_extent));
            env_bbox->color_ = prob_one_color;
        }
    }

    // Now we handle updates on the visualization thread
    void GMMMapViz::updateVisualizer(const vizMapAtomicFlags& sliderFlags, bool force_gmm_visibility_update) {
        // Update visualizer (Read and Write)
        {
            const std::lock_guard<std::mutex> g_lock(*geometry_lock_ptr);
            // Clear deleted geometries
            clearVisualizer(gmm_update_avaliable, occ_var_update_avaliable,
                            occ_voxels_update_avaliable, env_visibility_update_avaliable);

            bool update_gmm_visibility_only = curMapVizFlags.update_gmm_visibility || force_gmm_visibility_update;
            bool update_occvar_visibility_only = curMapVizFlags.update_occ_var_visibility;
            bool update_env_bbox_visibility_only = curMapVizFlags.update_env_visibility;
            bool update_occ_voxel_visibility_only = curMapVizFlags.update_occ_voxels_visibility;

            // Push updated geometry and visibility to the renderer
            if (gmm_update_avaliable){
                //std::cout << "GMMVisibility Updated Separately!" << std::endl;
                updateGMMPartitionVisibility();
                gmm_update_avaliable = false;
                update_gmm_visibility_only = false;
            }

            // Remove geometries if source changed!
            if (occ_var_update_avaliable || curMapVizFlags.update_occ_var_source){
                updateOccVarPointCloudVisibility();
                occ_var_update_avaliable = false;
                update_occvar_visibility_only = false;
            }

            if (occ_voxels_update_avaliable) {
                updateOccVoxelVisibility();
                occ_voxels_update_avaliable = false;
                update_occ_voxel_visibility_only = false;
            }

            if (env_visibility_update_avaliable){
                updateEnvBBoxVisibility();
                // Need to clear this flag last!
                env_visibility_update_avaliable = false;
                update_env_bbox_visibility_only = false;
            }

            // Push visibility for the remaining objects (geometry source did not change) to the renderer
            updatePartitionVisibility(update_gmm_visibility_only,
                                      update_occvar_visibility_only,
                                      update_occ_voxel_visibility_only,
                                      update_env_bbox_visibility_only);

            curMapVizFlags.clearVisibilityUpdateFlags();
        }
    }

    bool GMMMapViz::isEnvBBoxUpdated(const vizMapAtomicFlags& curFlags) {
        const std::lock_guard<std::mutex> g_lock(*geometry_lock_ptr);
        return curMapVizFlags.isEnvBBoxGeometryUpdated(curFlags);
    }

    void GMMMapViz::adjustGlobalBBox(const vizMapAtomicFlags& sliderFlags, Eigen::Vector3d& bbox_min, Eigen::Vector3d& bbox_max){
        const std::lock_guard<std::mutex> g_lock(*geometry_lock_ptr);
        if (sliderFlags.env_bbox_low_x != 0){
            bbox_min(0) = this->adj_BBox.min()(0);
        }
        if (sliderFlags.env_bbox_high_x != 1){
            bbox_max(0) = this->adj_BBox.max()(0);
        }
        if (sliderFlags.env_bbox_low_y != 0){
            bbox_min(1) = this->adj_BBox.min()(1);
        }
        if (sliderFlags.env_bbox_high_y != 1){
            bbox_max(1) = this->adj_BBox.max()(1);
        }
        if (sliderFlags.env_bbox_low_z != 0){
            bbox_min(2) = this->adj_BBox.min()(2);
        }
        if (sliderFlags.env_bbox_high_z != 1){
            bbox_max(2) = this->adj_BBox.max()(2);
        }

    }

    bool GMMMapViz::isMapCropped(const vizMapAtomicFlags& sliderFlags){
        if (sliderFlags.env_bbox_low_x == 0 &&
            sliderFlags.env_bbox_high_x == 1 &&
            sliderFlags.env_bbox_low_y == 0 &&
            sliderFlags.env_bbox_high_y == 1 &&
            sliderFlags.env_bbox_low_z == 0 &&
            sliderFlags.env_bbox_high_z == 1){
            return false;
        } else {
            return true;
        }
    }

    void GMMMapViz::updateVisibility(bool update_gmm, bool update_occvar, bool update_env) {
        if (!update_gmm && !update_occvar && !update_env){
            return;
        }

        //std::cout << fmt::format("Size of the hidden gaussians: {}, update GMM visibility {}", hidden_clusters.size(), update_gmm) << std::endl;
        auto geometry_labels = env_widget->GetScene()->GetGeometries();
        for (auto &label: geometry_labels) {
            if (update_gmm){
                if (label.find(gau_obs_base) != std::string::npos) {
                    //std::cout << fmt::format("Found obs gaussians with label: {}, Found: {}", label, hidden_clusters.find(label) == hidden_clusters.end()) << std::endl;
                    bool isActive = gmm_linesets.active_obs_gaussians.find(label) != gmm_linesets.active_obs_gaussians.end() &&
                            hidden_clusters.find(label) == hidden_clusters.end();
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_gmm_obs & isActive);
                } else if (label.find(gau_free_base) != std::string::npos) {
                    bool isActive = gmm_linesets.active_free_gaussians.find(label) != gmm_linesets.active_free_gaussians.end() &&
                                    hidden_clusters.find(label) == hidden_clusters.end();
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_gmm_free & isActive);
                } else if (label.find(gau_free_near_obs_base) != std::string::npos) {
                    bool isActive = gmm_linesets.active_free_near_obs_gaussians.find(label) != gmm_linesets.active_free_near_obs_gaussians.end() &&
                                    hidden_clusters.find(label) == hidden_clusters.end();
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_gmm_free_near_obs & isActive);
                } else if (label.find(bbox_obs_base) != std::string::npos) {
                    bool isActive = gmm_linesets.active_obs_gaussians.find(changePrefix(label, gau_obs_base)) != gmm_linesets.active_obs_gaussians.end() &&
                                    hidden_clusters.find(changePrefix(label, gau_obs_base)) == hidden_clusters.end();
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_gmm_bbox_obs & isActive);
                } else if (label.find(bbox_free_base) != std::string::npos) {
                    bool isActive = (gmm_linesets.active_free_gaussians.find(changePrefix(label, gau_free_base)) != gmm_linesets.active_free_gaussians.end() &&
                                    hidden_clusters.find(changePrefix(label, gau_free_base)) == hidden_clusters.end()) || (
                                    gmm_linesets.active_free_near_obs_gaussians.find(changePrefix(label, gau_free_near_obs_base)) != gmm_linesets.active_free_near_obs_gaussians.end() &&
                                    hidden_clusters.find(changePrefix(label, gau_free_near_obs_base)) == hidden_clusters.end());
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_gmm_bbox_free & isActive);
                } else if (label.find(gau_color_base) != std::string::npos) {
                    bool isActive = gmm_linesets.active_obs_gaussians.find(changePrefix(label, gau_obs_base)) != gmm_linesets.active_obs_gaussians.end() &&
                                    hidden_clusters.find(changePrefix(label, gau_obs_base)) == hidden_clusters.end();
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_gmm_color & isActive);
                } else if (label.find(gau_debug_color_base) != std::string::npos) {
                    bool isActive = gmm_linesets.active_obs_gaussians.find(changePrefix(label, gau_obs_base)) != gmm_linesets.active_obs_gaussians.end() &&
                                    hidden_clusters.find(changePrefix(label, gau_obs_base)) == hidden_clusters.end();
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_gmm_debug_color & isActive);
                }
            }

            if (update_occvar) {
                if (label.find(occ_pcd_base) != std::string::npos) {
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_occupancy_pts);
                } else if (label.find(occ_var_pcd_base) != std::string::npos) {
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_variance_pts);
                }
            }

            if (update_env){
                if (label.find(env_bbox_base) != std::string::npos) {
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_env_bbox);
                }
            }
        }
    }

    void GMMMapViz::updatePartitionVisibility(bool update_gmm, bool update_occvar, bool update_occ_voxels, bool update_env) {
        if (!update_gmm && !update_occvar && !update_env && !update_occ_voxels){
            return;
        }

        if (update_gmm){
            // Need redraw if necessary here
            updateGMMPartitionVisibility();
        }

        //std::cout << fmt::format("Size of the hidden gaussians: {}, update GMM visibility {}", hidden_clusters.size(), update_gmm) << std::endl;
        auto geometry_labels = env_widget->GetScene()->GetGeometries();
        for (auto &label: geometry_labels) {
            if (update_occvar) {
                if (label.find(occ_pcd_base) != std::string::npos) {
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_occupancy_pts);
                } else if (label.find(occ_var_pcd_base) != std::string::npos) {
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_variance_pts);
                }
            }

            if (update_env){
                if (label.find(env_bbox_base) != std::string::npos) {
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_env_bbox);
                }
            }

            if (update_occ_voxels){
                if (label.find(occ_voxel_base) != std::string::npos){
                    env_widget->GetScene()->ShowGeometry(label, curMapVizFlags.show_occupancy_voxels);
                }
            }
        }
    }


    void GMMMapViz::clearVisualizer(bool clear_gmm, bool clear_occvar, bool clear_occ_voxel, bool clear_env) {
        // Clear objects that are going to be updated or removed
        if (!clear_gmm && !clear_occvar && !clear_env){
            return;
        }

        // Note: GMMs are cleared in updateGMMVisibility
        if (clear_occ_voxel) {
            env_widget->GetScene()->RemoveGeometry(occ_voxel_base);
        }

        if (clear_occvar){
            env_widget->GetScene()->RemoveGeometry(occ_pcd_base);
            env_widget->GetScene()->RemoveGeometry(occ_var_pcd_base);
        }

        if (clear_env){
            env_widget->GetScene()->RemoveGeometry(env_bbox_base);
        }
    }

    void GMMMapViz::updateGMMVisibility() {
        // Update visibilities of the GMMs. Note the all GMMs within the env_bbox has valid geometry already!
        // 1) Add currently visible Gaussians into the renderer
        // 2) Update visibility of all clusters
        for (const auto& cluster_label : gmm_linesets.active_free_gaussians){
            // Remove the geometry if its needs to be redraw
            bool gaussian_has_lines = true;
            bool gaussian_visible = hidden_clusters.find(cluster_label) == hidden_clusters.end();
            if (gmm_linesets.gaussians.at(cluster_label).updateVisualizer){
                env_widget->GetScene()->RemoveGeometry(cluster_label);
                // We only draw if the geometry has valid lines
                if (gmm_linesets.gaussians.at(cluster_label).occ_geometry->HasLines()){
                    env_widget->GetScene()->AddGeometry(cluster_label, gmm_linesets.gaussians.at(cluster_label).occ_geometry.get(), mat);
                } else {
                    gaussian_has_lines = false;
                }

                env_widget->GetScene()->RemoveGeometry(changePrefix(cluster_label, bbox_free_base));
                env_widget->GetScene()->AddGeometry(changePrefix(cluster_label, bbox_free_base),
                                                    gmm_linesets.gaussians.at(cluster_label).bbox.get(), mat);

                gmm_linesets.gaussians.at(cluster_label).updateVisualizer = false;
            }

            env_widget->GetScene()->ShowGeometry(cluster_label, gaussian_has_lines & gaussian_visible & curMapVizFlags.show_gmm_free);
            env_widget->GetScene()->ShowGeometry(changePrefix(cluster_label, bbox_free_base),
                                                 gaussian_has_lines & gaussian_visible & curMapVizFlags.show_gmm_bbox_free);
        }

        for (const auto& cluster_label : gmm_linesets.active_free_near_obs_gaussians){
            // Remove the geometry if its needs to be redraw
            bool gaussian_has_lines = true;
            bool gaussian_visible = hidden_clusters.find(cluster_label) == hidden_clusters.end();
            if (gmm_linesets.gaussians.at(cluster_label).updateVisualizer){
                env_widget->GetScene()->RemoveGeometry(cluster_label);
                if (gmm_linesets.gaussians.at(cluster_label).occ_geometry->HasLines()){
                    env_widget->GetScene()->AddGeometry(cluster_label, gmm_linesets.gaussians.at(cluster_label).occ_geometry.get(), mat);
                } else {
                    gaussian_has_lines = false;
                }

                env_widget->GetScene()->RemoveGeometry(changePrefix(cluster_label, bbox_free_base));
                env_widget->GetScene()->AddGeometry(changePrefix(cluster_label, bbox_free_base),
                                                    gmm_linesets.gaussians.at(cluster_label).bbox.get(), mat);

                gmm_linesets.gaussians.at(cluster_label).updateVisualizer = false;
            }

            env_widget->GetScene()->ShowGeometry(cluster_label, gaussian_has_lines & gaussian_visible & curMapVizFlags.show_gmm_free_near_obs);
            env_widget->GetScene()->ShowGeometry(changePrefix(cluster_label, bbox_free_base),
                                                 gaussian_has_lines & gaussian_visible & curMapVizFlags.show_gmm_bbox_free);
        }

        for (const auto& cluster_label : gmm_linesets.active_obs_gaussians){
            // Remove the geometry if its needs to be redraw
            bool obs_gaussian_has_lines = true;
            bool color_gaussian_has_lines = true;
            bool debug_gaussian_has_lines = true;

            bool gaussian_visible = hidden_clusters.find(cluster_label) == hidden_clusters.end();
            if (gmm_linesets.gaussians.at(cluster_label).updateVisualizer){

                // Check occupancy clusters
                env_widget->GetScene()->RemoveGeometry(cluster_label);
                if (gmm_linesets.gaussians.at(cluster_label).occ_geometry->HasLines()){
                    env_widget->GetScene()->AddGeometry(cluster_label, gmm_linesets.gaussians.at(cluster_label).occ_geometry.get(), mat);
                } else {
                    obs_gaussian_has_lines = false;
                }

                // Check color clusters
                env_widget->GetScene()->RemoveGeometry(changePrefix(cluster_label, gau_color_base));
                if (gmm_linesets.gaussians.at(cluster_label).color_geometry->HasLines()){
                    env_widget->GetScene()->AddGeometry(changePrefix(cluster_label, gau_color_base),
                                                        gmm_linesets.gaussians.at(cluster_label).color_geometry.get(), mat);
                } else {
                    debug_gaussian_has_lines = false;
                }

                // Check debug clusters
                env_widget->GetScene()->RemoveGeometry(changePrefix(cluster_label, gau_debug_color_base));
                if (gmm_linesets.gaussians.at(cluster_label).color_debug_geometry->HasLines()){
                    env_widget->GetScene()->AddGeometry(changePrefix(cluster_label, gau_debug_color_base),
                                                        gmm_linesets.gaussians.at(cluster_label).color_debug_geometry.get(), mat);
                } else {
                    color_gaussian_has_lines = false;
                }

                env_widget->GetScene()->RemoveGeometry(changePrefix(cluster_label, bbox_obs_base));
                env_widget->GetScene()->AddGeometry(changePrefix(cluster_label, bbox_obs_base),
                                                    gmm_linesets.gaussians.at(cluster_label).bbox.get(), mat);
                gmm_linesets.gaussians.at(cluster_label).updateVisualizer = false;
            }

            env_widget->GetScene()->ShowGeometry(cluster_label, obs_gaussian_has_lines & gaussian_visible & curMapVizFlags.show_gmm_obs);
            env_widget->GetScene()->ShowGeometry(changePrefix(cluster_label, bbox_obs_base),
                                                 obs_gaussian_has_lines & gaussian_visible & curMapVizFlags.show_gmm_bbox_obs);
            env_widget->GetScene()->ShowGeometry(changePrefix(cluster_label, gau_color_base),
                                                 color_gaussian_has_lines & gaussian_visible & curMapVizFlags.show_gmm_color);
            env_widget->GetScene()->ShowGeometry(changePrefix(cluster_label, gau_debug_color_base),
                                                 debug_gaussian_has_lines & gaussian_visible & curMapVizFlags.show_gmm_debug_color);
        }

        // Turn off visibility for all other GMM objects. Remove geometry if it is not present in the map anymore.
        auto geometry_labels = env_widget->GetScene()->GetGeometries();
        for (auto &label: geometry_labels) {
            if (label.find(gau_obs_base) != std::string::npos) {
                if (gmm_linesets.gaussians.find(label) == gmm_linesets.gaussians.end()){
                    env_widget->GetScene()->RemoveGeometry(label);
                } else {
                    if (gmm_linesets.active_obs_gaussians.find(label) == gmm_linesets.active_obs_gaussians.end()){
                        env_widget->GetScene()->ShowGeometry(label, false);
                    }
                }
            } else if (label.find(gau_free_base) != std::string::npos) {
                if (gmm_linesets.gaussians.find(label) == gmm_linesets.gaussians.end()){
                    env_widget->GetScene()->RemoveGeometry(label);
                } else {
                    if (gmm_linesets.active_free_gaussians.find(label) == gmm_linesets.active_free_gaussians.end()) {
                        env_widget->GetScene()->ShowGeometry(label, false);
                    }
                }
            } else if (label.find(gau_free_near_obs_base) != std::string::npos) {
                if (gmm_linesets.gaussians.find(label) == gmm_linesets.gaussians.end()){
                    env_widget->GetScene()->RemoveGeometry(label);
                } else {
                    if (gmm_linesets.active_free_near_obs_gaussians.find(label) == gmm_linesets.active_free_near_obs_gaussians.end()) {
                        env_widget->GetScene()->ShowGeometry(label, false);
                    }
                }
            } else if (label.find(bbox_obs_base) != std::string::npos) {
                if (gmm_linesets.gaussians.find(changePrefix(label, gau_obs_base)) == gmm_linesets.gaussians.end()){
                    env_widget->GetScene()->RemoveGeometry(label);
                } else {
                    if (gmm_linesets.active_obs_gaussians.find(changePrefix(label, gau_obs_base)) == gmm_linesets.active_obs_gaussians.end()){
                        env_widget->GetScene()->ShowGeometry(label, false);
                    }
                }
            } else if (label.find(bbox_free_base) != std::string::npos) {
                if (gmm_linesets.gaussians.find(changePrefix(label, gau_free_base)) == gmm_linesets.gaussians.end() &&
                    gmm_linesets.gaussians.find(changePrefix(label, gau_free_near_obs_base)) == gmm_linesets.gaussians.end()){
                    env_widget->GetScene()->RemoveGeometry(label);
                } else {
                    if (gmm_linesets.active_free_gaussians.find(changePrefix(label, gau_free_base)) ==
                        gmm_linesets.active_free_gaussians.end() &&
                        gmm_linesets.active_free_near_obs_gaussians.find(changePrefix(label, gau_free_near_obs_base)) ==
                        gmm_linesets.active_free_near_obs_gaussians.end()) {
                        env_widget->GetScene()->ShowGeometry(label, false);
                    }
                }
            } else if (label.find(gau_color_base) != std::string::npos) {
                if (gmm_linesets.gaussians.find(changePrefix(label, gau_obs_base)) == gmm_linesets.gaussians.end()){
                    env_widget->GetScene()->RemoveGeometry(label);
                } else {
                    if (gmm_linesets.active_obs_gaussians.find(changePrefix(label, gau_obs_base)) ==
                        gmm_linesets.active_obs_gaussians.end()) {
                        env_widget->GetScene()->ShowGeometry(label, false);
                    }
                }
            } else if (label.find(gau_debug_color_base) != std::string::npos) {
                if (gmm_linesets.gaussians.find(changePrefix(label, gau_obs_base)) == gmm_linesets.gaussians.end()){
                    env_widget->GetScene()->RemoveGeometry(label);
                } else {
                    if (gmm_linesets.active_obs_gaussians.find(changePrefix(label, gau_obs_base)) ==
                        gmm_linesets.active_obs_gaussians.end()) {
                        env_widget->GetScene()->ShowGeometry(label, false);
                    }
                }
            }
        }
    }

    void GMMMapViz::updateGMMPartitionVisibility() {
        // Update visibilities of the GMMs in the partitions.
        for (auto& partition : gmm_linesets.obs_gaussian_partition.active_gaussian_linesets){
            // Remove the geometry if its needs to be redraw
            std::string occ_gaussian_partition_label = gmm_linesets.obs_gaussian_partition.getOccGaussianPartitionLabel(partition.first);
            std::string color_gaussian_partition_label = gmm_linesets.obs_gaussian_partition.getColorGaussianPartitionLabel(partition.first);
            std::string color_debug_gaussian_partition_label = gmm_linesets.obs_gaussian_partition.getColorDebugGaussianPartitionLabel(partition.first);
            std::string bbox_gaussian_partition_label = gmm_linesets.obs_gaussian_partition.getBBoxPartitionLabel(partition.first);

            bool valid_gau_geometry = partition.second.occ_gaussians.HasLines();
            if (partition.second.updateVisualizer){
                env_widget->GetScene()->RemoveGeometry(occ_gaussian_partition_label);
                env_widget->GetScene()->RemoveGeometry(color_gaussian_partition_label);
                env_widget->GetScene()->RemoveGeometry(color_debug_gaussian_partition_label);
                env_widget->GetScene()->RemoveGeometry(bbox_gaussian_partition_label);

                // We only add and draw geometry if it has valid lines
                if (valid_gau_geometry){
                    env_widget->GetScene()->AddGeometry(occ_gaussian_partition_label, &partition.second.occ_gaussians, mat);
                    env_widget->GetScene()->AddGeometry(color_gaussian_partition_label, &partition.second.color_gaussians, mat);
                    env_widget->GetScene()->AddGeometry(color_debug_gaussian_partition_label, &partition.second.color_debug_gaussians, mat);
                    env_widget->GetScene()->AddGeometry(bbox_gaussian_partition_label, &partition.second.bboxes, mat);
                }
                partition.second.updateVisualizer = false;
            }

            env_widget->GetScene()->ShowGeometry(occ_gaussian_partition_label, valid_gau_geometry & curMapVizFlags.show_gmm_obs);
            env_widget->GetScene()->ShowGeometry(color_gaussian_partition_label, valid_gau_geometry & curMapVizFlags.show_gmm_color);
            env_widget->GetScene()->ShowGeometry(color_debug_gaussian_partition_label, valid_gau_geometry & curMapVizFlags.show_gmm_debug_color);
            env_widget->GetScene()->ShowGeometry(bbox_gaussian_partition_label,valid_gau_geometry & curMapVizFlags.show_gmm_bbox_obs);
        }

        for (auto& partition : gmm_linesets.free_gaussian_partition.active_gaussian_linesets){
            // Remove the geometry if its needs to be redraw
            std::string occ_gaussian_partition_label = gmm_linesets.free_gaussian_partition.getOccGaussianPartitionLabel(partition.first);
            std::string bbox_gaussian_partition_label = gmm_linesets.free_gaussian_partition.getBBoxPartitionLabel(partition.first);

            bool valid_gau_geometry = partition.second.occ_gaussians.HasLines();
            if (partition.second.updateVisualizer){
                env_widget->GetScene()->RemoveGeometry(occ_gaussian_partition_label);
                env_widget->GetScene()->RemoveGeometry(bbox_gaussian_partition_label);

                // We only add and draw geometry if it has valid lines
                if (valid_gau_geometry){
                    env_widget->GetScene()->AddGeometry(occ_gaussian_partition_label, &partition.second.occ_gaussians, mat);
                    env_widget->GetScene()->AddGeometry(bbox_gaussian_partition_label, &partition.second.bboxes, mat);
                }
                partition.second.updateVisualizer = false;
            }

            env_widget->GetScene()->ShowGeometry(occ_gaussian_partition_label, valid_gau_geometry & curMapVizFlags.show_gmm_free);
            env_widget->GetScene()->ShowGeometry(bbox_gaussian_partition_label,valid_gau_geometry & curMapVizFlags.show_gmm_bbox_free);
        }

        for (auto& partition : gmm_linesets.free_near_obs_gaussian_partition.active_gaussian_linesets){
            // Remove the geometry if its needs to be redraw
            std::string occ_gaussian_partition_label = gmm_linesets.free_near_obs_gaussian_partition.getOccGaussianPartitionLabel(partition.first);
            std::string bbox_gaussian_partition_label = gmm_linesets.free_near_obs_gaussian_partition.getBBoxPartitionLabel(partition.first);

            bool valid_gau_geometry = partition.second.occ_gaussians.HasLines();
            if (partition.second.updateVisualizer){
                env_widget->GetScene()->RemoveGeometry(occ_gaussian_partition_label);
                env_widget->GetScene()->RemoveGeometry(bbox_gaussian_partition_label);

                // We only add and draw geometry if it has valid lines
                if (valid_gau_geometry){
                    env_widget->GetScene()->AddGeometry(occ_gaussian_partition_label, &partition.second.occ_gaussians, mat);
                    env_widget->GetScene()->AddGeometry(bbox_gaussian_partition_label, &partition.second.bboxes, mat);
                }
                partition.second.updateVisualizer = false;
            }

            env_widget->GetScene()->ShowGeometry(occ_gaussian_partition_label, valid_gau_geometry & curMapVizFlags.show_gmm_free_near_obs);
            env_widget->GetScene()->ShowGeometry(bbox_gaussian_partition_label,valid_gau_geometry & curMapVizFlags.show_gmm_bbox_free);
        }
    }

    void GMMMapViz::updateOccVarPointCloudVisibility(){
        if (curMapVizFlags.update_occ_var_source){
            this->env_widget->GetScene()->RemoveGeometry(occ_pcd_base);
            this->env_widget->GetScene()->RemoveGeometry(occ_var_pcd_base);
        }

        if (curMapVizFlags.from_ray){
            if (occ_var_from_rays->occ_pcd_cropped != nullptr && occ_var_from_rays->occ_var_pcd_cropped != nullptr){
                this->env_widget->GetScene()->AddGeometry(occ_pcd_base, occ_var_from_rays->occ_pcd_cropped.get(), adaptive_pt_mat);
                this->env_widget->GetScene()->AddGeometry(occ_var_pcd_base, occ_var_from_rays->occ_var_pcd_cropped.get(), adaptive_pt_mat);
                this->env_widget->GetScene()->ShowGeometry(occ_pcd_base, curMapVizFlags.show_occupancy_pts);
                this->env_widget->GetScene()->ShowGeometry(occ_var_pcd_base, curMapVizFlags.show_variance_pts);
            }
        } else {
            if (occ_var_from_bbox->occ_pcd_cropped != nullptr && occ_var_from_bbox->occ_var_pcd_cropped != nullptr) {
                this->env_widget->GetScene()->AddGeometry(occ_pcd_base, occ_var_from_bbox->occ_pcd_cropped.get(), adaptive_pt_mat);
                this->env_widget->GetScene()->AddGeometry(occ_var_pcd_base,
                                                          occ_var_from_bbox->occ_var_pcd_cropped.get(), adaptive_pt_mat);
                this->env_widget->GetScene()->ShowGeometry(occ_pcd_base, curMapVizFlags.show_occupancy_pts);
                this->env_widget->GetScene()->ShowGeometry(occ_var_pcd_base, curMapVizFlags.show_variance_pts);
            }
        }
    }

    void GMMMapViz::updateOccVoxelVisibility() {
        if (occ_voxels != nullptr){
            this->env_widget->GetScene()->RemoveGeometry(occ_voxel_base);
            this->env_widget->GetScene()->AddGeometry(occ_voxel_base, occ_voxels.get(), mat);
        }
    }

    void GMMMapViz::updateEnvBBoxVisibility() {
        if (env_bbox == nullptr){
            return;
        } else {
            this->env_widget->GetScene()->AddGeometry(env_bbox_base, env_bbox.get(), mat);
            this->env_widget->GetScene()->ShowGeometry(env_bbox_base, curMapVizFlags.show_env_bbox);
        }
    }

    std::string GMMMapViz::changePrefix(const std::string &source_label, const std::string &target_prefix) {
        return target_prefix + source_label.substr(source_label.find_last_of('_')+1);
    }

    void GMMMapViz::updateGeometrySources(){
        // Update the sources for the geometries from the map
        // 1) For GMMs
        gmm_linesets.updateGaussians(cur_map_updates);

        // 2) For environment bounding boxes directly from RTree of visualization object
        gmm_linesets.linesetRtree.GetBounds(map_BBox);

        // 3) For occupancy and variance points
        occ_var_from_rays->building_blocks_computed = false;
        occ_var_from_bbox->building_blocks_computed = false;
    }

    void GMMMapViz::appendHiddenGMMList(const std::list<GMMcluster_o*>& cluster_list){
        {
            const std::lock_guard<std::mutex> g_lock(*geometry_lock_ptr);
            //std::cout << fmt::format("Size of the cluster list: {}", cluster_list.size()) << std::endl;
            for (const auto &cluster: cluster_list) {
                hidden_clusters.insert(cluster->label);
                //std::cout << fmt::format("Appended cluster {} into the hidden list", cluster->label) << std::endl;
            }
        }
    }

    void GMMMapViz::replaceHiddenGMMList(const std::list<GMMcluster_o*>& cluster_list){
        {
            const std::lock_guard<std::mutex> g_lock(*geometry_lock_ptr);
            hidden_clusters.clear();
            for (const auto& cluster : cluster_list){
                hidden_clusters.insert(cluster->label);
            }
        }
    }

    void GMMMapViz::clearHiddenGMMList(){
        {
            const std::lock_guard<std::mutex> g_lock(*geometry_lock_ptr);
            hidden_clusters.clear();
        }
    }

    void GMMMapViz::generateCrossSectionOccupancyMap(const int& axis, const FP& value,
                                                     const std::string& filename, const int& max_resolution,
                                                     const FP& border_percentage) {
        const std::lock_guard<std::mutex> g_lock(*geometry_lock_ptr);
        if (axis < 0 || axis > 2){
            throw std::invalid_argument(fmt::format("Invalid axis choice {} for cross section.", axis));
        }

        V box_min = this->map_BBox.min();
        V box_max =  this->map_BBox.max();
        V box_extent = box_max - box_min;
        FP min_grid_size;

        int max_width, max_height;
        if (axis == 0){
            min_grid_size = std::max(box_extent(1), box_extent(2)) / (FP) max_resolution;
            max_height = (int) (box_extent(1) / min_grid_size);
            max_width = (int) (box_extent(2) / min_grid_size);
        } else if (axis == 1) {
            min_grid_size = std::max(box_extent(0), box_extent(2)) / (FP) max_resolution;
            max_height = (int) (box_extent(0) / min_grid_size);
            max_width = (int) (box_extent(2) / min_grid_size);
        } else {
            min_grid_size = std::max(box_extent(0), box_extent(1)) / (FP) max_resolution;
            max_height = (int) (box_extent(0) / min_grid_size);
            max_width = (int) (box_extent(1) / min_grid_size);
        }

        int border_size = (int) (border_percentage * (FP) std::max(max_height, max_width));

        // Generate image at fixed size
        RowMatrixXf r(2 * border_size + max_height, 2 * border_size + max_width);
        RowMatrixXf g(2 * border_size + max_height, 2 * border_size + max_width);
        RowMatrixXf b(2 * border_size + max_height, 2 * border_size + max_width);
        Eigen::Vector3d unexplored_color = colormap::interpolateNearestNeighbor(OccVarCMap, 0.5);
        r.setConstant((uint8_t) (unexplored_color(0) * 255));
        g.setConstant((uint8_t) (unexplored_color(1) * 255));
        b.setConstant((uint8_t) (unexplored_color(2) * 255));

        for (int v = 0; v < max_height; v++){
            for (int u = 0; u < max_width; u++){
                FP occ, variance;
                V pt;
                if (axis == 0){
                    pt << value, box_min(1) + (FP) v * min_grid_size, box_min(2) + (FP) u * min_grid_size;
                } else if (axis == 1){
                    pt << box_min(0) + (FP) v * min_grid_size, value, box_min(2) + (FP) u * min_grid_size;
                } else {
                    pt << box_min(0) + (FP) v * min_grid_size, box_min(1) + (FP) u * min_grid_size, value;
                }
                gmm_map->computeOccupancyAndVariance(pt, occ, variance, unexplored_evidence, unexplored_variance);
                Eigen::Vector3d color = colormap::interpolateNearestNeighbor(OccVarCMap, occ);
                r(border_size + v, border_size + u) = (uint8_t) (color(0) * 255);
                g(border_size + v, border_size + u) = (uint8_t) (color(1) * 255);
                b(border_size + v, border_size + u) = (uint8_t) (color(2) * 255);
            }
        }

        open3d::core::Tensor occRGB({2 * border_size + max_height, 2 * border_size + max_width, 3}, open3d::core::Dtype::UInt8);
        occRGB.Slice(2,0,1) = open3d::core::eigen_converter::EigenMatrixToTensor(r).Reshape(
                {2 * border_size + max_height, 2 * border_size + max_width, 1});
        occRGB.Slice(2,1,2) = open3d::core::eigen_converter::EigenMatrixToTensor(g).Reshape(
                {2 * border_size + max_height, 2 * border_size + max_width, 1});
        occRGB.Slice(2,2,3) = open3d::core::eigen_converter::EigenMatrixToTensor(b).Reshape(
                {2 * border_size + max_height, 2 * border_size + max_width, 1});

        std::string img_filename = dataset_param::result_path / fmt::format("{}.jpg", filename);
        open3d::t::io::WriteImageToJPG(img_filename, open3d::t::geometry::Image(occRGB));
        std::cout << fmt::format("Saved occupancy image with size [{}, {}] to file: {}",
                                 2 * border_size + max_height, 2 * border_size + max_width, img_filename) << std::endl;
    }

    // Extract voxels within certain probability range
    void GMMMapViz::generateOccupancyVoxels(float resolution, bool update_source) {

        // Regenerate the entire voxel map
        if (update_source){
            std::cout << fmt::format("Generating voxels due to map update. Please wait!") << std::endl;

            // Clear RTree
            occ_voxels_with_prob_rtree.RemoveAll();

            V box_min = this->map_BBox.min().cast<FP>();
            V box_max = this->map_BBox.max().cast<FP>();
            V center = this->map_BBox.center().cast<FP>();
            V neg_steps = (box_min - center)/resolution;
            V pos_steps = (box_max - center)/resolution;

            auto voxel = open3d::geometry::AxisAlignedBoundingBox({-resolution, -resolution, -resolution},
                                                                  {resolution, resolution, resolution});
            voxel.Scale(0.5, {0, 0, 0});

            // This allows the origin to be defined at the center
            for (int x = std::floor(neg_steps(0)); x < std::ceil(pos_steps(0)); x++) {
                for (int y = std::floor(neg_steps(1)); y < std::ceil(pos_steps(1)); y++) {
                    for (int z = std::floor(neg_steps(2)); z < std::ceil(pos_steps(2)); z++) {
                        FP occ, variance;
                        V pt = {(float) x, (float) y, (float) z};
                        pt = pt * resolution + center;

                        gmm_map->estimateMaxOccupancyAndVariance(pt + voxel.GetMinBound().cast<FP>(),
                                                                 pt + voxel.GetMaxBound().cast<FP>(),
                                                                 occ, variance, unexplored_evidence, unexplored_variance);
                        Eigen::Vector3d color = colormap::interpolateNearestNeighbor(OccVarCMap, occ);
                        //Eigen::Vector3d color = colormap::interpolateNearestNeighbor(OccVarCMap, 1);
                        auto cur_voxel = voxel;
                        cur_voxel.Translate(pt.cast<double>());
                        auto voxel_bbox_lineset = open3d::geometry::LineSet::CreateFromAxisAlignedBoundingBox(cur_voxel);
                        voxel_bbox_lineset->colors_.clear();
                        for (int j = 0; j < voxel_bbox_lineset->lines_.size(); j++) {
                            voxel_bbox_lineset->colors_.push_back(color);
                        }

                        // Insert into the RTree
                        Rect voxel_bbox = Rect(voxel_bbox_lineset->GetMinBound().cast<FP>(), voxel_bbox_lineset->GetMaxBound().cast<FP>());
                        occ_voxels_with_prob_rtree.Insert(voxel_bbox, std::make_pair(occ, *voxel_bbox_lineset));
                    }
                }
            }
            std::cout << fmt::format("Voxel generation completed at resolution {:.2f}m", resolution) << std::endl;
        }

        V box_min = this->env_bbox->GetMinBound().cast<FP>();
        V box_max = this->env_bbox->GetMaxBound().cast<FP>();
        Rect cur_box = Rect(box_min, box_max);

        int pt_counter = 0;
        int visible_voxels = 0;
        std::shared_ptr<open3d::geometry::LineSet> extractedVoxels(new open3d::geometry::LineSet);
        for (const auto& voxel_with_prob : occ_voxels_with_prob_rtree.Search(cur_box)){
            if (voxel_with_prob.first < curMapVizFlags.occ_var_pt_low || voxel_with_prob.first > curMapVizFlags.occ_var_pt_high) {
                continue;
            }
            visible_voxels++;

            //std::cout << "Transforming lines" << std::endl;
            for (int j = 0; j < voxel_with_prob.second.lines_.size(); j++) {
                //std::cout << "Transforming line " << j << std::endl;
                extractedVoxels->lines_.emplace_back(
                        voxel_with_prob.second.lines_.at(j) + Eigen::Vector2i::Constant(pt_counter));
            }
            //std::cout << "Update gmm lineset output " << std::endl;
            pt_counter += voxel_with_prob.second.points_.size();

            //std::cout << "Update gmm lineset output " << std::endl;
            extractedVoxels->colors_.insert(extractedVoxels->colors_.end(), voxel_with_prob.second.colors_.begin(),
                                            voxel_with_prob.second.colors_.end());
            extractedVoxels->points_.insert(extractedVoxels->points_.end(), voxel_with_prob.second.points_.begin(),
                                            voxel_with_prob.second.points_.end());
        }

        occ_voxels = extractedVoxels;
        std::cout << fmt::format("Active voxels in the 3D viewer: {}/{}", visible_voxels, occ_voxels_with_prob_rtree.CountLeafNodes()) << std::endl;
    }
}


