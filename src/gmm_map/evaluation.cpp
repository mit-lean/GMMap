//
// Created by peter on 3/12/22.
//
#include "gmm_map/map.h"
#include "gmm_map/evaluation.h"
#include "dataset_utils/dataset_utils.h"
#include "open3d/Open3D.h"

namespace gmm {
    void evaluateOccupancyAndVarianceBBox(GMMMap const* map, const Rect& GlobalBBox, int num_points,
                                          FP unexplored_evidence, FP unexplored_variance,
                                          std::vector<Eigen::Vector3d>& obs_and_free_pts,
                                          std::vector<FP>& occ_value, std::vector<FP>& variance_value, FP& throughput){
        // Compute occupancy and variance for all points along a ray
        throughput = 0;
        obs_and_free_pts.clear();
        occ_value.clear();
        variance_value.clear();

        obs_and_free_pts.reserve(num_points);
        occ_value.reserve(num_points);
        variance_value.reserve(num_points);

        for (int i = 0; i < num_points; i++){
            V pt = GlobalBBox.sample();
            FP occ, variance;
            auto start = std::chrono::steady_clock::now();
            map->computeOccupancyAndVariance(pt, occ, variance, unexplored_evidence, unexplored_variance);
            //map.computeOccupancyAndVarianceKNN(pt,  map.mapParameters.min_num_neighbor_clusters, map.mapParameters.min_num_neighbor_clusters, occ, variance,
            //                                   unexplored_evidence, unexplored_variance);
            //map.computeOccupancyAndVariance2Pass(pt, occ, variance, unexplored_evidence, unexplored_variance);
            auto stop = std::chrono::steady_clock::now();
            throughput += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
            obs_and_free_pts.emplace_back(pt.cast<double>());
            occ_value.push_back(occ);
            variance_value.push_back(variance);
        }
        std::cout << fmt::format("Querying {} points within the bbox in {}ns.", num_points, throughput) << std::endl;
        throughput = (FP) num_points * 1e9f / throughput; // pts/sec
    }

    void evaluateOccupancyAccuracyAndVarianceRay(GMMMap const* map, FP ray_sampling_dist, int num_frames,
                                                 FP unexplored_evidence, FP unexplored_variance,
                                                 std::vector<Eigen::Vector3d>& obs_and_free_pts,
                                                 std::vector<FP>& occ_value, std::vector<FP>& variance_value,
                                                 std::vector<bool>& error, FP& throughput){
        // Compute occupancy and variance for all points along a ray
        throughput = 0;
        obs_and_free_pts.clear();
        occ_value.clear();
        variance_value.clear();
        error.clear();

        // Prepare to read the point cloud
        // Read input depth files
        std::vector<std::string> rgb_files, depth_files, pose_files;
        std::vector<double> timestamps;
        dutil::LoadFilenames(rgb_files, depth_files, pose_files, timestamps, map->mapParameters.dataset);

        // Obtain camera intrinsic parameters
        open3d::camera::PinholeCameraIntrinsic intrinsic_legacy;
        open3d::core::Tensor intrinsic_t;
        std::tie(intrinsic_t, intrinsic_legacy) = dutil::LoadIntrinsics(map->mapParameters.dataset);

        // Read file
        std::string file_dir = dataset_param::result_path / "sorted_frame_roc.csv";
        std::ifstream sorted_accuracy_frames (file_dir, std::ios::in);
        std::string frame_info;
        int cur_num_frames = 0;
        long total = 0;
        if (sorted_accuracy_frames.is_open()){
            while (std::getline(sorted_accuracy_frames, frame_info)){
                // Parse information
                std::stringstream ss(frame_info);
                std::string info;
                std::getline(ss, info, ' ');
                int frame_idx = std::stoi(info);
                std::getline(ss, info, ' ');
                std::cout << fmt::format("Visualizing occupancy along rays at frame {} with auc {:.2f}", frame_idx, std::stod(info)) << std::endl;

                // Create pcd geometry
                std::shared_ptr<open3d::t::geometry::Image> input_depth, input_color;
                if (map->mapParameters.dataset == "tartanair") {
                    input_depth = dutil::createImageNpy2O3d(depth_files[frame_idx], map->mapParameters.depth_scale);
                } else {
                    input_depth = open3d::t::io::CreateImageFromFile(depth_files[frame_idx]);
                }
                input_color = open3d::t::io::CreateImageFromFile(rgb_files[frame_idx]);

                auto rgbd_image = open3d::geometry::RGBDImage::CreateFromColorAndDepth(input_color->ToLegacy(), input_depth->ToLegacy(),
                                                                               map->mapParameters.depth_scale, map->mapParameters.max_depth);
                auto cur_frame_pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(*rgbd_image, intrinsic_legacy);

                // Construct a transformed PCD
                Isometry3 curPose;
                dutil::processPose(pose_files, frame_idx, map->mapParameters.dataset, curPose);
                V origin = curPose.matrix().topRightCorner<3,1>();

                cur_frame_pcd->Transform(curPose.matrix().cast<double>());

                // Store accuracy information
                for (const auto& pt : cur_frame_pcd->points_){
                    FP ray_length = (pt.cast<FP>() - origin).norm();
                    V direction = ray_sampling_dist * (pt.cast<FP>() - origin).normalized();
                    int num_pts = (int) ceil(ray_length / ray_sampling_dist);
                    total += num_pts;
                    for (int i = 1; i <= num_pts; i++){
                        V cur_pt;
                        FP occ, variance;
                        if (i == num_pts){
                            cur_pt = pt.cast<FP>();
                            auto start = std::chrono::steady_clock::now();
                            map->computeOccupancyAndVariance(cur_pt, occ, variance, unexplored_evidence, unexplored_variance);
                            auto stop = std::chrono::steady_clock::now();
                            throughput += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                            if (occ < 0.5){
                                error.push_back(true);
                                //std::cout << fmt::format("Incorrect occupancy! Expected: {:.2f}, Estimated: {:.2f}", 1.0, occ) << std::endl;
                            } else {
                                error.push_back(false);
                            }
                        } else {
                            cur_pt = origin + i*direction;
                            auto start = std::chrono::steady_clock::now();
                            map->computeOccupancyAndVariance(cur_pt, occ, variance, unexplored_evidence, unexplored_variance);
                            auto stop = std::chrono::steady_clock::now();
                            throughput += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
                            if (occ < 0.5){
                                error.push_back(false);
                                //std::cout << fmt::format("Incorrect occupancy! Expected: {:.2f}, Estimated: {:.2f}", 1.0, occ) << std::endl;
                            } else {
                                error.push_back(true);
                            }
                        }
                        obs_and_free_pts.emplace_back(cur_pt.cast<double>());
                        occ_value.push_back(occ);
                        variance_value.push_back(variance);
                    }
                }

                cur_num_frames++;
                if (cur_num_frames >= num_frames){
                    break;
                }
            }
        } else {
            std::cout << fmt::format("File cannot be open at directory: {}", file_dir) << std::endl;
        }
        sorted_accuracy_frames.close();
        std::cout << fmt::format("Querying {} points across {} frames in {}ns.", total, num_frames, throughput) << std::endl;
        throughput = total*1e9/throughput; // pts/sec
    }
}
