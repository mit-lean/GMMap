// Note that the results on the TX2 might be a little bit different due to different compiler optimizations that affect floating point computation results
// The above issue seems to affect the RTree library, which can return the same items that intersects with a bounding box in different order!
// The proposed algorithm will lead to a slightly different result (<3%) due to such order differences.
#include <atomic>
#include <sstream>
#include <thread>
#include "open3d/Open3D.h"
#include "dataset_utils/dataset_utils.h"
#include "gmm_map/map_param_init.h"
#include "gmm_map/map.h"

using namespace open3d;
using namespace open3d::visualization;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4 || utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        std::cout << "Input format: dataset dataset_info_path start_frame_idx num_frames" <<std::endl;
        return 1;
    }

    std::string dataset_name = argv[1];
    std::string dataset_config_file_path = argv[2];
    int start_frame_idx = atoi(argv[3]);
    int num_frames = atoi(argv[4]);

    std::atomic<bool> update_obs_gmms = true;
    std::atomic<bool> update_free_gmms = true;
    std::atomic<bool> fuse_gmms_across_frames = true;
    dutil::loadDatasetInfoFromJSON(dataset_config_file_path, dataset_name);

    gmm::FP default_depth_scale = 1;
    if (dataset_name == "tum" || dataset_name == "tum_fd" || dataset_name == "stata"){
        default_depth_scale = dataset_param::scale;
    }

    // Initialize GMM Map parameters
    gmm::map_param map_parameter;
    gmm::initializeMapParameters(map_parameter, dataset_name);
    auto gmm_map = std::make_shared<gmm::GMMMap>(map_parameter,
                                                 &update_obs_gmms, &update_free_gmms,
                                                 &fuse_gmms_across_frames);
    gmm_map->mapParameters.track_color = false;
    std::cout << fmt::format("Start GMM map construction with {} threads (cores).", gmm_map->mapParameters.num_threads) << std::endl;
    std::cout << fmt::format("GMM fusion with hell_thresh_squard_free: {:.4f}, hell_thresh_squard_obs: {:.4f}",
                             gmm_map->mapParameters.hell_thresh_squard_free, gmm_map->mapParameters.hell_thresh_squard_obs) << std::endl;

    // Create dataset files
    std::vector<std::string> rgb_files, depth_files, pose_files;
    std::vector<double> timestamps;
    dutil::LoadFilenames(rgb_files, depth_files, pose_files, timestamps, dataset_name);
    int last_frame_idx = std::min<int>(start_frame_idx + num_frames, depth_files.size());
    long duration = 0;

    Eigen::Matrix4d T_eigen;
    // Load intrinsics
    core::Tensor intrinsic_t;
    camera::PinholeCameraIntrinsic intrinsic_legacy;
    // Obtain camera intrinsic parameters
    std::tie(intrinsic_t, intrinsic_legacy) = dutil::LoadIntrinsics(dataset_name);
    Matrix3d K = open3d::core::eigen_converter::TensorToEigenMatrixXd(intrinsic_t);

    // Mapping Statistics
    std::vector<std::string> statistic_strs;
    int actual_num_frames = last_frame_idx - start_frame_idx;
    statistic_strs.reserve(actual_num_frames);
    for (int f_idx = 0; f_idx < actual_num_frames; f_idx++){
        statistic_strs.emplace_back("");
    }

    for (int i = start_frame_idx; i < last_frame_idx; i++){
        auto mapping_start = std::chrono::steady_clock::now();
        std::cout << "\nFrame number " << i+1 << "/" << depth_files.size() << std::endl;

        // Create rgbd images
        std::shared_ptr<t::geometry::Image> input_depth;
        open3d::core::Tensor depth_tensor;
        gmm::RowMatrixXf depthmap;
        if (dataset_name == "tartanair") {
            // Note that we should obtain the depth directly here to maximize overall throughput
            depth_tensor = open3d::t::io::ReadNpy(depth_files[i]);
            auto t_shape = depth_tensor.GetShape();
            depthmap = open3d::core::eigen_converter::TensorToEigenMatrixXf(depth_tensor.Reshape({t_shape[0],t_shape[1]}));
        } else {
            input_depth = t::io::CreateImageFromFile(depth_files[i]);
            depth_tensor = input_depth->AsTensor();
            auto t_shape = depth_tensor.GetShape();
            if (dataset_name == "stata"){
                gmm::RowMatrixXf depthmap_full = open3d::core::eigen_converter::TensorToEigenMatrixXf(depth_tensor.Reshape({t_shape[0],t_shape[1]}));
                int stride = (int) (1.0 / dataset_param::sampling_rate);
                depthmap = Eigen::Map<gmm::RowMatrixXf, 0, Stride<Dynamic, Dynamic>>(depthmap_full.data(),
                                                                                depthmap_full.rows() / stride,
                                                                                depthmap_full.cols() / stride,
                                                                                Stride<Dynamic, Dynamic>(stride * depthmap_full.cols(), stride)) / default_depth_scale;
            } else {
                depthmap = open3d::core::eigen_converter::TensorToEigenMatrixXf(depth_tensor.Reshape({t_shape[0],t_shape[1]}))/default_depth_scale;
            }
        }

        // Perform gmm construction
        gmm::Isometry3 curPose;
        dutil::processPose(pose_files, i, dataset_name,curPose);

        // Perform map construction
        std::list<gmm::GMMcluster_o*> new_free_clusters, new_obs_clusters;
        std::list<gmm::GMMmetadata_c> new_obs_cluster_metadata;
        std::list<gmm::GMMmetadata_o> new_free_cluster_metadata;
        // Insert current frame into the map
        auto clustering_start = std::chrono::steady_clock::now();
        new_obs_cluster_metadata = gmm_map->extendedSPGFOpt(depthmap);

        if (update_free_gmms){
            new_free_cluster_metadata = gmm_map->constructFreeClustersFromObsClusters(new_obs_cluster_metadata);
        }
        gmm_map->transferMetadata2ClusterExtended(new_obs_cluster_metadata, new_free_cluster_metadata,
                                                  new_obs_clusters, new_free_clusters, gmm_map->mapParameters.gmm_frame_param);
        auto clustering_stop = std::chrono::steady_clock::now();
        std::cout << fmt::format("Clusters - Obstacle: {}, Free: {}",
                                 new_obs_clusters.size(), new_free_clusters.size()) << std::endl;
        long clustering_latency = std::chrono::duration_cast<std::chrono::microseconds>(clustering_stop - clustering_start).count();

        auto fusion_start = std::chrono::steady_clock::now();
        gmm_map->insertGMMsIntoCurrentMap(curPose, new_free_clusters, new_obs_clusters, clustering_latency);
        auto fusion_stop = std::chrono::steady_clock::now();
        //std::cout << fmt::format("GMM fusion latency: {}us",
        //                         std::chrono::duration_cast<std::chrono::microseconds>(fusion_stop - fusion_start).count()) << std::endl;

        auto mapping_stop = std::chrono::steady_clock::now();
        duration += std::chrono::duration_cast<std::chrono::microseconds>(mapping_stop - mapping_start).count();

        // Print relevant statistics
        std::cout << gmm_map->printStatistics(i, i, false) << std::endl;
        statistic_strs.at(i - start_frame_idx) = gmm_map->printFrameStatisticsCSV(i, (i - start_frame_idx + 1) * 1000000.0 / (double) duration);
    }

    std::ofstream mapping_statistics;
    fs::path mapping_statistics_file;
    std::filesystem::create_directories(dataset_param::result_path);
    if (gmm_map->isCUDAEnabled()){
        mapping_statistics_file = dataset_param::result_path / "mapping_statistics_cuda.csv";
    } else {
        if (gmm_map->mapParameters.num_threads == 1){
            mapping_statistics_file = dataset_param::result_path / "mapping_statistics_cpu_single_core.csv";
        } else {
            mapping_statistics_file = dataset_param::result_path / "mapping_statistics_cpu_multi_core.csv";
        }
    }
    std::cout << fmt::format("Writing mapping statistics to: {}", mapping_statistics_file.string()) << std::endl;
    mapping_statistics.open(mapping_statistics_file.string(), std::ios::out);
    for (int f_idx = 0; f_idx < actual_num_frames; f_idx++){
        mapping_statistics << statistic_strs.at(f_idx) << std::endl;
    }
    mapping_statistics.close();

    std::cout << gmm_map->printStatistics(last_frame_idx, last_frame_idx) << std::endl;
    std::cout << fmt::format("Sequence completed with overall throughput (including dataset loading) of {:.3f}fps", gmm_map->total_processed_frames * 1000000.0 / duration) << std::endl;
    // We exclude dataset loading from disk from throughput computation
    std::cout << fmt::format("Please use \"GMM Mapping FPS\" for reporting the throughput of CPU implementation.") << std::endl;
    return 0;
}