//
// Created by peter on 5/30/21.
//
#ifndef GMM_REG_DATASET_UTILS_H
#define GMM_REG_DATASET_UTILS_H
#include <iostream>
#include <list>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include "open3d/Open3D.h"
#include <json/json.h> // Note that jsoncpp is automatically included in Open3D

using std::string;
using namespace Eigen;
namespace fs = std::filesystem;
using image_channels_t = std::vector<std::vector<std::vector<unsigned char>>>;

// Define camera intrinsic parameters
namespace dataset_param
{
    extern Json::Value dataset_info;
    extern int width;
    extern int height;
    extern float fx;
    extern float fy;
    extern float cx;
    extern float cy;
    extern float scale;
    extern float sampling_rate;
    extern float max_depth;
    extern bool use_slam_traj;

    // Paths and files (not all will be used!)
    extern fs::path raw_data_path;
    extern fs::path rgb_path;
    extern fs::path depth_path;
    extern fs::path rgb_txt;
    extern fs::path depth_txt;
    extern fs::path pose_txt;
    extern fs::path slam_traj_txt;
    extern fs::path slam_path;
    extern fs::path slam_pose_txt;
    extern fs::path uav1_imu_txt;
    extern fs::path uav1_sim_txt;

    // Camera locations
    extern Eigen::Matrix4d cam_rel_pose;

    // Result paths
    extern fs::path result_path;
    extern fs::path acc_result_prediction_path;
    extern fs::path acc_result_variance_path;

    // Path for rendering video
    extern fs::path rendering_path;
    extern fs::path rendering_pcd_path;
    extern fs::path rendering_free_gmm_path;
    extern fs::path rendering_obs_gmm_path;

    // visualization settings
    extern int vis_window_width;
    extern int vis_window_height;
    extern float vis_zoom_factor;
    extern float vis_height_factor;
    extern int vis_delay_per_frame;
    extern int vis_updated_interval;
    extern bool vis_show_obstacle_gmm;
    extern bool vis_show_free_gmm;
}

namespace dutil
{
    using namespace open3d;

    void loadDatasetInfoFromJSON(const fs::path& dataset_info_path, const std::string& dataset_name);

    void getImagesRaw(const fs::path& directory, std::vector<string>& filenames);

    void getPosesRaw(const fs::path& filename, std::vector<string>& poses);

    void processTUMData(const fs::path& txt_path, std::vector<double>& timestamp, std::vector<string>& content);

    void getTUMDatasetRaw(const string& dataset_name, const fs::path& rgb_path, const fs::path& depth_path, const fs::path& pose_path,
                          std::vector<string>& rgb_txt, std::vector<string>& depth_txt, std::vector<string>& pose_txt,
                          std::vector<double>& timestamps, double max_difference = 0.02);

    std::string convertPoseToQuaternionString(const Matrix4d& pose, const string& dataset_name);

    void poseProcessRaw(const string& poseRaw, Matrix4d& pose, const string& dataset_name);

    void poseProcessRaw(const string& poseRaw, Matrix4f& pose, const string& dataset_name);

    void processQuaternionRaw(const string& poseRaw, Matrix4d& pose, const string& dataset_name);

    void retrieveCameraIntrinsics(const string& dataset, float& cx, float& cy, float& fx, float& fy, int& img_width, int& img_height);

    void forwardProject(const float& v, const float& u, const float& d, Eigen::Vector3d& point, const string& dataset, const float& max_depth);

    void forwardProject(const float& v, const float& u, const float& d, Eigen::Vector3f& point, const string& dataset, const float& max_depth);

    void forwardProjectVariance(const int& v, const int& u, const double& variance, Eigen::Matrix3d& covariance, const string& dataset);

    void forwardProjectVariance(const int& v, const int& u, const float& variance, Eigen::Matrix3f& covariance, const string& dataset);

    void backProject(double& v, double& u, double& d, const Eigen::Vector3d& point, const string& dataset);

    void backProject(float& v, float& u, float& d, const Eigen::Vector3f& point, const string& dataset);

    void forwardProjectJacobian(const Eigen::Vector3d& point, Matrix3d& J, const string& dataset);

    void forwardProjectJacobian(const Eigen::Vector3f& point, Matrix3f& J, const string& dataset);

    void backProjectJacobian(const Eigen::Vector3d& point, Matrix3d& J, const string& dataset);

    void backProjectJacobian(const Eigen::Vector3f& point, Matrix3f& J, const string& dataset);

    void o3dimage2vec2D(const t::geometry::Image& image, std::vector<std::vector<float>>& result, const float& scale = 1, const float& max = 20);

    void CreatePointCloud(const std::vector<std::vector<float>>& depth_image, const image_channels_t& rgb_image,
                          const string& dataset, const int& max_depth, std::shared_ptr<geometry::PointCloud>& pcd);
    void CreatePointCloudFromDepth(const std::vector<std::vector<float>>& depth_image,
                          const string& dataset, const int& max_depth, std::shared_ptr<geometry::PointCloud>& pcd, int downsample = 1);

    void Convert2PointCloud(const std::list<Vector3d>& points, std::shared_ptr<geometry::PointCloud>& pcd);

    void writeEigenToCSV(const string& fileName, const MatrixXd&  matrix);

    MatrixXd readEigenFromCSV(const string& , bool ignore_first_row = false);

    void saveBBoxToCSV(const string& dataset, const Eigen::Vector3d& BBoxHigh, const Eigen::Vector3d& BBoxLow);

    // Returns false if BBox CSV file does not exists!
    bool readBBoxFromCSV(const string& dataset, Eigen::Vector3d& BBoxHigh, Eigen::Vector3d& BBoxLow);

    std::shared_ptr<t::geometry::Image> createImageNpy2O3d(const string& filepath, float depth_scale);

    void processPoseO3d(const std::vector<std::string>& poses, int idx, const string& dataset, core::Tensor& T_frame_to_model);

    void processPose(const std::vector<std::string>& poses, int idx, const string& dataset,
                     Eigen::Isometry3d& T_frame_to_model);

    void processPose(const std::vector<std::string>& poses, int idx, const string& dataset,
                     Eigen::Isometry3f& T_frame_to_model);

    void poseError(const Matrix4d& Qi, const Matrix4d& Qn, const Matrix4d& Pi, const Matrix4d& Pn, double& transError, double& rotError);

    std::vector<double> updateErrors(std::vector<double>& accum_errors,
                      const Matrix4d& Qcur, const Matrix4d& Qpre, const Matrix4d& Pcur, const Matrix4d& Ppre,
                      const Matrix4d& Q0, const Matrix4d& P0);

    void gaussianImageProj(const Vector3d& cluster_mean, const Matrix3d& cluster_cov,
                           Vector2d& img_mean, Matrix2d& img_cov, const string& dataset);

    void gaussianPBox(const Vector3d& cluster_mean, const Matrix3d& cluster_cov,
                      int(&PBoxLow)[2], int (&PBoxHigh)[2], const string& dataset, double scale = 2);

    Matrix3d sensorNoiseCov(int v, int u, double d, const string& dataset);

    void LoadFilenames(std::vector<std::string>& rgb_files, std::vector<std::string>& depth_files,
                       std::vector<std::string>& pose_files, std::vector<double>& timestamps, const string& dataset);

    std::pair<core::Tensor, camera::PinholeCameraIntrinsic> LoadIntrinsics(const string& dataset);

    std::list<std::vector<int>> nearestNeighbors(const std::vector<int>& coord, int v_max, int u_max);

    // Used to obtain the bounding box of the entire environment
    void obtainEnvBBox(const std::string& dataset_name, double depth_max, double depth_scale, Eigen::Vector3d& lowerBound, Eigen::Vector3d& upperBound);

    // Obtain peak memory usage of the program up to this function call
    // https://man7.org/linux/man-pages/man5/proc.5.html
    void getPeakRSS(double& peakRSS);

    // Get the current memory usage up to this function call
    // https://man7.org/linux/man-pages/man5/proc.5.html
    void getCurrentRSS(double& curRSS, double& shareRSS, double& privateRSS);

    Matrix3d cameraIntrinsic(const std::string& dataset);

    Matrix3d cameraIntrinsicInv(const std::string& dataset);

    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    TensorToEigenMatrixRowMajor(const core::Tensor &tensor) {
        static_assert(std::is_same<T, double>::value ||
                      std::is_same<T, float>::value ||
                      std::is_same<T, int>::value,
                      "Only supports double, float and int (MatrixXd, MatrixXf and "
                      "MatrixXi).");
        core::Dtype dtype = core::Dtype::FromType<T>();
        core::SizeVector dim = tensor.GetShape();
        if (dim.size() != 2) {
            utility::LogError(
                    " [TensorToEigenMatrix]: Number of dimensions supported = 2, "
                    "but got {}.",
                    dim.size());
        }

        core::Tensor tensor_cpu_contiguous =
                tensor.Contiguous().To(core::Device("CPU:0"), dtype);
        T *data_ptr = tensor_cpu_contiguous.GetDataPtr<T>();

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                eigen_matrix(data_ptr, dim[0], dim[1]);

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                eigen_matrix_copy(eigen_matrix);
        return eigen_matrix_copy;
    }

    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
    TensorToEigenMatrixColMajor(const core::Tensor &tensor) {
        static_assert(std::is_same<T, double>::value ||
                      std::is_same<T, float>::value ||
                      std::is_same<T, int>::value,
                      "Only supports double, float and int (MatrixXd, MatrixXf and "
                      "MatrixXi).");
        core::Dtype dtype = core::Dtype::FromType<T>();
        core::SizeVector dim = tensor.GetShape();
        if (dim.size() != 2) {
            utility::LogError(
                    " [TensorToEigenMatrix]: Number of dimensions supported = 2, "
                    "but got {}.",
                    dim.size());
        }

        core::Tensor tensor_cpu_contiguous =
                tensor.Contiguous().To(core::Device("CPU:0"), dtype);
        T *data_ptr = tensor_cpu_contiguous.GetDataPtr<T>();

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                eigen_matrix(data_ptr, dim[0], dim[1]);

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
                eigen_matrix_copy(eigen_matrix);
        return eigen_matrix_copy;
    }
}

#endif //GMM_REG_DATASET_UTILS_H