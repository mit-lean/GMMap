//
// Created by peter on 5/30/21.
//
#include "dataset_utils/dataset_utils.h"
#include <cmath>
#include <Eigen/Eigenvalues>
#include <limits>
#include <sys/resource.h>
#include <fstream>
#include <unistd.h>

using namespace dutil;
namespace dataset_param {
    Json::Value dataset_info;
    int width;
    int height;
    float fx;
    float fy;
    float cx;
    float cy;
    float scale;
    float sampling_rate;
    float max_depth;
    bool use_slam_traj;

    fs::path raw_data_path;
    fs::path rgb_path;
    fs::path depth_path;
    fs::path rgb_txt;
    fs::path depth_txt;
    fs::path pose_txt;
    fs::path slam_path;
    fs::path slam_traj_txt;
    fs::path slam_pose_txt;
    fs::path uav1_imu_txt;
    fs::path uav1_sim_txt;
    Eigen::Matrix4d cam_rel_pose;

    fs::path result_path;
    fs::path acc_result_prediction_path;
    fs::path acc_result_variance_path;

    fs::path rendering_path;
    fs::path rendering_pcd_path;
    fs::path rendering_free_gmm_path;
    fs::path rendering_obs_gmm_path;

    int vis_window_width;
    int vis_window_height;
    float vis_zoom_factor;
    float vis_height_factor;
    int vis_delay_per_frame;
    int vis_updated_interval;
    bool vis_show_obstacle_gmm;
    bool vis_show_free_gmm;
}

void dutil::loadDatasetInfoFromJSON(const fs::path& dataset_info_path, const std::string& dataset_name){
    using namespace dataset_param;
    std::ifstream ifs;
    ifs.open(dataset_info_path / (dataset_name + ".json"));
    if (!ifs.is_open()){
        throw std::invalid_argument(fmt::format("Cannot read dataset {}.json from: {}", dataset_name, dataset_info_path.string()));
    }

    Json::Reader reader;
    dataset_info.clear();
    bool b = reader.parse(ifs, dataset_param::dataset_info); // reader can also read strings
    ifs.close();
    if (!b){
        std::cout << "Error: " << reader.getFormattedErrorMessages();
        throw std::invalid_argument(fmt::format("Cannot read from: {}", dataset_info_path.string()));
    }
    std::cout << fmt::format("Dataset info successfully loaded from: {}", dataset_info_path.string()) << std::endl;

    // Store parameters
    width = dataset_info["width"].asInt();
    height = dataset_info["height"].asInt();
    scale = dataset_info["scale"].asFloat();
    max_depth = dataset_info["max_depth"].asFloat();
    if (dataset_name == "stata") {
        float fov = dataset_info["fov"].asFloat() * 3.14159265358979323846f / 180.0f;
        fx = (float) width / std::tan(fov / 2) / 2;
        fy = fx;
        cx = (float) width / 2;
        cy = (float) height / 2;
        sampling_rate = dataset_info["sampling_rate"].asFloat();
    } else {
        fx = dataset_info["fx"].asFloat();
        fy = dataset_info["fy"].asFloat();
        cx = dataset_info["cx"].asFloat();
        cy = dataset_info["cy"].asFloat();
    }

    auto path_info = dataset_info["dataset_path"];

    if (dataset_name == "tartanair"){
        fs::path extra_info = fs::path(path_info["scene"].asString()) / path_info["scene"].asString() / path_info["difficulty"].asString() / path_info["sequence"].asString();
        raw_data_path =  getenv("HOME") / fs::path(path_info["location"].asString()) / path_info["scene"].asString() / path_info["difficulty"].asString();
        rgb_path = raw_data_path / fs::path("image_" + path_info["camera"].asString()) / extra_info / fs::path("image_" + path_info["camera"].asString());
        depth_path = raw_data_path / fs::path("depth_" + path_info["camera"].asString()) / extra_info / fs::path("depth_" + path_info["camera"].asString());
        pose_txt = raw_data_path / fs::path("depth_" + path_info["camera"].asString()) / extra_info / fs::path("pose_" + path_info["camera"].asString() + ".txt");
        result_path = fs::path(dataset_info["result_path"].asString()) / dataset_name / path_info["scene"].asString();
    } else if (dataset_name == "tum") {
        raw_data_path = getenv("HOME") / fs::path(path_info["location"].asString()) / path_info["category"].asString() / fs::path("rgbd_dataset_" + path_info["scene"].asString());
        rgb_txt = raw_data_path / fs::path("rgb.txt");
        depth_txt = raw_data_path / fs::path("depth.txt");
        pose_txt = raw_data_path / fs::path("groundtruth.txt");
        slam_path = fs::path(path_info["location"].asString()) / "rgbd_benchmark_tools/data/rgbdslam";
        slam_pose_txt = slam_path / fs::path(path_info["scene"].asString() + "-rgbdslam.txt");
        result_path = fs::path(dataset_info["result_path"].asString()) / dataset_name / path_info["scene"].asString();
    } else if (dataset_name == "tum_fd") {
        raw_data_path = getenv("HOME") / fs::path(path_info["location"].asString()) / path_info["category"].asString() / fs::path("rgbd_dataset_" + path_info["scene"].asString());
        rgb_txt = raw_data_path / fs::path("rgb.txt");
        depth_txt = raw_data_path / fs::path("depth.txt");
        pose_txt = raw_data_path / fs::path("groundtruth.txt");
        slam_path = fs::path(path_info["location"].asString()) / "rgbd_benchmark_tools/data/rgbdslam";
        slam_pose_txt = slam_path / fs::path(path_info["scene"].asString() + "-rgbdslam.txt");
        result_path = fs::path(dataset_info["result_path"].asString()) / dataset_name / path_info["scene"].asString();
    } else if (dataset_name == "stata") {
        raw_data_path = getenv("HOME") / fs::path(path_info["location"].asString()) / path_info["scene"].asString();
        Matrix4d cam_pose_wrt_camsys;
        if (path_info["right_camera"].asBool()){
            rgb_path = raw_data_path / fs::path("Camera_RGB_Right");
            depth_path = raw_data_path / fs::path("Camera_Depth_Right");
            processQuaternionRaw(dataset_info["right_cam_pose_wrt_camsys"].asString(), cam_pose_wrt_camsys, dataset_name);
        } else {
            rgb_path = raw_data_path / fs::path("Camera_RGB");
            depth_path = raw_data_path / fs::path("Camera_Depth");
            processQuaternionRaw(dataset_info["left_cam_pose_wrt_camsys"].asString(), cam_pose_wrt_camsys, dataset_name);
        }
        processQuaternionRaw(dataset_info["camsys_pose_wrt_robot"].asString(), cam_rel_pose, dataset_name);
        cam_rel_pose = cam_rel_pose * cam_pose_wrt_camsys;
        uav1_imu_txt = raw_data_path / fs::path("uav1_imu.csv");
        uav1_sim_txt = raw_data_path / fs::path("uav1_sim.csv");
        result_path = fs::path(dataset_info["result_path"].asString()) / dataset_name / path_info["scene"].asString();
    } else {
        throw std::invalid_argument("Dataset: " + dataset_name + " is not supported!");
    }
    use_slam_traj = dataset_info["dataset_path"]["use_slam_traj"].asBool();
    slam_traj_txt = raw_data_path / "slam_traj.txt";

    acc_result_prediction_path = result_path / fs::path("prediction");
    acc_result_variance_path = result_path / fs::path("variance");
    rendering_path = result_path / fs::path("rendering");
    rendering_pcd_path = rendering_path / fs::path("pointcloud");
    rendering_free_gmm_path = rendering_path / fs::path("free");
    rendering_obs_gmm_path = rendering_path / fs::path("obstacle");

    vis_window_width = dataset_info["visualization_parameters"]["window_width"].asInt();
    vis_window_height = dataset_info["visualization_parameters"]["window_height"].asInt();
    vis_zoom_factor = dataset_info["visualization_parameters"]["zoom_factor"].asFloat();
    vis_height_factor = dataset_info["visualization_parameters"]["height_factor"].asFloat();
    vis_delay_per_frame = dataset_info["visualization_parameters"]["delay_per_frame"].asInt();
    vis_updated_interval = dataset_info["visualization_parameters"]["update_interval"].asInt();
    vis_show_obstacle_gmm = dataset_info["visualization_parameters"]["show_obstacle_gmm"].asBool();
    vis_show_free_gmm = dataset_info["visualization_parameters"]["show_free_gmm"].asBool();
}

void dutil::getImagesRaw(const fs::path& directory, std::vector<string>& filenames){
    // Get a vector of file names to the iterator
    for (const auto & entry : fs::directory_iterator(directory.string())){
        filenames.push_back(entry.path().string());
        //std::cout << entry.path() << std::endl;
    }
    std::sort(filenames.begin(), filenames.end());
}

void dutil::getPosesRaw(const fs::path& filename, std::vector<string>& poses){
    // Get a vector of poses
    std::fstream newfile;
    newfile.open(filename.string(),std::ios::in); //open a file to perform read operation using file object
    if (newfile.is_open()){   //checking whether the file is open
        string tp;
        while(getline(newfile, tp)){ //read data from file object and put it into string.
            poses.push_back(tp);
            // std::cout << tp << "\n";
        }
        newfile.close(); //close the file object.
    } else {
        throw std::invalid_argument(fmt::format("Cannot open file: {}", filename.string()));
    }
}

void dutil::processTUMData(const fs::path& txt_path, std::vector<double>& timestamp, std::vector<string>& content){
    std::fstream newfile;
    newfile.open(txt_path.string(),std::ios::in); //open a file to perform read operation using file object
    if (newfile.is_open()){   //checking whether the file is open
        string tp;
        int line_idx = 0;
        while(getline(newfile, tp)){ //read data from file object and put it into string.
            // Start from the 3rd line
            if (line_idx > 2){
                double time = std::stod(tp.substr(0, tp.find(' ')));
                string relative_path = tp.substr(tp.find(' ')+1);
                timestamp.push_back(time);
                content.push_back(relative_path);
            }
            line_idx += 1;
        }
        newfile.close(); //close the file object.
    }
}


void dutil::getTUMDatasetRaw(const string& dataset_name, const fs::path& rgb_path, const fs::path& depth_path, const fs::path& pose_path,
                      std::vector<string>& rgb_txt, std::vector<string>& depth_txt, std::vector<string>& pose_txt,
                      std::vector<double>& timestamps, double max_difference){
    // Process TUM dataset with correspondance calculation
    std::vector<string> rgb_files, depth_files, pose_files;
    std::vector<double> rgb_timestamps, depth_timestamps, pose_timestamps;
    processTUMData(rgb_path, rgb_timestamps, rgb_files);
    processTUMData(depth_path, depth_timestamps, depth_files);
    processTUMData(pose_path, pose_timestamps, pose_files);

    if (dataset_name != "tum" && dataset_name != "tum_fd"){
        throw std::invalid_argument("This type of TUM dataset is not supported!");
    }

    // Finding correspondance
    int rgb_idx = 0;
    int depth_idx = 0;
    int pose_idx = 0;
    while (rgb_idx < rgb_timestamps.size() && depth_idx < depth_timestamps.size() && pose_idx < pose_timestamps.size()){
        // Find appropriate depthmap
        double cur_depth_diff = depth_timestamps[depth_idx] - rgb_timestamps[rgb_idx];
        double pre_depth_diff = std::numeric_limits<double>::max();
        int modified;
        while (std::abs(cur_depth_diff) < std::abs(pre_depth_diff)){
            modified = 0;
            pre_depth_diff = cur_depth_diff;
            if (cur_depth_diff > 0){
                rgb_idx++;
            } else {
                depth_idx++;
                modified = 1;
            }
            // End condition checks
            if (rgb_idx == rgb_timestamps.size() || depth_idx == depth_timestamps.size()){
                if (std::abs(cur_depth_diff) > max_difference){
                    return;
                } else {
                    break;
                }
            } else {
                cur_depth_diff = depth_timestamps[depth_idx] - rgb_timestamps[rgb_idx];
            }
        }
        if (modified == 0){
            rgb_idx--;
        } else {
            depth_idx--;
        }

        if (std::abs(pre_depth_diff) > max_difference){
            rgb_idx++;
            continue;
        }

        // Find appropriate pose (fix rgb_idx)
        double cur_pose_diff = pose_timestamps[pose_idx] - rgb_timestamps[rgb_idx];
        double pre_pose_diff = std::numeric_limits<double>::max();
        while (std::abs(cur_pose_diff) < std::abs(pre_pose_diff)){
            modified = 0;
            pre_pose_diff = cur_pose_diff;
            if (cur_pose_diff <= 0){
                pose_idx++;
            } else {
                modified = 1;
                break;
            }
            // End condition checks
            if (pose_idx == pose_timestamps.size()){
                if (std::abs(cur_pose_diff) > max_difference){
                    return;
                } else {
                    break;
                }
            } else {
                cur_pose_diff = pose_timestamps[pose_idx] - rgb_timestamps[rgb_idx];
            }
        }

        if (std::abs(pre_pose_diff) <= max_difference){
            if (modified == 0){
                pose_idx--;
            }
            // Update correspondance
            // std::cout << std::fixed << "Total: " << rgb_txt.size() << ", RGB: " << rgb_timestamps[rgb_idx] << ", Depth: " << depth_timestamps[depth_idx] << ", Pose: " << pose_timestamps[pose_idx] << std::endl;
            timestamps.push_back(rgb_timestamps[rgb_idx]);
            rgb_txt.push_back(dataset_param::raw_data_path / fs::path(rgb_files[rgb_idx]));
            depth_txt.push_back(dataset_param::raw_data_path / fs::path(depth_files[depth_idx]));
            pose_txt.push_back(pose_files[pose_idx]);
            depth_idx++;
            pose_idx++;
        }
        rgb_idx++;
    }
}


void dutil::poseProcessRaw(const string& poseRaw, Matrix4d& pose, const string& dataset){
    if (dataset == "stata" && !dataset_param::use_slam_traj) {
        // Convert quaternion pose in string to Eigen format
        std::vector<double> poseElements;
        std::string element;
        std::stringstream RowStringStream(poseRaw); //convert matrixRowString that is a string to a stream variable.
        while (getline(RowStringStream, element, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            poseElements.push_back(stod(element));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }

        // Extract Translation
        pose = Matrix<double,4,4>::Identity();
        pose(0,3) = poseElements[1];
        pose(1,3) = poseElements[2];
        pose(2,3) = poseElements[3];

        // Extract Rotation
        Eigen::AngleAxisd rollAngle(poseElements[9], Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd yawAngle( poseElements[8], Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd pitchAngle(poseElements[7], Eigen::Vector3d::UnitX());

        Eigen::Quaterniond q = rollAngle * yawAngle * pitchAngle;

        Eigen::Matrix3d R = q.normalized().toRotationMatrix();
        pose.topLeftCorner(3,3) = R;
    } else {
        // Convert quaternion pose in string to Eigen format
        processQuaternionRaw(poseRaw, pose, dataset);
    }
}

void dutil::processQuaternionRaw(const string& poseRaw, Matrix4d& pose, const string& dataset_name){
    // Convert quaternion pose in string to Eigen format
    double poseElements[7];
    int i = 0;
    std::stringstream ssin(poseRaw);
    while (ssin.good()){
        ssin >> poseElements[i];
        ++i;
    }

    // Extract Translation
    pose = Matrix<double,4,4>::Identity();
    pose(0,3) = poseElements[0];
    pose(1,3) = poseElements[1];
    pose(2,3) = poseElements[2];
    Eigen::Quaterniond q;

    // Extract Rotation
    if (dataset_name == "stata"){
        q.x() = poseElements[4];
        q.y() = poseElements[5];
        q.z() = poseElements[6];
        q.w() = poseElements[3];
    } else {
        q.x() = poseElements[3];
        q.y() = poseElements[4];
        q.z() = poseElements[5];
        q.w() = poseElements[6];
    }
    Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    pose.topLeftCorner(3,3) = R;
}

std::string dutil::convertPoseToQuaternionString(const Matrix4d& pose, const string& dataset_name){
    Eigen::Quaterniond q(pose.topLeftCorner<3,3>());
    if (dataset_name == "stata"){
        return fmt::format("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                           pose(0,3), pose(1,3), pose(2,3), q.w(), q.x(), q.y(), q.z());
    } else {
        return fmt::format("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                           pose(0,3), pose(1,3), pose(2,3), q.x(), q.y(), q.z(), q.w());
    }
}

void dutil::poseProcessRaw(const string& poseRaw, Matrix4f& pose, const string& dataset){
    // Convert quaternion pose in string to Eigen format
    Matrix4d pose_d;
    poseProcessRaw(poseRaw, pose_d, dataset);
    pose = pose_d.cast<float>();
}

void dutil::retrieveCameraIntrinsics(const string& dataset, float& cx, float& cy, float& fx, float& fy, int& img_width, int& img_height) {
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "nyudepthv2" || dataset == "tartanair" || dataset == "stata"){
        cx = dataset_param::cx;
        cy = dataset_param::cy;
        fx = dataset_param::fx;
        fy = dataset_param::fy;
        img_width = dataset_param::width;
        img_height = dataset_param::height;
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
}

void dutil::forwardProject(const float& v, const float& u, const float& d, Eigen::Vector3d& point, const string& dataset, const float& max_depth)
{
    // Assume that the depth is already scaled
    // Obtain the 3D point of a pixel on the depth map
    // Eigen::Vector3d camera_coord;
    // camera_coord << v, u, d;
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "stata" || dataset == "tartanair"){
        // Nan will be zero
        if (d <= 0 || d > max_depth){
            point << NAN, NAN, NAN;
        } else {
            //point = dataset_param::tum_K_inv * camera_coord;
            point(0) = (u - dataset_param::cx) * d / dataset_param::fx;
            point(1) = (v - dataset_param::cy) * d / dataset_param::fy;
            point(2) = d;

        }

    } else if (dataset == "nyudepthv2"){
        // Nan will be zero
        if (d <= 0 || d > max_depth){
            point << NAN, NAN, NAN;
        } else {
            //point = dataset_param::nyu_K_inv * camera_coord;

            point(0) = (v - dataset_param::cy) * d / dataset_param::fy;
            point(1) = (u - dataset_param::cx) * d / dataset_param::fx;
            point(2) = d;

        }
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
}

void dutil::forwardProject(const float& v, const float& u, const float& d, Eigen::Vector3f& point, const string& dataset, const float& max_depth)
{
    // Obtain the 3D point of a pixel on the depth map
    // Eigen::Vector3d camera_coord;
    // camera_coord << v, u, d;
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "tartanair" || dataset == "stata"){
        // Nan will be zero
        if (d <= 0 || d > max_depth){
            point << NAN, NAN, NAN;
        } else {
            //point = dataset_param::tum_K_inv * camera_coord;
            point(0) = (u - dataset_param::cx) * d / dataset_param::fx;
            point(1) = (v - dataset_param::cy) * d / dataset_param::fy;
            point(2) = d;
        }
    } else if (dataset == "nyudepthv2"){
        // Nan will be zero
        if (d <= 0 || d > max_depth){
            point << NAN, NAN, NAN;
        } else {
            //point = dataset_param::nyu_K_inv * camera_coord;

            point(0) = (v - dataset_param::cy) * d / dataset_param::fy;
            point(1) = (u - dataset_param::cx) * d / dataset_param::fx;
            point(2) = d;

        }
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
}

void dutil::forwardProjectVariance(const int& v, const int& u, const double& variance, Eigen::Matrix3d& covariance, const string& dataset){
    Eigen::Vector3d scale;
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "tartanair" || dataset == "stata"){
        scale = {((float) u - dataset_param::cx) / dataset_param::fx, ((float) v - dataset_param::cy) / dataset_param::fy, 1};
    } else if (dataset == "nyudepthv2"){
        scale = {((float) v - dataset_param::cy) / dataset_param::fy, ((float) u - dataset_param::cx) / dataset_param::fx,  1};
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
    covariance = variance * scale * scale.transpose();
}

void dutil::forwardProjectVariance(const int& v, const int& u, const float& variance, Eigen::Matrix3f& covariance, const string& dataset){
    Eigen::Vector3f scale;
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "tartanair" || dataset == "stata"){
        scale = {((float) u - dataset_param::cx) / dataset_param::fx, ((float) v - dataset_param::cy) / dataset_param::fy, 1};
    } else if (dataset == "nyudepthv2"){
        scale = {((float) v - dataset_param::cy) / dataset_param::fy, ((float) u - dataset_param::cx) / dataset_param::fx,  1};
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
    covariance = variance * scale * scale.transpose();
}

void dutil::backProject(double& v, double& u, double& d, const Eigen::Vector3d& point, const string& dataset){
    // Obtain pixel location in the image plane from a 3d point in local camera frame.
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "tartanair" || dataset == "stata"){
        d = point(2);
        u = point(0)*dataset_param::fx/point(2) + dataset_param::cx;
        v = point(1)*dataset_param::fy/point(2) + dataset_param::cy;
    } else if (dataset == "nyudepthv2"){
        d = point(2);
        u = point(1)*dataset_param::fx/point(2) + dataset_param::cx;
        v = point(0)*dataset_param::fy/point(2) + dataset_param::cy;
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
}

void dutil::backProject(float& v, float& u, float& d, const Eigen::Vector3f& point, const string& dataset){
    // Obtain pixel location in the image plane from a 3d point in local camera frame.
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "tartanair" || dataset == "stata"){
        d = point(2);
        u = point(0)*dataset_param::fx/point(2) + dataset_param::cx;
        v = point(1)*dataset_param::fy/point(2) + dataset_param::cy;
    } else if (dataset == "nyudepthv2"){
        d = point(2);
        u = point(1)*dataset_param::fx/point(2) + dataset_param::cx;
        v = point(0)*dataset_param::fy/point(2) + dataset_param::cy;
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
}

void dutil::forwardProjectJacobian(const Eigen::Vector3d& point, Matrix3d& J, const string& dataset){
    // Input point is [u, v, d]
    J.setIdentity();
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "stata" || dataset == "tartanair"){
        J(0,0) = point(2) / dataset_param::fx;
        J(0,2) = (point(0) - dataset_param::cx) /  dataset_param::fx;
        J(1,1) = point(2) / dataset_param::fy;
        J(1,2) = (point(1) - dataset_param::cy) /  dataset_param::fy;
    } else if (dataset == "nyudepthv2"){
        J(1,0) = point(2) / dataset_param::fx;
        J(1,2) = (point(0) - dataset_param::cx) /  dataset_param::fx;
        J(0,1) = point(2) / dataset_param::fy;
        J(0,2) = (point(1) - dataset_param::cy) /  dataset_param::fy;
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
}

void dutil::forwardProjectJacobian(const Eigen::Vector3f& point, Matrix3f& J, const string& dataset){
    // Input point is [u, v, d]
    J.setIdentity();
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "stata" || dataset == "tartanair"){
        J(0,0) = point(2) / dataset_param::fx;
        J(0,2) = (point(0) - dataset_param::cx) /  dataset_param::fx;
        J(1,1) = point(2) / dataset_param::fy;
        J(1,2) = (point(1) - dataset_param::cy) /  dataset_param::fy;
    } else if (dataset == "nyudepthv2"){
        J(1,0) = point(2) / dataset_param::fx;
        J(1,2) = (point(0) - dataset_param::cx) /  dataset_param::fx;
        J(0,1) = point(2) / dataset_param::fy;
        J(0,2) = (point(1) - dataset_param::cy) /  dataset_param::fy;
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
}

void dutil::backProjectJacobian(const Eigen::Vector3d& point, Matrix3d& J, const string& dataset){
    // Input point is [x, y, z]
    J.setIdentity();
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "stata" || dataset == "tartanair"){
        J(0,0) = dataset_param::fx / point(2);
        J(0,2) = -dataset_param::fx * point(0) / (point(2) * point(2));
        J(1,1) = dataset_param::fy / point(2);
        J(1,2) = -dataset_param::fy * point(1) / (point(2) * point(2));
    } else if (dataset == "nyudepthv2"){
        J(1,0) = dataset_param::fx / point(2);
        J(1,2) = -dataset_param::fx * point(0) / (point(2) * point(2));
        J(0,1) = dataset_param::fy / point(2);
        J(0,2) = -dataset_param::fy * point(1) / (point(2) * point(2));
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
}

void dutil::backProjectJacobian(const Eigen::Vector3f& point, Matrix3f& J, const string& dataset){
    J.setIdentity();
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "stata" || dataset == "tartanair"){
        J(0,0) = dataset_param::fx / point(2);
        J(0,2) = -dataset_param::fx * point(0) / (point(2) * point(2));
        J(1,1) = dataset_param::fy / point(2);
        J(1,2) = -dataset_param::fy * point(1) / (point(2) * point(2));
    } else if (dataset == "nyudepthv2"){
        J(1,0) = dataset_param::fx / point(2);
        J(1,2) = -dataset_param::fx * point(0) / (point(2) * point(2));
        J(0,1) = dataset_param::fy / point(2);
        J(0,2) = -dataset_param::fy * point(1) / (point(2) * point(2));
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
}

void dutil::o3dimage2vec2D(const t::geometry::Image& image, std::vector<std::vector<float>>& result, const float& scale, const float& max){
    // Extract depth values from png images read using open3d
    // Read raw png images is difficult in C++ without specialized library!
    using namespace open3d::core;
    auto shape = image.AsTensor().GetShape();
    SizeVector reduced_shape = {shape[0], shape[1]};
    MatrixXf img = eigen_converter::TensorToEigenMatrixXf(image.AsTensor().Reshape(reduced_shape));
    for (int v = 0; v < img.rows() ; ++v){
        std::vector<float> row;
        for (int u = 0; u < img.cols(); ++u){
            float depth = img(v,u) / scale;
            if (depth <= max){
                row.push_back(depth);
            } else {
                row.push_back(0);
            }
        }
        result.push_back(row);
    }
    //std::cout << "Number of rows: " << img.rows() << ", columns: " << img.cols() << std::endl;
}

void dutil::CreatePointCloud(const std::vector<std::vector<float>>& depth_image, const image_channels_t& rgb_image,
                             const string& dataset, const int& max_depth, std::shared_ptr<geometry::PointCloud>& pcd){
    geometry::PointCloud pcd_obj;
    for (int v = 0; v < depth_image.size(); v++){
        for (int u = 0; u < depth_image[0].size(); u++){
            //std::cout << depth_image[v][u] << std::endl;
            if (depth_image[v][u] > 0){
                Eigen::Vector3d point;
                Eigen::Vector3d color;
                forwardProject(v,u,(double) depth_image[v][u], point, dataset, 20);
                if (!std::isnan(point[0])){
                    pcd_obj.points_.push_back(point);
                    color << (double) rgb_image[0][v][u]/255, (double) rgb_image[1][v][u]/255, (double) rgb_image[2][v][u]/255;
                    pcd_obj.colors_.push_back(color);
                }
            }
        }
    }
    *pcd = pcd_obj;
}

void dutil::CreatePointCloudFromDepth(const std::vector<std::vector<float>>& depth_image,
                             const string& dataset, const int& max_depth, std::shared_ptr<geometry::PointCloud>& pcd, int downsample){
    geometry::PointCloud pcd_obj;
    for (int v = 0; v < depth_image.size(); v = v + downsample){
        for (int u = 0; u < depth_image[0].size(); u = u + downsample){
            //std::cout << depth_image[v][u] << std::endl;
            if (depth_image[v][u] > 0){
                Eigen::Vector3d point;
                Eigen::Vector3d color;
                forwardProject(v,u,(double) depth_image[v][u], point, dataset, 20);
                if (!std::isnan(point[0])){
                    pcd_obj.points_.push_back(point);
                }
            }
        }
    }
    *pcd = pcd_obj;
}

void dutil::writeEigenToCSV(const string& fileName, const MatrixXd&  matrix){
    //https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");

    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();
    }
}

MatrixXd dutil::readEigenFromCSV(const string& fileName, bool ignore_first_row ){
    // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

    // the input is the file: "fileToOpen.csv":
    // a,b,c
    // d,e,f
    // This function converts input file data into the Eigen matrix format
    // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
    // M=[a b c
    //    d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
    // later on, this vector is mapped into the Eigen matrix format
    std::vector<double> matrixEntries;

    // in this object we store the data from the matrix
    std::ifstream matrixDataFile(fileName);

    // this variable is used to store the row of the matrix that contains commas
    string matrixRowString;

    // this variable is used to store the matrix entry;
    string matrixEntry;

    // this variable is used to track the number of rows
    int matrixRowNumber = 0;

    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        //std::cout << fmt::format("Processed row {}", matrixRowNumber) << std::endl;
        if (!ignore_first_row ||  matrixRowNumber != 0){
            std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.
            while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
            {
                matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
            }
        }
        matrixRowNumber++; //update the column numbers
    }

    // here we convet the vector variable into the matrix and return the resulting object,
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    if (ignore_first_row){
        return Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber - 1, matrixEntries.size() / (matrixRowNumber - 1));
    } else {
        return Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
    }

}

void dutil::Convert2PointCloud(const std::list<Eigen::Vector3d>& points, std::shared_ptr<geometry::PointCloud>& pcd){
    // Creating a pointcloud with a list of points
    using namespace open3d;
    geometry::PointCloud pcd_obj;
    for (auto it = points.begin(); it != points.end(); ++it){
        //std::cout << *it << std::endl;
        pcd_obj.points_.push_back(*it);
    }
    pcd = std::make_shared<geometry::PointCloud>(pcd_obj);
}

void dutil::processPoseO3d(const std::vector<std::string>& poses, int idx, const string& dataset, core::Tensor& T_frame_to_model){
    Eigen::Isometry3d pose;
    processPose(poses, idx, dataset, pose);
    // eigen_converter::EigneMatrixToTensor does not work very well with mlpack libraries!
    T_frame_to_model = open3d::core::eigen_converter::EigenMatrixToTensor(pose.matrix());
}

void dutil::processPose(const std::vector<std::string>& poses, int idx, const string& dataset,
                        Eigen::Isometry3d& T_frame_to_model){
    // Obtain ground truth pose
    using namespace open3d::core;
    Matrix4d pose_mat;
    dutil::poseProcessRaw(poses.at(idx), pose_mat, dataset);
    if (dataset == "tum" || dataset == "tum_fd" || dataset_param::use_slam_traj){
        // No change in pose matrix!
        pose_mat = pose_mat;
    } else if (dataset == "nyudepthv2"){
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    } else if (dataset == "tartanair"){
        // Rotate pose to the correct orientation
        Matrix3d R;
        R << 0, 0, 1,
             1, 0, 0,
             0, 1, 0;
        pose_mat.topLeftCorner<3,3>() = pose_mat.topLeftCorner<3,3>() * R;
        pose_mat.topRightCorner<3,1>() = pose_mat.topRightCorner<3,1>();
    } else if (dataset == "stata") {
        //std::cout << "Raw pose" << std::endl;
        //std::cout << pose_mat << std::endl;
        //std::cout << "Camera rel pose" << std::endl;
        //std::cout << dataset_param::cam_rel_pose << std::endl;
        pose_mat = pose_mat * dataset_param::cam_rel_pose;
        Matrix3d R;
        R << 0, 0, 1,
                1, 0, 0,
                0, 1, 0;
        pose_mat.topLeftCorner<3,3>() = pose_mat.topLeftCorner<3,3>() * R;
        pose_mat.topRightCorner<3,1>() = pose_mat.topRightCorner<3,1>();
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
    T_frame_to_model = pose_mat;
}

void dutil::processPose(const std::vector<std::string>& poses, int idx, const string& dataset,
                        Eigen::Isometry3f& T_frame_to_model){
    Eigen::Isometry3d T_frame_to_model_d;
    processPose(poses, idx, dataset, T_frame_to_model_d);
    T_frame_to_model = T_frame_to_model_d.cast<float>();
}

std::shared_ptr<t::geometry::Image> dutil::createImageNpy2O3d(const string& filepath, float depth_scale){
    auto img_tensor = open3d::t::io::ReadNpy(filepath) * depth_scale;
    // Clip to prevent overflow when converting to UInt16
    // If depth_scale = 100, we can produce and depth image whose max range is 65.5m
    return std::make_shared<t::geometry::Image>(img_tensor.Clip_(0, 65535).To(open3d::core::UInt16));
}

void dutil::poseError(const Matrix4d& Qi, const Matrix4d& Qn, const Matrix4d& Pi, const Matrix4d& Pn, double& transError, double& rotError){
    // Returns the rotational error
    // See: http://www.boris-belousov.net/2016/12/01/quat-dist/
    // See paper: A Benchmark for the Evaluation of RGB-D SLAM Systems

    auto Qr = Qi.inverse() * Qn;
    auto Pr = Pi.inverse() * Pn;
    auto E = Qr.inverse() * Pr;

    // Squared norm (L2 norm, sqrt not applied yet!)
    transError = E.topRightCorner(3,1).squaredNorm();
    rotError = acos((E.topLeftCorner(3,3).trace() - 1)/2);
}

std::vector<double> dutil::updateErrors(std::vector<double>& accum_errors,
                                 const Matrix4d& Qcur, const Matrix4d& Qpre, const Matrix4d& Pcur, const Matrix4d& Ppre,
                                 const Matrix4d& Q0, const Matrix4d& P0){
    // idx: current pose index
    // cur_errors: contains intermediate forms of the current error states
    // Q, P: predicted and actual pose of the robot
    // Returns a vector of instantaneous errors: [idx, distance, ATE(m), ATE(rad), RPE(m), RPE(rad)]

    accum_errors.at(0) += 1;
    accum_errors.at(1) += (Pcur.topRightCorner(3,1) - Ppre.topRightCorner(3,1)).norm();

    double ATErmse, ATErad, RPErmse, RPErad;
    dutil::poseError(Qpre, Qcur, Ppre, Pcur, RPErmse, RPErad);
    dutil::poseError(Q0, Qcur, P0, Pcur, ATErmse, ATErad);
    accum_errors.at(2) += ATErmse;
    accum_errors.at(3) += ATErad;
    accum_errors.at(4) += RPErmse;
    accum_errors.at(5) += RPErad;

    // Instantaneous errors
    std::vector<double> results;
    results.push_back(accum_errors.at(0));
    results.push_back(accum_errors.at(1));
    results.push_back(sqrt(ATErmse));
    results.push_back(ATErad);
    results.push_back(sqrt(RPErmse));
    results.push_back(RPErad);

    return results;
}

void dutil::gaussianImageProj(const Vector3d& cluster_mean, const Matrix3d& cluster_cov,
                       Vector2d& img_mean, Matrix2d& img_cov, const string& dataset){
    double v, u, d;
    backProject(v, u, d, cluster_mean, dataset);
    img_mean << v, u;
    Matrix<double, 2, 3> proj_jacob;

    // Obtain camera intrinsic from a dataset
    // https://openaccess.thecvf.com/content_cvpr_2018/papers/Dhawale_Fast_Monte-Carlo_Localization_CVPR_2018_paper.pdf
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "tartanair" || dataset == "stata"){
        proj_jacob << 0, dataset_param::fy/cluster_mean(2), -dataset_param::fy*cluster_mean(1)/pow(cluster_mean(2),2),
                    dataset_param::fx/cluster_mean(2), 0, -dataset_param::fx*cluster_mean(0)/pow(cluster_mean(2),2);
    } else if (dataset == "nyudepthv2"){
        proj_jacob << dataset_param::fx/cluster_mean(2), 0, -dataset_param::fx*cluster_mean(0)/pow(cluster_mean(2),2),
                    0, dataset_param::fy/cluster_mean(2), -dataset_param::fy*cluster_mean(1)/pow(cluster_mean(2),2);
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
    img_cov = proj_jacob * cluster_cov * proj_jacob.transpose();
}

void dutil::gaussianPBox(const Vector3d& cluster_mean, const Matrix3d& cluster_cov,
                  int(&PBoxLow)[2], int (&PBoxHigh)[2], const string& dataset, double scale){
    // Calculate the bounding box of Gaussian reprojection on the image plane.
    Vector2d img_mean;
    Matrix2d img_cov;
    gaussianImageProj(cluster_mean, cluster_cov, img_mean, img_cov, dataset);
    SelfAdjointEigenSolver<Matrix2d> Sol(img_cov);
    auto evec0 = Sol.eigenvectors().col(0).cwiseAbs2() * Sol.eigenvalues()[0];
    auto evec1 = Sol.eigenvectors().col(1).cwiseAbs2() * Sol.eigenvalues()[1];
    double vbound = sqrt(evec0(0) + evec1(0));
    double ubound = sqrt(evec0(1) + evec1(1));
    PBoxLow[0] = (int) round(img_mean(0) - scale * vbound);
    PBoxLow[1] = (int) round(img_mean(1) - scale * ubound);
    PBoxHigh[0] = (int) round(img_mean(0) + scale * vbound);
    PBoxHigh[1] = (int) round(img_mean(1) + scale * ubound);
}

Matrix3d dutil::sensorNoiseCov(int v, int u, double d, const string& dataset){
    // Predict the noise of the depth sensor
    // See publications: 1) Efficient Parametric Multi-Fidelity Surface Mapping
    double var_z = 0.0012 + 0.0019*pow(d-0.4,2) + 0.0001/sqrt(d);
    Matrix3d cov, J_x;
    cov <<  1.0/12, 0, 0,
            0, 1.0/12, 0,
            0, 0, pow(var_z,2);
    // TODO: TUM Fastdepth
    if (dataset == "tum" || dataset == "tartanair" || dataset == "stata"){
        J_x <<  1/dataset_param::fx, 0, (u - dataset_param::cx)/dataset_param::fx,
                0, 1/dataset_param::fy, (v - dataset_param::cy)/dataset_param::fy,
                0, 0, 1;
    } else if (dataset == "nyudepthv2"){
        J_x <<  0, 1/dataset_param::fy, (v - dataset_param::cy)/dataset_param::fy,
                1/dataset_param::fx, 0, (u - dataset_param::cx)/dataset_param::fx,
                0, 0, 1;
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
    cov = J_x * cov * J_x.transpose();
    return cov;
}

void dutil::LoadFilenames(std::vector<std::string>& rgb_files, std::vector<std::string>& depth_files,
                   std::vector<std::string>& pose_files, std::vector<double>& timestamps, const string& dataset) {
    std::string dataset_rgb_path;
    std::string dataset_depth_path;
    std::cout << fmt::format("Loading RGB and depth data files for dataset: {} ...", dataset) << std::endl;
    if (dataset == "tum"){
        // Load appropriate filenames in a list of strings
        dataset_rgb_path = dataset_param::rgb_txt;
        dataset_depth_path = dataset_param::depth_txt;
        dutil::getTUMDatasetRaw(dataset, dataset_param::rgb_txt, dataset_param::depth_txt, dataset_param::pose_txt,
                                rgb_files, depth_files, pose_files, timestamps);
        if (dataset_param::use_slam_traj){
            std::cout << fmt::format("Poses from SLAM is loaded from {}", dataset_param::slam_traj_txt.string()) << std::endl;
            pose_files.clear();
            dutil::getPosesRaw(dataset_param::slam_traj_txt, pose_files);
        }
    } else if (dataset == "tum_fd"){
        // Load appropriate filenames in a list of strings
        dataset_rgb_path = dataset_param::rgb_txt;
        dataset_depth_path = dataset_param::depth_txt;
        dutil::getTUMDatasetRaw(dataset, dataset_param::rgb_txt, dataset_param::depth_txt, dataset_param::pose_txt,
                                rgb_files, depth_files, pose_files, timestamps);
        if (dataset_param::use_slam_traj){
            std::cout << fmt::format("Poses from SLAM is loaded from {}", dataset_param::slam_traj_txt.string()) << std::endl;
            pose_files.clear();
            dutil::getPosesRaw(dataset_param::slam_traj_txt, pose_files);
        }
    } else if (dataset == "nyudepthv2"){
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    } else if (dataset == "tartanair"){
        // Load appropriate filenames in a list of strings
        dataset_rgb_path = dataset_param::rgb_path;
        dataset_depth_path = dataset_param::depth_path;
        dutil::getImagesRaw(dataset_param::rgb_path, rgb_files);
        dutil::getImagesRaw(dataset_param::depth_path, depth_files);
        if (dataset_param::use_slam_traj) {
            std::cout << fmt::format("Poses from SLAM is loaded from {}", dataset_param::slam_traj_txt.string()) << std::endl;
            dutil::getPosesRaw(dataset_param::slam_traj_txt, pose_files);
        } else {
            dutil::getPosesRaw(dataset_param::pose_txt, pose_files);
        }
        for (int idx = 0; idx < rgb_files.size(); idx++){
            timestamps.push_back(idx);
        }
    } else if (dataset == "stata") {
        // Load appropriate filenames in a list of strings
        dataset_rgb_path = dataset_param::rgb_path;
        dataset_depth_path = dataset_param::depth_path;
        dutil::getImagesRaw(dataset_param::rgb_path, rgb_files);
        dutil::getImagesRaw(dataset_param::depth_path, depth_files);

        // Obtain timestamps from the filename
        timestamps.clear();
        for (auto& path : rgb_files){
            std::size_t found_ext = path.find_last_of(".");
            std::size_t found_file = path.find_last_of("/\\");
            auto time_str = path.substr(found_file+1, found_ext - found_file - 1);
            time_str.erase(0, std::min(time_str.find_first_not_of('0'), time_str.size()-1));
            timestamps.push_back(std::stod(time_str));
        }

        // Generate pose information using the timestamps
        std::ifstream matrixDataFile(dataset_param::uav1_sim_txt);
        string matrixRowString;
        getline(matrixDataFile, matrixRowString); // Skip the first row.
        int idx = 0;
        while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
        {
            string matrixEntry;
            std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.
            getline(matrixRowStringStream, matrixEntry, ',');
            if (stol(matrixEntry) + 1 == (long) timestamps[idx] || stol(matrixEntry) == (long) timestamps[idx]){
                pose_files.push_back(matrixRowStringStream.str());
                idx++;
                if (idx == timestamps.size()){
                    break;
                }
            }
        }

        if (dataset_param::use_slam_traj){
            std::cout << fmt::format("Poses from SLAM is loaded from {}", dataset_param::slam_traj_txt.string()) << std::endl;
            pose_files.clear();
            dutil::getPosesRaw(dataset_param::slam_traj_txt, pose_files);
        }

        if (timestamps.size() != pose_files.size()){
            throw std::invalid_argument(fmt::format("Length of ground truth pose {} is not the same as the number of RGB-D frames {}!", pose_files.size(), timestamps.size()));
        }
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }

    // Load rgb files
    if (rgb_files.empty()) {
        utility::LogError(
                "RGB images not found! Please ensure directory: {} exists!",
                dataset_rgb_path);
    }

    if (depth_files.empty()) {
        utility::LogError(
                "Depth images not found! Please ensure directory: {} exists!",
                dataset_depth_path);
    }

    if (depth_files.size() != rgb_files.size()) {
        utility::LogError(
                "Number of depth images ({}) and color image ({}) "
                "mismatch!");
    }
    std::cout << "Loaded RGB, depth and pose data files ..." << std::endl;
}

std::pair<core::Tensor, camera::PinholeCameraIntrinsic> dutil::LoadIntrinsics(const string& dataset) {
    std::cout << "Loading camera intrinsics ..." << std::endl;
    core::Tensor intrinsic_t;
    camera::PinholeCameraIntrinsic intrinsic_legacy;
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "tartanair" || dataset == "stata"){
        // Define the intrinsic calibration parameters
        intrinsic_t = core::Tensor::Init<double>(
                {{dataset_param::fx, 0, dataset_param::cx},
                 {0, dataset_param::fy, dataset_param::cy},
                 {0, 0, 1}});

        intrinsic_legacy = camera::PinholeCameraIntrinsic(
                -1, -1, dataset_param::fx, dataset_param::fy,
                dataset_param::cx, dataset_param::cy);
    } else if (dataset == "nyudepthv2"){
        // Define the intrinsic calibration parameters
        intrinsic_t = core::Tensor::Init<double>(
                {{dataset_param::fx, 0, dataset_param::cx},
                 {0, dataset_param::fy, dataset_param::cy},
                 {0, 0, 1}});

        intrinsic_legacy = camera::PinholeCameraIntrinsic(
                -1, -1, dataset_param::fx, dataset_param::fy,
                dataset_param::cx, dataset_param::cy);
    } else {
        std::ostringstream ss;
        ss << "Dataset: " << dataset << " is not supported!";
        throw std::invalid_argument(ss.str());
    }
    std::cout << "Loaded camera intrinsics ..." << std::endl;
    return std::make_pair(intrinsic_t, intrinsic_legacy);
}

std::list<std::vector<int>> dutil::nearestNeighbors(const std::vector<int>& coord, int v_max, int u_max){
    // Determine the 8 neighbors of a given image coordinate
    std::list<std::vector<int>> neighbors;
    int v = coord[0];
    int u = coord[1];
    if (v-1 >= 0){
        neighbors.push_back({v-1,u});
        if (u-1 >= 0){
            neighbors.push_back({v-1,u-1});
        }
        if (u+1 < u_max){
            neighbors.push_back({v-1,u+1});
        }
    }
    if (v+1 < v_max){
        neighbors.push_back({v+1,u});
        if (u-1 >= 0){
            neighbors.push_back({v+1,u-1});
        }
        if (u+1 < u_max){
            neighbors.push_back({v+1,u+1});
        }
    }
    if (u-1 >= 0){
        neighbors.push_back({v,u-1});
    }
    if (u+1 < u_max){
        neighbors.push_back({v,u+1});
    }
    return neighbors;
}

void dutil::getPeakRSS(double& peakRSS){
    // Return results in kB
    struct rusage ru;
    getrusage( RUSAGE_SELF, &ru );
    peakRSS = ru.ru_maxrss;
}

void dutil::getCurrentRSS(double& curRSS, double& shareRSS, double& privateRSS){
    int tSize = 0, resident = 0, share = 0;
    std::ifstream buffer("/proc/self/statm");
    buffer >> tSize >> resident >> share;
    buffer.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    curRSS = resident * page_size_kb;
    shareRSS = share * page_size_kb;
    privateRSS = curRSS - shareRSS;
}

void dutil::obtainEnvBBox(const std::string& dataset_name, double depth_max, double depth_scale, Eigen::Vector3d& lowerBound, Eigen::Vector3d& upperBound){
    // Contain the bounding box associated with a sequence in the environment
    // Does not filter out spurious measurements that are within depth_max
    std::cout << "Computing bounding box for the environment ..." << std::endl;
    std::cout << fmt::format("Dataset: {}, Max depth: {:.2f}m, Depth scale: {}", dataset_name, depth_max, depth_scale) << std::endl;
    std::vector<std::string> rgb_files, depth_files, pose_files;
    std::vector<double> timestamps;
    dutil::LoadFilenames(rgb_files, depth_files, pose_files, timestamps, dataset_name);

    // Load intrinsics
    core::Tensor intrinsic_t;
    camera::PinholeCameraIntrinsic intrinsic_legacy;
    // Obtain camera intrinsic parameters
    std::tie(intrinsic_t, intrinsic_legacy) = dutil::LoadIntrinsics(dataset_name);

    // Iterate through all frames to compute the bounding box
    for (int idx = 0; idx < depth_files.size(); idx++){
        std::shared_ptr<t::geometry::Image> input_depth, input_color;
        if (dataset_name == "tartanair") {
            input_depth = dutil::createImageNpy2O3d(depth_files[idx], depth_scale);
        } else {
            input_depth = t::io::CreateImageFromFile(depth_files[idx]);
        }
        input_color = t::io::CreateImageFromFile(rgb_files[idx]);

        if (dataset_name == "stata"){
            *input_color = input_color->Resize(dataset_param::sampling_rate);
            *input_depth = input_depth->Resize(dataset_param::sampling_rate);
        }

        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(input_color->ToLegacy(), input_depth->ToLegacy(),
                                                                       depth_scale, depth_max);
        auto cur_frame_pcd = geometry::PointCloud::CreateFromRGBDImage(*rgbd_image, intrinsic_legacy);

        Eigen::Isometry3d curPose;
        dutil::processPose(pose_files, idx, dataset_name,curPose);

        // Compute bounding box
        if (idx == 0){
            lowerBound = curPose.translation();
            upperBound = curPose.translation();
        } else {
            lowerBound = lowerBound.cwiseMin(curPose.translation());
            upperBound = upperBound.cwiseMax(curPose.translation());
        }

        /*
        auto transformed_pcd = cur_frame_pcd->Transform(curPose.matrix());
        for (int i = 0; i < transformed_pcd.points_.size(); i++){
            lowerBound = lowerBound.cwiseMin(transformed_pcd.points_[i]);
            upperBound = upperBound.cwiseMax(transformed_pcd.points_[i]);
        }
        */

        auto bbox = cur_frame_pcd->Transform(curPose.matrix()).GetAxisAlignedBoundingBox();
        lowerBound = lowerBound.cwiseMin(bbox.GetMinBound());
        upperBound = upperBound.cwiseMax(bbox.GetMaxBound());

        if (idx % 10 == 0){
            //std::cout << curPose.matrix() << std::endl;
            std::cout << fmt::format("Frame {}/{}, Global min. bound: [{:.2f}, {:.2f}, {:.2f}], Global max. bound: [{:.2f}, {:.2f}, {:.2f}]\t\r", idx+1, depth_files.size(),
                                     lowerBound[0], lowerBound[1], lowerBound[2],
                                     upperBound[0], upperBound[1], upperBound[2]) << std::flush;
        }
    }
    std::cout << fmt::format("Frames: {}, Global min. bound: [{:.2f}, {:.2f}, {:.2f}], Global max. bound: [{:.2f}, {:.2f}, {:.2f}]", depth_files.size(),
                             lowerBound[0], lowerBound[1], lowerBound[2],
                             upperBound[0], upperBound[1], upperBound[2]) << std::endl;
}

void dutil::saveBBoxToCSV(const string& dataset, const Eigen::Vector3d& BBoxHigh, const Eigen::Vector3d& BBoxLow){
    //https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", ", ");
    fs::path fileName = dataset_param::raw_data_path / fs::path("bbox.txt");

    // Open with replacement mode
    std::ofstream file(fileName, std::ios::out);
    if (file.is_open())
    {
        file << BBoxHigh.format(CSVFormat) << std::endl;
        file << BBoxLow.format(CSVFormat) << std::endl;
        file.close();
    }
}

// Returns false if BBox CSV file does not exists!
bool dutil::readBBoxFromCSV(const string& dataset, Eigen::Vector3d& BBoxHigh, Eigen::Vector3d& BBoxLow){
    fs::path fileName = dataset_param::raw_data_path / fs::path("bbox.txt");
    // Return false if the bbox file is not found!
    if (!fs::exists(fileName))
        return false;

    // Prepare to read the file and obtain the bbox!
    // in this object we store the data from the matrix
    std::ifstream matrixDataFile(fileName);

    // this variable is used to store the row of the matrix that contains commas
    string matrixRowString;
    int row_idx = 0;

    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        // this variable is used to store the matrix entry;
        string matrixEntry;
        std::vector<double> matrixEntries;
        std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

        while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        // here we convet the vector variable into the matrix and return the resulting object,
        // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
        if (row_idx == 0)
            BBoxHigh << matrixEntries.at(0), matrixEntries.at(1), matrixEntries.at(2);
        else if (row_idx == 1)
            BBoxLow << matrixEntries.at(0), matrixEntries.at(1), matrixEntries.at(2);
        else {
            std::cout << fmt::format("File contains more than two entries. Ignoring content!") << std::endl;
            return false;
        }
        row_idx++;
    }
    std::cout << fmt::format("Bounding Box: [{:.2f}, {:.2f}, {:.2f}] to [{:.2f}, {:.2f}, {:.2f}]", BBoxLow[0], BBoxLow[1], BBoxLow[2], BBoxHigh[0], BBoxHigh[1], BBoxHigh[2]) << std::endl;
    return true;
}

Matrix3d dutil::cameraIntrinsic(const std::string &dataset) {
    // Obtain the camera intrinsics matrix
    using namespace dataset_param;
    Matrix3d K;
    if (dataset == "tum" || dataset == "tum_fd" || dataset == "tartanair" || dataset == "stata"){
        K << fx, 0, cx,
             0, fy, cy,
             0, 0, 1;
    } else
        throw std::runtime_error("Dataset: " + dataset + " is not supported!");
    return K;
}

Matrix3d dutil::cameraIntrinsicInv(const std::string &dataset) {
    // Obtain the inverse camera intrinsics matrix
    return cameraIntrinsic(dataset).inverse();
}