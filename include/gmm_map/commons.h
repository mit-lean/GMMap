#ifndef GMM_COMMONS_H
#define GMM_COMMONS_H
// Define data types used in the mapping framework
#include <Eigen/Eigen>
#include <list>
#include <string>
#include <cmath>
#include <stdexcept>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

namespace gmm {
    using RowMatrixXd = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowMatrixXi = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowDepthScanlineXf = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
    using RowRGBScanlineXi = Eigen::Matrix<uint8_t, 3, Eigen::Dynamic, Eigen::RowMajor>;
    using RowColorChannelScanlineXi = Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor>;

    // Define the floating point precision used in mapping
    // We place this under the gmm namespace to prevent potential pollution
    using FP = float;

    typedef Eigen::Transform<FP, 3, Eigen::Isometry> Isometry3;

    typedef Eigen::Matrix<FP, 2, 1> V_2;
    typedef Eigen::Matrix<FP, 2, 2> M_2;

    typedef Eigen::Matrix<FP, 3, 1> V;
    typedef Eigen::Matrix<FP, 3, 3> M;

    // 6D vector and matrices to track color and position
    typedef Eigen::Matrix<FP,6,1> V_c;
    typedef Eigen::Matrix<FP,6,6> M_c;
    typedef Eigen::Matrix<FP,3,6> M_c_eff; // Only track part of the color covariance matrix for efficiency

    // 4D vector and matrices for tracking occupancy
    typedef Eigen::Matrix<FP,4,1> V_o;
    typedef Eigen::Matrix<FP,4,4> M_o;

    // 7D vector and matrices and arrays
    typedef Eigen::Array<FP, 7, 1> Array7;
    typedef Eigen::Matrix<FP, 7, 1> Vector7;
    typedef Eigen::Matrix<FP, 3, 7> Matrix3x7;

    // BBox information (3D)
    typedef Eigen::AlignedBox<FP, 3> Rect;
    typedef Eigen::Matrix<FP, 3, 1> Rect_Vec;
    typedef Eigen::Matrix<FP, 3, 1> Rect_Real_Vec;

    // PBox information (2D in pixel space)
    typedef Eigen::AlignedBox<int, 2> PRect;
    typedef Eigen::Matrix<int, 2, 1> PRect_Vec;
    typedef Eigen::Matrix<int, 2, 1> PRect_Real_Vec;

    // Constants
    constexpr FP PI_GREEK = 3.14159f;
    static const FP LOG_2_PI = logf(2.0f * PI_GREEK);
    static const FP RT_2_PI_3 = sqrtf(powf(2.0f*PI_GREEK,3));

    template<typename T1, typename T2, typename T3>
    inline void forwardProject(const T3 v, const T3 u, const T1 d,
                        const T1 fx, const T1 fy,
                        const T1 cx, const T1 cy,
                        Eigen::Matrix<T2, 3, 1>& point,
                        const std::string& dataset, const T1 max_depth)
    {
        // Assume that the depth is already scaled
        // Obtain the 3D point of a pixel on the depth map
        // Eigen::Vector3d camera_coord;
        // camera_coord << v, u, d;
        if (dataset == "tum" || dataset == "tartanair" || dataset == "tum_fd" || dataset == "stata"){
            // Nan will be zero
            if (d <= 0 || d > max_depth){
                point << NAN, NAN, NAN;
            } else {
                //point = dataset_param::tum_K_inv * camera_coord;
                point(0) = ((T1) u - cx) * d / fx;
                point(1) = ((T1) v - cy) * d / fy;
                point(2) = d;

            }

        } else if (dataset == "nyudepthv2"){
            // Nan will be zero
            if (d <= 0 || d > max_depth){
                point << NAN, NAN, NAN;
            } else {
                //point = dataset_param::nyu_K_inv * camera_coord;

                point(0) = ((T1) v - cy) * d / fy;
                point(1) = ((T1) u - cx) * d / fx;
                point(2) = d;

            }
        } else {
            std::ostringstream ss;
            ss << "Dataset: " << dataset << " is not supported!";
            throw std::invalid_argument(ss.str());
        }
    }

    template<typename T1, typename T2, typename T3>
    inline void forwardProjectVariance(const T3 v, const T3 u,
                                const T1 fx, const T1 fy,
                                const T1 cx, const T1 cy,
                                const T1 variance,
                                Eigen::Matrix<T2, 3, 3>& covariance, const std::string& dataset){
        Eigen::Matrix<T2, 3, 1> scale;
        if (dataset == "tum" || dataset == "tartanair" || dataset == "tum_fd" || dataset == "stata"){
            scale = {((T1) u - cx) / fx, ((T1) v - cy) / fy, 1};
        } else if (dataset == "nyudepthv2"){
            scale = {((T1) v - cy) / fy, ((T1) u - cx) / fx,  1};
        } else {
            std::ostringstream ss;
            ss << "Dataset: " << dataset << " is not supported!";
            throw std::invalid_argument(ss.str());
        }
        covariance = variance * scale * scale.transpose();
    }
}

#endif
