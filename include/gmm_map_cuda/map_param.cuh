//
// Created by peter on 7/11/22.
// Parameters transferred to CUDA for computation

#ifndef GMM_MAPPING_MAP_PARAM_CUH
#define GMM_MAPPING_MAP_PARAM_CUH
#include "cuda_common.cuh"
#include "matrix.cuh"

class frame_param_basic {
public:
    // Define some constants within this program. These constants will be replaced by input args!
    int img_width; // Width of the depthmap
    int img_height; // Height of the depthmap
    float cx;
    float cy;
    float fx;
    float fy;
    float f;

    int num_threads;
    bool measure_memory; // Remove intermediate representations for region growing to save memory usage.
    bool update_free_gmm; // Track if free space metadata should be updated.
    bool track_color;
    bool preserve_details_far_objects;

    // Define clustering parameters for proposed incremental method
    int occ_x_t; // Maximum number of pixels in the x direction  for occlusion region
    float noise_thresh; // Maximum noise threshold of the sensor
    int sparse_t; // Minimum number of points in a cluster
    int ncheck_t; // Number of points after which fitting is applied

    float gau_bd_scale; // Bounds for computing boundary of Gaussians (BBox)
    float adaptive_thresh_scale; // Adaptive threshold scale: 0.35 for Tartanair, 1.2 for TUM (more noisy)
    //float max_len; // Maximum projected length of each line segment
    float max_depth;
    float line_t;
    float angle_t; // Dot product between line vectors
    float noise_floor; // Minimum noisy bound to avoid over segmentation (5cm)
    int num_line_t; // Line threshold (Related to size of features to track)
    int num_pixels_t; // Pixel threshold
    //int max_incomplete_clusters; // Maximum number of previous segment to consider during line segmentation

    // Parameters for advanced free space modelling to ensure accuracy of unexplored space.
    float free_space_start_len;
    float free_space_max_length; // Maximum length for generating free-space GMMs
    float free_space_dist_scale; // Multiplicative constant to increase free space merging within the frame

    // Debug parameters
    int debug_row_idx;

    __host__ __device__  frame_param_basic() = default;

};

class frame_param_cuda : public frame_param_basic {
public:
    // Dynamically allocated by the user
    float* depth_dists;
    float* depth_dists_cum; // Cumulative depth distance
    int depth_dist_array_size;

    __host__ __device__ int determineDepthIndex(const float& depth) const {
        int index = 0;
        for (int i = 0; i < depth_dist_array_size; i++){
            float cum_depth = depth_dists_cum[i];
            if (cum_depth < depth){
                index++;
            } else {
                break;
            }
        }
        return index;
    }

    __host__ __device__ void determineDepthBounds(const int& index, float& lowerBound, float& upperBound) const {
        if (index == 0){
            lowerBound = 0;
        } else {
            lowerBound = depth_dists_cum[index - 1];
        }
        upperBound = depth_dists_cum[index];
    }

    __host__ __device__ float determineDepthLength(const int& index) const {
        return depth_dists[index];
    }

    __host__ __device__ cuEigen::Vector3f forwardProject(const int& v, const int& u, const float& d) const {
#ifdef __CUDA_ARCH__
        if (d <= 0 || d > max_depth){
            // Return invalid zero depth
            return cuEigen::Vector3f{__fdividef(((float) u - cx) * d, fx), __fdividef(((float) v - cy) * d, fy), 0};
        } else {
            return cuEigen::Vector3f{__fdividef(((float) u - cx) * d, fx), __fdividef(((float) v - cy) * d, fy), d};
        }
#else
        if (d <= 0 || d > max_depth){
            // Return invalid zero depth
            return cuEigen::Vector3f{((float) u - cx) * d / fx, ((float) v - cy) * d / fy, 0};
        } else {
            return cuEigen::Vector3f{((float) u - cx) * d / fx, ((float) v - cy) * d / fy, d};
        }
#endif
    }

    __host__ __device__ void forwardProject(const int& v, const int& u, const float& d, float& x, float& y) const {
#ifdef __CUDA_ARCH__
        x = __fdividef(((float) u - cx) * d, fx);
        y = __fdividef(((float) v - cy) * d, fy);
#else
        x = ((float) u - cx) * d / fx;
        y = ((float) v - cy) * d / fy;
#endif
    }

    __host__ __device__ void adaptThreshold(const cuEigen::Vector3f& point, float& line_t_adapt, float& depth_t_adapt) const {
        // Determine the adaptive threshold for RGBD sensors
#ifdef __CUDA_ARCH__
        line_t_adapt = adaptive_thresh_scale * point.z * __fdividef(point.z, f);
#else
        line_t_adapt = adaptive_thresh_scale * point.z * point.z / f;
#endif
        depth_t_adapt = 6.0f * line_t_adapt;
    }

    void allocate_depth_array(int size) {
        depth_dist_array_size = size;
        depth_dists = new float[size];
        depth_dists_cum = new float[size];
    }

    void deallocate_depth_array() {
        delete [] depth_dists;
        delete [] depth_dists_cum;
        depth_dist_array_size = 0;
        depth_dists = nullptr;
        depth_dists_cum = nullptr;
    }

    // To use this struct as a constant variable, we need to avoid dynamic initialization associated with constructors
    /*
    frame_param_cuda() : frame_param_basic() {};

    frame_param_cuda(const frame_param_cuda& param) : frame_param_basic(param) {
        if (param.depth_dist_array_size > 0){
            depth_dists = new float[param.depth_dist_array_size];
            depth_dists_cum = new float[param.depth_dist_array_size];
            depth_dist_array_size = param.depth_dist_array_size;

            for (int i = 0; i < param.depth_dist_array_size; i++){
                depth_dists[i] = param.depth_dists[i];
                depth_dists_cum[i] = param.depth_dists_cum[i];
            }
        }
    }

    friend void swap( frame_param_cuda& first, frame_param_cuda& second) {
        using std::swap;
        swap(static_cast<frame_param_basic&>(first), static_cast<frame_param_basic&>(second));
        swap(first.depth_dist_array_size, second.depth_dist_array_size);
        swap(first.depth_dists, second.depth_dists);
        swap(first.depth_dists_cum, second.depth_dists_cum);
    }

    frame_param_cuda(frame_param_cuda&& param) noexcept : frame_param_basic() {
        swap(*this, param);
    }

    frame_param_cuda& operator=(frame_param_cuda param) {
        swap(*this, param);
        return *this;
    }

    ~frame_param_cuda() {
        deallocate_depth_array();
    }
    */
};

#endif //GMM_MAPPING_MAP_PARAM_CUH
