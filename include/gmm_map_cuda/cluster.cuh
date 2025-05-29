//
// Created by peterli on 7/9/22.
//
#ifndef GMM_MAPPING_CLUSTER_CUH
#define GMM_MAPPING_CLUSTER_CUH
#include "cuda_common.cuh"
#include "matrix.cuh"

namespace cuGMM {
    using V = cuEigen::Vector3f;
    using M = cuEigen::Matrix3f;

    // GPU compatible data structure
    struct GMMmetadata_o {
        V     S = {0,0,0};
        M     J = M(true);

        // Note that these only tracks the center of the segments
        V       LeftPoint;
        V       RightPoint;

        int     NumLines = 1;
        int     N = 0;
        float   W = 0;
        int     LeftPixel = 0;
        int     RightPixel = 0;
        float   depth = 0; // Current z-distance from the camera

        bool    Updated = false; // Used during segment fusion to track if it is fused with another segment from the following scanline
        bool    near_obs = false; // Flag to track if the free cluster appears near an obstacle cluster
        bool    cur_pt_obs = false; // Flag to check if the current cluster is near an obstacle
        bool    fused = false; // Flag to check if the segment is fused wth other segments

        // Bounding Box Information
        V   BBoxLow;
        V   BBoxHigh;

        __host__ __device__ void printInfo() const;
    };

    // Metadata needed to track free space
    struct freeSpaceBasis {
        V   S_ray_ends = {0,0,0};
        M   J_ray_ends = M(true);
        V   S_basis = {0,0,0};
        M   J_basis = M(true);
        float   W_ray_ends = 0;
        float   W_basis = 0;
        int     depth_idx = 0;

        // Bounding Box Information
        V   BBoxLow;
        V   BBoxHigh;

        bool    cluster_cross_depth_boundary = false;

        __host__ __device__ void merge(const freeSpaceBasis& new_data);
    };

    // Extra metadata is needed for scanline segmentation
    struct GMMmetadata_c : GMMmetadata_o {
        V       LineVec = {0,0,0};
        V       PlaneVec = {0,0,0};
        V       UpMean = {0,0,0};
        float   PlaneLen = 0;
        float   PlaneWidth = 0;

        // Index of the depth array used for storing depth bounds
        bool    track_color = false;

        // Free space metadata
        freeSpaceBasis freeBasis;

        __host__ __device__ float distDepth(const V& pt) const;

        __host__ __device__ float distLine(const V& pt) const;

        __host__ __device__ float distDepthEst(const V& pt) const;

        __host__ float distPlaneSquared(const V& pt) const;
    };


}
#endif //GMM_MAPPING_CLUSTER_CUH
