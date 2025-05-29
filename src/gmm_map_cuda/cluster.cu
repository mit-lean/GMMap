//
// Created by peterli on 7/9/22.
// Implementation of some functions used during SPGF
#include "gmm_map_cuda/cluster.cuh"
namespace cuGMM {
    __host__ __device__ void GMMmetadata_o::printInfo() const {
        printf("Printing info for a cluster metadata with N: %d, W: %.4f as follows:\n", N, W);
        /*
        printf("S:\n");
        S.print();
        printf("J:\n");
        J.print();
         */
    }

    __host__ __device__ float GMMmetadata_c::distDepth(const V& pt) const {
        // Distance alone the depth
        if (N == 0){
            return 0;
        } else {
    #ifdef __CUDA_ARCH__
            return fabsf (pt.z - RightPoint.z);
    #else
            return std::abs (pt.z - RightPoint.z);
    #endif
        }
    }

    __host__ __device__ float GMMmetadata_c::distLine(const V& pt) const {
        // Distance alone the scanline
        if (N == 0){
            return 0;
        } else {
    #ifdef __CUDA_ARCH__
            return fabsf (pt.x - RightPoint.x);
    #else
            return std::abs (pt.x - RightPoint.x);
    #endif
        }
    }

    __host__ __device__ float GMMmetadata_c::distDepthEst(const V& pt) const {
    #ifdef __CUDA_ARCH__
        return LeftPoint.z + LineVec.z * __fdividef(pt.x-LeftPoint.x, LineVec.x);
    #else
        return LeftPoint.z + LineVec.z*(pt.x-LeftPoint.x)/LineVec.x;
    #endif
    }

    __host__ float GMMmetadata_c::distPlaneSquared(const V& pt) const {
        // Check if a point lies on the same plane as cluster described by the metadata
        // Used for obstacle cluster only!
        if (NumLines == 1){
            float dist = std::abs(S.z/ (float) N - pt.z);
            return dist * dist;
        } else {
            V normal = cross(LineVec, PlaneVec);
            float normal_scale = dot(normal,normal);
            float dist = dot(normal,S / (float) N) - dot(normal,pt);
            return dist * dist / normal_scale;
        }
    }

    __host__ __device__ void freeSpaceBasis::merge(const freeSpaceBasis &new_data) {
        // Merge metadata associated with free space
        W_ray_ends = W_ray_ends + new_data.W_ray_ends;
        S_ray_ends = S_ray_ends + new_data.S_ray_ends;
        J_ray_ends = J_ray_ends + new_data.J_ray_ends;

        // The back accumulates the bases used to infer all other free space clusters
        W_basis = W_basis + new_data.W_basis;
        S_basis = S_basis + new_data.S_basis;
        J_basis = J_basis + new_data.J_basis;
        depth_idx = min(depth_idx, new_data.depth_idx);
    }
}
