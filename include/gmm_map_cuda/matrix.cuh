//
// Created by peter on 7/8/22.
// Define common operations for matrix classes
#ifndef GMM_MAPPING_MATRIX_CUH
#define GMM_MAPPING_MATRIX_CUH
#include "cuda_common.cuh"

// Simulate some matrix behaviour on the CUDA device
// Row major format for storage. Note that the Eigen C++ library uses column major storage by default
namespace cuEigen {
    // Pre-declaration
    using Vector3f = float3;
    class Matrix3f;

    __host__ __device__ void print(const Vector3f& vec);
    __host__ __device__ Matrix3f dotRowVec(const Vector3f& a, const Vector3f& b);
    __host__ __device__ float normSquared(const Vector3f& a);
    __host__ __device__ float norm(const Vector3f& a);

    class Matrix3f {
    public:
        // Assume column major storage
        Vector3f data[3];
        __host__ __device__ Matrix3f(bool init_zero = false);
        __host__ __device__ Matrix3f(const Vector3f& col_vec_0, const Vector3f& col_vec_1, const Vector3f& col_vec_2);
        __host__ __device__ void setZeros();
        __host__ __device__ void setOnes();
        __host__ __device__ Matrix3f operator+(const Matrix3f& mat) const;
        __host__ __device__ Matrix3f operator-(const Matrix3f& mat) const;
        __host__ __device__ Matrix3f operator*(const float& c) const;
        __host__ __device__ Vector3f operator*(const Vector3f& vec) const;
        __host__ __device__ Matrix3f operator*(const Matrix3f& mat) const;
        __host__ __device__ Matrix3f operator/(const float& c) const;
        __host__ __device__ float& operator()(int row_idx, int col_idx);
        __host__ __device__ float operator()(int row_idx, int col_idx) const;
        __host__ __device__ void print() const;
    };
}
#endif //GMM_MAPPING_MATRIX_CUH
