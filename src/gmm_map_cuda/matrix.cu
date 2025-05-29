//
// Created by peter on 7/8/22.
//
#include "gmm_map_cuda/matrix.cuh"
namespace cuEigen {
    __host__ __device__ void print(const Vector3f& vec){
        printf("[%.4f]\n[%.4f]\n[%.4f]\n", vec.x, vec.y, vec.z);
    }

    __host__ __device__ Matrix3f dotRowVec(const Vector3f& a, const Vector3f& b){
        // a * b^T
        return Matrix3f{a * b.x, a * b.y, a * b.z};
    }

    __host__ __device__ float normSquared(const Vector3f& a){
        return dot(a,a);
    }

    __host__ __device__ float norm(const Vector3f& a){
        return sqrtf(dot(a,a));
    }

    // APIs for Matrix3f
    __host__ __device__ Matrix3f::Matrix3f(bool init_zero){
        if (init_zero) {
            setZeros();
        }
    }

    __host__ __device__ Matrix3f::Matrix3f(const Vector3f& col_vec_0, const Vector3f& col_vec_1, const Vector3f& col_vec_2){
        data[0] = col_vec_0;
        data[1] = col_vec_1;
        data[2] = col_vec_2;
    }

    __host__ __device__ void Matrix3f::setZeros(){
        data[0] = {0, 0, 0};
        data[1] = {0, 0, 0};
        data[2] = {0, 0, 0};
    }

    __host__ __device__ void Matrix3f::setOnes(){
        data[0] = {1, 1, 1};
        data[1] = {1, 1, 1};
        data[2] = {1, 1, 1};
    }

    __host__ __device__ Matrix3f Matrix3f::operator+(const Matrix3f& mat) const {
        return Matrix3f{data[0] + mat.data[0],
                        data[1] + mat.data[1],
                        data[2] + mat.data[2]};
    }

    __host__ __device__ Matrix3f Matrix3f::operator-(const Matrix3f& mat) const {
        return Matrix3f{data[0] - mat.data[0],
                        data[1] - mat.data[1],
                        data[2] - mat.data[2]};
    }

    __host__ __device__ Matrix3f Matrix3f::operator*(const float& c) const {
        return Matrix3f{data[0] * c,
                        data[1] * c,
                        data[2] * c};
    }

    __host__ __device__ Vector3f Matrix3f::operator*(const Vector3f& vec) const {
        return data[0] * vec.x + data[1] * vec.y + data[2] * vec.z;
    }

    __host__ __device__ Matrix3f Matrix3f::operator*(const Matrix3f& mat) const {
        return Matrix3f{data[0] * mat.data[0].x + data[1] * mat.data[0].y + data[2] * mat.data[0].z,
                        data[0] * mat.data[1].x + data[1] * mat.data[1].y + data[2] * mat.data[1].z,
                        data[0] * mat.data[2].x + data[1] * mat.data[2].y + data[2] * mat.data[2].z};
    }

    __host__ __device__ Matrix3f Matrix3f::operator/(const float& c) const {
        return Matrix3f{data[0] / c,
                        data[1] / c,
                        data[2] / c};
    }

    __host__ __device__ float& Matrix3f::operator()(int row_idx, int col_idx) {
        switch(row_idx){
            case 0: return data[col_idx].x;
            case 1: return data[col_idx].y;
            default: return data[col_idx].z;
        }
    }

    __host__ __device__ float Matrix3f::operator()(int row_idx, int col_idx) const {
        switch(row_idx){
            case 0: return data[col_idx].x;
            case 1: return data[col_idx].y;
            default: return data[col_idx].z;
        }
    }

    __host__ __device__ void Matrix3f::print() const {
        printf("[%.4f, %.4f, %.4f]\n", data[0].x, data[1].x, data[2].x);
        printf("[%.4f, %.4f, %.4f]\n", data[0].y, data[1].y, data[2].y);
        printf("[%.4f, %.4f, %.4f]\n", data[0].z, data[1].z, data[2].z);
    }
}