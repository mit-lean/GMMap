//
// Created by peterli on 7/9/22.
// Circular buffer implementation

#ifndef RING_BUFFER_CUH
#define RING_BUFFER_CUH
#include "cuda_common.cuh"
#include "cluster.cuh"

template <typename T, int buf_size>
class CircularBuffer {
private:
    T buffer[buf_size];
    int head_i = 0; // Points at the first valid data
    int tail_i = 0; // Points to where the next data should be inserted
    int len = 0; // Track current length
public:
    __host__ __device__ CircularBuffer() = default;

    __host__ __device__ void push_back(T val) {
        buffer[tail_i] = val;
        tail_i = increment(tail_i);
        len++;
    }

    __host__ __device__ T pop_back() {
        tail_i = decrement(tail_i);
        T val = buffer[tail_i];
        len--;
        return val;
    }

    __host__ __device__ T& back() {
        return buffer[decrement(tail_i)];
    }

    __host__ __device__ void push_front(T val) {
        head_i = decrement(head_i);
        buffer[head_i] = val;
        len++;
    }

    __host__ __device__ T pop_front() {
        T val = buffer[head_i];
        head_i = increment(head_i);
        len--;
        return val;
    }

    __host__ __device__ T& front() {
        return  buffer[head_i];
    }

    __host__ __device__ T& operator()(int offset) {
        if (head_i + offset >= buf_size)
            return buffer[head_i + offset - buf_size];
        else
            return buffer[head_i + offset];
    }

    __host__ __device__ T operator()(int offset) const {
        if (head_i + offset >= buf_size)
            return buffer[head_i + offset - buf_size];
        else
            return buffer[head_i + offset];
    }

    __host__ __device__ void clear() {
        head_i = 0;
        tail_i = 0;
        len = 0;
    }

    __host__ __device__ int increment(int i) const {
        return (i + 1 < buf_size) ? (i + 1) : (0);
    }


    __host__ __device__ int decrement(int i) const {
        return (i - 1 >= 0) ? (i - 1) : (buf_size - 1);
    }

    __host__ __device__ bool empty() const {
        return (head_i == tail_i) && len == 0;
    }

    __host__ __device__ bool full() const {
        return (head_i == tail_i) && len == buf_size;
    }

    __host__ __device__ int size() const {
        return len;
    }

    __host__ __device__ int max_size() const {
        return buf_size;
    }
};

template <typename T>
class Buffer2D {
public:
    T* buffer = nullptr;
    int* index = nullptr;
    int max_size_per_consumer = 0;
    int num_consumers = 0;
    __host__ __device__ Buffer2D() = default;
    __host__ __device__ Buffer2D(int num_consumers, int max_size_per_consumer){
        this->buffer = new T[num_consumers*max_size_per_consumer];
        this->index = new int[num_consumers];
        this->num_consumers = num_consumers;
        this->max_size_per_consumer = max_size_per_consumer;
        this->clearAll();
    }
    __host__ __device__ T& operator()(int consumer_idx, int data_idx) {
        return buffer[consumer_idx*max_size_per_consumer + data_idx];
    }
    __host__ __device__ T operator()(int consumer_idx, int data_idx) const {
        return buffer[consumer_idx*max_size_per_consumer + data_idx];
    }
    __host__ __device__ int size(int consumer_idx) const {
        return index[consumer_idx];
    }
    __host__ __device__ bool empty(int consumer_idx) const {
        return size(consumer_idx) == 0;
    }
    __host__ __device__ bool full(int consumer_idx) const {
        return size(consumer_idx) == max_size_per_consumer;
    }
    __host__ __device__ void push_back(int consumer_idx, const T& data){
        if (!full(consumer_idx)){
            buffer[consumer_idx*max_size_per_consumer + index[consumer_idx]] = data;
            index[consumer_idx]++;
        }
    }
    __host__ __device__ void clear(int consumer_idx){
        index[consumer_idx] = 0;
    }
    __host__ __device__ void clearAll(){
        for (int i = 0; i < num_consumers; i++){
            index[i] = 0;
        }
    }

    // Note that we do not have a destructor here. One needs to destruct the allocated memory before deleting this object!
    __host__ __device__ void freeBuffer(){
        delete [] buffer;
        delete [] index;
        buffer = nullptr;
        index = nullptr;
        max_size_per_consumer = 0;
        num_consumers = 0;
    }
};

namespace cuGMM {
    class Buffer2D {
    public:
        V*     S = nullptr;
        M*     J = nullptr;
        V*       LeftPoint = nullptr;
        V*       RightPoint = nullptr;
        int*     NumLines = nullptr;
        int*     N = nullptr;
        float*   W = nullptr;
        int*     LeftPixel = nullptr;
        int*     RightPixel = nullptr;
        float*   depth = nullptr; // Current z-distance from the camera
        bool*    Updated = nullptr; // Used during segment fusion to track if it is fused with another segment from the following scanline
        bool*    near_obs = nullptr; // Flag to track if the free cluster appears near an obstacle cluster
        bool*    cur_pt_obs = nullptr; // Flag to check if the current cluster is near an obstacle
        bool*    fused = nullptr; // Flag to check if the segment is fused wth other segments
        // Bounding Box Information
        V*   BBoxLow = nullptr;
        V*   BBoxHigh = nullptr;

        // Obstacle cluster
        V*       LineVec = nullptr;
        V*       PlaneVec = nullptr;
        V*       UpMean = nullptr;
        float*   PlaneLen = nullptr;
        float*   PlaneWidth = nullptr;
        // Index of the depth array used for storing depth bounds
        bool*    track_color = nullptr;

        // Free space metadata
        V*   S_ray_ends = nullptr;
        M*   J_ray_ends = nullptr;
        V*   S_basis = nullptr;
        M*   J_basis = nullptr;
        float*   W_ray_ends = nullptr;
        float*   W_basis = nullptr;
        int*     depth_idx = nullptr;
        // Bounding Box Information
        V*   freeBBoxLow = nullptr;
        V*   freeBBoxHigh = nullptr;
        int* index = nullptr;
        int max_size_per_consumer = 0;
        int num_consumers = 0;

        __host__ __device__ Buffer2D() = default;
        __host__ __device__ Buffer2D(int num_consumers, int max_size_per_consumer){
            S = new V[num_consumers*max_size_per_consumer];
            J = new M[num_consumers*max_size_per_consumer];
            LeftPoint = new V[num_consumers*max_size_per_consumer];
            RightPoint = new V[num_consumers*max_size_per_consumer];
            NumLines = new int[num_consumers*max_size_per_consumer];
            N = new int[num_consumers*max_size_per_consumer];
            W = new float[num_consumers*max_size_per_consumer];
            LeftPixel = new int[num_consumers*max_size_per_consumer];
            RightPixel = new int[num_consumers*max_size_per_consumer];
            depth = new float[num_consumers*max_size_per_consumer];
            Updated = new bool[num_consumers*max_size_per_consumer];
            near_obs = new bool[num_consumers*max_size_per_consumer];
            cur_pt_obs = new bool[num_consumers*max_size_per_consumer];
            fused = new bool[num_consumers*max_size_per_consumer];
            // Bounding Box Information
            BBoxLow = new V[num_consumers*max_size_per_consumer];
            BBoxHigh = new V[num_consumers*max_size_per_consumer];

            // Obstacle cluster
            LineVec = new V[num_consumers*max_size_per_consumer];
            PlaneVec = new V[num_consumers*max_size_per_consumer];
            UpMean = new V[num_consumers*max_size_per_consumer];
            PlaneLen = new float[num_consumers*max_size_per_consumer];
            PlaneWidth = new float[num_consumers*max_size_per_consumer];
            // Index of the depth array used for storing depth bounds
            track_color = new bool[num_consumers*max_size_per_consumer];

            // Free space metadata
            S_ray_ends = new V[num_consumers*max_size_per_consumer];
            J_ray_ends = new M[num_consumers*max_size_per_consumer];
            S_basis = new V[num_consumers*max_size_per_consumer];
            J_basis = new M[num_consumers*max_size_per_consumer];
            W_ray_ends = new float[num_consumers*max_size_per_consumer];
            W_basis = new float[num_consumers*max_size_per_consumer];
            depth_idx = new int[num_consumers*max_size_per_consumer];
            // Bounding Box Information
            freeBBoxLow = new V[num_consumers*max_size_per_consumer];
            freeBBoxHigh = new V[num_consumers*max_size_per_consumer];

            this->index = new int[num_consumers];
            this->num_consumers = num_consumers;
            this->max_size_per_consumer = max_size_per_consumer;
            this->clearAll();
        }
        __host__ __device__ int size(int consumer_idx) const {
            return index[consumer_idx];
        }
        __host__ __device__ bool empty(int consumer_idx) const {
            return size(consumer_idx) == 0;
        }
        __host__ __device__ bool full(int consumer_idx) const {
            return size(consumer_idx) == max_size_per_consumer;
        }
        __host__ __device__ int bufferIdx(int consumer_idx, int cluster_idx) const {
            return consumer_idx + cluster_idx * num_consumers;
        }
        __host__ __device__ void push_back(int consumer_idx, const GMMmetadata_c& cluster){
            if (!full(consumer_idx)){
                int idx = bufferIdx(consumer_idx, index[consumer_idx]);
                S[idx] = cluster.S;
                J[idx] = cluster.J;
                LeftPoint[idx] = cluster.LeftPoint;
                RightPoint[idx] = cluster.RightPoint;
                NumLines[idx] = cluster.NumLines;
                N[idx] = cluster.N;
                W[idx] = cluster.W;
                LeftPixel[idx] = cluster.LeftPixel;
                RightPixel[idx] = cluster.RightPixel;
                depth[idx] = cluster.depth;
                Updated[idx] = cluster.Updated;
                near_obs[idx] = cluster.near_obs;
                cur_pt_obs[idx] = cluster.cur_pt_obs;
                fused[idx] = cluster.fused;
                // Bounding Box Information
                BBoxLow[idx] = cluster.BBoxLow;
                BBoxHigh[idx] = cluster.BBoxHigh;

                // Obstacle cluster
                LineVec[idx] = cluster.LineVec;
                PlaneVec[idx] = cluster.PlaneVec;
                UpMean[idx] = cluster.UpMean;
                PlaneLen[idx] = cluster.PlaneLen;
                PlaneWidth[idx] = cluster.PlaneWidth;
                // Index of the depth array used for storing depth bounds
                track_color[idx] = cluster.track_color;

                // Free space metadata
                S_ray_ends[idx] = cluster.freeBasis.S_ray_ends;
                J_ray_ends[idx] = cluster.freeBasis.J_ray_ends;
                S_basis[idx] = cluster.freeBasis.S_basis;
                J_basis[idx] = cluster.freeBasis.J_basis;
                W_ray_ends[idx] = cluster.freeBasis.W_ray_ends;
                W_basis[idx] = cluster.freeBasis.W_basis;
                depth_idx[idx] = cluster.freeBasis.depth_idx;
                // Bounding Box Information
                freeBBoxLow[idx] = cluster.freeBasis.BBoxLow;
                freeBBoxHigh[idx] = cluster.freeBasis.BBoxHigh;

                index[consumer_idx]++;
            }
        }
        __host__ __device__ void clear(int consumer_idx){
            index[consumer_idx] = 0;
        }

        __host__ __device__ void clearAll(){
            for (int i = 0; i < num_consumers; i++){
                index[i] = 0;
            }
        }

        // Note that we do not have a destructor here. One needs to destruct the allocated memory before deleting this object!
        __host__ __device__ void freeBuffer(){
            delete [] S;
            delete [] J;
            delete [] LeftPoint;
            delete [] RightPoint;
            delete [] NumLines;
            delete [] N;
            delete [] W;
            delete [] LeftPixel;
            delete [] RightPixel;
            delete [] depth; // Current z-distance from the camera
            delete [] Updated; // Used during segment fusion to track if it is fused with another segment from the following scanline
            delete [] near_obs; // Flag to track if the free cluster appears near an obstacle cluster
            delete [] cur_pt_obs; // Flag to check if the current cluster is near an obstacle
            delete [] fused; // Flag to check if the segment is fused wth other segments
            // Bounding Box Information
            delete [] BBoxLow;
            delete [] BBoxHigh;

            // Obstacle cluster
            delete [] LineVec;
            delete [] PlaneVec;
            delete [] UpMean;
            delete [] PlaneLen;
            delete [] PlaneWidth;
            // Index of the depth array used for storing depth bounds
            delete [] track_color;

            // Free space metadata
            delete [] S_ray_ends;
            delete [] J_ray_ends;
            delete [] S_basis;
            delete [] J_basis;
            delete [] W_ray_ends;
            delete [] W_basis;
            delete [] depth_idx;
            // Bounding Box Information
            delete [] freeBBoxLow;
            delete [] freeBBoxHigh;
            delete [] index;

            S = nullptr;
            J = nullptr;
            LeftPoint = nullptr;
            RightPoint = nullptr;
            NumLines = nullptr;
            N = nullptr;
            W = nullptr;
            LeftPixel = nullptr;
            RightPixel = nullptr;
            depth = nullptr; // Current z-distance from the camera
            Updated = nullptr; // Used during segment fusion to track if it is fused with another segment from the following scanline
            near_obs = nullptr; // Flag to track if the free cluster appears near an obstacle cluster
            cur_pt_obs = nullptr; // Flag to check if the current cluster is near an obstacle
            fused = nullptr; // Flag to check if the segment is fused wth other segments
            // Bounding Box Information
            BBoxLow = nullptr;
            BBoxHigh = nullptr;

            // Obstacle cluster
            LineVec = nullptr;
            PlaneVec = nullptr;
            UpMean = nullptr;
            PlaneLen = nullptr;
            PlaneWidth = nullptr;
            // Index of the depth array used for storing depth bounds
            track_color = nullptr;

            // Free space metadata
            S_ray_ends = nullptr;
            J_ray_ends = nullptr;
            S_basis = nullptr;
            J_basis = nullptr;
            W_ray_ends = nullptr;
            W_basis = nullptr;
            depth_idx = nullptr;
            // Bounding Box Information
            freeBBoxLow = nullptr;
            freeBBoxHigh = nullptr;
            index = nullptr;

            max_size_per_consumer = 0;
            num_consumers = 0;
        }
    };
}

// This is a wrapper!
class cudaStreamBuffer {
private:
    cudaStream_t* buffer;
    int num_streams;
public:
    cudaStreamBuffer(int num_streams) {
        this->buffer = new cudaStream_t[num_streams];
        this->num_streams = num_streams;
        for (int i = 0; i < num_streams; i++){
            gpuErrchk(cudaStreamCreate(&buffer[i]));
        }
    }
    cudaStream_t& at(int index) {
        return buffer[index];
    }
    cudaStream_t at(int index) const {
        return buffer[index];
    }
    void synchronizeStream(int index) {
        printf("Synchronizing CUDA stream %d\n", index);
        gpuErrchk(cudaStreamSynchronize(buffer[index]));
    }
    void destroyAllStreams(){
        for (int i = 0; i < num_streams; i++){
            gpuErrchk(cudaStreamDestroy(buffer[i]));
        }
        delete [] buffer;
        buffer = nullptr;
    }
    void destroyStream(int index) {
        gpuErrchk(cudaStreamDestroy(buffer[index]));
    }
};
#endif //RING_BUFFER_CUH
