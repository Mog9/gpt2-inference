#include <cuda_runtime.h>
#include <cfloat>

#include "reduction.h"

#define BLOCK_SIZE 1024

__device__ float warp_reduce_sum(float val) {
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void reduce_sum(float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int i = threadIdx.x;
    float local_sum = 0.0f;
    for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < N; j += gridDim.x * blockDim.x) {
        local_sum += input[j];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 16; s >>= 1) {
        if(i < s) {
            sdata[i] += sdata[i + s];
        }
        __syncthreads();
    }
    if(i < 32){
        float val = sdata[i];
        val = warp_reduce_sum(val);
        if(i==0) output[blockIdx.x] = val;
    }
}

__global__ void reduce_max(float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int i = threadIdx.x;
    float local_max = -FLT_MAX;
    for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < N; j += gridDim.x * blockDim.x) {
        local_max = fmaxf(local_max, input[j]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 16; s >>= 1) {
        if(i < s) {
            sdata[i] = fmaxf(sdata[i], sdata[i + s]);
        }
        __syncthreads();
    }
    if(i < 32){
        float val = sdata[i];
        val = warp_reduce_max(val);
        if(i==0) output[blockIdx.x] = val;
    }
}

void launch_sum_reduction(
    float* d_input,
    float* d_output,
    int N,
    int blocks
) {
    reduce_sum<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input,
        d_output,
        N
    );
}

void launch_max_reduction(
    float* d_input,
    float* d_output,
    int N,
    int blocks
) {
    reduce_max<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input,
        d_output,
        N
    );
}