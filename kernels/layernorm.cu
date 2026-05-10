/*
Both 3-pass (naive) and 2-pass Welford LayerNorm implementations
were tested and benchmarked here.

Results on GPT-2 hidden sizes (768–1600) showed the 3-pass version
consistently outperforming the 2-pass Welford version on RTX 3050,
mainly due to lower arithmetic overhead, lower register pressure,
and better occupancy characteristics.

The 2-pass Welford implementation began outperforming the 3-pass
version only at much larger hidden dimensions (e.g. 8192+), where
reduction pass savings outweighed the additional compute cost.

Since this project currently targets GPT-2 inference workloads,
the 3-pass LayerNorm implementation is used as the primary kernel.
*/


#include <cuda_runtime.h>
#include <iostream>
#include "../include/cuda_utils.cuh"

__global__ void layernorm(float* input, float* output, float* gamma, float* beta, int rows, int cols, float eps){
    int row = blockIdx.x;
    
    __shared__ float mean;
    __shared__ float row_inv_std;
    __shared__ float warp_sum[32];
    
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    float local_sum = 0.0f;
    for(int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_sum += input[row * cols + col];
    }
    local_sum = warp_reduce_sum(local_sum);
    
    if(lane == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    if(warp_id == 0) {
        local_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_sum[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        
        if(lane == 0) {
            mean = local_sum / cols;
        }
    }
    __syncthreads();
    
    float local_sum_sq = 0.0f;
    for(int col = threadIdx.x; col < cols; col += blockDim.x) {
        float x = input[row * cols + col];
        float diff = x - mean;
        local_sum_sq += diff * diff;
    }
    local_sum_sq = warp_reduce_sum(local_sum_sq);

    if(lane == 0) {
        warp_sum[warp_id] = local_sum_sq;
    }

    __syncthreads();

    if(warp_id == 0) {
        local_sum_sq = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_sum[lane] : 0.0f;
        local_sum_sq = warp_reduce_sum(local_sum_sq);

        if(lane == 0) {
            row_inv_std = rsqrtf(local_sum_sq / cols + eps);
        }
    }

    __syncthreads();

    for(int col = threadIdx.x; col < cols; col += blockDim.x) {
        float x = input[row * cols + col];
        float normalized = (x - mean) * row_inv_std;
        output[row * cols + col] = normalized * gamma[col] + beta[col];
    }
}


void launch_layernorm(
    float* d_input,
    float* d_output,
    float* d_gamma,
    float* d_beta,
    int rows,
    int cols,
    float eps
) {
    int threads = 256;

    layernorm<<<rows, threads>>>(
        d_input,
        d_output,
        d_gamma,
        d_beta,
        rows,
        cols,
        eps
    );
}



//! 3 pass
// struct layer {
//     float mean;
//     float m2;
//     int count;
// };

// __device__ layer l_combine(layer a, layer b) {
//     if(a.count == 0) return b;
//     if(b.count == 0) return a;
//     layer out;

//     float delta = b.mean - a.mean;
//     out.count = a.count + b.count;
//     out.mean = a.mean + delta * ((float)b.count / out.count);
//     out.m2 = a.m2 + b.m2 + delta * delta * ((float)(a.count * b.count) / out.count);

//     return out;
// }

// __device__ layer warp_reduce_l(layer val) {
//     for(int offset = warpSize / 2; offset > 0; offset /= 2) {
//         layer other;

//         other.mean = __shfl_xor_sync(0xffffffff,val.mean, offset);
//         other.m2 = __shfl_xor_sync(0xffffffff, val.m2,offset);
//         other.count = __shfl_xor_sync(0xffffffff, val.count,offset);
//         val = l_combine(val, other);
//     }
//     return val;
// }

// __global__ void layernorm(float* input, float* output, float* gamma, float* beta, int rows, int cols, float eps){
//     int row = blockIdx.x;
    
//     __shared__ float mean;
//     __shared__ float row_inv_std;
    
//     __shared__ layer warp_layer[32];

//     layer local;
//     local.mean = 0.0f;
//     local.m2 = 0.0f;
//     local.count = 0;

//     int lane = threadIdx.x % warpSize;
//     int warp_id = threadIdx.x / warpSize;
    
//     //mean and variance fuse
//     for(int col = threadIdx.x; col < cols; col += blockDim.x) {
//         float x = input[row * cols + col];
//         local.count++;
//         float delta = x - local.mean;
//         local.mean += delta / (float)local.count;
//         float delta2 = x - local.mean;
//         local.m2 += delta * delta2;  
//     }
//     local = warp_reduce_l(local);

//     if(lane == 0) {
//         warp_layer[warp_id] = local;
//     }
//     __syncthreads();

//     if(warp_id == 0) {
//         layer block_val;

//         int warp_count = (blockDim.x + warpSize - 1) / warpSize;
//         if(lane < warp_count) {
//             block_val = warp_layer[lane];
//         }
//         else {
//             block_val.mean = 0.0f;
//             block_val.m2 = 0.0f;
//             block_val.count = 0;
//         }
//         block_val = warp_reduce_l(block_val);

//         if(lane == 0) {
//             mean = block_val.mean;
//             row_inv_std = rsqrtf(block_val.m2 / block_val.count + eps);
//         }
//     }
//     __syncthreads();

//     //normalize
//     for(int col = threadIdx.x; col < cols; col += blockDim.x) {
//     float x = input[row * cols + col];
//     float normalized = (x - mean) * row_inv_std;
//     output[row * cols + col] = normalized * gamma[col] + beta[col];
//     }
// }

