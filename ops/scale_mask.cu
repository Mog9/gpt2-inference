#include <cuda_runtime.h>
#include "../include/scale_mask.h"
#define TILE 16

__global__ void scale_mask_kernel(float* scores, int seq_len, float scale) {
    int q_token = blockIdx.y * blockDim.y + threadIdx.y;
    int k_token = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(q_token >= seq_len || k_token >= seq_len) {return;}

    int idx = q_token * seq_len + k_token;

    //mask
    if(k_token > q_token) {
        scores[idx] = -1e9f;
        return;
    }

    scores[idx] *= scale; //scaling
}

void launch_scale_mask(
    float* scores,
    int seq_len,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    dim3 block(TILE, TILE);
    dim3 grid((seq_len + TILE - 1) / TILE, (seq_len + TILE - 1) / TILE);

    scale_mask_kernel<<<grid, block>>>(scores, seq_len, scale);
}