#include <cuda_runtime.h>
#include <math.h>
#include "../include/attention_score.h"
#define TILE 16

__global__ void attention_scores_kernel(QKVHeadView view, float* scores) {
    int q_token = blockIdx.y * blockDim.y + threadIdx.y;

    int k_token = blockIdx.x * blockDim.x + threadIdx.x;

    if(q_token >= view.seq_len ||k_token >= view.seq_len)
        return;

    float score = 0.0f;
    for(int d = 0; d < view.head_dim; d++) {

        int q_idx = q_token * (3 * view.hidden_dim)
            + (view.head_idx * view.head_dim)
            + d;

        int k_idx = k_token * (3 * view.hidden_dim)
            + view.hidden_dim
            + (view.head_idx * view.head_dim)
            + d;

        float q = view.qkv[q_idx];

        float k =view.qkv[k_idx];

        score += q * k;
    }

    scores[q_token * view.seq_len + k_token] = score;
}

void launch_attention_scores(
    QKVHeadView view,
    float* scores
) {

    dim3 block(TILE, TILE);

    dim3 grid(
        (view.seq_len + TILE - 1) / TILE,
        (view.seq_len + TILE - 1) / TILE
    );

    attention_scores_kernel<<<grid, block>>>(view,scores);
}