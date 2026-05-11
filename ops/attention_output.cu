#include <cuda_runtime.h>
#include "../include/attention_output.h"
#define TILE 16

/*

attention_output =
    attention_weights @ V

attention_weights:
    [seq_len, seq_len]

V:
    [seq_len, head_dim]

output:
    [seq_len, head_dim]

*/

__global__ void attention_output_kernel(float* attention_weights, QKVHeadView view, float* output) {
    int token = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if(token >= view.seq_len || d >= view.head_dim) {return;}

    float sum = 0.0f;
    for(int k = 0; k < view.seq_len; k++) {
        float weight = attention_weights[token * view.seq_len + k];
        float v = view.v[k * view.row_stride + d];

        sum += weight * v;
    }

    output[token * view.head_dim + d] = sum;
}

void launch_attention_output(
    float* attention_weights,
    QKVHeadView view,
    float* output
) {

    dim3 block(TILE, TILE);
    dim3 grid((view.head_dim + TILE - 1) / TILE, (view.seq_len + TILE - 1) / TILE);

    attention_output_kernel<<<grid, block>>>(attention_weights, view, output);
}