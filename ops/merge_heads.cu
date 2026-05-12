#include <cuda_runtime.h>
#include "../include/merge_heads.h"
#define THREADS 256

/*

layout: [num_heads, seq_len, head_dim]

merged_output layout: [seq_len, hidden_dim]

where:
hidden_dim = num_heads * head_dim

for gpt2 if we got [12,1024,64], this means 12 seperate attention heads
after merge(concat)
12 * 64 which is 768, we get:

[1024, 768]

*/

__global__ void merge_heads_kernel(float* head_outputs, float* merged_output, int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int hidden_dim = num_heads * head_dim;
    int total = seq_len * hidden_dim;

    if(idx >= total) return;

    int token = idx / hidden_dim;
    int hidden_idx = idx % hidden_dim;

    int head = hidden_idx / head_dim;
    int d = hidden_idx % head_dim;

    int src = head * seq_len * head_dim + token * head_dim + d;

    merged_output[idx] = head_outputs[src];
}

void launch_merge_heads(float* head_outputs, float* merged_output, int seq_len, int num_heads, int head_dim) {
    int hidden_dim = num_heads * head_dim;
    int total = seq_len * hidden_dim;

    int blocks = (total + THREADS - 1) / THREADS;

    merge_heads_kernel<<<blocks, THREADS>>>(head_outputs, merged_output, seq_len, num_heads, head_dim);
}