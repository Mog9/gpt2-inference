#include <cuda_runtime.h>
#include "../include/embedding.h"
#define THREADS 256

__global__ void embedding_lookup_kernel(
    int* token_ids,

    float* token_embedding_table,
    float* positional_embedding_table,

    float* output,

    int seq_len,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * hidden_dim;
    if(idx >= total) return;

    int token_pos = idx / hidden_dim;
    int dim = idx % hidden_dim;

    int token_id= token_ids[token_pos];

    float tok = token_embedding_table[token_id * hidden_dim + dim];
    float pos =positional_embedding_table[token_pos * hidden_dim + dim];

    output[idx] = tok + pos;
}

void launch_embedding_lookup(
    int* token_ids,

    float* token_embedding_table,
    float* positional_embedding_table,

    float* output,

    int seq_len,
    int hidden_dim
) {
    int total = seq_len * hidden_dim;

    int blocks =(total + THREADS - 1) / THREADS;

    embedding_lookup_kernel<<<blocks, THREADS>>>(
        token_ids,
        token_embedding_table,
        positional_embedding_table,
        output,
        seq_len,
        hidden_dim
    );
}