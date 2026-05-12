#include <cuda_runtime.h>
#include <cmath>
#include "../include/gelu.h"
#define THREADS 256

__global__ void gelu_kernel(float* input, float* output,int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= total) return;

    float x = input[idx];
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);

    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

void launch_gelu(
    float* input,
    float* output,
    int rows,
    int cols
) {
    int total = rows * cols;

    int blocks = (total + THREADS - 1) / THREADS;

    gelu_kernel<<<blocks, THREADS>>>(input, output, total);
}