#include <cuda_runtime.h>
#include "../include/residual_add.h"

#define THREADS 256

__global__ void residual_add_kernel(
    float* x,
    float* residual,
    float* output,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total) output[idx] = x[idx] + residual[idx];
}

void launch_residual_add(
    float* x,
    float* residual,
    float* output,
    int rows,
    int cols
) {
    int total = rows * cols;
    int blocks = (total + THREADS - 1) / THREADS;
    residual_add_kernel<<<blocks, THREADS>>>(x, residual, output, total);
}