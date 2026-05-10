#include <cuda_runtime.h>
#include "../include/linear.h"
#include "../include/matmul.h"


__global__ void bias_add_kernel(float* output, float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = M * N;

    if(idx < total) {
        int col = idx % N;
        output[idx] += bias[col];
    }
}

void launch_linear(
    float* input,
    float* weight,
    float* bias,
    float* output,
    int M,
    int K,
    int N
) {
    launch_matmul(M,K,N,input,weight, output);

    if(bias != nullptr) {
        int total = M * N;

        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        bias_add_kernel<<<blocks, threads>>>(output, bias, M,N);
    }
}
