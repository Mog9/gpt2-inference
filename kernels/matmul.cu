#include <cuda_runtime.h>
#include <iostream>
#include "../include/matmul.h"
#define TILE 16

__global__ void matmul_kernel(
    int I,
    int J,
    int K,
    float* M,
    float* N,
    float* P
) {
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float PValue = 0.0f;

    __shared__ float tileM[TILE][TILE];
    __shared__ float tileN[TILE][TILE];

    for(int phase = 0; phase < (J + TILE - 1) / TILE; phase++) {
        if(rows < I && phase * TILE + tx < J)
            tileM[ty][tx] = M[rows * J + phase * TILE + tx];
        else
            tileM[ty][tx] = 0.0f;
        if(col < K && phase * TILE + ty < J)
            tileN[ty][tx] = N[(phase * TILE + ty) * K + col];
        else
            tileN[ty][tx] = 0.0f;
        __syncthreads();

        #pragma unroll
        for(int d = 0; d < TILE; d++) {
            PValue += tileM[ty][d] * tileN[d][tx];
        }
        __syncthreads();
    }

    if(rows < I && col < K) {
        P[rows * K + col] = PValue;
    }
}

void launch_matmul(
    int I,
    int J,
    int K,
    float* M,
    float* N,
    float* P
) {
    dim3 block(TILE, TILE);
    dim3 grid(
        (K + TILE - 1) / TILE,
        (I + TILE - 1) / TILE
    );

    matmul_kernel<<<grid, block>>>(I, J, K, M, N, P);

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "\n";
    }
}