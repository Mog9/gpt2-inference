#pragma once
#include <cuda_runtime.h>

void launch_matmul(
    int I,
    int J,
    int K,
    float* M,
    float* N,
    float* P
);