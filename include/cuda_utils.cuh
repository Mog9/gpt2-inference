#pragma once

#include <cuda_runtime.h>
#include <cfloat>

__device__ inline float warp_reduce_sum(float val) {
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}