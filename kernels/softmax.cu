#include <cuda_runtime.h>
#include <cfloat>
#include "../include/softmax.h"

struct MD {
    float m;
    float d;
};

__device__ MD md_combine(MD a, MD b) {
    MD out;

    out.m = fmaxf(a.m, b.m);
    out.d = a.d * expf(a.m - out.m) + b.d * expf(b.m - out.m);
    // this rescales both the partial dinominators into the SAME normalization frame

    return out;
}

//combined warp_reduce_max + warp_reduce_sum
__device__ MD warp_reduce_md(MD val) {
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        MD other;

        other.m = __shfl_xor_sync(0xffffffff,val.m, offset);
        other.d = __shfl_xor_sync(0xffffffff, val.d,offset);
        val = md_combine(val, other);
    }
    return val;
}

__global__ void softmax(
    float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.x;

    int lane = threadIdx.x % warpSize; //thread pos inside warp
    int warp_id = threadIdx.x / warpSize; //tells me which warp the thread belongs to

    __shared__ MD warp_md[32]; //32 warps per block and each stores (m,d) pair per warp
    __shared__ MD final_md;

    MD local;

    local.m = -FLT_MAX;
    local.d = 0.0f;

    for(int col = threadIdx.x; col < cols; col += blockDim.x){
        float x = input[row * cols + col];
        float new_m =fmaxf(local.m, x);

        local.d =local.d * expf(local.m - new_m) + expf(x - new_m);
        local.m = new_m;
    }

    /*
    this combines all threads in the warp
    into ONE warp level (m,d) result bascially, 
    
    before this local = warp_reduce_md(local);
    - each thread only knows its own partial state, but after this, 
    - every thread in the warp has combined warp state
    */
    local = warp_reduce_md(local);

    if(lane == 0) {
        warp_md[warp_id] = local; //only one thread per warp writies the result
    }
    __syncthreads();

    if(warp_id == 0) { //only first warp
        MD block_val;

        int warp_count = (blockDim.x + warpSize - 1) / warpSize;

        if(lane < warp_count) { //active lanes
            block_val = warp_md[lane];
        }
        else {
            block_val.m = -FLT_MAX;
            block_val.d = 0.0f;
        }

        block_val = warp_reduce_md(block_val);

        if(lane == 0) {
            final_md = block_val;
        }
    }
    __syncthreads();

    for(int col = threadIdx.x; col < cols; col += blockDim.x) {
        float x = input[row * cols + col];

        output[row * cols + col] = expf(x - final_md.m) / final_md.d;
        //the final normalization, where m and d are final max and denominator for the row
    }
}

void launch_softmax(
    float* d_input,
    float* d_output,
    int rows,
    int cols
) {
    int threads = 256;
    softmax<<<rows, threads>>>(d_input,d_output,rows, cols);
}