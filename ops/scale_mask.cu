#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>
#include "../include/scale_mask.h"

struct MD{
    float m;
    float d;
};

__device__ MD md_combine(MD a,MD b){
    MD out;
    out.m = fmaxf(a.m,b.m);
    out.d = a.d*expf(a.m-out.m)+b.d*expf(b.m-out.m);
    return out;
}

__device__ MD warp_reduce_md(MD val){

    for(int offset=warpSize/2;offset>0;offset/=2){

        MD other;

        other.m=__shfl_xor_sync(0xffffffff,val.m,offset);
        other.d=__shfl_xor_sync(0xffffffff,val.d,offset);

        val=md_combine(val,other);
    }

    return val;
}

__global__ void fused_scale_mask_softmax_kernel(
    float* scores,
    int seq_len,
    float scale
){

    int row=blockIdx.x;

    int lane=threadIdx.x%warpSize;
    int warp_id=threadIdx.x/warpSize;

    __shared__ MD warp_md[32];
    __shared__ MD final_md;

    MD local;

    local.m=-FLT_MAX;
    local.d=0.0f;

    for(int col=threadIdx.x;col<seq_len;col+=blockDim.x){

        float x;

        if(col>row) x=-1e9f;
        else x=scores[row*seq_len+col]*scale;

        float new_m=fmaxf(local.m,x);

        local.d=local.d*expf(local.m-new_m)+expf(x-new_m);

        local.m=new_m;
    }

    local=warp_reduce_md(local);

    if(lane==0) warp_md[warp_id]=local;

    __syncthreads();

    if(warp_id==0){

        MD block_val;

        int warp_count=(blockDim.x+warpSize-1)/warpSize;

        if(lane<warp_count) block_val=warp_md[lane];
        else{
            block_val.m=-FLT_MAX;
            block_val.d=0.0f;
        }

        block_val=warp_reduce_md(block_val);

        if(lane==0) final_md=block_val;
    }

    __syncthreads();

    for(int col=threadIdx.x;col<seq_len;col+=blockDim.x){

        float x;

        if(col>row) x=-1e9f;
        else x=scores[row*seq_len+col]*scale;

        scores[row*seq_len+col]=expf(x-final_md.m)/final_md.d;
    }
}

void launch_scale_mask(
    float* scores,
    int seq_len,
    int head_dim
){

    float scale=1.0f/sqrtf((float)head_dim);

    fused_scale_mask_softmax_kernel<<<seq_len,256>>>(
        scores,
        seq_len,
        scale
    );
}