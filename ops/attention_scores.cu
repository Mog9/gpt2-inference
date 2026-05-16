#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>
#include "../include/attention_score.h"

__global__ void fused_attention_scores_softmax_kernel(
    QKVHeadView view,
    float* scores
){

    int row=blockIdx.x;

    int tid=threadIdx.x;

    if(row>=view.seq_len)
        return;

    extern __shared__ float shared[];

    float* logits=shared;

    float local_max=-FLT_MAX;

    for(int col=tid;col<view.seq_len;col+=blockDim.x){

        float score=0.0f;

        for(int d=0;d<view.head_dim;d++){

            int q_idx=
                row*view.hidden_dim+
                (view.head_idx*view.head_dim)+
                d;

            int k_idx=
                col*view.hidden_dim+
                (view.head_idx*view.head_dim)+
                d;

            float q=view.q[q_idx];

            float k=view.k_cache[k_idx];

            score+=q*k;
        }

        score/=sqrtf((float)view.head_dim);

        if(col>row)
            score=-1e9f;

        logits[col]=score;

        local_max=fmaxf(local_max,score);
    }

    __syncthreads();

    for(int stride=blockDim.x/2;stride>0;stride/=2){
        local_max=fmaxf(
            local_max,
            __shfl_down_sync(0xffffffff,local_max,stride)
        );
    }

    __shared__ float max_val;

    if(tid==0)
        max_val=local_max;

    __syncthreads();

    float local_sum=0.0f;

    for(int col=tid;col<view.seq_len;col+=blockDim.x){

        float val=expf(logits[col]-max_val);

        logits[col]=val;

        local_sum+=val;
    }

    for(int stride=blockDim.x/2;stride>0;stride/=2){
        local_sum+=__shfl_down_sync(
            0xffffffff,
            local_sum,
            stride
        );
    }

    __shared__ float sum_val;

    if(tid==0)
        sum_val=local_sum;

    __syncthreads();

    for(int col=tid;col<view.seq_len;col+=blockDim.x){

        scores[row*view.seq_len+col]=
            logits[col]/sum_val;
    }
}

void launch_attention_scores(
    QKVHeadView view,
    float* scores
){

    int threads=256;

    fused_attention_scores_softmax_kernel<<<
        view.seq_len,
        threads,
        view.seq_len*sizeof(float)
    >>>(
        view,
        scores
    );
}