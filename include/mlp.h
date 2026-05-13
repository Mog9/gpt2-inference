#pragma once

void launch_mlp(

    float* input,

    float* up_weight,
    float* up_bias,

    float* down_weight,
    float* down_bias,

    float* up_proj,
    float* gelu_out,
    float* down_proj,

    int seq_len,
    int hidden_dim
);