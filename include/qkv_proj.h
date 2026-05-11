#pragma once

void launch_qkv_projection(
    float* input,
    float* weight,
    float* bias,
    float* qkv,
    int seq_len,
    int hidden_dim
);