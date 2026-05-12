#pragma once

void launch_output_projection(
    float* input,
    float* weight,
    float* bias,
    float* output,
    int seq_len,
    int hidden_dim
);