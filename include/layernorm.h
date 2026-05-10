#pragma once

void launch_layernorm(
    float* d_input,
    float* d_output,
    float* d_gamma,
    float* d_beta,
    int rows,
    int cols,
    float eps
);