#pragma once

void launch_sum_reduction(
    float* d_input,
    float* d_output,
    int N,
    int blocks
);

void launch_max_reduction(
    float* d_input,
    float* d_output,
    int N,
    int blocks
);
