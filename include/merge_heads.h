#pragma once

void launch_merge_heads(
    float* head_outputs,
    float* merged_output,
    int seq_len,
    int num_heads,
    int head_dim
);