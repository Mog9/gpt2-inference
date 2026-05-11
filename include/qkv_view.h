#pragma once

struct QKVHeadView {
    float* q;
    float* k;
    float* v;

    int seq_len;
    int hidden_dim;
    int num_heads;
    int head_dim;

    int row_stride;
    int head_stride;
};

QKVHeadView create_qkv_head_view(
    float* qkv,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_idx
);