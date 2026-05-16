#pragma once

struct QKVHeadView{

    float* q;

    float* k_cache;

    float* v_cache;

    int seq_len;

    int hidden_dim;

    int num_heads;

    int head_dim;

    int head_idx;
};

QKVHeadView create_qkv_head_view(
    float* q,
    float* k_cache,
    float* v_cache,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_idx
);