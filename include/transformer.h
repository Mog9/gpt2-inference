#pragma once

void launch_transformer_block(

    float* input,
    float* output,

    float* ln1_gamma,
    float* ln1_beta,
    float* ln1_out,

    float* qkv_weight,
    float* qkv_bias,
    float* qkv,

    float* scores,
    float* heads,
    float* merged,

    float* attn_proj_weight,
    float* attn_proj_bias,
    float* attn_proj,

    float* attn_residual,

    float* ln2_gamma,
    float* ln2_beta,
    float* ln2_out,

    float* mlp_up_weight,
    float* mlp_up_bias,

    float* mlp_down_weight,
    float* mlp_down_bias,

    float* mlp_up,
    float* mlp_gelu,
    float* mlp_down,

    float* k_cache,
    float* v_cache,

    int cache_pos,
    int seq_len,
    int hidden_dim,
    int num_heads
);