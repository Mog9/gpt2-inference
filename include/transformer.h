#pragma once

void launch_transformer_block(

    //input and output
    float* input,
    float* output,

    //layernorm 1
    float* ln1_gamma,
    float* ln1_beta,
    float* ln1_out,

    // attention qkv
    float* qkv_weight,
    float* qkv_bias,
    float* qkv,

    //attention temp buffers
    float* scores,
    float* heads,
    float* merged,

    //attention output projection
    float* attn_proj_weight,
    float* attn_proj_bias,
    float* attn_proj,

    //residual after attention
    float* attn_residual,

    // layernorm 2
    float* ln2_gamma,
    float* ln2_beta,
    float* ln2_out,

    //mlp
    float* mlp_up_weight,
    float* mlp_up_bias,

    float* mlp_down_weight,
    float* mlp_down_bias,

    float* mlp_up,
    float* mlp_gelu,
    float* mlp_down,

    //config
    int seq_len,
    int hidden_dim,
    int num_heads
);