#pragma once
#include <vector>

struct TransformerBlockWeights {
    float* ln1_gamma;
    float* ln1_beta;

    float* qkv_weight;
    float* qkv_bias;

    float* attn_proj_weight;
    float* attn_proj_bias;

    float* ln2_gamma;
    float* ln2_beta;

    float* mlp_up_weight;
    float* mlp_up_bias;

    float* mlp_down_weight;
    float* mlp_down_bias;
};

struct GPT2Weights {
    float* token_embedding_table;
    float* positional_embedding_table;

    std::vector<TransformerBlockWeights> blocks;

    float* final_ln_gamma;
    float* final_ln_beta;

    float* lm_head_weight;
};