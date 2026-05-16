#pragma once

struct GPT2Buffers {
    float* hidden_states_1;
    float* hidden_states_2;
    float* ln1_out;
    float* qkv;
    float* scores;
    float* heads;
    float* merged;
    float* attn_proj;
    float* attn_residual;
    float* ln2_out;
    float* mlp_up;
    float* mlp_gelu;
    float* mlp_down;
    float* logits;
};