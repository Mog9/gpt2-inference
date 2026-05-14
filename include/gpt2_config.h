#pragma once

struct GPT2Config {
    int vocab_size=50257;
    int max_seq_len = 1024;
    int hidden_dim = 768;
    int num_heads = 12;
    int num_layers = 12;
    int mlp_dim = 4 * hidden_dim;
};