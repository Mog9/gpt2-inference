#pragma once

void launch_embedding_lookup(
    int* token_ids,

    float* token_embedding_table,
    float* positional_embedding_table,

    float* output,

    int seq_len,
    int hidden_dim
);