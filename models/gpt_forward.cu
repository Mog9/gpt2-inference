#include "../include/gpt2_forward.h"
#include "../include/embedding.h"
#include "../include/transformer.h"
#include "../include/layernorm.h"
#include "../include/linear.h"

void gpt2_forward(
    int* token_ids,
    GPT2Weights& weights,
    GPT2Buffers& buffers,
    int seq_len,
    const GPT2Config& config
){

    launch_embedding_lookup(
        token_ids,
        weights.token_embedding_table,
        weights.positional_embedding_table,
        buffers.hidden_states_1,
        seq_len,
        config.hidden_dim
    );

    for(int layer=0;layer<config.num_layers;layer++){

        TransformerBlockWeights& block=
            weights.blocks[layer];

        launch_transformer_block(
            buffers.hidden_states_1,
            buffers.hidden_states_2,

            block.ln1_gamma,
            block.ln1_beta,
            buffers.ln1_out,

            block.qkv_weight,
            block.qkv_bias,
            buffers.qkv,

            buffers.scores,
            buffers.heads,
            buffers.merged,

            block.attn_proj_weight,
            block.attn_proj_bias,
            buffers.attn_proj,
            buffers.attn_residual,

            block.ln2_gamma,
            block.ln2_beta,
            buffers.ln2_out,

            block.mlp_up_weight,
            block.mlp_up_bias,

            block.mlp_down_weight,
            block.mlp_down_bias,

            buffers.mlp_up,
            buffers.mlp_gelu,
            buffers.mlp_down,

            buffers.kv_cache[layer].k_cache,
            buffers.kv_cache[layer].v_cache,

            seq_len -1,
            seq_len,
            config.hidden_dim,
            config.num_heads
        );

        float* temp=
            buffers.hidden_states_1;

        buffers.hidden_states_1=
            buffers.hidden_states_2;

        buffers.hidden_states_2=
            temp;
    }

    launch_layernorm(
        buffers.hidden_states_1,
        buffers.hidden_states_2,
        weights.final_ln_gamma,
        weights.final_ln_beta,
        seq_len,
        config.hidden_dim,
        1e-5f
    );

    launch_linear(
        buffers.hidden_states_2,
        weights.lm_head_weight,
        nullptr,
        buffers.logits,
        seq_len,
        config.hidden_dim,
        config.vocab_size
    );
}