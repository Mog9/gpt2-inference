#include "../include/transformer.h"
#include "../include/layernorm.h"
#include "../include/qkv_proj.h"
#include "../include/qkv_view.h"
#include "../include/attention_score.h"
#include "../include/scale_mask.h"
#include "../include/softmax.h"
#include "../include/attention_output.h"
#include "../include/merge_heads.h"
#include "../include/output_projection.h"
#include "../include/residual_add.h"
#include "../include/mlp.h"

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

    int seq_len,
    int hidden_dim,
    int num_heads
) {

    int head_dim = hidden_dim / num_heads;

    launch_layernorm(input,ln1_out,ln1_gamma,ln1_beta, seq_len, hidden_dim,1e-5f);

    launch_qkv_projection(ln1_out, qkv_weight, qkv_bias, qkv, seq_len, hidden_dim);

    for(int h = 0; h < num_heads; h++) {

        QKVHeadView view = create_qkv_head_view(qkv, seq_len, hidden_dim, num_heads, h);

        float* head_out = heads + h * seq_len * head_dim;

        launch_attention_scores(view, scores);

        launch_scale_mask(scores, seq_len, head_dim);

        launch_softmax(scores, scores, seq_len, seq_len);

        launch_attention_output(scores, view, head_out);
    }

    launch_merge_heads(heads, merged, seq_len, num_heads, head_dim);

    launch_output_projection(merged, attn_proj_weight, attn_proj_bias, attn_proj, seq_len, hidden_dim);

    launch_residual_add(input, attn_proj, attn_residual, seq_len, hidden_dim);

    launch_layernorm(attn_residual, ln2_out,ln2_gamma,ln2_beta,seq_len,hidden_dim, 1e-5f);

    launch_mlp(
        ln2_out,
        mlp_up_weight,
        mlp_up_bias,
        mlp_down_weight,
        mlp_down_bias,
        mlp_up,
        mlp_gelu,
        mlp_down,
        seq_len,
        hidden_dim
    );

    launch_residual_add(
        attn_residual,
        mlp_down,
        output,
        seq_len,
        hidden_dim
    );
}