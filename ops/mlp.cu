#include "../include/mlp.h"
#include "../include/linear.h"
#include "../include/gelu.h"
#include "../include/residual_add.h"

void launch_mlp(
    float* input,

    float* up_weight,
    float* up_bias,

    float* down_weight,
    float* down_bias,

    float* up_proj,
    float* gelu_out,
    float* down_proj,
    float* output,

    int seq_len,
    int hidden_dim
) {

    int mlp_dim = hidden_dim * 4;

    //up projection
    launch_linear(
        input,
        up_weight,
        up_bias,
        up_proj,
        seq_len,
        hidden_dim,
        mlp_dim
    );

    //gelu
    launch_gelu(
        up_proj,
        gelu_out,
        seq_len,
        mlp_dim
    );

    //down projection
    launch_linear(
        gelu_out,
        down_weight,
        down_bias,
        down_proj,
        seq_len,
        mlp_dim,
        hidden_dim
    );

    //residual add
    launch_residual_add(
        input,
        down_proj,
        output,
        seq_len,
        hidden_dim
    );
}