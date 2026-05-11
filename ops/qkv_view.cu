#include <cuda_runtime.h>
#include "../include/qkv_view.h"

QKVHeadView create_qkv_head_view(
    float* qkv,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_idx
) {
    QKVHeadView view;

    int head_dim = hidden_dim / num_heads;

    int q_offset = head_idx * head_dim;
    int k_offset = hidden_dim + head_idx * head_dim;
    int v_offset = 2 * hidden_dim + head_idx * head_dim;

    view.q = qkv + q_offset;
    view.k = qkv + k_offset;
    view.v = qkv + v_offset;

    view.seq_len = seq_len;
    view.hidden_dim = hidden_dim;

    view.num_heads = num_heads;
    view.head_dim = head_dim;

    /*
    row_stride: distance to next row in memory

    each row contains: Q + K + V

    total row width: 3 * hidden_dim
    */

    view.row_stride =3 * hidden_dim;

    // head_stride: distance between heads inside Q/K/V regions.

    view.head_stride = head_dim;
    return view;
}