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

    view.qkv = qkv;

    view.seq_len = seq_len;

    view.hidden_dim = hidden_dim;

    view.num_heads = num_heads;

    view.head_dim = hidden_dim / num_heads;

    view.head_idx = head_idx;

    return view;
}