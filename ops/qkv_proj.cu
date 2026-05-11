#include <cuda_runtime.h>
#include "../include/qkv_proj.h"
#include "../include/linear.h"

/*

input:
    X = [seq_len, hidden_dim]

weights:
    Wqkv = [hidden_dim, 3 * hidden_dim]

output:
    QKV = [seq_len, 3 * hidden_dim]

Q - K - V


example (GPT-2 small):
input:
    [1024, 768]

weights:
    [768, 2304]

output:
    [1024, 2304]

each row layout:
-- 768 Q -- 768 K -- 768 V

later:
    Q = qkv + 0
    K = qkv + hidden_dim
    V = qkv + 2 * hidden_dim

*/

void launch_qkv_projection(
    float* input,
    float* weight,
    float* bias,
    float* qkv,
    int seq_len,
    int hidden_dim
) {
    launch_linear(
        input,
        weight,
        bias,
        qkv,
        seq_len,
        hidden_dim,
        3 * hidden_dim
    );
}