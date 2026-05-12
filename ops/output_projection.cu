#include "../include/output_projection.h"
#include "../include/linear.h"

void launch_output_projection(
    float* input,
    float* weight,
    float* bias,
    float* output,
    int seq_len,
    int hidden_dim
) {
    launch_linear(input, weight, bias, output, seq_len, hidden_dim, hidden_dim);
}