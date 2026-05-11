#pragma once
#include "qkv_view.h"

void launch_attention_output(
    float* attention_weights,
    QKVHeadView view,
    float* output
);