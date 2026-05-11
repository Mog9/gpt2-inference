#pragma once
#include "qkv_view.h"

void launch_attention_scores(
    QKVHeadView view,
    float* scores
);