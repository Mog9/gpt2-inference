#pragma once
#include "gpt2_config.h"
#include "gpt2_weights.h"
#include "gpt2_buffers.h"

void gpt2_forward(
    int* token_ids,
    GPT2Weights& weights,
    GPT2Buffers& buffers,
    int seq_len,
    const GPT2Config& config
);