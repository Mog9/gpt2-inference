#pragma once
#include "gpt2_config.h"
#include "gpt2_weights.h"

void load_gpt2_weights(
    GPT2Weights& weights,
    const GPT2Config& config,
    const char* weight_dir
);