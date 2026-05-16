#include "../include/gpt2_loader.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

void load_tensor(
    float** d_ptr,
    const std::string& path,
    size_t num_elements,
    bool transpose,
    int rows,
    int cols
) {
    std::vector<float> host(num_elements);

    std::ifstream file(path, std::ios::binary);

    if(!file) {
        std::cerr << "Failed to open: " << path << "\n";
        exit(1);
    }

    file.read(
        reinterpret_cast<char*>(host.data()),
        num_elements * sizeof(float)
    );

    file.close();

    if(transpose) {

        std::vector<float> transposed(num_elements);

        for(int r = 0; r < rows; r++) {
            for(int c = 0; c < cols; c++) {
                transposed[c * rows + r] =
                    host[r * cols + c];
            }
        }

        host.swap(transposed);
    }

    cudaMalloc(
        d_ptr,
        num_elements * sizeof(float)
    );

    cudaMemcpy(
        *d_ptr,
        host.data(),
        num_elements * sizeof(float),
        cudaMemcpyHostToDevice
    );
}

void load_gpt2_weights(
    GPT2Weights& weights,
    const GPT2Config& config,
    const char* weight_dir
) {

    weights.blocks.resize(config.num_layers);

    load_tensor(
        &weights.token_embedding_table,
        std::string(weight_dir) + "/transformer_wte_weight.bin",
        config.vocab_size * config.hidden_dim,
        false,
        0,
        0
    );

    load_tensor(
        &weights.lm_head_weight,
        std::string(weight_dir) + "/transformer_wte_weight.bin",
        config.vocab_size * config.hidden_dim,
        true,
        config.vocab_size,
        config.hidden_dim
    );

    load_tensor(
        &weights.positional_embedding_table,
        std::string(weight_dir) + "/transformer_wpe_weight.bin",
        config.max_seq_len * config.hidden_dim,
        false,
        0,
        0
    );

    load_tensor(
        &weights.final_ln_gamma,
        std::string(weight_dir) + "/transformer_ln_f_weight.bin",
        config.hidden_dim,
        false,
        0,
        0
    );

    load_tensor(
        &weights.final_ln_beta,
        std::string(weight_dir) + "/transformer_ln_f_bias.bin",
        config.hidden_dim,
        false,
        0,
        0
    );

    for(int i = 0; i < config.num_layers; i++) {

        auto& block = weights.blocks[i];

        std::string prefix =
            std::string(weight_dir)
            + "/transformer_h_"
            + std::to_string(i);

        load_tensor(
            &block.ln1_gamma,
            prefix + "_ln_1_weight.bin",
            config.hidden_dim,
            false,
            0,
            0
        );

        load_tensor(
            &block.ln1_beta,
            prefix + "_ln_1_bias.bin",
            config.hidden_dim,
            false,
            0,
            0
        );

        load_tensor(
            &block.qkv_weight,
            prefix + "_attn_c_attn_weight.bin",
            config.hidden_dim * (3 * config.hidden_dim),
            false,
            config.hidden_dim,
            3 * config.hidden_dim
        );

        load_tensor(
            &block.qkv_bias,
            prefix + "_attn_c_attn_bias.bin",
            3 * config.hidden_dim,
            false,
            0,
            0
        );

        load_tensor(
            &block.attn_proj_weight,
            prefix + "_attn_c_proj_weight.bin",
            config.hidden_dim * config.hidden_dim,
            false,
            config.hidden_dim,
            config.hidden_dim
        );

        load_tensor(
            &block.attn_proj_bias,
            prefix + "_attn_c_proj_bias.bin",
            config.hidden_dim,
            false,
            0,
            0
        );

        load_tensor(
            &block.ln2_gamma,
            prefix + "_ln_2_weight.bin",
            config.hidden_dim,
            false,
            0,
            0
        );

        load_tensor(
            &block.ln2_beta,
            prefix + "_ln_2_bias.bin",
            config.hidden_dim,
            false,
            0,
            0
        );

        load_tensor(
            &block.mlp_up_weight,
            prefix + "_mlp_c_fc_weight.bin",
            config.hidden_dim * config.mlp_dim,
            false,
            config.hidden_dim,
            config.mlp_dim
        );

        load_tensor(
            &block.mlp_up_bias,
            prefix + "_mlp_c_fc_bias.bin",
            config.mlp_dim,
            false,
            0,
            0
        );

        load_tensor(
            &block.mlp_down_weight,
            prefix + "_mlp_c_proj_weight.bin",
            config.mlp_dim * config.hidden_dim,
            false,
            config.mlp_dim,
            config.hidden_dim
        );

        load_tensor(
            &block.mlp_down_bias,
            prefix + "_mlp_c_proj_bias.bin",
            config.hidden_dim,
            false,
            0,
            0
        );

        std::cout
            << "Loaded transformer block "
            << i
            << "\n";
    }

    std::cout
        << "\nGPT-2 weights loaded successfully\n";
}