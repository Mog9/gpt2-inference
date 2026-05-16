#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <string>
#include <cstdlib>
#include <chrono>
#include <nlohmann/json.hpp>
#include "../include/gpt2_config.h"
#include "../include/gpt2_weights.h"
#include "../include/gpt2_buffers.h"
#include "../include/gpt2_loader.h"
#include "../include/gpt2_forward.h"

int sample_next_token(const std::vector<float>& logits,const std::vector<int>& generated,int seq_len,int vocab_size,float temperature=0.8f,int top_k=40,float top_p=0.9f,float repetition_penalty=1.2f) {
    int offset=(seq_len-1)*vocab_size;
    std::vector<std::pair<float,int>> scores;
    scores.reserve(vocab_size);

    for(int i=0;i<vocab_size;i++) {
        float logit=logits[offset+i];

        for(int prev:generated) {
            if(prev==i) {
                if(logit>0.0f)
                    logit/=repetition_penalty;
                else
                    logit*=repetition_penalty;
            }
        }

        logit/=temperature;
        scores.push_back({logit,i});
    }

    std::partial_sort(
        scores.begin(),
        scores.begin()+top_k,
        scores.end(),
        [](const auto& a,const auto& b) {
            return a.first>b.first;
        }
    );

    scores.resize(top_k);

    float max_logit=scores[0].first;
    std::vector<float> probs(top_k);
    float sum=0.0f;

    for(int i=0;i<top_k;i++) {
        probs[i]=std::exp(scores[i].first-max_logit);
        sum+=probs[i];
    }

    for(int i=0;i<top_k;i++)
        probs[i]/=sum;

    float cumulative=0.0f;
    int cutoff=top_k;

    for(int i=0;i<top_k;i++) {
        cumulative+=probs[i];

        if(cumulative>=top_p) {
            cutoff=i+1;
            break;
        }
    }

    probs.resize(cutoff);
    scores.resize(cutoff);

    float renorm=0.0f;

    for(float p:probs)
        renorm+=p;

    for(float& p:probs)
        p/=renorm;

    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::discrete_distribution<> dist(
        probs.begin(),
        probs.end()
    );

    int sampled_idx=dist(gen);

    return scores[sampled_idx].second;
}

int main() {
    GPT2Config config;
    GPT2Weights weights;

    load_gpt2_weights(
        weights,
        config,
        "weights/gpt2"
    );

    GPT2Buffers buffers;

    cudaMalloc(&buffers.hidden_states_1,config.max_seq_len*config.hidden_dim*sizeof(float));
    cudaMalloc(&buffers.hidden_states_2,config.max_seq_len*config.hidden_dim*sizeof(float));
    cudaMalloc(&buffers.ln1_out,config.max_seq_len*config.hidden_dim*sizeof(float));
    cudaMalloc(&buffers.qkv,config.max_seq_len*(3*config.hidden_dim)*sizeof(float));
    cudaMalloc(&buffers.scores,config.num_heads*config.max_seq_len*config.max_seq_len*sizeof(float));
    cudaMalloc(&buffers.heads,config.num_heads*config.max_seq_len*(config.hidden_dim/config.num_heads)*sizeof(float));
    cudaMalloc(&buffers.merged,config.max_seq_len*config.hidden_dim*sizeof(float));
    cudaMalloc(&buffers.attn_proj,config.max_seq_len*config.hidden_dim*sizeof(float));
    cudaMalloc(&buffers.attn_residual,config.max_seq_len*config.hidden_dim*sizeof(float));
    cudaMalloc(&buffers.ln2_out,config.max_seq_len*config.hidden_dim*sizeof(float));
    cudaMalloc(&buffers.mlp_up,config.max_seq_len*config.mlp_dim*sizeof(float));
    cudaMalloc(&buffers.mlp_gelu,config.max_seq_len*config.mlp_dim*sizeof(float));
    cudaMalloc(&buffers.mlp_down,config.max_seq_len*config.hidden_dim*sizeof(float));
    cudaMalloc(&buffers.logits,config.max_seq_len*config.vocab_size*sizeof(float));

    while(true) {
        std::string prompt = "The history of artificial intelligence began in antiquity with myths...";
        std::cout<<"\n> ";

        if(prompt=="exit")
            break;

        std::string tokenize_cmd=
            "python scripts/tokenize_prompt.py \""+
            prompt+
            "\"";

        system(tokenize_cmd.c_str());

        std::ifstream file("data/prompt_tokens.json");

        nlohmann::json j;

        file>>j;

        std::vector<int> tokens=
            j.get<std::vector<int>>();

        std::vector<int> generated;

        int max_new_tokens=1; //low because of profile, test
        std::cout<<"\nGPT2:\n";

        auto start=
            std::chrono::high_resolution_clock::now();

        for(int step=0;step<max_new_tokens;step++) {
            int seq_len=tokens.size();

            int* d_tokens;

            cudaMalloc(
                &d_tokens,
                seq_len*sizeof(int)
            );

            cudaMemcpy(
                d_tokens,
                tokens.data(),
                seq_len*sizeof(int),
                cudaMemcpyHostToDevice
            );

            gpt2_forward(
                d_tokens,
                weights,
                buffers,
                seq_len,
                config
            );

            std::vector<float> h_logits(
                seq_len*config.vocab_size
            );

            cudaMemcpy(
                h_logits.data(),
                buffers.logits,
                seq_len*config.vocab_size*sizeof(float),
                cudaMemcpyDeviceToHost
            );

            int next_token=
                sample_next_token(
                    h_logits,
                    generated,
                    seq_len,
                    config.vocab_size
                );

            generated.push_back(next_token);
            tokens.push_back(next_token);

            cudaFree(d_tokens);

            if(next_token==50256)
                break;

            if(tokens.size()>=config.max_seq_len)
                break;
        }

        auto end=
            std::chrono::high_resolution_clock::now();

        float seconds=
            std::chrono::duration<float>(
                end-start
            ).count();

        float tok_per_sec=
            generated.size()/seconds;

        std::ofstream out(
            "data/generated_tokens.json"
        );

        nlohmann::json out_json=
            generated;

        out<<out_json;

        out.close();

        system(
            "python scripts/decode_token.py"
        );

        std::cout
            <<"\n\nTokens/sec: "
            <<tok_per_sec
            <<"\n";
    }

    return 0;
}