# GPT-2 CUDA Inference Engine

CUDA implementations of transformer inference kernels and runtime components written entirely from scratch in C++ and CUDA.

The project builds a complete GPT-2 style inference pipeline directly from low-level CUDA primitives, including custom GEMM kernels, normalization, multi-head attention, KV cache management, transformer blocks, and autoregressive token generation.

---

# Features

## CUDA Kernels

### Tiled Matrix Multiplication
Shared-memory tiled GEMM kernel used for transformer linear projections and feed-forward layers.

### LayerNorm
Custom LayerNorm implementation using CUDA reduction primitives for mean and variance computation.

### GELU
GPT-2 style GELU activation using the tanh approximation formulation.

### Embedding Lookup
CUDA embedding gather kernel for token and positional embeddings.

### Reduction Primitives
Warp-level and block-level reductions used internally for normalization and softmax operations.

---

# Transformer Runtime

## Reusable Linear Layer Wrapper
Built reusable transformer linear layers on top of the tiled GEMM implementation with optional bias support.

Used for:
- QKV projection
- attention output projection
- MLP projections
- logits projection

---

## Fused QKV Projection
Implemented GPT-style fused Query/Key/Value projection:

```text
[seq_len, hidden_dim]
    ->
[seq_len, 3 * hidden_dim]
```

using a single projection pass.

---

## Stride-Based Head Views
Implemented pointer-offset based QKV head views so attention heads can operate directly on fused tensors without reshaping or memory copies.

---

## Fused Attention Scores + Softmax
Implemented fused scaled dot-product attention score computation, causal masking, and numerically stable softmax in CUDA.

Current pipeline computes:

```text
QK^T
-> scale
-> causal mask
-> softmax
```

inside a fused attention kernel.

---

## Attention Output Computation
Implemented:

```text
attention_weights @ V
```

for contextualized token representations across all attention heads.

---

## Multi-Head Merge
Implemented merge operation from:

```text
[num_heads, seq_len, head_dim]
```

into:

```text
[seq_len, hidden_dim]
```

before final projection.

---

## Residual Connections
Implemented GPT-style residual routing:

```text
x = x + block_output
```

for both attention and MLP blocks.

---

## Transformer MLP Block
Implemented full GPT-2 feed-forward pipeline:

```text
linear_up
-> GELU
-> linear_down
```

using transformer dimensions:

```text
[seq_len, 768]
-> [seq_len, 3072]
-> [seq_len, 768]
```

---

## KV Cache
Implemented transformer KV cache storage for autoregressive inference.

Key and value tensors are stored per transformer layer to support incremental decoding workflows.

---

## Full Transformer Block
Implemented full GPT-style transformer execution:

```text
LayerNorm
-> Multi-Head Causal Self-Attention
-> Residual Add
-> LayerNorm
-> MLP
-> Residual Add
```

inside a complete CUDA transformer runtime.

---

## GPT-2 Weight Loading
Implemented GPT-2 checkpoint loading and runtime execution using pretrained GPT-2 Small weights.

---

## Autoregressive Token Generation
Implemented autoregressive GPT-style token generation pipeline using CUDA transformer forward execution.

---

# Performance

```text
Hardware:
RTX 3050 Laptop GPU (4GB VRAM)

Model:
GPT-2 Small

Inference:
CUDA FP32

Peak Throughput:
~190+ tokens/sec
```

---

# Profiling and Optimization

Profiling performed using NVIDIA Nsight Compute.

Implemented optimizations include:
- shared-memory tiled GEMM
- fused attention score + softmax kernel
- stride-based tensor views
- reduced memory copies inside attention pipeline
- reusable transformer projection kernels
- KV cache integration

---

# Hardware

Benchmarks and profiling performed on:

- NVIDIA RTX 3050 Laptop GPU
- 4GB VRAM