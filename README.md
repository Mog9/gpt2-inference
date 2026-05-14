# GPT-2 CUDA Inference Engine

CUDA implementations of core transformer inference kernels and transformer runtime components written from scratch in C++/CUDA.

The current focus of the project is building a complete GPT-style transformer inference runtime from low-level CUDA primitives upward, including custom GEMM kernels, normalization, softmax, multi-head attention, transformer blocks, embeddings, and feed-forward networks.

The project is currently at the stage where a complete GPT-style transformer block and embedding pipeline are functioning end-to-end in CUDA.

## Kernels Implemented

### Tiled Matrix Multiplication
Shared-memory tiled GEMM kernel used as the base compute primitive for transformer linear layers and projections.

### Softmax
Numerically stable row-wise softmax kernels for attention probability computation.

### LayerNorm
Custom LayerNorm implementations using reduction-based mean and variance computation.

### GELU
Transformer-style GELU activation kernel using the tanh approximation formulation used in GPT-2.

### Embedding Lookup
CUDA embedding gather kernel for token and positional embeddings used as transformer input generation.

### Reduction Primitives
Warp-level and block-level CUDA reductions used internally for normalization and softmax operations.

---

# Transformer Operations Implemented

## Reusable Linear Layer Wrapper
Built a reusable linear operation on top of tiled GEMM with optional bias support so transformer projections can share the same kernel path.

Used for:
- QKV projection
- output projection
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

using a single linear projection pass.

---

## Stride-Based QKV Head Views
Implemented per-head tensor views using pointer offsets and row strides so individual attention heads can access Q/K/V slices without physical reshaping or memory copies.

This allows attention kernels to operate directly on the fused GPT-2 tensor layout.

---

## Attention Score Computation (QKᵀ)
Implemented scaled dot-product attention score computation using custom CUDA kernels over strided Q/K head views.

Current implementation computes:

```text
QK^T
```

for each attention head independently.

---

## Scaling + Causal Masking
Implemented fused scaling and autoregressive causal masking pass over attention scores before softmax.

Masked future-token positions are replaced with large negative values to enforce GPT-style causal attention behavior.

---

## Softmax Over Attention Scores
Applied numerically stable row-wise softmax over masked attention scores to produce normalized attention probability distributions.

---

## Attention Output Computation
Implemented attention output computation using:

```text
attention_weights @ V
```

to generate contextualized token representations for each attention head.

---

## Multi-Head Merge
Implemented head merge operation that concatenates:

```text
[num_heads, seq_len, head_dim]
```

into:

```text
[seq_len, hidden_dim]
```

before final attention output projection.

---

## Output Projection
Implemented final transformer attention output projection using the reusable linear/GEMM pipeline.

This mixes information learned across all attention heads back into the transformer hidden representation space.

---

## Residual Connections
Implemented residual add kernels for GPT-style skip connections after attention and MLP blocks.

Current residual form:

```text
x = x + block_output
```

---

## Transformer GELU Activation
Implemented GPT-2 style GELU activation using the tanh approximation formulation:

```text
0.5x(1 + tanh(...))
```

used inside transformer feed-forward networks.

---

## Transformer MLP Block
Implemented full GPT-style feed-forward network pipeline:

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

## Full Transformer Block
Implemented full GPT-style transformer block execution:

```text
LayerNorm
-> Multi-Head Causal Self-Attention
-> Residual Add
-> LayerNorm
-> MLP
-> Residual Add
```

including attention, normalization, residual routing, and feed-forward execution inside a single transformer layer runtime.

---

## Embedding Lookup Pipeline
Implemented GPT-2 style token and positional embedding lookup:

```text
token_embedding[token_id]
+
positional_embedding[position]
```

which converts integer token ids into transformer hidden-state vectors:

```text
[token_ids]
    ->
[seq_len, hidden_dim]
```

used as input to transformer blocks.

---

# Current Status

The project currently supports:

### Transformer Runtime Components
- embedding lookup
- transformer block execution
- multi-head causal self-attention
- transformer MLP execution
- residual routing
- layer normalization
- output projection

### GPT-Style Attention Pipeline
- fused QKV projection
- head splitting
- QKᵀ attention scores
- scaling + masking
- softmax
- attention @ V
- head merging
- output projection

### GPT-Style Feed-Forward Pipeline
- linear up projection
- GELU activation
- linear down projection
- residual add

---

# Optimization Status

The current implementation is focused on correctness and architecture first.

Kernel optimization and inference-specific performance work will be done after the complete transformer engine is functioning end-to-end.

---

# Hardware

Benchmarks and profiling were performed on:

RTX 3050 Laptop GPU