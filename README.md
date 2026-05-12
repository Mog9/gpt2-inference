# GPT-2 CUDA Inference Engine

CUDA implementations of core transformer inference kernels and attention pipeline components written from scratch in C++/CUDA.

The current focus of the project is building a complete GPT-style transformer inference pipeline from low-level CUDA primitives upward, including custom GEMM kernels, normalization, softmax, multi-head attention, tensor views, and transformer projections.

The project is currently at the stage where a full multi-head causal self-attention forward pass is working end-to-end in CUDA.

## Kernels Implemented

### Tiled Matrix Multiplication
Shared-memory tiled GEMM kernel used as the base compute primitive for transformer linear layers and projections.

### Softmax
Numerically stable row-wise softmax kernels for attention probability computation.

### LayerNorm
Custom LayerNorm implementations using reduction-based mean and variance computation.

### Reduction Primitives
Warp-level and block-level CUDA reductions used internally for normalization and softmax operations.

---

# Transformer Operations Implemented

## Reusable Linear Layer Wrapper
Built a reusable linear operation on top of tiled GEMM with optional bias support so transformer projections can share the same kernel path.

Used for:
- QKV projection
- output projection
- future MLP layers

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

# Current Status

The project currently supports a full multi-head causal self-attention forward pipeline in CUDA including:
- fused QKV projection
- head splitting
- QKᵀ attention scores
- scaling + masking
- softmax
- attention @ V
- head merging
- output projection

---

# Optimization Status

The current implementation is focused on correctness and architecture first.

Kernel optimization and inference-specific performance work will be done after the complete transformer engine is functioning end-to-end.

---

# Hardware

Benchmarks and profiling were performed on:

RTX 3050 Laptop GPU