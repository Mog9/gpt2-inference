# GPT-2 CUDA Inference Kernels

CUDA implementations of core transformer inference kernels written from scratch in C++/CUDA.

Current implementations include:

- tiled matrix multiplication
- reusable linear layer operator
- fused QKV projection
- QKV tensor head views
- stable softmax
- online softmax
- 3-pass LayerNorm
- 2-pass Welford LayerNorm
- warp-level reductions
- block-level reductions
- custom struct/state reductions

## Implemented Kernels

### Tiled Matrix Multiplication

Shared-memory tiled GEMM kernel with cooperative tile loading and threadblock tiling.

### Linear Layer Operator

Reusable transformer-style linear layer wrapper built on top of GEMM with optional bias support.

### Fused QKV Projection

Single GEMM-based fused Query/Key/Value projection matching GPT-2 tensor layouts.

### QKV Tensor Views

Stride-based tensor view system for per-head Q/K/V access without physical reshaping or copies.

### Stable Softmax

Numerically stable row-wise softmax using separate max and sum reductions.

### Online Softmax

2-pass online softmax implementation using fused `(m, d)` reductions and online normalization updates.

### 3-Pass LayerNorm

LayerNorm implementation using separate mean, variance, and normalization passes.

### 2-Pass Welford LayerNorm

Online Welford-based LayerNorm using associative statistical state reductions.

### Reduction Utilities

Custom CUDA reduction primitives include:

- warp sum reductions
- warp max reductions
- warp shuffle communication
- shared-memory block reductions
- struct/state reductions

## Hardware

Benchmarks and profiling were performed on:

- RTX 3050 Laptop GPU