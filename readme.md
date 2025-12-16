# ONNX Runtime ROCm Custom Ops Optimization

**English** | [简体中文](./README.zh-CN.md)

This repository hosts the custom operators, kernels, and tooling we used for the "ONNX Runtime 算子性能优化" track of the 先导杯/智能计算创新设计赛. We focus on five latency-critical operators on AMD ROCm DCUs—Conv2d, Attention, BatchNormalization, LeakyReLU, and GroupNormalization—and provide a full toolchain for compiling, validating, and benchmarking our optimized implementations.

## Highlights

| Operator | Baseline Latency (ms) | Optimized Latency (ms) | Speedup | Notes |
| --- | --- | --- | --- | --- |
| Conv2d | 1.118 | 0.052 | 21.5× | im2col+GEMM baseline plus Winograd F(2×2,3×3) for 3×3 kernels |
| Attention | 0.268 | 0.012 | 22.3× | hipBLAS strided batched GEMM + warp-level softmax |
| BatchNormalization | 6.891 | 0.177 | 38.9× | parameter fusion + float4 grid-stride kernel |
| LeakyReLU | 1.258 | 0.956 | 1.32× | vectorized activation with branch-free fmaxf |
| GroupNormalization | 0.260 | 0.182 | 1.43× | warp-level reduction + rsqrtf + float4 IO |

Accuracy is verified with SNR and cosine similarity checks (see `benchmark.py`); all optimized kernels stay within numerical noise of the reference implementation.

## Repository Layout

```
onnx/
├── benchmark.py           # Latency + accuracy harness for ROcm EP
├── compile.sh             # hipcc/clang++ compilation pipeline for libcustom_op_library.so
├── custom_op_library.cc   # ORT custom op registrations
├── rocm_ops.cc/.hip.cpp   # Host dispatchers + HIP kernels (Conv2d/Attention/BN/Leaky/GroupNorm)
├── winograd/              # Winograd-specific kernels (shared with Conv2d dispatcher)
├── include/onnxruntime/   # Minimal ORT headers needed for out-of-tree builds
├── docker/Dockerfile      # Reference image (DTK 25.04, PyTorch 2.4.1, ROCm stack)
└── baseline_latency.json  # Official baseline traces (used for comparison)
```

Top-level documents `ONNX_Runtime算子性能优化设计文档.pdf` and `ONNX_码疯冲击.pptx` contain the full technical report and slide deck that informed this README.

## Build & Installation

### Prerequisites

- ROCm 5.7+ toolchain with `hipcc`, `clang++`, `hipBLAS`, `rocBLAS`, and `MIOpen`
- DTK 25.04 header/libs layout (matching our competition environment)
- ONNX Runtime headers (already vendored under `include/onnxruntime`)

> Tip: use the provided base image in `docker/Dockerfile` to reproduce the contest environment quickly.

### Compile the Custom Operator Library

From the `onnx/` directory:

```bash
bash compile.sh
```

This script builds

- HIP object files for `rocm_ops.hip.cpp` and all Winograd kernels
- Host-side objects for `rocm_ops.cc` and `custom_op_library.cc`
- A shared library `libcustom_op_library.so` with proper `rpath` entries for ROCm libs

Copy or symlink `libcustom_op_library.so` next to your ONNX Runtime application, or register it explicitly through `SessionOptions.register_custom_ops_library` as shown in `benchmark.py`.

## Optimization Playbook

### Conv2d
- Recast convolution as GEMM via `im2col`, then add Winograd F(2×2,3×3) for 3×3 hot paths.
- Hierarchical tiling: global → shared memory double buffering → register blocking with software pipelining.
- float4 vector I/O, bank-conflict-free shared memory layout, and occupancy-aware tile sizing make the kernel compute-bound.

### Attention
- Offload QKᵀ and softmax-weighted V multiplication to `hipblasSgemmStridedBatched`, fusing the 1/√H scaling factor.
- Dedicated warp-level softmax kernel keeps each scores row resident in shared memory, applies butterfly reductions via `__shfl_down`, and writes normalized weights without extra global reads.
- Achieves 22× latency reduction and shifts the operator from memory-bound to compute-bound behavior.

### BatchNormalization
- Algebraic fusion: precompute per-channel `scale` and `shift` (α, ζ) so the main kernel only performs `y = α * x + ζ`.
- float4 grid-stride kernels saturate memory bandwidth; a lightweight remainder kernel handles tail elements.
- Host-side reusable device buffers eliminate repetitive `hipMalloc/hipFree` pairs.

### LeakyReLU
- Replace branchy element-wise logic with vectorized `fmaxf(x, alpha * x)` loops plus tail handling.
- Grid-stride loops ensure high occupancy for large tensors; dynamic block/grid sizing prevents launch overhead from dominating small inputs.
- Realizes 1.3× in the standard benchmark and up to ~2× on ≥1M-element tensors.

### GroupNormalization
- Single-kernel design with warp-level reductions: each warp accumulates partial sums via shuffle ops, reducing shared-memory usage from 256 slots to 8.
- float4 loads/stores and rsqrtf-based inverse std computation keep the kernel bandwidth-efficient.
- 1.43× speedup verified on N=256, C=64, G=32 workloads with numerical parity (SNR≈6e-14).

## Profiling Summary

hipprof traces show the five kernels’ cumulative duration dropping from 1,062,381 ns to 87,039 ns post-optimization. Attention and BatchNormalization, previously the top bottlenecks (>80% combined), now consume <40% of total kernel time, with Winograd convolution becoming the dominant (yet far faster) kernel.

## Further Reading

- `ONNX_Runtime算子性能优化设计文档.pdf`: 35-page technical deep dive with derivations, profiling screenshots, and extended benchmarks.
- `ONNX_码疯冲击.pptx`: slide deck used during the competition defense, summarizing methodology, innovation points, and ranking records.

Feel free to adapt these kernels to other ROCm projects—each section above is modular, and the build + benchmark tooling is designed to be a drop-in starting point for additional operator optimizations.

