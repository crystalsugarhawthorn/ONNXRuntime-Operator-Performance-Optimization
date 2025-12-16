# ONNX Runtime ROCm 自定义算子优化

[English](./readme.md) | **简体中文**

本仓库包含我们在先导杯「ONNX Runtime 算子性能优化」赛题中的自定义算子、核函数以及完整的编译与验证工具链。我们针对 AMD ROCm DCU 上的 5 个关键算子——Conv2d、Attention、BatchNormalization、LeakyReLU 与 GroupNormalization——进行了系统优化，并提供可复现的测试脚本与文档资料。

## 亮点成果

| 算子 | 基线时延 (ms) | 优化时延 (ms) | 加速比 | 关键手段 |
| --- | --- | --- | --- | --- |
| Conv2d | 1.118 | 0.052 | 21.5× | im2col+GEMM 框架，3×3 场景叠加 Winograd F(2×2,3×3) |
| Attention | 0.268 | 0.012 | 22.3× | hipBLAS 批量 GEMM + Warp 级 Softmax |
| BatchNormalization | 6.891 | 0.177 | 38.9× | 参数融合 + float4 Grid-Stride 主核 |
| LeakyReLU | 1.258 | 0.956 | 1.32× | 向量化 fmaxf 激活、分支消除 |
| GroupNormalization | 0.260 | 0.182 | 1.43× | Warp 规约 + rsqrtf + float4 读写 |

所有优化算子均通过 `benchmark.py` 中的 SNR 与 Cosine 指标校验，数值误差控制在浮点噪声范围内。

## 仓库结构

```
onnx/
├── benchmark.py           # ROCm EP 延迟与精度基准脚本
├── compile.sh             # 构建 libcustom_op_library.so 的 hipcc/clang++ 流程
├── custom_op_library.cc   # ORT 自定义算子注册入口
├── rocm_ops.cc/.hip.cpp   # 主机端调度 + 各算子的 HIP kernel
├── winograd/              # Conv2d 共享的 Winograd kernel
├── include/onnxruntime/   # 精简版 ORT 头文件
├── docker/Dockerfile      # 竞赛使用的 DTK 25.04 ROCm 镜像
└── baseline_latency.json  # 官方基线性能数据
```

根目录下的 `ONNX_Runtime算子性能优化设计文档.pdf` 与 `ONNX_码疯冲击.pptx` 提供了完整技术报告与答辩材料。

## 构建与安装

### 环境依赖

- ROCm 5.7 及以上工具链：`hipcc`、`clang++`、`hipBLAS`、`rocBLAS`、`MIOpen`
- 与大赛一致的 DTK 25.04 头文件/库路径
- ONNX Runtime 头文件（已随仓库提供）

> 建议直接沿用 `docker/Dockerfile` 对应的基础镜像，可快速还原官方环境。

### 编译自定义算子库

在 `onnx/` 目录执行：

```bash
bash compile.sh
```

脚本会完成：

- 编译 `rocm_ops.hip.cpp` 及全部 Winograd kernel
- 编译 `rocm_ops.cc` 与 `custom_op_library.cc`
- 链接生成 `libcustom_op_library.so`，并写入所需 ROCm 库依赖

随后将 `libcustom_op_library.so` 拷贝或软链至 ONNX Runtime 应用目录，或按 `benchmark.py` 示例通过 `SessionOptions.register_custom_ops_library` 动态加载。

## 优化手册

### Conv2d
- 通过 `im2col` 将卷积重写为 GEMM，再对 3×3 核使用 Winograd F(2×2,3×3)。
- 分层分块：全局内存 → 共享内存双缓冲 → 寄存器阻塞与软件流水。
- float4 I/O、Bank Conflict 规避以及占用率导向的 tile 选择，使算子从内存受限转向计算受限。

### Attention
- 利用 `hipblasSgemmStridedBatched` 处理 QKᵀ 与 Scores·V，并内置 1/√H 缩放。
- 自定义 Warp 级 Softmax kernel 在共享内存中缓存每行分数，并通过 `__shfl_down` 蝶式规约实现数值稳定化。
- 22× 加速后，注意力算子由原先的主要瓶颈转为可忽略的耗时。

### BatchNormalization
- 先行代数融合，预计算每通道 `scale` 与 `shift`（α、ζ），主核只保留 `y = α * x + ζ`。
- float4 Grid-Stride kernel 最大化带宽利用率，独立尾核处理不足 4 的元素。
- 主机端缓存设备内存，避免重复 `hipMalloc/hipFree`。

### LeakyReLU
- 使用向量化 `fmaxf(x, alpha * x)` 消除分支，配合网格步长循环保持高占用。
- 检测对齐后启用 float4 访存，大规模张量下可获得 1.3×~2× 提升。

### GroupNormalization
- 单 kernel 方案结合 Warp 级规约：64 线程通过 shuffle 完成均值/方差累积，仅需极少共享内存与同步。
- float4 读写 + `rsqrtf` 计算标准差倒数，使内核保持带宽高效。
- 在 N=256, C=64, G=32 场景下实现 1.43× 加速，SNR≈6e-14，精度无损。

## Profiling 摘要

五个算子的累计执行时间由 1,062,381 ns 降至 87,039 ns。优化后 Attention 与 BatchNormalization 的占比从 80%+ 降到不足 40%，Winograd Conv 成为新的主要耗时但绝对时间远小于原实现。

## 延伸阅读

- `ONNX_Runtime算子性能优化设计文档.pdf`：35 页技术报告，包含推导、profiling 截图与更全面的基准实验。
- `ONNX_码疯冲击.pptx`：答辩汇报用幻灯片，概述方法、创新点及平台排名记录。

欢迎将这些 kernel 作为 ROCm 项目的起点，我们的编译与基准脚本也可直接复用，帮助你快速验证新的算子优化思路。

