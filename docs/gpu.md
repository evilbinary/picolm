# GPU 加速使用说明

## 概述

picolm 支持通过 Vulkan Compute 将模型推理的矩阵乘法（matmul）卸载到 GPU 执行，覆盖约 88% 的权重读取带宽。RMSNorm、RoPE、Softmax、SiLU、KV cache 等操作仍由 CPU 处理。

## 编译

### 前置条件

- **Vulkan SDK** 1.2+
  - Windows: 从 https://vulkan.lunarg.com/ 下载安装
  - Linux: `apt install vulkan-sdk` 或 `dnf install vulkan-sdk`
  - macOS: 安装 MoltenVK（通过 Homebrew: `brew install molten-vk`）

### Windows 编译

```bash
# 下载 Vulkan SDK 后设置路径（或修改 Makefile 中的 VULKAN_SDK_ROOT）
# 默认路径: E:/soft/VulkanSDK/1.4.350.0
make vulkan
```

### Linux 编译

```bash
make vulkan-linux
```

### macOS 编译

```bash
make vulkan-mac
```

### 编译产物

编译后生成 `picolm` 可执行文件。编译启用了 `USE_VULKAN` 宏，包含 GPU 代码路径。

## 运行

```bash
# 基本用法（自动检测 GPU）
./picolm model.gguf -p "你好" -n 128

# 强制使用集成 GPU（如 Intel UHD Graphics）
PICOLM_GPU_FORCE=1 ./picolm model.gguf -p "你好" -n 128

# 只卸载部分层到 GPU，其余走 CPU（VRAM 不足时使用）
PICOLM_GPU_LAYERS=8 ./picolm model.gguf -p "你好" -n 128

# 组合使用
PICOLM_GPU_FORCE=1 PICOLM_GPU_LAYERS=8 ./picolm model.gguf -p "你好" -n 128
```

## 运行时环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PICOLM_GPU_FORCE=1` | — | 跳过集成 GPU 检测，强制使用（默认 iGPU 会被自动跳过） |
| `PICOLM_GPU_LAYERS=N` | 全部层 | 只卸载前 N 层 transformer 到 GPU，其余层走 CPU |

## 硬件要求

### 支持的 GPU

任何支持 Vulkan 1.2+ 的 GPU：

| GPU 类型 | 情况 |
|----------|------|
| 独立显卡（NVIDIA GTX/RTX, AMD RX） | 推荐，性能最佳 |
| 集成显卡（Intel UHD / Iris Xe, AMD APU） | 支持，但通常比 CPU 慢，需 `PICOLM_GPU_FORCE=1` |
| Apple Metal（通过 MoltenVK） | 支持 |
| Android / iOS（通过 MoltenVK） | 支持 |

### VRAM 需求

模型权重以量化格式（Q4_K/Q6_K/Q8_0）存储在 VRAM 中。预估用量：

| 模型大小 | 量化格式 | 预估 VRAM |
|----------|----------|-----------|
| 1.1B (TinyLlama) | Q4_K | ~0.6 GB |
| 7B (Qwen3-7B) | Q4_K | ~3.9 GB |
| 7B (Qwen3-7B) | Q8_0 | ~7.0 GB |
| 8B (Qwen3-8B) | Q4_K | ~4.5 GB |

### VRAM 不足

如果 VRAM 不够容纳全部模型层，可以用 `PICOLM_GPU_LAYERS=N` 只卸载部分层：

```bash
# 假设 VRAM 只能容纳 40% 的权重
PICOLM_GPU_LAYERS=12 ./picolm qwen3-8b-q4.gguf -p "你好" -n 128
```

此时前 12 层走 GPU，其余层走 CPU。性能介于全 GPU 和全 CPU 之间。

## 回退机制

| 场景 | 行为 |
|------|------|
| 未编译 `USE_VULKAN` | CPU 路径，无影响 |
| 无 Vulkan 驱动 | `gpu_init` 失败，自动走 CPU |
| VRAM 不足 | 启动时报错，设置 `PICOLM_GPU_LAYERS` 后重试 |
| 集成 GPU | 自动跳过（`PICOLM_GPU_FORCE=1` 覆盖） |
| 某层权重不在 GPU 上 | 自动走 CPU matmul |

推理过程中如果 GPU 调用失败（如超时、设备丢失），自动回退 CPU，不影响输出正确性。

## 数值精度

GPU 计算结果与 CPU 存在微小差异（约 1e-5 量级），原因：

- 浮点运算顺序不同（GPU 并行累加 vs CPU 串行累加）
- GPU 使用 IEEE 754 标准 FP32

最终生成的 token 与 CPU 一致的。如发现 token 差异，通常是 sampler 的随机性导致（设置相同 `-s` seed 可复现）。

## 性能调优

### 集成 GPU 特别说明

集成 GPU（如 Intel UHD Graphics 630）没有独立显存，与系统共享内存。推理时存在额外的数据搬运开销，通常比 CPU 慢。建议：

1. 用 `PICOLM_GPU_LAYERS` 控制卸载层数，找到最佳平衡点
2. 或完全走 CPU（不设 `PICOLM_GPU_FORCE` 即可）

### 分析性能

观察启动日志中的 `GPU:` 行：

```
GPU: VK_AMD (Radeon RX 7800 XT)            # GPU 名称
GPU: ~3847.2 MB model weights to upload     # 上传的权重总量
GPU: uploaded 196 tensors                    # 上传的 tensor 数量
GPU: Vulkan init OK                         # 初始化成功
GPU: matmul n=3584 d=3584 qtype=12 ...      # 每步 matmul 日志
```

## 常见问题

**Q: 编译报错找不到 vkCreateInstance？**
A: 确认 Vulkan SDK 路径正确，`VULKAN_SDK_ROOT` 指向 SDK 根目录。

**Q: 运行提示 "no Vulkan physical devices found"？**
A: 没有安装 GPU 驱动，或驱动不支持 Vulkan。更新显卡驱动后重试。

**Q: 提示 "skipping — set PICOLM_GPU_FORCE=1 to use it"？**
A: 检测到集成 GPU。如果仍想使用，设 `PICOLM_GPU_FORCE=1`。

**Q: 程序卡死无响应？**
A: GPU dispatch 超时。通常是 iGPU 计算太慢或驱动 bug。加 `PICOLM_GPU_LAYERS=4` 减少 GPU 负载，或放弃使用 GPU。

## 技术架构

详见 [vulkan_gpu_offload.md](vulkan_gpu_offload.md) 设计文档。
