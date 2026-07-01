# model_forward vs Standard Transformer Decoder

picolm 的 `model_forward` 和标准 Transformer decoder 本质是同一个架构，但有几处简化/差异：

## 相同点

- **Decoder-only** 自回归架构，逐 token 生成
- **RMSNorm** -> Attention -> 残差连接 -> RMSNorm -> FFN -> 残差连接
- **SwiGLU FFN**: gate/up/down 三个投影 + SiLU 激活
- **RoPE** 旋转位置编码
- **GQA** (Grouped Query Attention)

## 差异

| 特性 | 标准 Transformer (LLaMA/Qwen 参考实现) | picolm model_forward |
|---|---|---|
| **Attention score** | `softmax(Q @ K^T / sqrt(d)) @ V`，需 `[n_heads, seq_len]` score 矩阵 | **Flash attention / online softmax** — 单 pass 流式累加，无 score buffer，节省显存 |
| **K/V cache 精度** | FP32 | **FP16** — 存储和计算都在 FP16，减半 cache 内存带宽 |
| **Q/K/V 偏置** | LLaMA 无，Qwen2 有 | **可选偏置** — 通过 `matmul_bias` 统一处理，NULL 则跳过 |
| **QK-norm** | 仅 Qwen3 有，每 head RMSNorm | **条件执行** — GGUF 有 `attn_q_norm.weight` 才启用，向后兼容 |
| **FFN gate+up** | 两次独立 matmul | **matmul_dual 融合**（`#ifdef USE_BARRIER_POOL`）— 两次 matmul 在同一个线程循环里并行完成 |
| **线程池** | 通常 OMP/pthread 通用模型 | **事件驱动线程池** — Windows: WaitForSingleObject/SetEvent，Linux: 条件变量，无 spin-wait |
| **RoPE 计算** | 每个位置实时 powf/cosf/sinf | **预计算查表** — 加载时生成 cos/sin 表，推理时直接索引 |
| **mmap** | 全量加载或逐 tensor 读取 | **mmap + 预缺页** — mmap 映射后主动 touch 每页触发 page fault，避免推理时缺页抖动 |
| **数学库** | 可能依赖 BLAS | **手写 SIMD** — AVX2 量化向量点积（Q4_K, Q8_0, Q6_K），无外部依赖 |
| **数据类型** | 加载时可能转 FP32 | **量化原位推理** — Q4_K/Q6_K/Q8_0 直接在量化表示上计算，仅反量化行到临时 buffer |

## 核心取舍

picolm 的设计围绕 **单文件、无外部依赖、内存带宽瓶颈优化**：

- Flash attention 去掉了 `O(n^2)` 的 score buffer，seq_len 只受 cache 大小限制
- FP16 KV cache 把 cache 读写带宽减半（~10GB/s 下意味着 ~0.5 tok/s 的差距）
- mmap 零拷贝加载，不需要 decode/convert 整个模型到内存
- matmul_dual 把 FFN 的 4 次大 matmul（q/k/v/output）融合计算，减少线程同步开销
