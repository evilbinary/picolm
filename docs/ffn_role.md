# FFN (Feed-Forward Network) 在 Transformer 中的作用

## 一句话

Attention 负责 **位置间通信**，FFN 负责 **每个位置独立做特征变换**。

```
Attention: 位置 A 看懂了位置 B 的上下文 → 传给 FFN
FFN:       "好的，基于我看到的信息，我应该..."
```

## 具体作用

### 1. 特征维度膨胀 → 筛选 → 压缩

SwiGLU FFN 结构：`down(silu(gate(x)) * up(x))`

| 子层 | 维度变化 | 作用 |
|------|---------|------|
| `gate` | d → 4d | 产生门控信号 |
| `up`   | d → 4d | 特征膨胀，提供容量 |
| `silu(gate) * up` | 逐元素乘 | 非线性门控，决定哪些信息保留 |
| `down` | 4d → d | 压缩回原始维度 |

相当于每个位置有一个独立的**小前馈网络**做特征提取。

### 2. 存储事实知识

Geva et al. 2021 的研究发现 FFN 本质是**键值存储器**：

- `gate` 层：查找器（哪些知识相关）
- `down` 层：读取器（把相关事实读出来）

LLM 能回答"法国首都是什么"，靠的不是 attention（attention 只看上下文），而是 FFN 里固化存储的这个世界知识。

```
Attention 传递信息，FFN 提供信息。
```

### 3. 为什么 FFN 占大部分参数量

典型分布（Qwen3-8B）：

```
总参数 8B
  Attention: ~2.5B (31%)
  FFN:       ~5B   (63%)
  Embedding: ~0.5B (6%)
```

FFN 每层每个位置独立运行、不共享参数，所以需要足够"胖"才能装下足够多的知识。

### 4. 为什么 FFN 是推理瓶颈

推理时每层有 4 次大 matmul：

```
Q = x @ Wq      ← d × d
K = x @ Wk      ← d × d
V = x @ Wv      ← d × d
O = attn @ Wo   ← d × d

gate = x_norm @ Wg   ← d × 4d   ← 2/3 的 FLOPs
up   = x_norm @ Wu   ← d × 4d
down = silu_gate_up @ Wd  ← 4d × d
```

FFN 的 gate/up/down 三次 matmul **占了每层 ~88% 的权重读取量**，这也是为什么 picolm 的 `matmul_dual`（融合 gate+up）和量化主要针对 FFN 优化。
