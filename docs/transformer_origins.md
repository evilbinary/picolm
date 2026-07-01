# Transformer 的科学依据与起源

## 背景：RNN 的瓶颈

Transformer 之前，序列建模靠 RNN/LSTM：

- **顺序计算**：t 时刻的隐状态依赖 t-1 时刻，无法并行
- **长程依赖**：梯度消失/爆炸，100 步后的信息基本没了
- LSTM 用门控缓解了梯度问题，但顺序计算瓶颈没解决

## 核心洞察

**Attention Is All You Need** (Vaswani et al., 2017) 的关键发现：

1. **序列建模可以不用循环** — 一个位置的信息不需要通过时间步传递，通过"查表"（attention）直接看所有位置
2. **位置编码替代顺序结构** — 扔掉 RNN 后，位置信息通过三角函数注入
3. **矩阵乘法代替循环** — `Q @ K^T` 一次算完所有 pairwise 分数，GPU 上极度高效

## 发明路线

### Step 1 — Attention 机制的出现

- **Bahdanau 2014**: NMT 机器翻译。在 RNN decoder 上加了 attention，翻译时动态看源句子的不同位置。这是第一个"让模型自己决定看哪"的机制
- **Luong 2015**: 简化 attention 为 dot product（内积相似度），效果好且计算更快

### Step 2 — 去掉 RNN 的尝试

- **ConvS2S (2017)**: 用 CNN 代替 RNN，通过层级卷积获得全局感受野，局部可以并行，但长距离仍需加深层数
- 学术界已经意识到 RNN 的顺序约束是瓶颈，但还缺一个能完全替代它的组件

### Step 3 — Transformer 的诞生

**Attention 堆叠**：既然 attention 能一次看所有位置，那就不用 RNN 了，只用 attention 层堆起来就行。但有几个问题需要解决：

**Scale 问题**：`softmax(Q @ K^T)` 在维度大时，内积的方差很大（随 `d` 增大），导致 softmax 梯度消失。解决：`÷ sqrt(d)` 把方差归一化到 1。

**Multi-head**：一次 attention 只能看一种关系，分成 h 个头各看各的再拼接 — 等价于 ensemble 多个 attention，捕获不同类型的相关性。

**残差连接 + LayerNorm**：30 层叠加时，残差让梯度直达底层，LayerNorm 稳定每层激活分布。

### Step 4 — 为什么取代 RNN

**数学等价性**：

- RNN: `h_t = f(h_{t-1}, x_t)` — **线性时间信息聚合**，复杂度 O(n)，但必须串行
- Attention: 每层所有位置全连接图，复杂度 O(n²)，但**完全并行**

Transformer 没有从本质上改变"信息聚合"的数学目标，而是把 RNN 的"按时间步递归"换成了"按层一次递归" — 每层所有位置互相看一次，层间依然顺序。

## 发展路线

```
2014 Bahdanau Attention        ← 第一次引入 attention
2015 Luong dot-product attn    ← 简化
2017 ConvS2S                   ← 去掉 RNN 尝试
2017 Transformer               ← Only attention, 完全并行
2018 BERT (encoder-only)       ← 双向理解
2018 GPT (decoder-only)        ← 单向生成
2022 ChatGPT (GPT-3.5)         ← 规模 + RLHF
2023 LLaMA/Qwen               ← 开源生态
```
