# Speculative Decoding 加速方案

## 概述

在 picolm 中实现投机解码（Speculative Decoding），通过 draft 模型快速生成候选 token + 目标模型并行验证，在不改变输出分布的前提下提升推理速度。

## 架构

```
当前生成循环（串行）:

pos → model_forward(tok, pos) → logits → sample → next_tok → pos+1 → ...
                                                                    ↑
                                                    每个 token 一次完整 forward

投机解码后:

draft 生成 K 个候选 [t1..tK]  ──→  model_forward_batch()  ──→  K 组 logits
（前半层模型，~50% matmul）         （一次 forward，K 路并行）        ↓
                                                             逐位置 accept/reject
                                                                    ↓
                                                            产出 1~K 个 token
```

## 核心改动

### 1. Draft 策略：Self-Speculation

不用加载第二个模型。同一个模型只跑前 N/2 层作为 draft：

```c
float *model_forward_draft(model_t *m, int token, int pos, int max_layers);
```

与 `model_forward()` 共享全部代码，层循环上限为 `max_layers`。跳过 final rmsnorm 和 output projection。

### 2. Batched Verification：`model_forward_batch()`

新增函数：

```c
int model_forward_batch(model_t *m, const int *tokens, int pos, int K,
                         float *logits_out);  // [K * vocab_size]
```

一次验证 K 个 draft token，返回 K 组 logits。

#### 中间状态

`run_state_t` 新增 K 份缓冲：

```c
float *x_batch;     // [K * n_embd]
float *xb_batch;    // [K * n_embd]
float *q_batch;     // [K * n_embd]
float *hb_batch;    // [K * n_ffn]
float *hb2_batch;   // [K * n_ffn]
```

对于 Qwen3-8B（n_embd=3584, K=4）：~180KB 额外内存。

#### Attention 改动

当前（单 query 位置）：

```c
for (int h = 0; h < n_heads; h++) {
    float *qh = s->q + h * head_dim;
    for (int t = 0; t <= pos; t++) {
        score = dot(qh, kt) / sqrt(d);
        // online softmax
    }
}
```

改为 K 路并行（K 个 query 位置）：

```c
for (int h = 0; h < n_heads; h++) {
    for (int v = 0; v < K; v++) {
        float *qh = q_batch + v * n_heads * head_dim + h * head_dim;
        for (int t = 0; t <= pos + v; t++) {
            score = dot(qh, kt) / sqrt(d);
            // online softmax
        }
    }
}
```

第 v 路可以 attend 到之前所有位置（包括 v-1 的 KV 输出），KV cache 写入 `pos+v`。

#### FFN 改动

FFN 没有序列依赖，直接循环 K 次：

```c
for (int v = 0; v < K; v++) {
    float *xv = x_batch + v * dim;
    float *hb = hb_batch + v * n_ffn;
    // rmsnorm, matmul gate/up, silu, mul, matmul down
}
```

### 3. Accept/Reject 逻辑

```
(1) Draft 模型生成 K 个候选 token [t1, t2, ..., tK]
(2) model_forward_batch() 验证全部候选
(3) 每位置 i 检查：
    - argmax(logits[i]) == t_i → accept
    - 否则 → 从 adjusted distribution 采样，截断于 i
(4) 已接受 token → 直接产出
(5) 截断处补一个 target 采样 token
(6) 下一轮从截断位置继续
```

### 4. 配置

用环境变量控制（与 GPU 部分卸载风格一致）：

```c
int K = 4;  // default
const char *env = getenv("PICOLM_SPEC_K");
if (env) K = atoi(env);
if (K < 1 || K > 16) K = 4;

int draft_layers = m->config.n_layers / 2;
const char *l_env = getenv("PICOLM_SPEC_LAYERS");
if (l_env) draft_layers = atoi(l_env);
```

## 改动范围

| 文件 | 改动 |
|------|------|
| `model.h` | `model_forward_batch()` 声明, `model_forward_draft()` 声明, `draft_state_t` |
| `model.c` | `model_forward_batch()` ~200 行, `model_forward_draft()` 改层循环上限, KV cache save/restore |
| `run_state_t` | K 份批处理缓冲指针 |
| `picolm.c` | generation loop 改 spec_decode 主循环 |

## 预估加速

条件：draft 跑一半层（50% matmul），K=4，接受率 0.7/token

- 每轮目标模型 **1 次** forward 替代原来 **~3.3 次**（期望接受数 1/(1-0.7)）
- draft 成本 ≈ 2 次半程 forward
- 理论加速比：层数越少 draft 越快，预估 **1.8x — 2.6x**

## 与 GPU 后端的关系

- GPU 只加速 matmul → 与 speculative decoding 正交
- 可以叠加使用：draft 阶段 GPU 加速（少量层），batch verification 也走 GPU（K 路并行 matmul）
- 在 iGPU 内存不足的场景下（Intel UHD 630），draft 走全部 CPU 少量层，大模型验证走 GPU，减少 VRAM 压力



 运行时控制 — 环境变量：

  PICOLM_SPEC_K=4       # 每轮 draft 候选数（默认 4）
  PICOLM_SPEC_LAYERS=16 # draft 用多少层（默认 n_layers/2）