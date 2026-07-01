# Vulkan Compute GPU Offload 设计方案

## 目标

在 picolm 中增加 Vulkan Compute 后端，把 8 次 per-layer matmul 卸载到 GPU 执行，**完全不改动现有 CPU 代码路径**。跨 Windows/Linux/macOS/iOS/Android 五平台。

## 架构

```
matmul() → gpu_try_matmul()  ← 查到 GPU buffer 就 dispatch compute shader
             ↓ 查不到则退回 CPU（原路径）
           CPU matmul  ← 完全不变
```

- GPU 只做 **matmul**（占 ~88% 权重读取），RMSNorm/RoPE/softmax/SiLU/KV cache 全留 CPU
- 权重以量化格式 (Q4_K/Q6_K/Q8_0) 常驻 VRAM，每步只传 `x` 和读回 `out`
- 编译时通过 `USE_VULKAN` 开关，不启用时 gpu.h 提供 inline stub，零开销

## 新增文件

| 文件 | 用途 |
|------|------|
| `picolm/gpu.h` | 公开 API + 非 USE_VULKAN 时的 inline stub |
| `picolm/gpu.c` | Vulkan 全实现：设备/管线/buffer/descriptor/dispatch |
| `picolm/shaders/q4_k.comp` | Q4_K 矩阵乘 GLSL compute shader |
| `picolm/shaders/q6_k.comp` | Q6_K 矩阵乘 GLSL compute shader |
| `picolm/shaders/q8_0.comp` | Q8_0 矩阵乘 GLSL compute shader |
| `picolm/shaders/compile.sh` | glslangValidator 编译 SPIR-V 脚本 |
| `picolm/shaders/spv/*.spv` | 预编译 SPIR-V 二进制 |

## 存在文件改动

| 文件 | 改动 |
|------|------|
| `picolm/tensor.c` | `matmul()` 开头加 `#ifdef USE_VULKAN` 短路，3 行 |
| `picolm/tensor.c` | `matmul_dual()` / `matmul_bias()` 同理 |
| `picolm/picolm.c` | `model_load` 后加 `gpu_init()` + `gpu_upload_model()` |
| `picolm/Makefile` | 加 `vulkan` 和 `shaders` target |

## API 设计 (gpu.h)

```c
int  gpu_init(const model_t *m);         // 创建 Vulkan device, 检查 VRAM
int  gpu_upload_model(const model_t *m);  // 上传所有权重到 VRAM
int  gpu_available(void);                 // 1 = ready
int  gpu_try_matmul(out, x, W, n, d, qtype); // 尝试 GPU matmul，失败返回 -1
void gpu_shutdown(void);                 // 释放所有资源

// 非 USE_VULKAN 时，以上全部展开为 inline { return -1; } / {}，零指令
```

## 核心设计

### Buffer 生命周期

1. **模型加载**: `gpu_init()` 创建 Vulkan device/pipeline/descriptor，检查 VRAM 是否足够
2. **权重上传**: `gpu_upload_model()` 遍历所有权重 tensor，`vkCreateBuffer(DEVICE_LOCAL)` + staging copy
3. **每步推理**: `gpu_try_matmul()` 查 cpu_ptr → GPU buffer → memcpy x 到 staging → dispatch → memcpy out 回来
4. **卸载**: `gpu_shutdown()` vkDeviceWaitIdle + 释放所有资源

### GPU 权重查找

所有权重指针指向 mmap 地址。上传时用 `cpu_ptr = mmap_addr` 做 key 存入查找表（线性扫描 `weights[]` 数组，最多 ~100 个 tensor），dispatch 时通过 cpu_ptr 查到对应 VkBuffer。

### Compute Shader 设计

每个 workgroup 处理 256 行，每个 invocation 处理 1 行。

**Q4_K shader** (144 bytes/block, 256 elts):
- 读取 d/dmin (FP16 at bytes 0-3)
- 解包 scales[12] (6-bit packed, 8 sub-blocks)
- 提取 qs[128] 半字节，按 4 对子块 (64 elts each) 累加
- 公式: `sum += d*sc*nibble*x - dmin*mn*x`

**Q6_K shader** (210 bytes/block, 256 elts):
- 读取 d (FP16 at byte 208)
- 读取 scales[16] (int8)
- 从 ql[128]/qh[64] 解包 6-bit 值
- 16 组累加后合并: `sum += d * sum(scales[j] * sums[j])`

**Q8_0 shader** (34 bytes/block, 32 elts):
- 读取 d (FP16)
- `sum += d * qs[i] * x[i]`

Push constants 传 `{ n, d, row_bytes }`，descriptor set 绑定 `{ x staging buffer, W device buffer, out staging buffer }`。

### 回退策略

| 场景 | 行为 |
|------|------|
| `USE_VULKAN` 未定义 | stub 全 return -1，CPU path 编译时完全不变 |
| 无 Vulkan driver/设备 | `gpu_init` 失败，`gpu_available()=0` |
| VRAM 不足 | `gpu_init` 失败 |
| 某 tensor 类型不支持 | 不在 lookup 表里，`gpu_try_matmul` 失败，CPU 处理 |

## 实现步骤

### Phase 1: 骨架
- 创建 `gpu.h` + `gpu.c`（Vulkan instance/device 初始化，staging buffer 管理）
- 创建空 shader 文件
- `matmul()` 插入 `#ifdef USE_VULKAN` 短路

### Phase 2: 权重上传
- 实现 `gpu_upload_model()` — 遍历 model weights，创建 DEVICE_LOCAL buffer
- staging copy + fence wait
- 按 cpu_ptr 查找的 lookup 表

### Phase 3: Q4_K compute shader + dispatch
- descriptor set layout 创建（x/W/out 三个 binding）
- push constants (n, d, row_bytes)
- `vkCmdDispatch` + fence 同步
- **验证**: 单 Q4_K matmul 数值 vs CPU

### Phase 4: Q6_K + Q8_0 shader
- 重复 Phase 3 的 pipeline 和 shader

### Phase 5: matmul_dual / matmul_bias
- 加 `#ifdef USE_VULKAN` 短路（各自调两次 `gpu_try_matmul`）
- bias 加法留 CPU（bias 很小，~4K floats）

### Phase 6: 集成
- `picolm.c` 加 `gpu_init`/`gpu_upload_model`/`gpu_shutdown`
- Makefile 加 `vulkan` target
- shader 编译规则

## 验证

1. **编译隔离**: `make native` — 二进制与之前完全一致（USE_VULKAN 未定义）
2. **Fallback**: `make vulkan` + 无 GPU 环境 — 自动回退 CPU，输出 token 一致
3. **数值正确**: 逐 matmul 对比 CPU vs GPU 结果（容忍 1e-5 差异）
4. **端到端**: `./picolm model.gguf -p "Hello" -n 20 -s 42` CPU vs GPU token 一致
