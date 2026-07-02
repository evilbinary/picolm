#ifndef TENSOR_H
#define TENSOR_H

#include "quant.h"

#define MAX_THREADS 16

/* Set the scratch buffer used for row dequantization (embedding lookup, etc).
 * Must be called once at init with a buffer of at least max_row_size floats. */
void tensor_init_scratch(float *buf, int size);

/* Set number of threads for matmul (default: 1) */
void tensor_set_threads(int t);
int  tensor_get_threads(void);

/* Batch matmul: out[n_batch × d] = x[n_batch × n] @ W[n × d]
 * Weight W is read once for all n_batch inputs (prefill optimization). */
void matmul_batch(float *out, const float *x, int n_batch,
                   const void *W, int n, int d, gguf_type_t qtype);

/* Dual batch matmul: same as matmul_batch but two output matrices. */
void matmul_dual_batch(float *out1, float *out2, const float *x, int n_batch,
                        const void *W1, const void *W2,
                        int n, int d, gguf_type_t qtype1, gguf_type_t qtype2);

/* Matrix-vector multiply: out[d] = W[d, n] @ x[n]
 * W is quantized in the given type, stored row-major.
 * Uses fused dequant+dot (no scratch buffer) and optional threading. */
void matmul(float *out, const float *x, const void *W, int n, int d, gguf_type_t qtype);

/* Dual matmul: out1[d] = W1[d,n] @ x[n], out2[d] = W2[d,n] @ x[n]
 * Both matmuls share the same input x and are processed in one thread cycle.
 * W1 and W2 must have the same input dimension n and output dimension d. */
void matmul_dual(float *out1, float *out2, const float *x,
                  const void *W1, const void *W2,
                  int n, int d, gguf_type_t qtype1, gguf_type_t qtype2);

/* Matrix-vector multiply with bias: out[d] = W[d, n] @ x[n] + b[d]
 * If b is NULL, bias is skipped. */
void matmul_bias(float *out, const float *x, const void *W, const void *b,
                 int n, int d, gguf_type_t w_type, gguf_type_t b_type,
                 float *scratch);

/* RMS normalization: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i] */
void rmsnorm(float *out, const float *x, const float *weight, int size, float eps);

/* In-place softmax over x[0..size-1] */
void softmax(float *x, int size);

/* Rotary position encoding using pre-computed cos/sin tables.
 * cos_pos and sin_pos point to the tables for the current position:
 *   cos_pos[i] = cos(pos / freq_base^(2i/head_dim))
 *   sin_pos[i] = sin(pos / freq_base^(2i/head_dim))
 * Each has head_dim/2 entries.
 * type: ROPE_TYPE_LLAMA (pairwise) or ROPE_TYPE_QWEN (interleaved) */
void rope(float *q, float *k, int head_dim, int n_heads, int n_kv_heads,
          const float *cos_pos, const float *sin_pos, int type);

/* In-place SiLU: x[i] = x[i] / (1 + exp(-x[i])) */
void silu(float *x, int size);

/* Element-wise multiply: out[i] = a[i] * b[i] */
void elemwise_mul(float *out, const float *a, const float *b, int size);

/* Vector add in-place: a[i] += b[i] */
void vec_add(float *a, const float *b, int size);

#endif /* TENSOR_H */
