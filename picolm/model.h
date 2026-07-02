#ifndef MODEL_H
#define MODEL_H

#include "quant.h"
#include <stdint.h>
#include <stddef.h>

#define GGUF_MAGIC 0x46554747
#define MAX_LAYERS 64

/* Magic for KV cache files */
#define KVCACHE_MAGIC 0x4B564350  /* "KVCP" */

/* ---- RoPE type: LLaMA uses pairwise, Qwen2 uses interleaved ---- */
typedef enum {
    ROPE_TYPE_LLAMA = 0,  /* pairwise: q[i*2] and q[i*2+1] */
    ROPE_TYPE_QWEN  = 1,  /* interleaved: q[i] and q[i+half] */
} rope_type_t;

/* ---- Configuration ---- */

typedef struct {
    int n_embd;         /* embedding dimension (e.g. 2048) */
    int n_ffn;          /* feed-forward hidden size (e.g. 5632) */
    int n_heads;        /* number of attention heads (e.g. 32) */
    int n_kv_heads;     /* number of KV heads for GQA (e.g. 4) */
    int n_layers;       /* number of transformer layers (e.g. 22) */
    int vocab_size;     /* vocabulary size (e.g. 32000) */
    int max_seq_len;    /* maximum sequence length (e.g. 2048) */
    int head_dim;       /* = n_embd / n_heads */
    float rope_freq_base; /* RoPE theta base (e.g. 10000.0) */
    float rms_norm_eps;   /* RMSNorm epsilon (LLaMA: 1e-5, Qwen2: 1e-7) */
    rope_type_t rope_type; /* RoPE implementation type */
    int alignment;      /* GGUF data alignment */
    gguf_type_t weight_type; /* default weight quantization type */
} model_config_t;

/* ---- Per-layer weight pointers (into mmap) ---- */

typedef struct {
    const void *attn_norm;
    const void *attn_q;
    const void *attn_k;
    const void *attn_v;
    const void *attn_output;
    const void *attn_q_norm;   /* QK-norm (Qwen3): per-head RMSNorm weight [head_dim] */
    const void *attn_k_norm;   /* QK-norm (Qwen3): per-head RMSNorm weight [head_dim] */
    const void *ffn_norm;
    const void *ffn_gate;
    const void *ffn_down;
    const void *ffn_up;
    /* Bias pointers (Qwen2, etc. have these; LLaMA doesn't) */
    const void *attn_q_b;
    const void *attn_k_b;
    const void *attn_v_b;
    const void *attn_output_b;
    /* Per-tensor quantization types */
    gguf_type_t type_attn_norm;
    gguf_type_t type_attn_q;
    gguf_type_t type_attn_k;
    gguf_type_t type_attn_v;
    gguf_type_t type_attn_output;
    gguf_type_t type_ffn_norm;
    gguf_type_t type_ffn_gate;
    gguf_type_t type_ffn_down;
    gguf_type_t type_ffn_up;
    /* Bias quantization types */
    gguf_type_t type_attn_q_norm;
    gguf_type_t type_attn_k_norm;
    gguf_type_t type_attn_q_b;
    gguf_type_t type_attn_k_b;
    gguf_type_t type_attn_v_b;
    gguf_type_t type_attn_output_b;
} layer_weights_t;

typedef struct {
    const void *token_embd;
    gguf_type_t type_token_embd;
    const void *output_norm;
    gguf_type_t type_output_norm;
    const void *output;        /* final output projection (may alias token_embd) */
    gguf_type_t type_output;
    layer_weights_t layers[MAX_LAYERS];
} model_weights_t;

/* ---- Runtime state (pre-allocated buffers) ---- */

typedef struct {
    float *x;            /* current activation [n_embd] */
    float *xb;           /* buffer after norm / attention output [n_embd] */
    float *xb2;          /* second buffer [n_embd] */
    float *q;            /* query vector [n_embd] */
    /* att buffer REMOVED — flash attention uses online softmax */
    float *hb;           /* FFN hidden buffer [n_ffn] */
    float *hb2;          /* FFN hidden buffer 2 [n_ffn] */
    float *logits;       /* output logits [vocab_size] */

    /* KV cache stored as FP16 to halve memory */
    uint16_t *key_cache;    /* [n_layers * max_seq_len * n_kv_heads * head_dim] as FP16 */
    uint16_t *val_cache;    /* [n_layers * max_seq_len * n_kv_heads * head_dim] as FP16 */

    float *dequant_scratch; /* scratch for matmul dequant [max(n_embd, n_ffn)] */

    /* Pre-computed RoPE cos/sin tables [max_seq_len * head_dim/2] */
    float *rope_cos;
    float *rope_sin;

    /* Pre-dequantized norm weights (small, keep in RAM) */
    float *norm_weights;
    float *attn_norm_w[MAX_LAYERS];
    float *ffn_norm_w[MAX_LAYERS];
    float *attn_q_norm_w[MAX_LAYERS];  /* QK-norm (Qwen3) */
    float *attn_k_norm_w[MAX_LAYERS];
    float *output_norm_w;

    /* Pre-dequantized bias buffers (Qwen2, etc.) */
    float *attn_q_bias[MAX_LAYERS];
    float *attn_k_bias[MAX_LAYERS];
    float *attn_v_bias[MAX_LAYERS];
    float *attn_output_bias[MAX_LAYERS];

    /* Single allocation base */
    void *mem_block;
    size_t mem_size;

    /* Separate allocation for FP16 KV cache */
    void *kv_block;
    size_t kv_size;
} run_state_t;

/* ---- Model ---- */

typedef struct {
    model_config_t  config;
    model_weights_t weights;
    run_state_t     state;

    /* mmap bookkeeping */
    void  *mmap_addr;
    size_t mmap_size;
#ifdef _WIN32
    void  *file_handle;
    void  *map_handle;
#else
    int    fd;
#endif

    /* Tokenizer data offsets (filled by GGUF parser, used by tokenizer_load) */
    const void *tok_tokens_data;
    uint64_t    tok_n_tokens;
    const void *tok_scores_data;
    uint64_t    tok_n_scores;
    uint32_t    tok_bos_id;
    uint32_t    tok_eos_id;
    int         tok_add_bos;   /* 1=prepend BOS token on encode (default for LLaMA) */
} model_t;

/* Load a GGUF model file. Returns 0 on success. */
int model_load(model_t *m, const char *path, int max_seq_len);

/* Run one forward pass. Returns pointer to logits[vocab_size]. */
float *model_forward(model_t *m, int token, int pos);

/* Free all resources. */
void model_free(model_t *m);

/* ---- KV cache persistence ---- */

int kvcache_save(const model_t *m, const char *path, int n_pos);
int kvcache_load(model_t *m, const char *path);

/* ---- Speculative Decoding (optional, compile with -DUSE_SPECULATIVE) ---- */

#ifdef USE_SPECULATIVE

/* Draft state for self-speculation */
#define MAX_SPEC_K 16

typedef struct {
    int candidates[MAX_SPEC_K];      /* K draft tokens */
    int K;                           /* number of candidates per round */
    int draft_layers;                /* layers to run in draft mode */
} draft_state_t;

/* Initialize draft state (call once after model_load) */
void draft_init(draft_state_t *ds, const model_t *m);

/* Draft forward: run only the first max_layers of the model.
 * Returns pointer to draft logits [vocab_size]. */
float *model_forward_draft(model_t *m, int token, int pos, int max_layers);

/* Batched verification: run K tokens through the full model in one pass.
 * Fills logits_out[K * vocab_size] with one logit vector per candidate.
 * Returns 0 on success, -1 to fall back to sequential. */
int model_forward_batch(model_t *m, const int *tokens, int pos, int K,
                         float *logits_out);

#else
/* Stub: speculative decoding not compiled in */
#define MAX_SPEC_K 1
#endif

#endif /* MODEL_H */
