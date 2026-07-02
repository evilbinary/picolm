#include "model.h"
#include "tensor.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ---- GGUF metadata value types ---- */
enum {
    GGUF_META_UINT8   = 0,
    GGUF_META_INT8    = 1,
    GGUF_META_UINT16  = 2,
    GGUF_META_INT16   = 3,
    GGUF_META_UINT32  = 4,
    GGUF_META_INT32   = 5,
    GGUF_META_FLOAT32 = 6,
    GGUF_META_BOOL    = 7,
    GGUF_META_STRING  = 8,
    GGUF_META_ARRAY   = 9,
    GGUF_META_UINT64  = 10,
    GGUF_META_INT64   = 11,
    GGUF_META_FLOAT64 = 12,
};

/* ---- Helpers for reading GGUF binary format ---- */

typedef struct {
    const uint8_t *data;
    size_t pos;
    size_t size;
} reader_t;

static uint8_t read_u8(reader_t *r) {
    uint8_t v = r->data[r->pos];
    r->pos += 1;
    return v;
}

static uint16_t read_u16(reader_t *r) {
    uint16_t v;
    memcpy(&v, r->data + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(reader_t *r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static int32_t read_i32(reader_t *r) {
    int32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(reader_t *r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(reader_t *r) {
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

typedef struct { const char *str; uint64_t len; } gguf_str_t;

static gguf_str_t read_gguf_string(reader_t *r) {
    gguf_str_t s;
    s.len = read_u64(r);
    s.str = (const char *)(r->data + r->pos);
    r->pos += s.len;
    return s;
}

static int str_eq(gguf_str_t s, const char *lit) {
    size_t n = strlen(lit);
    return s.len == n && memcmp(s.str, lit, n) == 0;
}

static uint64_t skip_meta_value(reader_t *r, uint32_t vtype, int *is_numeric) {
    *is_numeric = 1;
    switch (vtype) {
        case GGUF_META_UINT8:   return read_u8(r);
        case GGUF_META_INT8:    return (uint64_t)(int64_t)(int8_t)read_u8(r);
        case GGUF_META_UINT16:  return read_u16(r);
        case GGUF_META_INT16:   return (uint64_t)(int64_t)(int16_t)read_u16(r);
        case GGUF_META_UINT32:  return read_u32(r);
        case GGUF_META_INT32:   return (uint64_t)(int64_t)read_i32(r);
        case GGUF_META_UINT64:  return read_u64(r);
        case GGUF_META_INT64:   return read_u64(r);
        case GGUF_META_FLOAT32: { read_f32(r); *is_numeric = 0; return 0; }
        case GGUF_META_FLOAT64: { r->pos += 8; *is_numeric = 0; return 0; }
        case GGUF_META_BOOL:    return read_u8(r);
        case GGUF_META_STRING:  { read_gguf_string(r); *is_numeric = 0; return 0; }
        case GGUF_META_ARRAY: {
            *is_numeric = 0;
            uint32_t arr_type = read_u32(r);
            uint64_t arr_len  = read_u64(r);
            int dummy;
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_meta_value(r, arr_type, &dummy);
            }
            return 0;
        }
        default:
            fprintf(stderr, "Unknown GGUF metadata type: %u\n", vtype);
            exit(1);
    }
}

/* ---- mmap abstraction ---- */

/* Prefault the mmap so inference doesn't pay page-fault cost.
 * Touches one byte per page to bring all pages into RAM upfront.
 * A ~4GB model has ~1M pages; prefaulting takes a few seconds at load
 * but prevents a 30s+ page-fault storm during first inference. */
static void prefault_mmap(const void *addr, size_t size) {
    const volatile char *p = (const volatile char *)addr;
    for (size_t off = 0; off < size; off += 4096)
        (void)p[off];
#ifdef __linux__
    /* Hint transparent huge pages to reduce TLB misses on large model scans.
     * The kernel will try to back this region with 2MB pages instead of 4KB. */
    madvise((void*)addr, size, MADV_HUGEPAGE);
    madvise((void*)addr, size, MADV_WILLNEED | MADV_SEQUENTIAL);
#endif
}

#ifdef _WIN32
/* Try to load the model into 2MB huge pages to reduce TLB misses.
 * This reduces TLB entries from ~1.1M (4KB) to ~2200 (2MB) for a 4.5GB model.
 * Requires SeLockMemoryPrivilege (configurable via Local Security Policy).
 * Falls back silently if huge pages are unavailable. */
static int mmap_file_huge(model_t *m, const char *path) {
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) return -1;

    LARGE_INTEGER fsize;
    GetFileSizeEx(fh, &fsize);
    size_t file_size = (size_t)fsize.QuadPart;
    SIZE_T large_min = GetLargePageMinimum();
    if (large_min == 0) { CloseHandle(fh); return -1; }

    SIZE_T alloc_size = ((file_size + large_min - 1) / large_min) * large_min;

    HANDLE hToken;
    TOKEN_PRIVILEGES tp;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &hToken)) {
        tp.PrivilegeCount = 1;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        LookupPrivilegeValueA(NULL, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid);
        AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(tp), NULL, NULL);
        CloseHandle(hToken);
    }

    void *buf = VirtualAlloc(NULL, alloc_size,
                             MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                             PAGE_READWRITE);
    if (!buf) { CloseHandle(fh); return -1; }

    uint8_t *p = (uint8_t *)buf;
    size_t remaining = file_size;
    while (remaining > 0) {
        DWORD chunk = (DWORD)(remaining > 0x7FFFFFFF ? 0x7FFFFFFF : remaining);
        DWORD read;
        if (!ReadFile(fh, p, chunk, &read, NULL) || read != chunk) {
            VirtualFree(buf, 0, MEM_RELEASE);
            CloseHandle(fh);
            return -1;
        }
        p += read;
        remaining -= read;
    }
    CloseHandle(fh);

    m->mmap_addr  = buf;
    m->file_handle = NULL;
    m->map_handle  = (void*)(intptr_t)-1;
    m->mmap_size   = file_size;
    fprintf(stderr, "Model loaded: %zu MB (huge pages, 2MB)\n", file_size >> 20);
    return 0;
}
#endif

static int mmap_file(model_t *m, const char *path) {
#ifdef _WIN32
    /* Huge pages: best-effort, reduces TLB misses on large models */
    if (mmap_file_huge(m, path) == 0) return 0;

    /* Normal file mapping — demand-paged, no extra memory overhead */
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return -1;
    }

    LARGE_INTEGER fsize;
    GetFileSizeEx(fh, &fsize);
    m->mmap_size = (size_t)fsize.QuadPart;

    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        fprintf(stderr, "CreateFileMapping failed\n");
        CloseHandle(fh);
        return -1;
    }

    void *addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        fprintf(stderr, "MapViewOfFile failed\n");
        CloseHandle(mh);
        CloseHandle(fh);
        return -1;
    }

    m->mmap_addr  = addr;
    m->file_handle = fh;
    m->map_handle  = mh;
    prefault_mmap(addr, m->mmap_size);
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return -1;
    }

    struct stat st;
    fstat(fd, &st);
    m->mmap_size = (size_t)st.st_size;

    void *addr = mmap(NULL, m->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        close(fd);
        return -1;
    }

    m->mmap_addr = addr;
    m->fd = fd;
    prefault_mmap(addr, m->mmap_size);
#endif
    return 0;
}

static void munmap_file(model_t *m) {
    if (!m->mmap_addr) return;
#ifdef _WIN32
    if (m->map_handle == (void*)(intptr_t)-1) {
        /* Allocated via VirtualAlloc huge pages */
        VirtualFree(m->mmap_addr, 0, MEM_RELEASE);
    } else {
        UnmapViewOfFile(m->mmap_addr);
        CloseHandle(m->map_handle);
        CloseHandle(m->file_handle);
    }
#else
    munmap(m->mmap_addr, m->mmap_size);
    close(m->fd);
#endif
    m->mmap_addr = NULL;
}

/* ---- GGUF Parser ---- */

static int parse_gguf(model_t *m, int max_seq_len) {
    reader_t r = { .data = (const uint8_t *)m->mmap_addr, .pos = 0, .size = m->mmap_size };
    model_config_t *cfg = &m->config;

    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
        return -1;
    }

    uint32_t version = read_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", version);
        return -1;
    }

    uint64_t n_tensors  = read_u64(&r);
    uint64_t n_metadata = read_u64(&r);

    cfg->alignment = 32;
    cfg->rope_freq_base = 10000.0f;
    cfg->rms_norm_eps = 1e-5f;  /* LLaMA default, Qwen2 uses 1e-7 */
    cfg->rope_type = ROPE_TYPE_LLAMA;  /* LLaMA default, Qwen2 uses interleaved */
    cfg->max_seq_len = 2048;
    cfg->weight_type = GGUF_TYPE_F16;
    m->tok_bos_id = 1;
    m->tok_eos_id = 2;
    m->tok_add_bos = 1;

    for (uint64_t i = 0; i < n_metadata; i++) {
        gguf_str_t key = read_gguf_string(&r);
        uint32_t vtype = read_u32(&r);

        if (str_eq(key, "llama.embedding_length") || str_eq(key, "general.embedding_length")
            || str_eq(key, "qwen2.embedding_length") || str_eq(key, "qwen3.embedding_length")) {
            int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.feed_forward_length") || str_eq(key, "general.feed_forward_length")
            || str_eq(key, "qwen2.feed_forward_length") || str_eq(key, "qwen3.feed_forward_length")) {
            int dummy; cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count")
            || str_eq(key, "qwen2.attention.head_count") || str_eq(key, "qwen3.attention.head_count")) {
            int dummy; cfg->n_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count_kv")
            || str_eq(key, "qwen2.attention.head_count_kv") || str_eq(key, "qwen3.attention.head_count_kv")) {
            int dummy; cfg->n_kv_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.block_count")
            || str_eq(key, "qwen2.block_count") || str_eq(key, "qwen3.block_count")) {
            int dummy; cfg->n_layers = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.context_length")
            || str_eq(key, "qwen2.context_length") || str_eq(key, "qwen3.context_length")) {
            int dummy; cfg->max_seq_len = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.rope.freq_base")
            || str_eq(key, "qwen2.rope.freq_base") || str_eq(key, "qwen3.rope.freq_base")) {
            if (vtype == GGUF_META_FLOAT32) {
                cfg->rope_freq_base = read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "llama.attention.layer_norm_rms_epsilon")
            || str_eq(key, "qwen2.attention.layer_norm_rms_epsilon")
            || str_eq(key, "qwen3.attention.layer_norm_rms_epsilon")) {
            if (vtype == GGUF_META_FLOAT32) {
                cfg->rms_norm_eps = read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "general.alignment")) {
            int dummy; cfg->alignment = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "general.architecture")) {
            /* Detect model architecture for RoPE type selection */
            if (vtype == GGUF_META_STRING) {
                gguf_str_t arch = read_gguf_string(&r);
                if (str_eq(arch, "qwen2") || str_eq(arch, "qwen3")) {
                    cfg->rope_type = ROPE_TYPE_QWEN;
                }
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "llama.vocab_size")
            || str_eq(key, "qwen2.vocab_size") || str_eq(key, "qwen3.vocab_size")) {
            int dummy; cfg->vocab_size = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.bos_token_id")) {
            int dummy; m->tok_bos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.eos_token_id")) {
            int dummy; m->tok_eos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.add_bos_token")) {
            int dummy; m->tok_add_bos = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.tokens")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                m->tok_tokens_data = r.data + r.pos;
                m->tok_n_tokens = arr_len;
                int dummy;
                for (uint64_t j = 0; j < arr_len; j++) {
                    skip_meta_value(&r, arr_type, &dummy);
                }
            }
        } else if (str_eq(key, "tokenizer.ggml.scores")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                (void)arr_type;
                m->tok_scores_data = r.data + r.pos;
                m->tok_n_scores = arr_len;
                r.pos += arr_len * 4;
            }
        } else {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        }
    }

    if (max_seq_len > 0 && max_seq_len < cfg->max_seq_len) {
        cfg->max_seq_len = max_seq_len;
    }
    cfg->head_dim = cfg->n_embd / cfg->n_heads;

    /* Parse tensor info entries */
    typedef struct {
        gguf_str_t name;
        uint32_t   n_dims;
        uint64_t   dims[4];
        uint32_t   type;
        uint64_t   offset;
    } tensor_info_t;

    tensor_info_t *tinfos = (tensor_info_t *)malloc(n_tensors * sizeof(tensor_info_t));
    if (!tinfos) { fprintf(stderr, "OOM allocating tensor info\n"); return -1; }

    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfos[i].name   = read_gguf_string(&r);
        tinfos[i].n_dims = read_u32(&r);
        for (uint32_t d = 0; d < tinfos[i].n_dims; d++) {
            tinfos[i].dims[d] = read_u64(&r);
        }
        tinfos[i].type   = read_u32(&r);
        tinfos[i].offset = read_u64(&r);
    }

    size_t alignment = (size_t)cfg->alignment;
    size_t tensor_data_base = (r.pos + alignment - 1) & ~(alignment - 1);

    model_weights_t *w = &m->weights;
    memset(w, 0, sizeof(*w));

    for (uint64_t i = 0; i < n_tensors; i++) {
        const void *ptr = (const uint8_t *)m->mmap_addr + tensor_data_base + tinfos[i].offset;
        gguf_type_t qtype = (gguf_type_t)tinfos[i].type;

        if (str_eq(tinfos[i].name, "token_embd.weight")) {
            w->token_embd = ptr; w->type_token_embd = qtype;
        } else if (str_eq(tinfos[i].name, "output_norm.weight")) {
            w->output_norm = ptr; w->type_output_norm = qtype;
        } else if (str_eq(tinfos[i].name, "output.weight")) {
            w->output = ptr; w->type_output = qtype;
        } else {
            int layer = -1;
            char suffix[64] = {0};

            if (tinfos[i].name.len > 4 && memcmp(tinfos[i].name.str, "blk.", 4) == 0) {
                const char *p = tinfos[i].name.str + 4;
                const char *end = tinfos[i].name.str + tinfos[i].name.len;
                layer = 0;
                while (p < end && *p >= '0' && *p <= '9') {
                    layer = layer * 10 + (*p - '0');
                    p++;
                }
                if (p < end && *p == '.') {
                    p++;
                    size_t slen = (size_t)(end - p);
                    if (slen < sizeof(suffix)) {
                        memcpy(suffix, p, slen);
                        suffix[slen] = '\0';
                    }
                }
            }

            if (layer >= 0 && layer < MAX_LAYERS) {
                layer_weights_t *lw = &w->layers[layer];
                if (strcmp(suffix, "attn_norm.weight") == 0) {
                    lw->attn_norm = ptr; lw->type_attn_norm = qtype;
                } else if (strcmp(suffix, "attn_q.weight") == 0) {
                    lw->attn_q = ptr; lw->type_attn_q = qtype;
                } else if (strcmp(suffix, "attn_k.weight") == 0) {
                    lw->attn_k = ptr; lw->type_attn_k = qtype;
                } else if (strcmp(suffix, "attn_v.weight") == 0) {
                    lw->attn_v = ptr; lw->type_attn_v = qtype;
                } else if (strcmp(suffix, "attn_output.weight") == 0) {
                    lw->attn_output = ptr; lw->type_attn_output = qtype;
                } else if (strcmp(suffix, "attn_q_norm.weight") == 0) {
                    lw->attn_q_norm = ptr; lw->type_attn_q_norm = qtype;
                } else if (strcmp(suffix, "attn_k_norm.weight") == 0) {
                    lw->attn_k_norm = ptr; lw->type_attn_k_norm = qtype;
                } else if (strcmp(suffix, "attn_q.bias") == 0) {
                    lw->attn_q_b = ptr; lw->type_attn_q_b = qtype;
                } else if (strcmp(suffix, "attn_k.bias") == 0) {
                    lw->attn_k_b = ptr; lw->type_attn_k_b = qtype;
                } else if (strcmp(suffix, "attn_v.bias") == 0) {
                    lw->attn_v_b = ptr; lw->type_attn_v_b = qtype;
                } else if (strcmp(suffix, "attn_output.bias") == 0) {
                    lw->attn_output_b = ptr; lw->type_attn_output_b = qtype;
                } else if (strcmp(suffix, "ffn_norm.weight") == 0) {
                    lw->ffn_norm = ptr; lw->type_ffn_norm = qtype;
                } else if (strcmp(suffix, "ffn_gate.weight") == 0) {
                    lw->ffn_gate = ptr; lw->type_ffn_gate = qtype;
                } else if (strcmp(suffix, "ffn_down.weight") == 0) {
                    lw->ffn_down = ptr; lw->type_ffn_down = qtype;
                } else if (strcmp(suffix, "ffn_up.weight") == 0) {
                    lw->ffn_up = ptr; lw->type_ffn_up = qtype;
                }
            }
        }
    }

    if (!w->output) {
        w->output = w->token_embd;
        w->type_output = w->type_token_embd;
    }

    if (cfg->vocab_size == 0) {
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (str_eq(tinfos[i].name, "token_embd.weight")) {
                if (tinfos[i].n_dims >= 2) {
                    int d0 = (int)tinfos[i].dims[0];
                    int d1 = (int)tinfos[i].dims[1];
                    cfg->vocab_size = (d0 == cfg->n_embd) ? d1 : d0;
                }
                break;
            }
        }
    }
    if (cfg->vocab_size == 0 && m->tok_n_tokens > 0) {
        cfg->vocab_size = (int)m->tok_n_tokens;
    }

    cfg->weight_type = w->layers[0].type_attn_q;

    fprintf(stderr, "Model config:\n");
    fprintf(stderr, "  n_embd=%d, n_ffn=%d, n_heads=%d, n_kv_heads=%d\n",
            cfg->n_embd, cfg->n_ffn, cfg->n_heads, cfg->n_kv_heads);
    fprintf(stderr, "  n_layers=%d, vocab_size=%d, max_seq=%d\n",
            cfg->n_layers, cfg->vocab_size, cfg->max_seq_len);
    fprintf(stderr, "  head_dim=%d, rope_base=%.1f, rope_type=%s\n",
            cfg->head_dim, cfg->rope_freq_base,
            cfg->rope_type == ROPE_TYPE_QWEN ? "qwen" : "llama");

    free(tinfos);
    return 0;
}

/* ---- Pre-compute RoPE cos/sin lookup tables ---- */

static void init_rope_tables(run_state_t *s, const model_config_t *c) {
    int half_dim = c->head_dim / 2;
    for (int pos = 0; pos < c->max_seq_len; pos++) {
        float *cos_row = s->rope_cos + (size_t)pos * half_dim;
        float *sin_row = s->rope_sin + (size_t)pos * half_dim;
        for (int i = 0; i < half_dim; i++) {
            float theta = (float)pos / powf(c->rope_freq_base, (float)(2 * i) / (float)c->head_dim);
            cos_row[i] = cosf(theta);
            sin_row[i] = sinf(theta);
        }
    }
}

/* ---- Buffer allocation ---- */

static int allocate_run_state(model_t *m) {
    model_config_t *c = &m->config;
    run_state_t *s = &m->state;

    int kv_dim = c->n_kv_heads * c->head_dim;
    int half_dim = c->head_dim / 2;

    /* Calculate sizes for float buffers */
    size_t sz_x      = (size_t)c->n_embd * sizeof(float);
    size_t sz_xb     = (size_t)c->n_embd * sizeof(float);
    size_t sz_xb2    = (size_t)c->n_embd * sizeof(float);
    size_t sz_q      = (size_t)c->n_embd * sizeof(float);
    /* att buffer removed (flash attention) */
    size_t sz_hb     = (size_t)c->n_ffn * sizeof(float);
    size_t sz_hb2    = (size_t)c->n_ffn * sizeof(float);
    size_t sz_logits = (size_t)c->vocab_size * sizeof(float);

    int scratch_dim = c->n_embd > c->n_ffn ? c->n_embd : c->n_ffn;
    if (c->vocab_size > scratch_dim) scratch_dim = c->vocab_size;
    size_t sz_scratch = (size_t)scratch_dim * sizeof(float);

    /* RoPE tables: cos and sin for each (position, dim_pair) */
    size_t sz_rope = (size_t)c->max_seq_len * half_dim * sizeof(float) * 2;

    /* Norm weights: (n_layers * 2 + 1) * n_embd + n_layers * head_dim * 2 (QK-norm) */
    size_t n_norm = (size_t)(c->n_layers * 2 + 1) * c->n_embd
                  + (size_t)c->n_layers * c->head_dim * 2;
    size_t sz_norm = n_norm * sizeof(float);

    /* Bias buffers: per-layer Q/K/V/Output biases (Qwen2, etc.) */
    int q_dim = c->n_heads * c->head_dim;
    size_t sz_bias = (size_t)c->n_layers * (q_dim + kv_dim + kv_dim + q_dim) * sizeof(float);

    size_t total = sz_x + sz_xb + sz_xb2 + sz_q +
                   sz_hb + sz_hb2 + sz_logits +
                   sz_scratch + sz_rope + sz_norm + sz_bias;

    /* FP16 KV cache: separate allocation */
    size_t kv_elements = (size_t)c->n_layers * c->max_seq_len * kv_dim;
    size_t sz_kv = kv_elements * sizeof(uint16_t) * 2; /* key + val */

    fprintf(stderr, "Allocating %.2f MB for runtime state (+ %.2f MB FP16 KV cache)\n",
            (double)total / (1024.0 * 1024.0),
            (double)sz_kv / (1024.0 * 1024.0));

    s->mem_block = calloc(1, total);
    if (!s->mem_block) {
        fprintf(stderr, "OOM: cannot allocate %zu bytes\n", total);
        return -1;
    }
    s->mem_size = total + sz_kv;

    /* Allocate FP16 KV cache separately */
    s->kv_block = calloc(1, sz_kv);
    if (!s->kv_block) {
        fprintf(stderr, "OOM: cannot allocate %zu bytes for KV cache\n", sz_kv);
        free(s->mem_block);
        return -1;
    }
    s->kv_size = sz_kv;

    /* Carve float pointers */
    float *p = (float *)s->mem_block;
    s->x      = p; p += c->n_embd;
    s->xb     = p; p += c->n_embd;
    s->xb2    = p; p += c->n_embd;
    s->q      = p; p += c->n_embd;
    s->hb     = p; p += c->n_ffn;
    s->hb2    = p; p += c->n_ffn;
    s->logits = p; p += c->vocab_size;
    s->dequant_scratch = p; p += scratch_dim;

    /* RoPE tables */
    s->rope_cos = p; p += (size_t)c->max_seq_len * half_dim;
    s->rope_sin = p; p += (size_t)c->max_seq_len * half_dim;

    /* Norm weights */
    s->norm_weights = p;

    /* FP16 KV cache pointers */
    uint16_t *kp = (uint16_t *)s->kv_block;
    s->key_cache = kp; kp += kv_elements;
    s->val_cache = kp;

    /* Pre-dequantize norm weights */
    float *nw = s->norm_weights;
    for (int l = 0; l < c->n_layers; l++) {
        s->attn_norm_w[l] = nw;
        dequantize_row(m->weights.layers[l].attn_norm, nw, c->n_embd,
                       m->weights.layers[l].type_attn_norm);
        nw += c->n_embd;

        s->ffn_norm_w[l] = nw;
        dequantize_row(m->weights.layers[l].ffn_norm, nw, c->n_embd,
                       m->weights.layers[l].type_ffn_norm);
        nw += c->n_embd;

        /* Qwen3 QK-norm weights (per-head, if present) */
        s->attn_q_norm_w[l] = nw;
        if (m->weights.layers[l].attn_q_norm)
            dequantize_row(m->weights.layers[l].attn_q_norm, nw, c->head_dim,
                           m->weights.layers[l].type_attn_q_norm);
        else
            memset(nw, 0, (size_t)c->head_dim * sizeof(float));
        nw += c->head_dim;

        s->attn_k_norm_w[l] = nw;
        if (m->weights.layers[l].attn_k_norm)
            dequantize_row(m->weights.layers[l].attn_k_norm, nw, c->head_dim,
                           m->weights.layers[l].type_attn_k_norm);
        else
            memset(nw, 0, (size_t)c->head_dim * sizeof(float));
        nw += c->head_dim;
    }
    s->output_norm_w = nw;
    dequantize_row(m->weights.output_norm, nw, c->n_embd,
                   m->weights.type_output_norm);

    /* Pre-dequantize bias buffers (if present) */
    float *bp = nw + c->n_embd;
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &m->weights.layers[l];
        s->attn_q_bias[l] = bp;
        if (lw->attn_q_b)
            dequantize_row(lw->attn_q_b, bp, q_dim, lw->type_attn_q_b);
        else
            memset(bp, 0, (size_t)q_dim * sizeof(float));
        bp += q_dim;

        s->attn_k_bias[l] = bp;
        if (lw->attn_k_b)
            dequantize_row(lw->attn_k_b, bp, kv_dim, lw->type_attn_k_b);
        else
            memset(bp, 0, (size_t)kv_dim * sizeof(float));
        bp += kv_dim;

        s->attn_v_bias[l] = bp;
        if (lw->attn_v_b)
            dequantize_row(lw->attn_v_b, bp, kv_dim, lw->type_attn_v_b);
        else
            memset(bp, 0, (size_t)kv_dim * sizeof(float));
        bp += kv_dim;

        s->attn_output_bias[l] = bp;
        if (lw->attn_output_b)
            dequantize_row(lw->attn_output_b, bp, q_dim, lw->type_attn_output_b);
        else
            memset(bp, 0, (size_t)q_dim * sizeof(float));
        bp += q_dim;
    }

    /* Init tensor scratch */
    tensor_init_scratch(s->dequant_scratch, scratch_dim);

    /* Pre-compute RoPE tables (eliminates powf/cosf/sinf from hot path) */
    init_rope_tables(s, c);

    return 0;
}

/* ---- Public API ---- */

int model_load(model_t *m, const char *path, int max_seq_len) {
    memset(m, 0, sizeof(*m));

    if (mmap_file(m, path) != 0) return -1;
    if (parse_gguf(m, max_seq_len) != 0) return -1;
    if (allocate_run_state(m) != 0) return -1;

    return 0;
}

/* ================================================================
 * Forward pass with:
 *   - FP16 KV cache (halves memory bandwidth in attention)
 *   - Flash attention / online softmax (single pass, no score buffer)
 *   - Pre-computed RoPE tables (table lookup instead of trig)
 * ================================================================ */

/* Forward declaration for internal helper used by draft/verify */
static float *model_forward_ex(model_t *m, int token, int pos, int max_layers);

float *model_forward(model_t *m, int token, int pos) {
    return model_forward_ex(m, token, pos, m->config.n_layers);
}

static float *model_forward_ex(model_t *m, int token, int pos, int max_layers) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;

    int dim    = c->n_embd;
    int n_ffn  = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / n_kv_heads;
    int seq_len = c->max_seq_len;
    int half_dim = head_dim / 2;

    /* RoPE table pointers for this position */
    const float *cos_pos = s->rope_cos + (size_t)pos * half_dim;
    const float *sin_pos = s->rope_sin + (size_t)pos * half_dim;

    /* 1. Embedding lookup */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        dequantize_row(embd_row, s->x, dim, w->type_token_embd);
    }

    /* 2. Transformer layers */
    for (int l = 0; l < max_layers; l++) {
        layer_weights_t *lw = &w->layers[l];

        /* ---- Attention ---- */
        rmsnorm(s->xb, s->x, s->attn_norm_w[l], dim, c->rms_norm_eps);

        /* QKV projections (with optional bias for Qwen2, etc.) */
        int q_dim = n_heads * head_dim;
        matmul_bias(s->q, s->xb, lw->attn_q, s->attn_q_bias[l],
                    dim, q_dim, lw->type_attn_q, lw->type_attn_q_b, s->dequant_scratch);

        /* K+V fused: load both weight matrices in one traversal */
        float *k_tmp = s->xb2;
        float *v_tmp = s->hb; /* kv_dim fits; n_ffn=18944 >> 512 */
        matmul_dual(k_tmp, v_tmp, s->xb,
                    lw->attn_k, lw->attn_v,
                    dim, kv_dim, lw->type_attn_k, lw->type_attn_v);

        /* K bias (if present, e.g. Qwen2) */
        if (lw->attn_k_b) {
            dequantize_row(lw->attn_k_b, s->dequant_scratch, kv_dim, lw->type_attn_k_b);
            vec_add(k_tmp, s->dequant_scratch, kv_dim);
        }
        /* V bias (if present) */
        if (lw->attn_v_b) {
            dequantize_row(lw->attn_v_b, s->dequant_scratch, kv_dim, lw->type_attn_v_b);
            vec_add(v_tmp, s->dequant_scratch, kv_dim);
        }

        uint16_t *kcache_layer = s->key_cache + (size_t)l * seq_len * kv_dim;
        uint16_t *vcache_layer = s->val_cache + (size_t)l * seq_len * kv_dim;
        uint16_t *key_pos_fp16 = kcache_layer + (size_t)pos * kv_dim;
        uint16_t *val_pos_fp16 = vcache_layer + (size_t)pos * kv_dim;

        /* QK-norm (Qwen3): per-head RMSNorm applied before RoPE */
        if (lw->attn_q_norm) {
            float *qnw = s->attn_q_norm_w[l];
            float *knw = s->attn_k_norm_w[l];
            for (int h = 0; h < n_heads; h++)
                rmsnorm(s->q + h * head_dim, s->q + h * head_dim,
                        qnw, head_dim, c->rms_norm_eps);
            for (int h = 0; h < n_kv_heads; h++)
                rmsnorm(k_tmp + h * head_dim, k_tmp + h * head_dim,
                        knw, head_dim, c->rms_norm_eps);
        }

        /* Apply RoPE to Q and K (using pre-computed tables) */
        rope(s->q, k_tmp, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos, c->rope_type);

        /* Store K and V to FP16 cache */
        for (int d = 0; d < kv_dim; d++) {
            key_pos_fp16[d] = fp32_to_fp16(k_tmp[d]);
            val_pos_fp16[d] = fp32_to_fp16(v_tmp[d]);
        }

        /* ---- Flash Attention (online softmax) ----
         *
         * Instead of materializing the full [n_heads * seq_len] score array,
         * compute attention in a single pass using the online softmax trick:
         *
         *   max_s = -inf, sum_exp = 0, acc[d] = 0
         *   for each cached position t:
         *     s = dot(Q_h, K_t) / sqrt(d)
         *     if s > max_s:
         *       correction = exp(max_s - s)
         *       acc *= correction, sum_exp *= correction
         *       sum_exp += 1, acc += V_t
         *       max_s = s
         *     else:
         *       w = exp(s - max_s)
         *       sum_exp += w, acc += w * V_t
         *   result = acc / sum_exp
         *
         * This saves memory (no att[] buffer) and is more cache-friendly.
         */
        for (int h = 0; h < n_heads; h++) {
            float *qh = s->q + h * head_dim;
            int kv_h = h / kv_mul;
            float *xbh = s->xb + h * head_dim;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            /* Accumulator for weighted V values */
            float acc[256]; /* head_dim is typically 64-128 */
            memset(acc, 0, (size_t)head_dim * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                /* Compute score: dot(Q_h, K_t) / sqrt(head_dim) */
                const uint16_t *kt = kcache_layer + (size_t)t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += qh[d] * fp16_to_fp32(kt[d]);
                }
                score /= sqrtf((float)head_dim);

                /* Online softmax update */
                const uint16_t *vt = vcache_layer + (size_t)t * kv_dim + kv_h * head_dim;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    for (int d = 0; d < head_dim; d++) {
                        acc[d] = acc[d] * correction + fp16_to_fp32(vt[d]);
                    }
                    max_score = score;
                } else {
                    float w = expf(score - max_score);
                    sum_exp += w;
                    for (int d = 0; d < head_dim; d++) {
                        acc[d] += w * fp16_to_fp32(vt[d]);
                    }
                }
            }

            /* Normalize */
            float inv_sum = 1.0f / sum_exp;
            for (int d = 0; d < head_dim; d++) {
                xbh[d] = acc[d] * inv_sum;
            }
        }

        /* Output projection */
        matmul_bias(s->xb2, s->xb, lw->attn_output, s->attn_output_bias[l],
                    q_dim, dim, lw->type_attn_output, lw->type_attn_output_b, s->dequant_scratch);
        vec_add(s->x, s->xb2, dim);

        /* ---- FFN (SwiGLU) ---- */
        rmsnorm(s->xb, s->x, s->ffn_norm_w[l], dim, c->rms_norm_eps);

#ifdef USE_BARRIER_POOL
        matmul_dual(s->hb, s->hb2, s->xb,
                    lw->ffn_gate, lw->ffn_up,
                    dim, n_ffn, lw->type_ffn_gate, lw->type_ffn_up);
#else
        matmul(s->hb,  s->xb, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);
        matmul(s->hb2, s->xb, lw->ffn_up,   dim, n_ffn, lw->type_ffn_up);
#endif

        silu(s->hb, n_ffn);
        elemwise_mul(s->hb, s->hb, s->hb2, n_ffn);

        matmul(s->xb, s->hb, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
        vec_add(s->x, s->xb, dim);
    }

    /* 3. Final RMSNorm */
    rmsnorm(s->x, s->x, s->output_norm_w, dim, c->rms_norm_eps);

    /* 4. Output projection -> logits */
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);

    return s->logits;
}

void model_free(model_t *m) {
    if (m->state.mem_block) {
        free(m->state.mem_block);
        m->state.mem_block = NULL;
    }
    if (m->state.kv_block) {
        free(m->state.kv_block);
        m->state.kv_block = NULL;
    }
    munmap_file(m);
}

void model_forward_prefill(model_t *m, const int *tokens, int n_tokens, int start_pos) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;

    int dim     = c->n_embd;
    int n_ffn   = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim  = c->head_dim;
    int kv_dim    = n_kv_heads * head_dim;
    int kv_mul    = n_heads / n_kv_heads;
    int q_dim     = n_heads * head_dim;
    int seq_len   = c->max_seq_len;

    size_t bs = (size_t)n_tokens;

    /* Allocate batch workspace (one contiguous block) */
    size_t sz_x  = bs * (size_t)dim;
    size_t sz_xb = bs * (size_t)dim;
    size_t sz_xb2 = bs * (size_t)dim;
    size_t sz_q  = bs * (size_t)q_dim;
    size_t sz_kv = bs * (size_t)kv_dim;
    size_t sz_hb = bs * (size_t)n_ffn;
    size_t total = sz_x + sz_xb + sz_xb2 + sz_q + sz_kv * 2 + sz_hb * 2;

    float *buf = (float *)malloc(total * sizeof(float));
    if (!buf) { fprintf(stderr, "OOM: prefill batch buffer\n"); exit(1); }

    float *x_batch  = buf;  float *p = x_batch + sz_x;
    float *xb_batch = p;  p += sz_xb;
    float *xb2_batch = p; p += sz_xb2;
    float *q_batch  = p;  p += sz_q;
    float *k_batch  = p;  p += sz_kv;
    float *v_batch  = p;  p += sz_kv;
    float *hb_batch = p;  p += sz_hb;
    float *hb2_batch = p;

    /* 1. Embedding lookup for all tokens */
    for (int bi = 0; bi < n_tokens; bi++) {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)tokens[bi] * row_bytes;
        dequantize_row(embd_row, x_batch + bi * dim, dim, w->type_token_embd);
    }

    /* 2. Transformer layers */
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];

        /* a. Batch RMSNorm → xb_batch */
        for (int bi = 0; bi < n_tokens; bi++)
            rmsnorm(xb_batch + bi * dim, x_batch + bi * dim,
                    s->attn_norm_w[l], dim, c->rms_norm_eps);

        /* b. Batch Q projection */
        matmul_batch(q_batch, xb_batch, n_tokens,
                     lw->attn_q, dim, q_dim, lw->type_attn_q);

        /* c. Batch K+V projection */
        matmul_dual_batch(k_batch, v_batch, xb_batch, n_tokens,
                          lw->attn_k, lw->attn_v, dim, kv_dim,
                          lw->type_attn_k, lw->type_attn_v);

        /* d. Biases (per-tensor dequant + per-token vec_add) */
        if (lw->attn_q_b) {
            dequantize_row(lw->attn_q_b, s->dequant_scratch, q_dim, lw->type_attn_q_b);
            for (int bi = 0; bi < n_tokens; bi++)
                vec_add(q_batch + bi * q_dim, s->dequant_scratch, q_dim);
        }
        if (lw->attn_k_b) {
            dequantize_row(lw->attn_k_b, s->dequant_scratch, kv_dim, lw->type_attn_k_b);
            for (int bi = 0; bi < n_tokens; bi++)
                vec_add(k_batch + bi * kv_dim, s->dequant_scratch, kv_dim);
        }
        if (lw->attn_v_b) {
            dequantize_row(lw->attn_v_b, s->dequant_scratch, kv_dim, lw->type_attn_v_b);
            for (int bi = 0; bi < n_tokens; bi++)
                vec_add(v_batch + bi * kv_dim, s->dequant_scratch, kv_dim);
        }

        /* e. Per-position: QK-norm, RoPE, KV store, flash attention */
        for (int bi = 0; bi < n_tokens; bi++) {
            int pos = start_pos + bi;
            float *q_pos = q_batch + bi * q_dim;
            float *k_pos = k_batch + bi * kv_dim;
            float *v_pos = v_batch + bi * kv_dim;

            const float *cos_pos = s->rope_cos + (size_t)pos * (head_dim / 2);
            const float *sin_pos = s->rope_sin + (size_t)pos * (head_dim / 2);

            /* QK-norm (Qwen3) */
            if (lw->attn_q_norm) {
                for (int h = 0; h < n_heads; h++)
                    rmsnorm(q_pos + h * head_dim, q_pos + h * head_dim,
                            s->attn_q_norm_w[l], head_dim, c->rms_norm_eps);
                for (int h = 0; h < n_kv_heads; h++)
                    rmsnorm(k_pos + h * head_dim, k_pos + h * head_dim,
                            s->attn_k_norm_w[l], head_dim, c->rms_norm_eps);
            }

            /* RoPE */
            rope(q_pos, k_pos, head_dim, n_heads, n_kv_heads,
                 cos_pos, sin_pos, c->rope_type);

            /* KV store */
            uint16_t *kcache_l = s->key_cache + (size_t)l * seq_len * kv_dim;
            uint16_t *vcache_l = s->val_cache + (size_t)l * seq_len * kv_dim;
            uint16_t *kfp16 = kcache_l + (size_t)pos * kv_dim;
            uint16_t *vfp16 = vcache_l + (size_t)pos * kv_dim;
            for (int d = 0; d < kv_dim; d++) {
                kfp16[d] = fp32_to_fp16(k_pos[d]);
                vfp16[d] = fp32_to_fp16(v_pos[d]);
            }

            /* Flash attention */
            for (int h = 0; h < n_heads; h++) {
                float *qh = q_pos + h * head_dim;
                int kv_h = h / kv_mul;
                float *xbh = xb_batch + bi * dim + h * head_dim;

                float max_score = -1e30f;
                float sum_exp = 0.0f;
                float acc[256];
                memset(acc, 0, (size_t)head_dim * sizeof(float));

                for (int t = 0; t <= pos; t++) {
                    const uint16_t *kt = kcache_l + (size_t)t * kv_dim + kv_h * head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        score += qh[d] * fp16_to_fp32(kt[d]);
                    score /= sqrtf((float)head_dim);

                    const uint16_t *vt = vcache_l + (size_t)t * kv_dim + kv_h * head_dim;
                    if (score > max_score) {
                        float correction = expf(max_score - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        for (int d = 0; d < head_dim; d++)
                            acc[d] = acc[d] * correction + fp16_to_fp32(vt[d]);
                        max_score = score;
                    } else {
                        float w = expf(score - max_score);
                        sum_exp += w;
                        for (int d = 0; d < head_dim; d++)
                            acc[d] += w * fp16_to_fp32(vt[d]);
                    }
                }

                float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; d++)
                    xbh[d] = acc[d] * inv_sum;
            }
        }

        /* f. Batch output projection */
        matmul_batch(xb2_batch, xb_batch, n_tokens,
                     lw->attn_output, q_dim, dim, lw->type_attn_output);
        if (lw->attn_output_b) {
            dequantize_row(lw->attn_output_b, s->dequant_scratch, dim, lw->type_attn_output_b);
            for (int bi = 0; bi < n_tokens; bi++)
                vec_add(xb2_batch + bi * dim, s->dequant_scratch, dim);
        }

        /* g. Residual: x += attention_out */
        for (int bi = 0; bi < n_tokens; bi++) {
            float *xbi = x_batch + bi * dim;
            float *xoi = xb2_batch + bi * dim;
            for (int d = 0; d < dim; d++) xbi[d] += xoi[d];
        }

        /* h. Batch FFN RMSNorm */
        for (int bi = 0; bi < n_tokens; bi++)
            rmsnorm(xb_batch + bi * dim, x_batch + bi * dim,
                    s->ffn_norm_w[l], dim, c->rms_norm_eps);

        /* i. Batch FFN gate+up (dual matmul) */
        matmul_dual_batch(hb_batch, hb2_batch, xb_batch, n_tokens,
                          lw->ffn_gate, lw->ffn_up, dim, n_ffn,
                          lw->type_ffn_gate, lw->type_ffn_up);

        /* j. Per-token SiLU + elemwise_mul */
        for (int bi = 0; bi < n_tokens; bi++) {
            float *hb  = hb_batch + bi * n_ffn;
            float *hb2 = hb2_batch + bi * n_ffn;
            silu(hb, n_ffn);
            elemwise_mul(hb, hb, hb2, n_ffn);
        }

        /* k. Batch FFN down projection */
        matmul_batch(xb2_batch, hb_batch, n_tokens,
                     lw->ffn_down, n_ffn, dim, lw->type_ffn_down);

        /* l. Residual: x += ffn_out */
        for (int bi = 0; bi < n_tokens; bi++) {
            float *xbi = x_batch + bi * dim;
            float *xoi = xb2_batch + bi * dim;
            for (int d = 0; d < dim; d++) xbi[d] += xoi[d];
        }
    }

    /* 3. Final norm + output projection for the LAST token */
    float *last_x = x_batch + (n_tokens - 1) * dim;
    rmsnorm(s->x, last_x, s->output_norm_w, dim, c->rms_norm_eps);
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);

    free(buf);
}

/* ================================================================
 * KV Cache Persistence — save/load KV state to skip prompt prefill
 *
 * File format:
 *   [4 bytes] magic: KVCACHE_MAGIC
 *   [4 bytes] n_pos: number of cached positions
 *   [4 bytes] n_layers
 *   [4 bytes] kv_dim (n_kv_heads * head_dim)
 *   [N bytes] key_cache FP16 data (n_layers * n_pos * kv_dim * sizeof(uint16_t))
 *   [N bytes] val_cache FP16 data (same size)
 * ================================================================ */

int kvcache_save(const model_t *m, const char *path, int n_pos) {
    const model_config_t *c = &m->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int seq_len = c->max_seq_len;

    if (n_pos <= 0 || n_pos > seq_len) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "kvcache_save: cannot open %s\n", path);
        return -1;
    }

    uint32_t header[4] = {
        KVCACHE_MAGIC,
        (uint32_t)n_pos,
        (uint32_t)c->n_layers,
        (uint32_t)kv_dim
    };
    fwrite(header, sizeof(uint32_t), 4, f);

    /* Write KV cache for each layer, only the first n_pos positions */
    size_t row_size = (size_t)kv_dim * sizeof(uint16_t);
    for (int l = 0; l < c->n_layers; l++) {
        const uint16_t *kcache_l = m->state.key_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            fwrite(kcache_l + (size_t)p * kv_dim, 1, row_size, f);
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        const uint16_t *vcache_l = m->state.val_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            fwrite(vcache_l + (size_t)p * kv_dim, 1, row_size, f);
        }
    }

    fclose(f);
    fprintf(stderr, "KV cache saved: %d positions to %s\n", n_pos, path);
    return 0;
}

int kvcache_load(model_t *m, const char *path) {
    const model_config_t *c = &m->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int seq_len = c->max_seq_len;

    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    uint32_t header[4];
    if (fread(header, sizeof(uint32_t), 4, f) != 4) {
        fclose(f);
        return 0;
    }

    if (header[0] != KVCACHE_MAGIC) {
        fprintf(stderr, "kvcache_load: invalid magic\n");
        fclose(f);
        return 0;
    }

    int n_pos = (int)header[1];
    int file_layers = (int)header[2];
    int file_kv_dim = (int)header[3];

    if (file_layers != c->n_layers || file_kv_dim != kv_dim) {
        fprintf(stderr, "kvcache_load: model mismatch (layers=%d/%d, kv_dim=%d/%d)\n",
                file_layers, c->n_layers, file_kv_dim, kv_dim);
        fclose(f);
        return 0;
    }
    if (n_pos > seq_len) {
        fprintf(stderr, "kvcache_load: cached %d positions exceeds max_seq_len %d\n",
                n_pos, seq_len);
        fclose(f);
        return 0;
    }

    size_t row_size = (size_t)kv_dim * sizeof(uint16_t);
    for (int l = 0; l < c->n_layers; l++) {
        uint16_t *kcache_l = m->state.key_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            if (fread(kcache_l + (size_t)p * kv_dim, 1, row_size, f) != row_size) {
                fclose(f);
                return 0;
            }
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        uint16_t *vcache_l = m->state.val_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            if (fread(vcache_l + (size_t)p * kv_dim, 1, row_size, f) != row_size) {
                fclose(f);
                return 0;
            }
        }
    }

    fclose(f);
    fprintf(stderr, "KV cache loaded: %d positions from %s\n", n_pos, path);
    return n_pos;
}
