#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declarations */
static void build_gpt2_byte_table(void);
static int byte_linear_lookup(const tokenizer_t *t, const uint8_t *data, int len);

typedef struct { uint8_t data[4]; int len; } gpt2_byte_t;
static gpt2_byte_t gpt2_byte_table[256];

/* ---- GGUF string reader (reused from model.c logic) ---- */

static uint64_t read_u64_at(const uint8_t **p) {
    uint64_t v;
    memcpy(&v, *p, 8);
    *p += 8;
    return v;
}

/* ---- Sorted index for binary search ---- */

static char **g_vocab_for_sort; /* global for qsort comparison */

static int cmp_sorted(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return strcmp(g_vocab_for_sort[ia], g_vocab_for_sort[ib]);
}

static int vocab_lookup(const tokenizer_t *t, const char *str, int len) {
    /* Binary search in sorted vocabulary */
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int idx = t->sorted_idx[mid];
        int cmp = strncmp(t->vocab[idx], str, (size_t)len);
        if (cmp == 0) {
            /* Check exact length match.
             * Cast to unsigned char: bytes >= 0x80 are negative on signed-char platforms
             * (Windows/MinGW), which would send the binary search in the wrong direction
             * when the vocab entry is longer than the search string. */
            unsigned char next = (unsigned char)t->vocab[idx][len];
            if (next == 0) return idx;
            if (next > 0) { hi = mid - 1; }
            else { lo = mid + 1; }
        } else if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1; /* not found */
}

/* ---- BPE merge helper (shared between encoding paths) ---- */
static int bpe_merge(const tokenizer_t *t, int *tokens, int n) {
    while (n >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_tok = -1;

        for (int i = 0; i < n - 1; i++) {
            const char *s1 = t->vocab[tokens[i]];
            const char *s2 = t->vocab[tokens[i + 1]];
            int l1 = (int)strlen(s1);
            int l2 = (int)strlen(s2);

            char merged[512];
            if (l1 + l2 >= (int)sizeof(merged)) continue;
            memcpy(merged, s1, (size_t)l1);
            memcpy(merged + l1, s2, (size_t)l2);
            merged[l1 + l2] = '\0';

            int tok = vocab_lookup(t, merged, l1 + l2);
            if (tok >= 0 && t->scores[tok] > best_score) {
                best_score = t->scores[tok];
                best_idx = i;
                best_tok = tok;
            }
        }

        if (best_idx < 0) break;

        tokens[best_idx] = best_tok;
        for (int i = best_idx + 1; i < n - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n--;
    }
    return n;
}

/* ---- Public API ---- */

int tokenizer_load(tokenizer_t *t, const model_t *m) {
    int vs = m->config.vocab_size;
    t->vocab_size = vs;
    t->bos_id = m->tok_bos_id;
    t->eos_id = m->tok_eos_id;

    t->vocab = (char **)calloc((size_t)vs, sizeof(char *));
    t->scores = (float *)calloc((size_t)vs, sizeof(float));
    t->sorted_idx = (int *)malloc((size_t)vs * sizeof(int));
    if (!t->vocab || !t->scores || !t->sorted_idx) {
        fprintf(stderr, "OOM allocating tokenizer\n");
        return -1;
    }

    if (m->tok_tokens_data && m->tok_n_tokens > 0) {
        const uint8_t *p = (const uint8_t *)m->tok_tokens_data;
        uint64_t n = m->tok_n_tokens;
        if ((int)n > vs) n = (uint64_t)vs;

        for (uint64_t i = 0; i < n; i++) {
            uint64_t slen = read_u64_at(&p);
            t->vocab[i] = (char *)malloc((size_t)slen + 1);
            if (t->vocab[i]) {
                memcpy(t->vocab[i], p, (size_t)slen);
                t->vocab[i][slen] = '\0';
            }
            p += slen;
        }
    }

    for (int i = 0; i < vs; i++) {
        if (!t->vocab[i]) {
            t->vocab[i] = (char *)calloc(1, 1);
        }
    }

    if (m->tok_scores_data && m->tok_n_scores > 0) {
        uint64_t n = m->tok_n_scores;
        if ((int)n > vs) n = (uint64_t)vs;
        memcpy(t->scores, m->tok_scores_data, (size_t)n * sizeof(float));
    }

    for (int i = 0; i < vs; i++) {
        t->sorted_idx[i] = i;
    }
    g_vocab_for_sort = t->vocab;
    qsort(t->sorted_idx, (size_t)vs, sizeof(int), cmp_sorted);

    /* Detect tokenizer type: SentencePiece (LLaMA) has the ▁ prefix token
     * (U+2581). GPT-2/tiktoken (Qwen2) does not. */
    t->is_sentencepiece = 0;
    const char *space_marker = "\xE2\x96\x81"; /* ▁ (U+2581) */
    if (vocab_lookup(t, space_marker, 3) >= 0) {
        t->is_sentencepiece = 1;
    }

    fprintf(stderr, "Tokenizer loaded: %d tokens, bos=%u, eos=%u%s\n",
            vs, t->bos_id, t->eos_id,
            t->is_sentencepiece ? " (sentencepiece)" : " (bpe)");
    return 0;
}

/* ---- GPT-2/tiktoken byte-to-Unicode mapping· ---- *
 * tiktoken mapping: 
 *   Bytes 0x21-0x7E (printable ASCII) → themselves (U+0021-U+007E)
 *   Bytes 0xA1-0xAC, 0xAE-0xFF (Latin-1) → themselves (U+00A1-U+00FF)
 *   Everything else (0x00-0x20, 0x7F-0x9F, 0xA0, 0xAD) → U+0100..
 * See: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
 * Corrected mapping: bytes that stay as Latin-1: 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF */

/* tiktoken special byte mapping:
 * Special (need mapping): 0x00-0x20, 0x7F-0x9F, 0xA0, 0xAD = 68 bytes
 * Non-special (stay as Latin-1): 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF = 188 bytes */
static const unsigned char tiktoken_special[256] = {
    /* 0x00-0x0F: 16 special */
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    /* 0x10-0x1F: 16 special */
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    /* 0x20: special (space), 0x21-0x2F: 15 non-special */
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0x30-0x3F: 16 non-special */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0x40-0x4F: 16 non-special */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0x50-0x5F: 16 non-special */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0x60-0x6F: 16 non-special */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0x70-0x7E: 15 non-special, 0x7F: special (DEL) */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    /* 0x80-0x8F: 16 special (C1 controls) */
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    /* 0x90-0x9F: 16 special (C1 controls) */
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    /* 0xA0: special, 0xA1-0xAC: 12 non-special */
    1,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0xAD: 1 special (soft hyphen) */
    1,
    /* 0xAE-0xBF: 18 non-special */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0xC0-0xCF: 16 non-special */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0xD0-0xDF: 16 non-special */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0xE0-0xEF: 16 non-special */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0xF0-0xFF: 16 non-special */
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};
/* Verify: 16+16+1+15*6+1+16+16+13+1+18+16*4 = 33+90+33+14+82 = 256 ✓ */

static void build_gpt2_byte_table(void) {
    static int built = 0;
    if (built) return;
    int special_idx = 0;
    for (int b = 0; b < 256; b++) {
        if (tiktoken_special[b]) {
            unsigned int cp = 0x100 + (unsigned int)special_idx;
            if (cp < 0x800) {
                gpt2_byte_table[b].data[0] = (uint8_t)(0xC0 | (cp >> 6));
                gpt2_byte_table[b].data[1] = (uint8_t)(0x80 | (cp & 0x3F));
                gpt2_byte_table[b].len = 2;
            } else {
                gpt2_byte_table[b].data[0] = (uint8_t)(0xE0 | (cp >> 12));
                gpt2_byte_table[b].data[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                gpt2_byte_table[b].data[2] = (uint8_t)(0x80 | (cp & 0x3F));
                gpt2_byte_table[b].len = 3;
            }
            special_idx++;
        } else {
            /* Non-special: byte maps to itself as Unicode codepoint (U+00xx).
             * Store as UTF-8, since that's how it appears in the vocab. */
            unsigned int cp = (unsigned int)b;
            if (cp < 0x80) {
                gpt2_byte_table[b].data[0] = (uint8_t)cp;
                gpt2_byte_table[b].len = 1;
            } else if (cp < 0x800) {
                gpt2_byte_table[b].data[0] = (uint8_t)(0xC0 | (cp >> 6));
                gpt2_byte_table[b].data[1] = (uint8_t)(0x80 | (cp & 0x3F));
                gpt2_byte_table[b].len = 2;
            } else {
                gpt2_byte_table[b].data[0] = (uint8_t)(0xE0 | (cp >> 12));
                gpt2_byte_table[b].data[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
                gpt2_byte_table[b].data[2] = (uint8_t)(0x80 | (cp & 0x3F));
                gpt2_byte_table[b].len = 3;
            }
        }
    }
    built = 1;
}

/* Linear scan lookup: find a token matching the given byte sequence.
 * Necessary because binary search (strncmp) can fail on binary content
 * in GPT-2/tiktoken byte token strings. */
static int byte_linear_lookup(const tokenizer_t *t, const uint8_t *data, int len) {
    for (int vi = 0; vi < t->vocab_size; vi++) {
        if ((int)strlen(t->vocab[vi]) == len &&
            memcmp(t->vocab[vi], data, (size_t)len) == 0) {
            return vi;
        }
    }
    return -1;
}

/* Forward declaration */
static int encode_bpe_segment(const tokenizer_t *t, const char *text, int text_len,
                              int *tokens, int max_tokens, int add_space_marker);

/* ---- Core BPE encode: handle special tokens first, then BPE for normal text ---- */
static int bpe_encode_core(const tokenizer_t *t, const char *text,
                           int *tokens, int max_tokens,
                           int add_space_marker) {
    int n_tokens = 0;

    /* Qwen2 special tokens - handle before BPE */
    const char *specials[] = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<|extra_0|>", "<|extra_1|>", "<|extra_2|>", "<|extra_3|>"
    };
    int n_specials = sizeof(specials) / sizeof(specials[0]);

    const char *curr = text;
    while (*curr && n_tokens < max_tokens) {
        /* Try to match special token first */
        int special_id = -1;
        int special_len = 0;

        for (int i = 0; i < n_specials; i++) {
            int slen = (int)strlen(specials[i]);
            if (strncmp(curr, specials[i], (size_t)slen) == 0) {
                int id = vocab_lookup(t, specials[i], slen);
                if (id != -1) {
                    special_id = id;
                    special_len = slen;
                    break;
                }
            }
        }

        if (special_id != -1) {
            /* Found special token */
            tokens[n_tokens++] = special_id;
            curr += special_len;
        } else {
            /* Find next special token position */
            const char *next_special = NULL;
            for (const char *p = curr + 1; *p; p++) {
                for (int i = 0; i < n_specials; i++) {
                    if (strncmp(p, specials[i], strlen(specials[i])) == 0) {
                        next_special = p;
                        break;
                    }
                }
                if (next_special) break;
            }

            int seg_len = next_special ? (int)(next_special - curr) : (int)strlen(curr);
            if (seg_len == 0) break;

            /* Encode this segment with BPE */
            int encoded = encode_bpe_segment(t, curr, seg_len, tokens + n_tokens, max_tokens - n_tokens, add_space_marker);
            n_tokens += encoded;
            curr += seg_len;
        }
    }

    return n_tokens;
}

/* Encode a segment without special tokens */
static int encode_bpe_segment(const tokenizer_t *t, const char *text, int text_len,
                              int *tokens, int max_tokens, int add_space_marker) {
    if (text_len <= 0) return 0;

    /* Build normalized text */
    int norm_cap;
    if (add_space_marker) {
        norm_cap = text_len * 3 + 4;
    } else {
        norm_cap = text_len + 1;
    }
    char *norm = (char *)malloc((size_t)norm_cap);
    int norm_len = 0;

    if (add_space_marker) {
        norm[norm_len++] = (char)0xE2;
        norm[norm_len++] = (char)0x96;
        norm[norm_len++] = (char)0x81;
        for (int i = 0; i < text_len; i++) {
            if (text[i] == ' ') {
                norm[norm_len++] = (char)0xE2;
                norm[norm_len++] = (char)0x96;
                norm[norm_len++] = (char)0x81;
            } else {
                norm[norm_len++] = text[i];
            }
        }
    } else {
        memcpy(norm, text, (size_t)text_len);
        norm_len = text_len;
    }
    norm[norm_len] = '\0';

    /* Step 1: Convert to initial byte tokens */
    int *merge_buf = (int *)malloc((size_t)(norm_len + 1) * sizeof(int));
    int merge_len = 0;

    for (int i = 0; i < norm_len; ) {
        int clen = 1;
        unsigned char c = (unsigned char)norm[i];
        if (c >= 0xF0) clen = 4;
        else if (c >= 0xE0) clen = 3;
        else if (c >= 0xC0) clen = 2;
        if (i + clen > norm_len) clen = norm_len - i;

        int tok = -1;
        int tok_strategy = 0;

        /* Strategy 1: Try multi-byte UTF-8 character via binary search */
        if (clen > 1) {
            tok = vocab_lookup(t, norm + i, clen);
            if (tok >= 0) tok_strategy = 1;
        }

        /* Strategy 1b: Linear scan for multi-byte (if binary search failed,
         * might be due to binary content in GPT-2/tiktoken style vocabs) */
        if (tok < 0 && clen > 1) {
            for (int vi = 0; vi < t->vocab_size; vi++) {
                if ((int)strlen(t->vocab[vi]) == clen &&
                    memcmp(t->vocab[vi], norm + i, (size_t)clen) == 0) {
                    tok = vi;
                    tok_strategy = 1;
                    break;
                }
            }
        }

        if (tok < 0) {
            /* Strategy 2: Try GPT-2 byte encoding (linear scan).
             * Used by Qwen2/GPT-2 tokenizers in GGUF. */
            build_gpt2_byte_table();
            tok = byte_linear_lookup(t, gpt2_byte_table[(unsigned char)norm[i]].data,
                                      gpt2_byte_table[(unsigned char)norm[i]].len);
            if (tok >= 0) tok_strategy = 2;
        }

        if (tok < 0) {
            /* Strategy 3: Try <0xHH> byte token format (SentencePiece) */
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)norm[i]);
            tok = vocab_lookup(t, byte_tok, (int)strlen(byte_tok));
            if (tok >= 0) tok_strategy = 3;
        }

        if (tok < 0) {
            /* Strategy 4: Raw single-byte lookup (linear scan) */
            tok = byte_linear_lookup(t, (const uint8_t *)(norm + i), 1);
        }

        if (tok >= 0) {
            merge_buf[merge_len++] = tok;
        } else {
            fprintf(stderr, "Warning: byte 0x%02X has no token mapping\n",
                    (unsigned char)norm[i]);
        }
        i += (clen > 1 && tok_strategy == 1) ? clen : 1;
    }
    free(norm);

    /* Step 2: BPE merge */
    merge_len = bpe_merge(t, merge_buf, merge_len);

    int n = 0;
    for (int i = 0; i < merge_len && n < max_tokens; i++) {
        tokens[n++] = merge_buf[i];
    }
    free(merge_buf);
    return n;
}

int tokenizer_encode(const tokenizer_t *t, const char *text,
                     int *tokens, int max_tokens, int add_bos) {
    int n_tokens = 0;
    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = (int)t->bos_id;
    }
    if (!text || !*text) return n_tokens;

    int encoded = bpe_encode_core(t, text, tokens + n_tokens,
                                   max_tokens - n_tokens,
                                   t->is_sentencepiece);
    return n_tokens + encoded;
}

/* ---- UTF-8 byte accumulator for decode ---- */
static uint8_t utf8_acc[6];
static int     utf8_acc_len = 0;

/* Qwen2/GPT-2 decode: convert GPT-2 byte-encoded token string to clean text.
 * GPT-2 maps special bytes (0x00-0x20, 0x7F-0x9F, 0xA0, 0xAD) to U+0100-U+0143 range.
 * Non-special bytes (0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF) stay as Latin-1 (U+0021-U+007E, U+00A1-U+00AC, U+00AE-U+00FF).
 * We need to reverse the Latin-1 mapping back to raw bytes for UTF-8 reconstruction. */
static void decode_qwen_str(const char *str, char *out, int out_max) {
    int j = 0;
    for (int i = 0; str[i] && j < out_max - 1; i++) {
        unsigned char c = (unsigned char)str[i];

        /* Check for UTF-8 lead byte */
        int utf8_len = 1;
        if (c >= 0xF0) utf8_len = 4;
        else if (c >= 0xE0) utf8_len = 3;
        else if (c >= 0xC2) utf8_len = 2;

        if (utf8_len == 2) {
            /* 2-byte UTF-8: could be U+0080-U+00FF (Latin-1) or U+0100-U+0143 (special) */
            unsigned int cp = ((unsigned int)(c & 0x1F) << 6) | ((unsigned int)str[i+1] & 0x3F);

            if (cp >= 0x0100 && cp <= 0x0143) {
                /* Special bytes mapped to U+0100+ range */
                int orig_byte;
                if (cp <= 0x0120) {
                    /* U+0100-U+0120 -> bytes 0x00-0x20 */
                    orig_byte = cp - 0x0100;
                } else if (cp <= 0x0141) {
                    /* U+0121-U+0141 -> bytes 0x7F-0x9F */
                    orig_byte = 0x7F + (cp - 0x0121);
                } else if (cp == 0x0142) {
                    /* U+0142 -> byte 0xA0 (non-breaking space) */
                    orig_byte = 0xA0;
                } else if (cp == 0x0143) {
                    /* U+0143 -> byte 0xAD (soft hyphen) */
                    orig_byte = 0xAD;
                } else {
                    orig_byte = -1;
                }
                if (orig_byte >= 0) {
                    out[j++] = (char)orig_byte;
                } else {
                    out[j++] = str[i];
                    out[j++] = str[i+1];
                }
                i++;
            } else if ((cp >= 0x00A1 && cp <= 0x00AC) || cp >= 0x00AE) {
                /* Latin-1 non-special range: convert back to raw byte */
                out[j++] = (char)cp;
                i++;
            } else if (cp >= 0x0080 && cp <= 0x00FF) {
                /* Special bytes 0xA0/0xAD in Latin-1 range - output original UTF-8 */
                out[j++] = str[i];
                out[j++] = str[i+1];
                i++;
            } else {
                /* Other 2-byte UTF-8 (shouldn't happen for GPT-2 tokens) */
                out[j++] = str[i];
                out[j++] = str[i+1];
                i++;
            }
        } else if (utf8_len == 3) {
            /* 3-byte UTF-8: Chinese characters etc. - keep as-is */
            out[j++] = str[i];
            out[j++] = str[i+1];
            out[j++] = str[i+2];
            i += 2;
        } else if (utf8_len == 4) {
            /* 4-byte UTF-8 - keep as-is */
            out[j++] = str[i];
            out[j++] = str[i+1];
            out[j++] = str[i+2];
            out[j++] = str[i+3];
            i += 3;
        } else {
            /* ASCII byte - keep as-is */
            out[j++] = str[i];
        }
    }
    out[j] = '\0';
}

const char *tokenizer_decode(const tokenizer_t *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    const char *str = t->vocab[token];

    if (!t->is_sentencepiece) {
        /* Qwen2/GPT-2 style: decode byte encodings to clean text */
        static char qbuf[512];
        decode_qwen_str(str, qbuf, sizeof(qbuf));
        return qbuf;
    }

    /* ---- LLaMA/SentencePiece decode ---- */

    /* Handle byte tokens: <0xHH> -> accumulate for UTF-8 */
    if (str[0] == '<' && str[1] == '0' && str[2] == 'x' && str[5] == '>') {
        unsigned int val = 0;
        for (int i = 3; i < 5; i++) {
            val <<= 4;
            char c = str[i];
            if (c >= '0' && c <= '9') val += (unsigned)(c - '0');
            else if (c >= 'A' && c <= 'F') val += (unsigned)(c - 'A' + 10);
            else if (c >= 'a' && c <= 'f') val += (unsigned)(c - 'a' + 10);
        }
        utf8_acc[utf8_acc_len++] = (uint8_t)val;

        int expected = 1;
        if (utf8_acc[0] >= 0xF0) expected = 4;
        else if (utf8_acc[0] >= 0xE0) expected = 3;
        else if (utf8_acc[0] >= 0xC0) expected = 2;

        if (utf8_acc_len >= expected) {
            static char result[8];
            int n = utf8_acc_len;
            memcpy(result, utf8_acc, (size_t)n);
            result[n] = '\0';
            utf8_acc_len = 0;
            return result;
        }
        return "";
    }

    /* Non-byte token: reset accumulator */
    utf8_acc_len = 0;

    /* Handle SentencePiece leading space marker "▁" -> " " */
    if ((unsigned char)str[0] == 0xE2 && (unsigned char)str[1] == 0x96 && (unsigned char)str[2] == 0x81) {
        static char space_buf[256];
        if (prev_token == (int)t->bos_id) {
            int len = (int)strlen(str + 3);
            if (len >= (int)sizeof(space_buf)) len = (int)sizeof(space_buf) - 1;
            memcpy(space_buf, str + 3, (size_t)len);
            space_buf[len] = '\0';
            return space_buf;
        }
        space_buf[0] = ' ';
        int len = (int)strlen(str + 3);
        if (len >= (int)sizeof(space_buf) - 1) len = (int)sizeof(space_buf) - 2;
        memcpy(space_buf + 1, str + 3, (size_t)len);
        space_buf[1 + len] = '\0';
        return space_buf;
    }

    return str;
}

void tokenizer_free(tokenizer_t *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) {
            free(t->vocab[i]);
        }
        free(t->vocab);
        t->vocab = NULL;
    }
    free(t->scores);
    t->scores = NULL;
    free(t->sorted_idx);
    t->sorted_idx = NULL;
}
