#include "quant.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ================================================================
 * FP16 <-> FP32 conversion (software, no hardware dependency)
 * ================================================================ */

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign; /* +/- zero */
        } else {
            /* subnormal: renormalize */
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13); /* inf / nan */
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int      exp  = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0) {
        return (uint16_t)sign; /* zero or f32 subnormal -> fp16 zero */
    }
    if (((bits >> 23) & 0xFF) == 0xFF) {
        /* inf / nan */
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00); /* overflow -> inf */
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; /* too small -> zero */
        /* subnormal fp16 */
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        /* round to nearest */
        uint32_t round_bit = 1U << (shift - 1);
        mant = (mant + round_bit) >> shift;
        return (uint16_t)(sign | mant);
    }

    /* round to nearest even */
    mant += 0x00001000; /* bit 12 */
    if (mant & 0x00800000) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* ---- Q4_K helpers ---- */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >>  4) | ((q[j    ] >> 6) << 4);
    }
}

/* ================================================================
 * Dequantization kernels (scalar — used for embedding lookup etc.)
 * ================================================================ */

void dequantize_row_q4_K(const void *src, float *dst, int n) {
    const block_q4_K *blocks = (const block_q4_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);
        const uint8_t *q = b->qs;
        float *y = dst + i * 256;

        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(is, b->scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * (float)sc;
            float m2 = dmin * (float)mn;

            for (int l = 0; l < 32; l++) {
                y[l]      = d1 * (float)(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; l++) {
                y[l + 32] = d2 * (float)(q[l] >> 4)  - m2;
            }
            y  += 64;
            q  += 32;
            is += 2;
        }
    }
}

void dequantize_row_q3_K(const void *src, float *dst, int n) {
    const block_q3_K *blocks = (const block_q3_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q3_K *b = &blocks[i];
        float d = fp16_to_fp32(b->d);

        int32_t scales[16];
        {
            for (int j = 0; j < 8; j++) {
                scales[j] = (int32_t)(b->scales[j] & 0xF);
            }
            for (int j = 0; j < 8; j++) {
                scales[8 + j] = (int32_t)(b->scales[j] >> 4);
            }
            for (int j = 0; j < 4; j++) {
                scales[2*j]     |= ((b->scales[8 + j]     ) & 3) << 4;
                scales[2*j + 1] |= ((b->scales[8 + j] >> 2) & 3) << 4;
                scales[2*j + 8] |= ((b->scales[8 + j] >> 4) & 3) << 4;
                scales[2*j + 9] |= ((b->scales[8 + j] >> 6) & 3) << 4;
            }
            for (int j = 0; j < 16; j++) {
                scales[j] -= 32;
            }
        }

        const uint8_t *qs    = b->qs;
        const uint8_t *hmask = b->hmask;
        int out_idx = i * 256;

        for (int j = 0; j < 256; j++) {
            int q2 = (qs[j / 4] >> (2 * (j % 4))) & 3;
            int hbit = (hmask[j / 8] >> (j % 8)) & 1;
            int q3 = q2 | (hbit << 2);
            int sb = j / 16;
            dst[out_idx + j] = d * (float)scales[sb] * ((float)q3 - 4.0f);
        }
    }
}

void dequantize_row_q2_K(const void *src, float *dst, int n) {
    const block_q2_K *blocks = (const block_q2_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        const block_q2_K *b = &blocks[i];
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);

        const uint8_t *qs = b->qs;
        int out_idx = i * 256;

        for (int j = 0; j < 256; j++) {
            int q2 = (qs[j / 4] >> (2 * (j % 4))) & 3;
            int sb = j / 16;
            uint8_t sc = b->scales[sb] & 0xF;
            uint8_t mn = b->scales[sb] >> 4;
            dst[out_idx + j] = d * (float)sc * (float)q2 - dmin * (float)mn;
        }
    }
}

void dequantize_row_q6_K(const void *src, float *dst, int n) {
    const block_q6_K *blocks = (const block_q6_K *)src;
    int nb = n / 256;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        const uint8_t *ql = blocks[i].ql;
        const uint8_t *qh = blocks[i].qh;
        const int8_t  *sc = blocks[i].scales;
        float *y = dst + i * 256;

        for (int chunk = 0; chunk < 256; chunk += 128) {
            int is = chunk / 16;
            for (int l = 0; l < 32; l++) {
                int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                int is_l = is + (l / 16);
                y[l]      = d * (float)sc[is_l + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is_l + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is_l + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is_l + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
        }
    }
}

void dequantize_row_q8_0(const void *src, float *dst, int n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            dst[i * 32 + j] = d * (float)blocks[i].qs[j];
        }
    }
}

void dequantize_row_q4_0(const void *src, float *dst, int n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    int nb = n / 32;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            uint8_t nibble;
            if (j < 16) {
                nibble = blocks[i].qs[j] & 0xF;
            } else {
                nibble = blocks[i].qs[j - 16] >> 4;
            }
            dst[i * 32 + j] = d * ((float)nibble - 8.0f);
        }
    }
}

void dequantize_row_f16(const void *src, float *dst, int n) {
    const uint16_t *fp16 = (const uint16_t *)src;
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(fp16[i]);
    }
}

void dequantize_row_f32(const void *src, float *dst, int n) {
    memcpy(dst, src, n * sizeof(float));
}

void dequantize_row(const void *src, float *dst, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   dequantize_row_f32(src, dst, n);  break;
        case GGUF_TYPE_F16:   dequantize_row_f16(src, dst, n);  break;
        case GGUF_TYPE_Q4_0:  dequantize_row_q4_0(src, dst, n); break;
        case GGUF_TYPE_Q8_0:  dequantize_row_q8_0(src, dst, n); break;
        case GGUF_TYPE_Q2_K:  dequantize_row_q2_K(src, dst, n); break;
        case GGUF_TYPE_Q3_K:  dequantize_row_q3_K(src, dst, n); break;
        case GGUF_TYPE_Q4_K:  dequantize_row_q4_K(src, dst, n); break;
        case GGUF_TYPE_Q6_K:  dequantize_row_q6_K(src, dst, n); break;
        default:
            fprintf(stderr, "dequantize_row: unsupported type %d\n", type);
            exit(1);
    }
}

/* ---- Type info ---- */

int gguf_type_block_size(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   return 1;
        case GGUF_TYPE_F16:   return 1;
        case GGUF_TYPE_Q4_0:  return 32;
        case GGUF_TYPE_Q4_1:  return 32;
        case GGUF_TYPE_Q5_0:  return 32;
        case GGUF_TYPE_Q5_1:  return 32;
        case GGUF_TYPE_Q8_0:  return 32;
        case GGUF_TYPE_Q8_1:  return 32;
        case GGUF_TYPE_Q2_K:  return 256;
        case GGUF_TYPE_Q3_K:  return 256;
        case GGUF_TYPE_Q4_K:  return 256;
        case GGUF_TYPE_Q5_K:  return 256;
        case GGUF_TYPE_Q6_K:  return 256;
        default: return 0;
    }
}

int gguf_type_quant_size(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_F32:   return 4;
        case GGUF_TYPE_F16:   return 2;
        case GGUF_TYPE_Q4_0:  return 18;
        case GGUF_TYPE_Q4_1:  return 20;
        case GGUF_TYPE_Q5_0:  return 22;
        case GGUF_TYPE_Q5_1:  return 24;
        case GGUF_TYPE_Q8_0:  return 34;
        case GGUF_TYPE_Q8_1:  return 40;
        case GGUF_TYPE_Q2_K:  return 84;
        case GGUF_TYPE_Q3_K:  return 110;
        case GGUF_TYPE_Q4_K:  return 144;
        case GGUF_TYPE_Q5_K:  return 176;
        case GGUF_TYPE_Q6_K:  return 210;
        default: return 0;
    }
}

size_t gguf_type_row_size(gguf_type_t type, int n) {
    int bs = gguf_type_block_size(type);
    int qs = gguf_type_quant_size(type);
    if (bs == 0 || qs == 0) return 0;
    return (size_t)(n / bs) * qs;
}

/* ================================================================
 * Fused dequant + dot-product: compute dot(dequant(row), x) without
 * materializing the full dequantized row.
 *
 * Three tiers per format:
 *   1. NEON (ARM Pi 3/4/5)
 *   2. SSE2 (x86 development)
 *   3. Scalar fallback
 * ================================================================ */

/* ---- vec_dot_f32_f32 ---- */

float vec_dot_f32_f32(const void *src, const float *x, int n) {
    const float *w = (const float *)src;

#if defined(PICOLM_AVX2)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i),     _mm256_loadu_ps(x + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), acc1);
    }
    float sum = hsum_avx2(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#elif defined(PICOLM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(w + i),     vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
    }
    float sum = vaddvq_f32_compat(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#elif defined(PICOLM_SSE2)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(w + i),     _mm_loadu_ps(x + i)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(w + i + 4), _mm_loadu_ps(x + i + 4)));
    }
    float sum = hsum_sse(_mm_add_ps(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;

#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += w[i] * x[i];
    }
    return sum;
#endif
}

/* ---- vec_dot_q4_K_f32 ---- */

float vec_dot_q4_K_f32(const void *src, const float *x, int n) {
    const block_q4_K *blocks = (const block_q4_K *)src;
    int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        float d    = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);
        const uint8_t *q = b->qs;
        const float *xp = x + i * 256;

        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(is, b->scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * (float)sc;
            float m2 = dmin * (float)mn;

#if defined(PICOLM_AVX2)
            __m256 sum_qx1_v = _mm256_setzero_ps();
            __m256 sum_x1_v  = _mm256_setzero_ps();
            __m256 sum_qx2_v = _mm256_setzero_ps();
            __m256 sum_x2_v  = _mm256_setzero_ps();

            /* Prefetch next cache line */
            _mm_prefetch((const char*)(q + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(xp + 64), _MM_HINT_T0);

            /* Process 16 bytes at a time — 2 iterations instead of 4 */
            for (int l = 0; l < 32; l += 16) {
                __m128i q16 = _mm_loadu_si128((const __m128i*)(q + l));

                __m128i lo16 = _mm_and_si128(q16, _mm_set1_epi8(0x0F));
                __m128i hi16 = _mm_and_si128(_mm_srli_epi16(q16, 4), _mm_set1_epi8(0x0F));

                /* Low nibbles: bytes 0-7 → x[l..l+7], 8-15 → x[l+8..l+15] */
                __m256i lo16_0 = _mm256_cvtepu8_epi16(lo16);
                __m256 lo_f_a0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(lo16_0)));
                __m256 lo_f_a1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(lo16_0, 1)));
                __m256 x_l0 = _mm256_loadu_ps(xp + l);
                __m256 x_l1 = _mm256_loadu_ps(xp + l + 8);
                sum_qx1_v = _mm256_fmadd_ps(lo_f_a0, x_l0, sum_qx1_v);
                sum_qx1_v = _mm256_fmadd_ps(lo_f_a1, x_l1, sum_qx1_v);
                sum_x1_v  = _mm256_add_ps(sum_x1_v, _mm256_add_ps(x_l0, x_l1));

                /* High nibbles: bytes 0-7 → x[l+32..l+39], 8-15 → x[l+40..l+47] */
                __m256i hi16_0 = _mm256_cvtepu8_epi16(hi16);
                __m256 hi_f_a0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(hi16_0)));
                __m256 hi_f_a1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(hi16_0, 1)));
                __m256 x_h0 = _mm256_loadu_ps(xp + l + 32);
                __m256 x_h1 = _mm256_loadu_ps(xp + l + 40);
                sum_qx2_v = _mm256_fmadd_ps(hi_f_a0, x_h0, sum_qx2_v);
                sum_qx2_v = _mm256_fmadd_ps(hi_f_a1, x_h1, sum_qx2_v);
                sum_x2_v  = _mm256_add_ps(sum_x2_v, _mm256_add_ps(x_h0, x_h1));
            }

            float sum_qx1 = hsum_avx2(sum_qx1_v);
            float sum_x1  = hsum_avx2(sum_x1_v);
            float sum_qx2 = hsum_avx2(sum_qx2_v);
            float sum_x2  = hsum_avx2(sum_x2_v);
#elif defined(PICOLM_NEON)
            float32x4_t sum_qx1_v = vdupq_n_f32(0);
            float32x4_t sum_x1_v  = vdupq_n_f32(0);
            float32x4_t sum_qx2_v = vdupq_n_f32(0);
            float32x4_t sum_x2_v  = vdupq_n_f32(0);

            for (int l = 0; l < 32; l += 8) {
                uint8x8_t qbytes = vld1_u8(q + l);
                uint8x8_t q_lo_8 = vand_u8(qbytes, vdup_n_u8(0xF));
                uint8x8_t q_hi_8 = vshr_n_u8(qbytes, 4);

                uint16x8_t q_lo_16 = vmovl_u8(q_lo_8);
                uint16x8_t q_hi_16 = vmovl_u8(q_hi_8);

                float32x4_t qf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_lo_16)));
                float32x4_t xv0 = vld1q_f32(xp + l);
                sum_qx1_v = vmlaq_f32(sum_qx1_v, qf0, xv0);
                sum_x1_v  = vaddq_f32(sum_x1_v, xv0);

                float32x4_t qf0h = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q_hi_16)));
                float32x4_t xv0h = vld1q_f32(xp + l + 32);
                sum_qx2_v = vmlaq_f32(sum_qx2_v, qf0h, xv0h);
                sum_x2_v  = vaddq_f32(sum_x2_v, xv0h);

                float32x4_t qf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q_lo_16)));
                float32x4_t xv1 = vld1q_f32(xp + l + 4);
                sum_qx1_v = vmlaq_f32(sum_qx1_v, qf1, xv1);
                sum_x1_v  = vaddq_f32(sum_x1_v, xv1);

                float32x4_t qf1h = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q_hi_16)));
                float32x4_t xv1h = vld1q_f32(xp + l + 32 + 4);
                sum_qx2_v = vmlaq_f32(sum_qx2_v, qf1h, xv1h);
                sum_x2_v  = vaddq_f32(sum_x2_v, xv1h);
            }

            float sum_qx1 = vaddvq_f32_compat(sum_qx1_v);
            float sum_x1  = vaddvq_f32_compat(sum_x1_v);
            float sum_qx2 = vaddvq_f32_compat(sum_qx2_v);
            float sum_x2  = vaddvq_f32_compat(sum_x2_v);
#else
            float sum_qx1 = 0.0f, sum_x1 = 0.0f;
            float sum_qx2 = 0.0f, sum_x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                float x_lo = xp[l];
                float x_hi = xp[l + 32];
                sum_qx1 += (float)(q[l] & 0xF) * x_lo;
                sum_x1  += x_lo;
                sum_qx2 += (float)(q[l] >> 4) * x_hi;
                sum_x2  += x_hi;
            }
#endif
            sumf += d1 * sum_qx1 - m1 * sum_x1 + d2 * sum_qx2 - m2 * sum_x2;

            xp += 64;
            q  += 32;
            is += 2;
        }
    }
    return sumf;
}

/* ---- vec_dot_q6_K_f32 ---- */

float vec_dot_q6_K_f32(const void *src, const float *x, int n) {
    const block_q6_K *blocks = (const block_q6_K *)src;
    int nb = n / 256;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        const uint8_t *ql = blocks[i].ql;
        const uint8_t *qh = blocks[i].qh;
        const int8_t  *sc = blocks[i].scales;
        const float *xp = x + i * 256;

#if defined(PICOLM_AVX2)
        float sums[16];
        memset(sums, 0, sizeof(sums));
        for (int chunk = 0; chunk < 2; chunk++) {
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const float *xp_c = xp + chunk * 128;
            int is = chunk * 8;

            __m128i ql0 = _mm_loadu_si128((const __m128i*)(ql_c));
            __m128i ql1 = _mm_loadu_si128((const __m128i*)(ql_c + 16));
            __m128i ql2 = _mm_loadu_si128((const __m128i*)(ql_c + 32));
            __m128i ql3 = _mm_loadu_si128((const __m128i*)(ql_c + 48));
            __m128i qh0 = _mm_loadu_si128((const __m128i*)(qh_c));
            __m128i qh1 = _mm_loadu_si128((const __m128i*)(qh_c + 16));
            __m128i mF = _mm_set1_epi8(0x0F);
            __m128i m3 = _mm_set1_epi8(3);

            /* even groups */
            __m128i e_q1 = _mm_or_si128(_mm_and_si128(ql0, mF), _mm_slli_epi16(_mm_and_si128(qh0, m3), 4));
            __m128i e_q2 = _mm_or_si128(_mm_and_si128(ql2, mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 2), m3), 4));
            __m128i e_q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 4), m3), 4));
            __m128i e_q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql2, 4), mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh0, 6), m3), 4));

            /* odd groups */
            __m128i o_q1 = _mm_or_si128(_mm_and_si128(ql1, mF), _mm_slli_epi16(_mm_and_si128(qh1, m3), 4));
            __m128i o_q2 = _mm_or_si128(_mm_and_si128(ql3, mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 2), m3), 4));
            __m128i o_q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 4), m3), 4));
            __m128i o_q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql3, 4), mF), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh1, 6), m3), 4));

#define Q6_GROUP_SUM(qvals, xpos) ({ \
    __m256i _e16 = _mm256_cvtepu8_epi16(qvals); \
    __m128i _l = _mm256_castsi256_si128(_e16); \
    __m128i _h = _mm256_extracti128_si256(_e16, 1); \
    __m256 _fl = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_l)), _mm256_set1_ps(32.0f)); \
    __m256 _fh = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_h)), _mm256_set1_ps(32.0f)); \
    __m256 _p = _mm256_fmadd_ps(_fl, _mm256_loadu_ps(xp_c + (xpos)), \
               _mm256_mul_ps(_fh, _mm256_loadu_ps(xp_c + (xpos) + 8))); \
    hsum_avx2(_p); \
})

            sums[is+0] = Q6_GROUP_SUM(e_q1, 0);
            sums[is+2] = Q6_GROUP_SUM(e_q2, 32);
            sums[is+4] = Q6_GROUP_SUM(e_q3, 64);
            sums[is+6] = Q6_GROUP_SUM(e_q4, 96);
            sums[is+1] = Q6_GROUP_SUM(o_q1, 16);
            sums[is+3] = Q6_GROUP_SUM(o_q2, 48);
            sums[is+5] = Q6_GROUP_SUM(o_q3, 80);
            sums[is+7] = Q6_GROUP_SUM(o_q4, 112);

#undef Q6_GROUP_SUM
        }

        for (int j = 0; j < 16; j++) {
            sumf += d * (float)sc[j] * sums[j];
        }
#else
        /* Accumulate per-scale-group sums: 16 groups of 16 elements each */
        float sums[16] = {0};

        for (int chunk = 0; chunk < 2; chunk++) {
            int is = chunk * 8;
            const uint8_t *ql_c = ql + chunk * 64;
            const uint8_t *qh_c = qh + chunk * 32;
            const float *xp_c = xp + chunk * 128;

            for (int l = 0; l < 16; l++) {
                int q1 = (int)((ql_c[l]      & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_c[l]      >> 4)  | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_c[l + 32] >> 4)  | (((qh_c[l] >> 6) & 3) << 4)) - 32;
                sums[is + 0] += (float)q1 * xp_c[l];
                sums[is + 2] += (float)q2 * xp_c[l + 32];
                sums[is + 4] += (float)q3 * xp_c[l + 64];
                sums[is + 6] += (float)q4 * xp_c[l + 96];
            }
            for (int l = 16; l < 32; l++) {
                int q1 = (int)((ql_c[l]      & 0xF) | (((qh_c[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql_c[l + 32] & 0xF) | (((qh_c[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql_c[l]      >> 4)  | (((qh_c[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql_c[l + 32] >> 4)  | (((qh_c[l] >> 6) & 3) << 4)) - 32;
                sums[is + 1] += (float)q1 * xp_c[l];
                sums[is + 3] += (float)q2 * xp_c[l + 32];
                sums[is + 5] += (float)q3 * xp_c[l + 64];
                sums[is + 7] += (float)q4 * xp_c[l + 96];
            }
        }

        for (int j = 0; j < 16; j++) {
            sumf += d * (float)sc[j] * sums[j];
        }
#endif
    }
    return sumf;
}

/* ---- vec_dot_q8_0_f32 ---- */

float vec_dot_q8_0_f32(const void *src, const float *x, int n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    int nb = n / 32;
    float sumf = 0.0f;

#if defined(PICOLM_AVX2)
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        __m256i qs = _mm256_loadu_si256((const __m256i*)(blocks[i].qs));

        /* Expand int8 → int16 (low and high halves) */
        __m256i qs_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qs));
        __m256i qs_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qs, 1));

        /* int16 → int32 → float */
        __m256 f_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(qs_lo)));
        __m256 f_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(qs_lo, 1)));
        __m256 f_lo2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(qs_hi)));
        __m256 f_hi2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(qs_hi, 1)));

        __m256 x0 = _mm256_loadu_ps(x + i * 32);
        __m256 x1 = _mm256_loadu_ps(x + i * 32 + 8);
        __m256 x2 = _mm256_loadu_ps(x + i * 32 + 16);
        __m256 x3 = _mm256_loadu_ps(x + i * 32 + 24);

        __m256 p0 = _mm256_mul_ps(f_lo, x0);
        __m256 p1 = _mm256_mul_ps(f_hi, x1);
        __m256 p2 = _mm256_mul_ps(f_lo2, x2);
        __m256 p3 = _mm256_mul_ps(f_hi2, x3);

        __m256 acc = _mm256_add_ps(_mm256_add_ps(p0, p1), _mm256_add_ps(p2, p3));
        sumf += d * hsum_avx2(acc);
    }
#elif defined(PICOLM_NEON)
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        int8x16_t qs0 = vld1q_s8(blocks[i].qs);
        int8x16_t qs1 = vld1q_s8(blocks[i].qs + 16);

        int16x8_t qs0_lo = vmovl_s8(vget_low_s8(qs0));
        int16x8_t qs0_hi = vmovl_s8(vget_high_s8(qs0));
        int16x8_t qs1_lo = vmovl_s8(vget_low_s8(qs1));
        int16x8_t qs1_hi = vmovl_s8(vget_high_s8(qs1));

        float32x4_t x0 = vld1q_f32(x + i * 32);
        float32x4_t x1 = vld1q_f32(x + i * 32 + 4);
        float32x4_t x2 = vld1q_f32(x + i * 32 + 8);
        float32x4_t x3 = vld1q_f32(x + i * 32 + 12);
        float32x4_t x4 = vld1q_f32(x + i * 32 + 16);
        float32x4_t x5 = vld1q_f32(x + i * 32 + 20);
        float32x4_t x6 = vld1q_f32(x + i * 32 + 24);
        float32x4_t x7 = vld1q_f32(x + i * 32 + 28);

        float32x4_t acc = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs0_lo))), x0);
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs0_lo))), x1);
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs0_hi))), x2);
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs0_hi))), x3);
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs1_lo))), x4);
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs1_lo))), x5);
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(qs1_hi))), x6);
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(qs1_hi))), x7);

        sumf += d * vaddvq_f32_compat(acc);
    }
#else
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        float sum = 0.0f;
        /* process 2 at a time for better ILP */
        int j = 0;
        for (; j + 1 < 32; j += 2) {
            sum += (float)blocks[i].qs[j]   * x[j] +
                   (float)blocks[i].qs[j+1] * x[j+1];
        }
        for (; j < 32; j++) {
            sum += (float)blocks[i].qs[j] * x[j];
        }
        sumf += d * sum;
        x += 32;
    }
#endif
    return sumf;
}

/* ---- Generic dispatch ---- */

float vec_dot(const void *src, const float *x, int n, gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_Q4_K: return vec_dot_q4_K_f32(src, x, n);
        case GGUF_TYPE_Q6_K: return vec_dot_q6_K_f32(src, x, n);
        case GGUF_TYPE_Q8_0: return vec_dot_q8_0_f32(src, x, n);
        case GGUF_TYPE_F32:  return vec_dot_f32_f32(src, x, n);
        default: {
            /* Fallback: dequantize to temp buffer, then dot */
            float tmp[8192];
            float *buf = (n <= 8192) ? tmp : (float *)malloc((size_t)n * sizeof(float));
            dequantize_row(src, buf, n, type);
            float sum = vec_dot_f32_f32(buf, x, n);
            if (buf != tmp) free(buf);
            return sum;
        }
    }
}
