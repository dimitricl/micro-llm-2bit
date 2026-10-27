// Noyau x86_64 AVX2 : matvec int8 x ternaire.
//
// Même stratégie que neon.c : déballage scalaire des codes 2 bits puis
// arithmétique vectorielle (add/sub int16, accumulation int32 exacte).
// Repli scalaire si le compilateur ne définit pas __AVX2__.
#include "ternary.h"

#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#define LLM2_HAVE_AVX2 1
#endif

static inline void unpack8(const uint8_t *w, int16_t s[8]) {
    for (int j = 0; j < 8; j++) {
        uint8_t code = (uint8_t)((w[j / 4] >> (2 * (j % 4))) & 0x3);
        s[j] = (code == 0x1) ? (int16_t)1 : (code == 0x3 ? (int16_t)-1 : (int16_t)0);
    }
}

static void row_matvec(const int8_t *act, const uint8_t *w_row,
                       const float *scales_row, float act_scale, float *out,
                       int cols, int group_size) {
    float y = 0.0f;
    const int num_groups = cols / group_size;
    for (int g = 0; g < num_groups; g++) {
        int32_t acc = 0;
        const int8_t *a = act + g * group_size;
        const uint8_t *w = w_row + (g * group_size) / 4;
#if defined(LLM2_HAVE_AVX2)
        const __m128i zero = _mm_setzero_si128();
        const __m128i one = _mm_set1_epi16(1);
        const __m128i mone = _mm_set1_epi16(-1);
        for (int i = 0; i < group_size; i += 8) {
            int16_t s[8];
            unpack8(w + i / 4, s);
            __m128i va = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)(a + i)));
            __m128i vs = _mm_loadu_si128((const __m128i *)s);
            __m128i padd = _mm_and_si128(_mm_cmpeq_epi16(vs, one), va);
            __m128i nadd = _mm_and_si128(_mm_cmpeq_epi16(vs, mone), va);
            __m128i d = _mm_sub_epi16(padd, nadd);
            // |somme| <= 8*127 : aucun débordement int16.
            __m128i sum16 = _mm_add_epi16(d, _mm_srli_si128(d, 8));
            sum16 = _mm_add_epi16(sum16, _mm_srli_si128(sum16, 4));
            sum16 = _mm_add_epi16(sum16, _mm_srli_si128(sum16, 2));
            acc += (int32_t)_mm_extract_epi16(sum16, 0);
            (void)zero;
        }
#else
        for (int i = 0; i < group_size; i += 4) {
            uint8_t b = w[i / 4];
            for (int j = 0; j < 4; j++) {
                uint8_t code = (uint8_t)((b >> (2 * j)) & 0x3);
                if (code == 0x1) acc += a[i + j];
                else if (code == 0x3) acc -= a[i + j];
            }
        }
#endif
        y += (float)acc * scales_row[g];
    }
    *out = y * act_scale;
}

void ternary_matvec(const int8_t *act, const uint8_t *w_packed,
                    const float *scales, float act_scale, float *out,
                    int rows, int cols, int group_size) {
    const int row_stride = cols / 4;
    const int num_groups = cols / group_size;
    for (int n = 0; n < rows; n++) {
        row_matvec(act, w_packed + (size_t)n * (size_t)row_stride,
                   scales + (size_t)n * (size_t)num_groups, act_scale, out + n,
                   cols, group_size);
    }
}

const char *kernel_variant(void) {
#if defined(LLM2_HAVE_AVX2)
    return "avx2";
#else
    return "avx2-fallback-scalar";
#endif
}
