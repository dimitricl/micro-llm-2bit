// Noyau x86_64 AVX2 (fallback Intel Mac) : matvec int8 x ternaire.
//
// Même structure que neon.c : unpack scalaire des codes 2 bits + arithmétique
// vectorielle AVX2/SSSE3 (add/sub int16, accumulation int32 exacte).
// Série par défaut (pas d'OpenMP : clang Apple ne le fournit pas ; sur Linux
// on pourra ajouter -fopenmp si voulu, comportement inchangé).
#include "ternary.h"

#include <stdint.h>

#if defined(__AVX2__) && defined(__x86_64__)
#include <immintrin.h>
#define HAVE_AVX2 1
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
    int num_groups = cols / group_size;
    for (int g = 0; g < num_groups; g++) {
        int32_t acc = 0;
        const int8_t *a = act + g * group_size;
        const uint8_t *w = w_row + (g * group_size) / 4;
#if defined(HAVE_AVX2)
        int16_t s[8];
        __m128i one = _mm_set1_epi16(1);
        __m128i mone = _mm_set1_epi16(-1);
        __m128i ones = _mm_set1_epi16(1); // pour madd -> somme horizontale
        for (int i = 0; i < group_size; i += 8) {
            unpack8(w + i / 4, s);
            __m128i va = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)(a + i)));
            __m128i vs = _mm_loadu_si128((const __m128i *)s);
            __m128i padd = _mm_and_si128(_mm_cmpeq_epi16(vs, one), va);
            __m128i nadd = _mm_and_si128(_mm_cmpeq_epi16(vs, mone), va);
            __m128i contrib = _mm_sub_epi16(padd, nadd);
            __m128i sum32 = _mm_madd_epi16(contrib, ones); // 4 x int32
            sum32 = _mm_hadd_epi32(sum32, sum32);
            sum32 = _mm_hadd_epi32(sum32, sum32);
            acc += _mm_cvtsi128_si32(sum32);
        }
#else
        for (int i = 0; i < group_size; i += 4) {
            uint8_t b = w[i / 4];
            for (int j = 0; j < 4; j++) {
                uint8_t code = (uint8_t)((b >> (2 * j)) & 0x3);
                int8_t sign = (code == 0x1) ? 1 : (code == 0x3 ? -1 : 0);
                if (sign == 1) acc += a[i + j];
                else if (sign == -1) acc -= a[i + j];
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
    int row_stride = cols / 4;
    int num_groups = cols / group_size;
    for (int n = 0; n < rows; n++) {
        row_matvec(act, w_packed + (size_t)n * (size_t)row_stride,
                   scales + (size_t)n * (size_t)num_groups, act_scale,
                   out + n, cols, group_size);
    }
}

const char *kernel_variant(void) {
#if defined(HAVE_AVX2)
    return "avx2";
#else
    return "avx2-fallback-portable";
#endif
}
