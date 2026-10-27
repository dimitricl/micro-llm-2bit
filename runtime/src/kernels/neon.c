// Noyau ARM NEON (Apple Silicon, arm64) : matvec int8 x ternaire.
//
// Déballage scalaire des codes 2 bits (4 poids par octet) puis arithmétique
// NEON : wd[i] = sign(w[i]) * act[i] = (sign==+1 ? act : 0) - (sign==-1 ? act : 0),
// accumulation int16 par tranche de 8, puis somme exacte dans un int32.
// Aucune multiplication poids * activation.
//
// Si compilé hors AArch64, repli scalaire (mêmes résultats).
#include "ternary.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define LLM2_HAVE_NEON 1
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
#if defined(LLM2_HAVE_NEON)
        const int16x8_t zero = vdupq_n_s16(0);
        const int16x8_t one = vdupq_n_s16(1);
        const int16x8_t mone = vdupq_n_s16(-1);
        for (int i = 0; i < group_size; i += 8) {
            int16_t s[8];
            unpack8(w + i / 4, s);
            int16x8_t va = vmovl_s8(vld1_s8(a + i));
            int16x8_t vs = vld1q_s16(s);
            int16x8_t padd = vbslq_s16(vceqq_s16(vs, one), va, zero);
            int16x8_t nadd = vbslq_s16(vceqq_s16(vs, mone), va, zero);
            // |somme| <= 8*127 : aucun débordement int16.
            acc += (int32_t)vaddvq_s16(vsubq_s16(padd, nadd));
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
#if defined(LLM2_HAVE_NEON)
    return "neon";
#else
    return "neon-fallback-scalar";
#endif
}
