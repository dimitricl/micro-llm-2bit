// Noyau ARM NEON (Apple Silicon, arm64) : matvec int8 x ternaire.
//
// Stratégie : déballage scalaire des codes 2 bits (pas cher : 1 lecture
// d'octet pour 4 poids) + arithmétique vectorielle NEON (add/sub int16,
// accumulation int32 exacte). Aucune multiplication poids*activation.
// Multithread via Grand Central Dispatch (dispatch_apply).
//
// Si compilé hors AArch64, on replie sur le chemin scalaire (même résultats).
#include "ternary.h"

#include <stdint.h>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define HAVE_NEON 1
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
#if defined(HAVE_NEON)
        // Par tranches de 8 poids : unpack scalaire, calcul NEON.
        int16_t s[8];
        int16x8_t zero = vdupq_n_s16(0);
        int16x8_t one = vdupq_n_s16(1);
        int16x8_t mone = vdupq_n_s16(-1);
        for (int i = 0; i < group_size; i += 8) {
            unpack8(w + i / 4, s);
            int16x8_t va = vmovl_s8(vld1_s8(a + i)); // 8 act -> int16
            int16x8_t vs = vld1q_s16(s);            // 8 signes
            // +act là où sign==+1, -act là où sign==-1, 0 ailleurs.
            int16x8_t padd = vbslq_s16(vceqq_s16(vs, one), va, zero);
            int16x8_t nadd = vbslq_s16(vceqq_s16(vs, mone), va, zero);
            acc += (int32_t)vaddvq_s16(vsubq_s16(padd, nadd));
        }
#else
        int8_t s4[4];
        for (int i = 0; i < group_size; i += 4) {
            uint8_t b = w[i / 4];
            for (int j = 0; j < 4; j++) {
                uint8_t code = (uint8_t)((b >> (2 * j)) & 0x3);
                int8_t sign = (code == 0x1) ? 1 : (code == 0x3 ? -1 : 0);
                if (sign == 1) acc += a[i + j];
                else if (sign == -1) acc -= a[i + j];
            }
            (void)s4;
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
#if defined(__APPLE__)
    dispatch_apply((size_t)rows,
                   dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                   ^(size_t n) {
                       row_matvec(act, w_packed + n * (size_t)row_stride,
                                  scales + n * (size_t)num_groups, act_scale,
                                  out + n, cols, group_size);
                   });
#else
    for (int n = 0; n < rows; n++) {
        row_matvec(act, w_packed + (size_t)n * (size_t)row_stride,
                   scales + (size_t)n * (size_t)num_groups, act_scale,
                   out + n, cols, group_size);
    }
#endif
}

const char *kernel_variant(void) {
#if defined(HAVE_NEON)
    return "neon";
#else
    return "neon-fallback-portable";
#endif
}
