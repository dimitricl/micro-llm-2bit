// Noyau portable (C99 strict) : référence et dernier recours.
//
// Même API et mêmes résultats bit-exacts (accumulation entière int32 puis
// pondération float) que neon.c / avx2.c. Multithread via Grand Central
// Dispatch (dispatch_apply) sur macOS, boucle série ailleurs.
#include "ternary.h"

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

// Décode un octet packé en 4 signes {-1, 0, +1} (cf. ternary.h).
static inline void unpack4(uint8_t b, int8_t s[4]) {
    for (int j = 0; j < 4; j++) {
        uint8_t code = (uint8_t)((b >> (2 * j)) & 0x3);
        s[j] = (code == 0x1) ? (int8_t)1 : (code == 0x3 ? (int8_t)-1 : (int8_t)0);
    }
}

static void row_matvec(const int8_t *act, const uint8_t *w_row,
                       const float *scales_row, float act_scale, float *out,
                       int cols, int group_size) {
    float y = 0.0f;
    int num_groups = cols / group_size;
    int8_t s[4];
    for (int g = 0; g < num_groups; g++) {
        int32_t acc = 0; // accumulation entière exacte
        const int8_t *a = act + g * group_size;
        const uint8_t *w = w_row + (g * group_size) / 4;
        for (int i = 0; i < group_size; i += 4) {
            unpack4(w[i / 4], s);
            // Additions/soustractions pures, les zéros ne coûtent qu'un test.
            if (s[0] == 1) acc += a[i];
            else if (s[0] == -1) acc -= a[i];
            if (s[1] == 1) acc += a[i + 1];
            else if (s[1] == -1) acc -= a[i + 1];
            if (s[2] == 1) acc += a[i + 2];
            else if (s[2] == -1) acc -= a[i + 2];
            if (s[3] == 1) acc += a[i + 3];
            else if (s[3] == -1) acc -= a[i + 3];
        }
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

const char *kernel_variant(void) { return "portable"; }
