// API commune des noyaux matvec int8 x ternaire.
//
//   y[n] = act_scale * Σ_g scales[n*G + g] * Σ_{i ∈ g} sign(w[n][i]) * act[i]
//
// Codage 2 bits : 0b00 = 0, 0b01 = +1, 0b11 = -1, 0b10 = 0 (réservé).
// Premier poids du groupe dans les bits faibles de l'octet.
//
// Layout :
//   act      : int8[cols]
//   w_packed : uint8[rows * (cols/4)]  (ligne n à l'offset n*(cols/4))
//   scales   : float[rows * (cols/group)]
//   out      : float[rows]
//
// L'accumulation intra-groupe est un int32 exact : les trois variantes
// (scalaire, NEON, AVX2) produisent les mêmes bits.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ternary_matvec(const int8_t *act, const uint8_t *w_packed,
                    const float *scales, float act_scale, float *out,
                    int rows, int cols, int group_size);

const char *kernel_variant(void);

#ifdef __cplusplus
}
#endif
