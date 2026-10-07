// API commune des noyaux matvec int8 x ternaire (neon.c / avx2.c / portable.c).
//
// Rôle : y[n] = act_scale * Σ_g scales[n*G + g] * Σ_{i ∈ g} sign(w[n][i]) * act[i]
// avec sign : +1 -> addition, -1 -> soustraction, 0 -> ignoré (zéro saut de calcul).
// Codage 2 bits (cf. quant.py) : 0b00 = 0, 0b01 = +1, 0b11 = -1, 0b10 = 0 (réservé).
// Premier poids du groupe dans les bits faibles de l'octet.
//
// Layout :
//   act      : int8[K]
//   w_packed : uint8[rows * (K/4)] (K multiple de 4, ligne n à l'offset n*(K/4))
//   scales   : float[rows * num_groups], num_groups = K / group_size
//   out      : float[rows]
// Tous les pointeurs doivent être alignés au moins sur 1 octet (pas d'exigence SIMD
// stricte : les noyaux utilisent des loads non alignés sûrs).
#pragma once

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
