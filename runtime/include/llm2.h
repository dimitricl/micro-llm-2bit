// Runtime d'inférence LLM2 : structures et API.
#pragma once

#include <stddef.h>
#include <stdint.h>

#define LLM2_HEADER_SIZE 44
#define LLM2_MAX_LAYERS 64

typedef struct {
    const uint8_t *packed;  // rows * cols / 4 octets
    const float *scales;    // rows * (cols / group)
    int rows;
    int cols;
    int group;
} llm2_linear;

typedef struct {
    uint32_t version;
    uint32_t vocab;
    uint32_t d;
    uint32_t layers;
    uint32_t hq;
    uint32_t hkv;
    uint32_t ffn;
    uint32_t seq_max;
    uint32_t group;
    uint32_t head_dim;

    int fd;
    const uint8_t *map;
    size_t map_size;
    uint8_t *owned_map;    // repli lecture complète (Windows) : à libérer

    const int8_t *embed_q;
    const float *embed_s;

    llm2_linear lin[LLM2_MAX_LAYERS][7];
    const float *attn_norm[LLM2_MAX_LAYERS];
    const float *mlp_norm[LLM2_MAX_LAYERS];
    const float *final_norm;

    float *scales_owned;   // conversion fp16 -> fp32 (version 2)
    size_t scales_owned_n;

    int8_t *k_cache;       // [L][S][Hkv*Dh]
    int8_t *v_cache;
    float *k_scales;       // [L][S][Hkv]
    float *v_scales;

    float *rope_inv;       // [Dh/2]
    float *logits;         // [V]

    float *x, *h, *q, *k, *v, *attn, *gate, *up, *tmp;
    float *scores;         // [S]
    int8_t *act_q;         // [max(d, ffn)] quantification d'activation

    int pos;
} llm2_model;

// Charge un .bin (mmap). Retourne 0 en cas de succès, -1 sinon.
int llm2_load(llm2_model *m, const char *path);
void llm2_free(llm2_model *m);

// Avance d'un token ; retourne le buffer de logits (V floats), réutilisé.
const float *llm2_step(llm2_model *m, int token);

int llm2_argmax(const float *v, int n);

// Variante du noyau effectivement compilée ("neon", "avx2", "portable").
const char *kernel_variant(void);

// API du noyau matvec (fournie par un seul fichier de kernels/).
void ternary_matvec(const int8_t *act, const uint8_t *w_packed,
                    const float *scales, float act_scale, float *out,
                    int rows, int cols, int group_size);
