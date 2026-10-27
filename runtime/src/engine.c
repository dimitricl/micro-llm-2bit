// Moteur d'inférence LLM2 : mmap du .bin, forward pas-à-pas, KV cache int8.
//
// Portage fidèle de micro_llm/reference.py (oracle NumPy). Les écarts
// numériques attendus se limitent à l'ordre des sommes flottantes ; les
// accumulations ternaires sont exactes en int32.
#include "llm2.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define LLM2_MAGIC "LLM2"
#define LLM2_NORM_EPS 1e-6f
#define LLM2_ROPE_THETA 10000.0f

// ---------------------------------------------------------------------------
// Lecture du fichier
// ---------------------------------------------------------------------------

static int map_file(llm2_model *m, const char *path) {
#if defined(_WIN32)
    FILE *fh = fopen(path, "rb");
    if (!fh) return -1;
    if (fseek(fh, 0, SEEK_END) != 0) {
        fclose(fh);
        return -1;
    }
    long size = ftell(fh);
    if (size < LLM2_HEADER_SIZE) {
        fclose(fh);
        return -1;
    }
    rewind(fh);
    uint8_t *buf = (uint8_t *)malloc((size_t)size);
    if (!buf) {
        fclose(fh);
        return -1;
    }
    if (fread(buf, 1, (size_t)size, fh) != (size_t)size) {
        free(buf);
        fclose(fh);
        return -1;
    }
    fclose(fh);
    m->fd = -1;
    m->owned_map = buf;
    m->map = buf;
    m->map_size = (size_t)size;
    return 0;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < LLM2_HEADER_SIZE) {
        close(fd);
        return -1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        close(fd);
        return -1;
    }
    m->fd = fd;
    m->map = (const uint8_t *)p;
    m->map_size = (size_t)st.st_size;
    return 0;
#endif
}

static uint32_t read_le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            uint32_t e = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3ffu;
            f = (sign << 31) | ((e + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t off;
} cursor;

static const uint8_t *cur_take(cursor *c, size_t n) {
    if (n > c->size - c->off) return NULL;
    const uint8_t *p = c->base + c->off;
    c->off += n;
    return p;
}

static int aligned4(const void *p) { return ((uintptr_t)p & 3u) == 0; }

// ---------------------------------------------------------------------------
// Chargement
// ---------------------------------------------------------------------------

static int fail(llm2_model *m, const char *msg) {
    fprintf(stderr, "[llm2] %s\n", msg);
    llm2_free(m);
    return -1;
}

int llm2_load(llm2_model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;
    if (map_file(m, path) != 0) return fail(m, "impossible d'ouvrir le modele");

    if (m->map_size < LLM2_HEADER_SIZE || memcmp(m->map, LLM2_MAGIC, 4) != 0)
        return fail(m, "header LLM2 invalide");

    uint32_t h[10];
    for (int i = 0; i < 10; i++) h[i] = read_le32(m->map + 4 + 4 * i);
    m->version = h[0];
    m->vocab = h[1];
    m->d = h[2];
    m->layers = h[3];
    m->hq = h[4];
    m->hkv = h[5];
    m->ffn = h[6];
    m->seq_max = h[7];
    m->group = h[8];
    m->head_dim = h[9];

    if (m->version != 1 && m->version != 2)
        return fail(m, "version LLM2 non supportee");
    if (m->layers == 0 || m->layers > LLM2_MAX_LAYERS)
        return fail(m, "nombre de couches hors limites");
    if (m->d == 0 || m->vocab == 0 || m->seq_max == 0)
        return fail(m, "dimensions nulles");
    if (m->head_dim != m->d / m->hq || m->hq % m->hkv != 0)
        return fail(m, "tetes incompatibles");
    if (m->d % m->group != 0 || m->ffn % m->group != 0)
        return fail(m, "group_size incompatible");

    const int V = (int)m->vocab, D = (int)m->d, L = (int)m->layers;
    const int Hq = (int)m->hq, Hkv = (int)m->hkv, Dh = (int)m->head_dim;
    const int F = (int)m->ffn, G = (int)m->group, S = (int)m->seq_max;

    cursor c = {m->map, m->map_size, LLM2_HEADER_SIZE};

    m->embed_q = (const int8_t *)cur_take(&c, (size_t)V * (size_t)D);
    const uint8_t *embed_s = cur_take(&c, (size_t)V * 4);
    if (!m->embed_q || !embed_s) return fail(m, "embeddings tronques");
    m->embed_s = (const float *)embed_s;

    // Échelles fp16 -> fp32 : un seul tableau possédé.
    size_t total_scales = 0;
    if (m->version == 2) {
        size_t per_layer = (size_t)2 * D * D + (size_t)2 * Hkv * Dh * D +
                           (size_t)3 * F * D;
        total_scales = (size_t)L * (per_layer / (size_t)G);
        m->scales_owned = (float *)malloc(total_scales * sizeof(float));
        if (!m->scales_owned) return fail(m, "allocation echelles");
        m->scales_owned_n = total_scales;
    }

    const int rows[7] = {D, Hkv * Dh, Hkv * Dh, D, F, F, D};
    const int cols[7] = {D, D, D, Hq * Dh, D, D, F};
    size_t scale_idx = 0;
    for (int li = 0; li < L; li++) {
        for (int k = 0; k < 7; k++) {
            const int r = rows[k], cc = cols[k];
            const size_t n_packed = (size_t)r * (size_t)cc / 4;
            const size_t n_scales = (size_t)r * (size_t)cc / (size_t)G;
            const uint8_t *packed = cur_take(&c, n_packed);
            if (!packed) return fail(m, "lineaires tronques");
            llm2_linear *lin = &m->lin[li][k];
            lin->packed = packed;
            lin->rows = r;
            lin->cols = cc;
            lin->group = G;
            if (m->version == 1) {
                const uint8_t *raw = cur_take(&c, n_scales * 4);
                if (!raw) return fail(m, "echelles tronquees");
                lin->scales = (const float *)raw;
            } else {
                const uint8_t *raw = cur_take(&c, n_scales * 2);
                if (!raw) return fail(m, "echelles tronquees");
                for (size_t i = 0; i < n_scales; i++) {
                    uint16_t hv = (uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8));
                    m->scales_owned[scale_idx + i] = half_to_float(hv);
                }
                lin->scales = m->scales_owned + scale_idx;
                scale_idx += n_scales;
            }
        }
        const uint8_t *an = cur_take(&c, (size_t)D * 4);
        const uint8_t *mn = cur_take(&c, (size_t)D * 4);
        if (!an || !mn) return fail(m, "normes tronquees");
        m->attn_norm[li] = (const float *)an;
        m->mlp_norm[li] = (const float *)mn;
    }
    const uint8_t *fn = cur_take(&c, (size_t)D * 4);
    if (!fn) return fail(m, "norme finale tronquee");
    m->final_norm = (const float *)fn;

    if (c.off != m->map_size)
        return fail(m, "taille de fichier incoherente avec le header");

    // Les pointeurs float doivent être alignés sur 4 octets.
    if (!aligned4(m->embed_s) || !aligned4(m->final_norm))
        return fail(m, "donnees float mal alignees");
    for (int li = 0; li < L; li++) {
        if (!aligned4(m->attn_norm[li]) || !aligned4(m->mlp_norm[li]))
            return fail(m, "normes mal alignees");
        for (int k = 0; k < 7; k++) {
            if (!aligned4(m->lin[li][k].scales))
                return fail(m, "echelles mal alignees");
        }
    }

    // Caches KV int8 + échelles par tête.
    const size_t kv_n = (size_t)L * (size_t)S * (size_t)Hkv * (size_t)Dh;
    const size_t kv_s = (size_t)L * (size_t)S * (size_t)Hkv;
    m->k_cache = (int8_t *)calloc(kv_n, 1);
    m->v_cache = (int8_t *)calloc(kv_n, 1);
    m->k_scales = (float *)calloc(kv_s, sizeof(float));
    m->v_scales = (float *)calloc(kv_s, sizeof(float));

    const int max_dim = F > D ? F : D;
    m->x = (float *)calloc((size_t)D, sizeof(float));
    m->h = (float *)calloc((size_t)D, sizeof(float));
    m->q = (float *)calloc((size_t)Hq * Dh, sizeof(float));
    m->k = (float *)calloc((size_t)Hkv * Dh, sizeof(float));
    m->v = (float *)calloc((size_t)Hkv * Dh, sizeof(float));
    m->attn = (float *)calloc((size_t)Hq * Dh, sizeof(float));
    m->gate = (float *)calloc((size_t)F, sizeof(float));
    m->up = (float *)calloc((size_t)F, sizeof(float));
    m->tmp = (float *)calloc((size_t)F, sizeof(float));
    m->scores = (float *)calloc((size_t)S, sizeof(float));
    m->logits = (float *)calloc((size_t)V, sizeof(float));
    m->rope_inv = (float *)calloc((size_t)(Dh / 2), sizeof(float));
    m->act_q = (int8_t *)calloc((size_t)max_dim, 1);

    if (!m->k_cache || !m->v_cache || !m->k_scales || !m->v_scales || !m->x ||
        !m->h || !m->q || !m->k || !m->v || !m->attn || !m->gate || !m->up ||
        !m->tmp || !m->scores || !m->logits || !m->rope_inv || !m->act_q)
        return fail(m, "allocation memoire echouee");

    for (int i = 0; i < Dh / 2; i++) {
        m->rope_inv[i] = expf(-logf(LLM2_ROPE_THETA) * (float)(2 * i) / (float)Dh);
    }

    m->pos = 0;
    return 0;
}

void llm2_free(llm2_model *m) {
    if (!m) return;
#if !defined(_WIN32)
    if (m->map && !m->owned_map) {
        munmap((void *)m->map, m->map_size);
    }
    if (m->fd >= 0) close(m->fd);
#endif
    free(m->owned_map);
    free(m->scales_owned);
    free(m->k_cache);
    free(m->v_cache);
    free(m->k_scales);
    free(m->v_scales);
    free(m->x);
    free(m->h);
    free(m->q);
    free(m->k);
    free(m->v);
    free(m->attn);
    free(m->gate);
    free(m->up);
    free(m->tmp);
    free(m->scores);
    free(m->logits);
    free(m->rope_inv);
    free(m->act_q);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

// ---------------------------------------------------------------------------
// Briques numériques
// ---------------------------------------------------------------------------

static void rmsnorm(float *out, const float *x, const float *w, int d) {
    float acc = 0.0f;
    for (int i = 0; i < d; i++) acc += x[i] * x[i];
    const float inv = 1.0f / sqrtf(acc / (float)d + LLM2_NORM_EPS);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
}

static float quantize_act(const float *x, int n, int8_t *q) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    const float scale = amax > 1e-12f ? amax / 127.0f : 1e-8f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        float v = rintf(x[i] * inv);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        q[i] = (int8_t)v;
    }
    return scale;
}

static void rope(float *v, int heads, int dh, int pos, const float *inv) {
    for (int h = 0; h < heads; h++) {
        float *row = v + (size_t)h * (size_t)dh;
        for (int j = 0; j < dh / 2; j++) {
            const float ang = (float)pos * inv[j];
            const float cs = cosf(ang), sn = sinf(ang);
            const float x0 = row[2 * j], x1 = row[2 * j + 1];
            row[2 * j] = x0 * cs - x1 * sn;
            row[2 * j + 1] = x0 * sn + x1 * cs;
        }
    }
}

static void kv_store(int8_t *cache, float *scales, int li, int pos, int hkv,
                     int dh, int seq_max, const float *v) {
    int8_t *dst = cache + ((size_t)li * (size_t)seq_max + (size_t)pos) *
                               ((size_t)hkv * (size_t)dh);
    for (int h = 0; h < hkv; h++) {
        const float scale = quantize_act(v + (size_t)h * (size_t)dh, dh,
                                         dst + (size_t)h * (size_t)dh);
        scales[((size_t)li * (size_t)seq_max + (size_t)pos) * (size_t)hkv +
               (size_t)h] = scale;
    }
}

// ---------------------------------------------------------------------------
// Forward d'un token
// ---------------------------------------------------------------------------

const float *llm2_step(llm2_model *m, int token) {
    if (!m || token < 0 || (uint32_t)token >= m->vocab) return NULL;
    if (m->pos >= (int)m->seq_max) return NULL;

    const int D = (int)m->d, L = (int)m->layers;
    const int Hq = (int)m->hq, Hkv = (int)m->hkv, Dh = (int)m->head_dim;
    const int F = (int)m->ffn, V = (int)m->vocab, S = (int)m->seq_max;
    const int rep = Hq / Hkv;
    const int p = m->pos;
    const size_t kv_row = (size_t)Hkv * (size_t)Dh;

    for (int i = 0; i < D; i++)
        m->x[i] = (float)m->embed_q[(size_t)token * (size_t)D + (size_t)i] *
                  m->embed_s[token];

    for (int li = 0; li < L; li++) {
        llm2_linear *lq = &m->lin[li][0];
        llm2_linear *lk = &m->lin[li][1];
        llm2_linear *lv = &m->lin[li][2];
        llm2_linear *lo = &m->lin[li][3];
        llm2_linear *lg = &m->lin[li][4];
        llm2_linear *lu = &m->lin[li][5];
        llm2_linear *ld = &m->lin[li][6];

        // --- attention ---
        rmsnorm(m->h, m->x, m->attn_norm[li], D);
        float hs = quantize_act(m->h, D, m->act_q);
        ternary_matvec(m->act_q, lq->packed, lq->scales, hs, m->q, lq->rows,
                       lq->cols, lq->group);
        ternary_matvec(m->act_q, lk->packed, lk->scales, hs, m->k, lk->rows,
                       lk->cols, lk->group);
        ternary_matvec(m->act_q, lv->packed, lv->scales, hs, m->v, lv->rows,
                       lv->cols, lv->group);

        rope(m->q, Hq, Dh, p, m->rope_inv);
        rope(m->k, Hkv, Dh, p, m->rope_inv);

        kv_store(m->k_cache, m->k_scales, li, p, Hkv, Dh, S, m->k);
        kv_store(m->v_cache, m->v_scales, li, p, Hkv, Dh, S, m->v);
        const float *ks = m->k_scales + ((size_t)li * (size_t)S + (size_t)p) * (size_t)Hkv;
        const float *vs = m->v_scales + ((size_t)li * (size_t)S + (size_t)p) * (size_t)Hkv;
        const int8_t *kbase = m->k_cache + ((size_t)li * (size_t)S) * kv_row;
        const int8_t *vbase = m->v_cache + ((size_t)li * (size_t)S) * kv_row;
        const float attn_scale = 1.0f / sqrtf((float)Dh);

        for (int hh = 0; hh < Hq; hh++) {
            const int kvh = hh / rep;
            const float *qh = m->q + (size_t)hh * (size_t)Dh;
            float maxs = -INFINITY;
            for (int t = 0; t <= p; t++) {
                const int8_t *krow = kbase + (size_t)t * kv_row + (size_t)kvh * (size_t)Dh;
                float acc = 0.0f;
                for (int i = 0; i < Dh; i++) acc += (float)krow[i] * qh[i];
                const float s = acc * ks[t * Hkv + kvh] * attn_scale;
                m->scores[t] = s;
                if (s > maxs) maxs = s;
            }
            float sum = 0.0f;
            for (int t = 0; t <= p; t++) {
                const float e = expf(m->scores[t] - maxs);
                m->scores[t] = e;
                sum += e;
            }
            const float inv_sum = 1.0f / sum;
            float *ah = m->attn + (size_t)hh * (size_t)Dh;
            for (int i = 0; i < Dh; i++) {
                float acc = 0.0f;
                for (int t = 0; t <= p; t++) {
                    const int8_t *vrow = vbase + (size_t)t * kv_row + (size_t)kvh * (size_t)Dh;
                    acc += m->scores[t] * inv_sum * (float)vrow[i] *
                           vs[t * Hkv + kvh];
                }
                ah[i] = acc;
            }
        }

        float as = quantize_act(m->attn, Hq * Dh, m->act_q);
        ternary_matvec(m->act_q, lo->packed, lo->scales, as, m->h, lo->rows,
                       lo->cols, lo->group);
        for (int i = 0; i < D; i++) m->x[i] += m->h[i];

        // --- MLP SwiGLU ---
        rmsnorm(m->h, m->x, m->mlp_norm[li], D);
        float h2s = quantize_act(m->h, D, m->act_q);
        ternary_matvec(m->act_q, lg->packed, lg->scales, h2s, m->gate,
                       lg->rows, lg->cols, lg->group);
        ternary_matvec(m->act_q, lu->packed, lu->scales, h2s, m->up, lu->rows,
                       lu->cols, lu->group);
        for (int i = 0; i < F; i++) {
            const float g = m->gate[i];
            m->tmp[i] = (g / (1.0f + expf(-g))) * m->up[i];
        }
        float ds = quantize_act(m->tmp, F, m->act_q);
        ternary_matvec(m->act_q, ld->packed, ld->scales, ds, m->h, ld->rows,
                       ld->cols, ld->group);
        for (int i = 0; i < D; i++) m->x[i] += m->h[i];
    }

    rmsnorm(m->h, m->x, m->final_norm, D);
    for (int v = 0; v < V; v++) {
        const int8_t *row = m->embed_q + (size_t)v * (size_t)D;
        float acc = 0.0f;
        for (int i = 0; i < D; i++) acc += (float)row[i] * m->h[i];
        m->logits[v] = acc * m->embed_s[v];
    }

    m->pos++;
    return m->logits;
}

int llm2_argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}
