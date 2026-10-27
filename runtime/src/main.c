// CLI du runtime LLM2 : préfill + génération gloutonne, statistiques RSS.
//
// Exemples :
//   micro-llm --model exports/E-t08.bin --tokens 1,2,3 --gen 8
//   micro-llm --model exports/E-t08.bin --tokens 1 --dump-logits /tmp/l.bin
//   micro-llm --variant
#include "llm2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/resource.h>
#endif

static double self_maxrss_mb(void) {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
    return -1.0;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1.0;
#if defined(__APPLE__)
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#else
    return (double)ru.ru_maxrss / 1024.0;
#endif
#endif
}

static int parse_ids(const char *text, int *ids, int max_ids) {
    int n = 0;
    const char *p = text;
    while (*p && n < max_ids) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) return -1;
        ids[n++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
    }
    return n;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model FILE --tokens ID[,ID...] [--gen N] [--stats]\n"
            "       %s --variant\n",
            argv0, argv0);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *tokens_text = NULL;
    const char *dump_path = NULL;
    int gen = 0;
    int stats = 0;
    int show_variant = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(a, "--tokens") == 0 && i + 1 < argc) {
            tokens_text = argv[++i];
        } else if (strcmp(a, "--gen") == 0 && i + 1 < argc) {
            gen = atoi(argv[++i]);
        } else if (strcmp(a, "--dump-logits") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        } else if (strcmp(a, "--stats") == 0) {
            stats = 1;
        } else if (strcmp(a, "--variant") == 0) {
            show_variant = 1;
        } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "argument inconnu : %s\n", a);
            usage(argv[0]);
            return 2;
        }
    }

    if (show_variant) {
        printf("%s\n", kernel_variant());
        return 0;
    }
    if (!model_path || !tokens_text) {
        usage(argv[0]);
        return 2;
    }

    static llm2_model model;
    if (llm2_load(&model, model_path) != 0) return 1;

    int ids[4096];
    const int n_ids = parse_ids(tokens_text, ids, (int)(sizeof(ids) / sizeof(ids[0])));
    if (n_ids <= 0) {
        fprintf(stderr, "liste de tokens invalide\n");
        llm2_free(&model);
        return 2;
    }

    const float *logits = NULL;
    for (int i = 0; i < n_ids; i++) {
        int tok = ids[i];
        if (tok < 0 || (uint32_t)tok >= model.vocab) {
            fprintf(stderr, "token hors vocabulaire : %d\n", tok);
            llm2_free(&model);
            return 2;
        }
        logits = llm2_step(&model, tok);
        if (!logits) {
            fprintf(stderr, "contexte plein pendant le prefill\n");
            llm2_free(&model);
            return 1;
        }
    }

    int emitted = 0;
    for (int i = 0; i < gen; i++) {
        if (!logits) break;
        const int next = llm2_argmax(logits, (int)model.vocab);
        printf("%s%d", emitted ? "," : "", next);
        emitted++;
        logits = llm2_step(&model, next);
    }
    printf("\n");

    if (dump_path && logits) {
        FILE *fh = fopen(dump_path, "wb");
        if (!fh) {
            fprintf(stderr, "impossible d'ecrire %s\n", dump_path);
            llm2_free(&model);
            return 1;
        }
        if (fwrite(logits, sizeof(float), model.vocab, fh) != model.vocab) {
            fprintf(stderr, "ecriture logits incomplete\n");
            fclose(fh);
            llm2_free(&model);
            return 1;
        }
        fclose(fh);
    }

    if (stats) {
        long params = 0;
        const long emb = (long)model.vocab * (long)model.d;
        const long per_layer =
            2L * model.d * model.d + 2L * model.hkv * model.head_dim * model.d +
            3L * model.ffn * model.d;
        params = emb + (long)model.layers * per_layer + model.d;
        const double rss = self_maxrss_mb();
        fprintf(stderr,
                "[llm2] variant=%s version=%u vocab=%u d=%u layers=%u "
                "seq_max=%u pos=%d\n",
                kernel_variant(), model.version, model.vocab, model.d,
                model.layers, model.seq_max, model.pos);
        fprintf(stderr, "[llm2] params=%.3fM file=%.2fMB logits=%.2fMB "
                        "peak_rss=%.2fMB\n",
                (double)params / 1e6, (double)model.map_size / (1024.0 * 1024.0),
                (double)model.vocab * 4.0 / (1024.0 * 1024.0), rss);
    }

    llm2_free(&model);
    return 0;
}
