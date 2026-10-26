"""Phase 0 G — IC du taux hors lexique (t08 vs A/B/C), bootstrap apparié.

Le taux OOV de `eval_surface.py` est un rapport de deux comptes (mots hors
lexique / mots) : sans IC, l'écart 0,44 % (t08) vs 0,67 % (A) n'est pas
interprétable. Ici on rééchantillonne les **textes** (et non les mots, qui
sont corrélés dans un même texte), de façon stratifiée par source
(32 greedy + 32 sampling + 200 sonde), et on apparie les indices entre
modèles pour estimer l'IC de la différence.

Aucune génération : on relit les sorties déjà produites.

Sorties : docs/results/oov_bootstrap_G.json.
Usage : .venv/bin/python tools/oov_bootstrap_G.py
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from eval_surface import (lexicon_at, load_lexicon, load_probe_results,
                          parse_samples_md, words_of)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)

LEX = "runs/F-diag/lexique_fr.tsv.gz"
MIN_FREQ = 5
N_BOOT = 10_000
SEED = 0
RUNS = {
    "t08": ("eval/F-Et08-fixed/samples.md", "eval/Et08-probe/results.json"),
    "F-A": ("eval/F-A-fixed/samples.md", "eval/F-A-probe/results.json"),
    "F-B": ("eval/F-B-fixed/samples.md", "eval/F-B-probe/results.json"),
    "F-C": ("eval/F-C-fixed/samples.md", "eval/F-C-probe/results.json"),
}
OUT = "docs/results/oov_bootstrap_G.json"


def per_text(paths: tuple[str, str], lexicon: set[str]) -> dict[str, list[tuple[int, int]]]:
    samples_md, probe = paths
    parsed = parse_samples_md(samples_md)
    greedy = [v.get("greedy", "") for v in parsed.values()]
    sampling = [v.get("sampling", "") for v in parsed.values()]
    items = load_probe_results(probe)
    sonde = [it.get("gen", "") for it in items]

    def counts(texts: list[str]) -> list[tuple[int, int]]:
        out = []
        for t in texts:
            ws = words_of(t)
            out.append((sum(1 for w in ws if w not in lexicon), len(ws)))
        return out

    return {"greedy": counts(greedy), "sampling": counts(sampling),
            "sonde": counts(sonde)}


def rate(vec: list[tuple[int, int]]) -> float:
    o = sum(a for a, _ in vec)
    n = sum(b for _, b in vec)
    return o / n if n else 0.0


def main() -> int:
    counts = load_lexicon(LEX)
    lexicon = lexicon_at(counts, MIN_FREQ)
    data = {name: per_text(p, lexicon) for name, p in RUNS.items()}
    # Vérifie que les textes sont bien appariés (même nombre par strate).
    sizes = {name: {k: len(v) for k, v in d.items()} for name, d in data.items()}
    assert len({tuple(sorted(s.items())) for s in sizes.values()}) == 1, sizes
    strata = ["greedy", "sampling", "sonde"]
    point = {name: {k: rate(v[k]) for k in strata} for name, v in data.items()}
    for name in point:
        point[name]["combine"] = (
            sum(sum(a for a, _ in data[name][k]) for k in strata)
            / sum(sum(b for _, b in data[name][k]) for k in strata))

    rng = random.Random(SEED)
    n_by = {k: len(data["t08"][k]) for k in strata}
    boots: dict[str, list[float]] = {name: [] for name in data}
    diffs: dict[str, list[float]] = {name: [] for name in data if name != "t08"}
    for _ in range(N_BOOT):
        idx = {k: [rng.randrange(n_by[k]) for _ in range(n_by[k])] for k in strata}
        rates = {}
        for name, d in data.items():
            o = n = 0
            for k in strata:
                for i in idx[k]:
                    a, b = d[k][i]
                    o += a
                    n += b
            rates[name] = o / n if n else 0.0
            boots[name].append(rates[name])
        for name in diffs:
            diffs[name].append(rates[name] - rates["t08"])

    def ci(xs: list[float]) -> list[float]:
        s = sorted(xs)
        return [s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]]

    out = {"lexique": LEX, "min_freq": MIN_FREQ, "n_boot": N_BOOT,
           "appariement": sizes["t08"], "runs": {}}
    for name in data:
        d = {"point": point[name], "ic95_combine": ci(boots[name])}
        if name != "t08":
            ds = sorted(diffs[name])
            p = 2 * min(sum(1 for x in ds if x <= 0), sum(1 for x in ds if x >= 0)) / len(ds)
            d["diff_vs_t08"] = {
                "point": point[name]["combine"] - point["t08"]["combine"],
                "ic95": ci(diffs[name]),
                "p_bootstrap_bilateral": min(1.0, p),
            }
        out["runs"][name] = d
        print(f"[oov] {name:5s} combine {100*point[name]['combine']:.2f}% "
              f"IC95 [{100*d['ic95_combine'][0]:.2f}, {100*d['ic95_combine'][1]:.2f}]"
              + (f"  diff vs t08 {100*d['diff_vs_t08']['point']:+.2f} pp "
                 f"IC95 [{100*d['diff_vs_t08']['ic95'][0]:+.2f}, "
                 f"{100*d['diff_vs_t08']['ic95'][1]:+.2f}] "
                 f"p={d['diff_vs_t08']['p_bootstrap_bilateral']:.3f}"
                 if name != "t08" else ""), flush=True)

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[oov] -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
