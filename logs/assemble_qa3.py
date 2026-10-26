"""Assemble la source unique docs/results/probe_Epilotqa3_v1.json (CPU).

Lit eval/Epilotqa3-probe/results.json (sonde) + le meilleur val du train,
calcule les axes/IC95 Wilson, le proxy de corruption « capitain » et applique
la règle de décision PRÉ-ENREGISTRÉE de l'étape 3.
"""
import json
import math
import os
import re

ROOT = "/Volumes/Lexar/micro-llm-2bit"
PROBE_RES = os.path.join(ROOT, "eval/Epilotqa3-probe/results.json")
OUT = os.path.join(ROOT, "docs/results/probe_Epilotqa3_v1.json")
TRAIN_LOG = os.path.join(ROOT, "runs/E-pilot-qa3/train.log")
MANIFEST = os.path.join(ROOT, "data/qa/qa_fr_pilot_v3.manifest.json")

T08_CE, T08_PPL = 2.9851, 19.8
PILOT1_CAPITAIN = 44


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def best_val(path):
    best = None
    pat = re.compile(r"step (\d+)/\d+ .*val_ce[=~ ]([0-9.]+).*val_ppl[=~ ]([0-9.]+)")
    pat2 = re.compile(r"val_ce[=~ ]([0-9.]+).*val_ppl[=~ ]([0-9.]+)")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if m:
            ce, ppl = float(m.group(2)), float(m.group(3))
            if best is None or ce < best[1]:
                best = (int(m.group(1)), ce, ppl)
            continue
        m = pat2.search(line)
        if m:
            ce, ppl = float(m.group(1)), float(m.group(2))
            if best is None or ce < best[1]:
                best = (None, ce, ppl)
    return best


def main():
    r = json.load(open(PROBE_RES, encoding="utf-8"))
    items = r["items"]
    man = json.load(open(MANIFEST, encoding="utf-8"))

    axes = {}
    keys = [("global", None, None)]
    for cat in ("capitales", "dates", "definitions", "unites_sciences"):
        keys.append((cat, None, cat))
        for sl in ("par_coeur", "reformule"):
            keys.append((f"{cat}_{sl}", sl, cat))
    for name, sl, cat in keys:
        sel = [i for i in items
               if (sl is None or i["slice"] == sl)
               and (cat is None or i["categorie"] == cat)]
        n = len(sel)
        h = sum(1 for i in sel if i["hit"])
        axes[name] = {"hits": h, "n": n, "taux": h / n,
                      "IC95": [round(x, 4) for x in wilson(h, n)]}
    for sl in ("par_coeur", "reformule"):
        sel = [i for i in items if i["slice"] == sl]
        n = len(sel)
        h = sum(1 for i in sel if i["hit"])
        axes[sl] = {"hits": h, "n": n, "taux": h / n,
                    "IC95": [round(x, 4) for x in wilson(h, n)]}

    def cap_count(sel):
        return sum(i["gen"].lower().count("capitain") for i in sel)

    capitain = {
        "total_occurrences": cap_count(items),
        "docs_avec": sum(1 for i in items if "capitain" in i["gen"].lower()),
        "par_categorie": {
            c: cap_count([i for i in items if i["categorie"] == c])
            for c in ("capitales", "dates", "definitions", "unites_sciences")
        },
        "reference_pilote1": PILOT1_CAPITAIN,
        "reduction_pct_vs_pilote1": round(
            100 * (1 - cap_count(items) / PILOT1_CAPITAIN), 1),
    }

    bv = best_val(TRAIN_LOG)
    step, ce, ppl = bv if bv else (None, None, None)
    d_ce = 100 * (ce / T08_CE - 1) if ce else None
    d_ppl = 100 * (ppl / T08_PPL - 1) if ppl else None

    rf = axes["reformule"]
    lang_ok = d_ppl is not None and d_ppl <= 10.0
    facts_ok = rf["taux"] >= 0.20 and rf["IC95"][0] > 0.0503
    corr_ok = cap_count(items) <= 0.5 * PILOT1_CAPITAIN
    if facts_ok and lang_ok and corr_ok:
        label = "SUCCÈS"
    elif facts_ok:
        label = "PARTIEL"
    elif rf["taux"] < 0.15:
        label = "ÉCHEC"
    else:
        label = "PARTIEL"

    verdict = {
        "regle": ("SUCCÈS si reformule (catégories entraînées) >= 20 % ET borne "
                  "basse IC95 Wilson > 5,03 % ET val-PPL Δ <= +10 % ET "
                  "occurrences « capitain » réduites >= 50 % vs pilote #1 "
                  f"({PILOT1_CAPITAIN} -> <= {PILOT1_CAPITAIN // 2}) ; "
                  "PARTIEL si faits OK mais corruption non corrigée ; "
                  "ÉCHEC si faits < 15 % malgré 3x le budget"),
        "reformule_global": (f"{rf['hits']}/{rf['n']} = {rf['taux']:.4f}, "
                             f"IC95 [{rf['IC95'][0]:.4f} ; {rf['IC95'][1]:.4f}]"),
        "critere_faits": facts_ok,
        "critere_langue": lang_ok,
        "critere_corruption": corr_ok,
        "capitain": capitain,
        "langue_val_ppl": (f"{d_ppl:+.2f} % (<= +10 %)"
                           if d_ppl is not None else "indisponible"),
        "label": label,
    }

    out = {
        "version": "probe_fr_v1",
        "source_unique": "docs/results/probe_Epilotqa3_v1.json",
        "genere_depuis": "eval/Epilotqa3-probe/results.json",
        "probe_sha256": r["probe_sha256"],
        "model": r["model"],
        "corpus": {
            "fichier": man["fichier"], "manifeste": os.path.relpath(
                MANIFEST, ROOT), "sha256": man["sha256"],
            "n_faits": man["n_faits"], "faits": man["faits"],
            "repeat": man["repeat"], "n_variants_par_fait":
            man["n_variants_par_fait"],
            "part_general_tokens": man["part_general_tokens"],
            "steps_par_epoque": man["steps_par_epoque"],
            "corpus_reel": man["corpus_reel"],
        },
        "decodage": r["decodage"],
        "val_gelée": {
            "run_best_ce": {"step": step, "val_ce": ce, "val_ppl": ppl},
            "baseline_t08": {"t08_val_ce": T08_CE, "t08_val_ppl": T08_PPL},
            "delta_pct_val_ce": round(d_ce, 2) if d_ce is not None else None,
            "delta_pct_val_ppl": round(d_ppl, 2) if d_ppl is not None else None,
        },
        "baselines": {
            "t08": {"global": "4/200 = 0.0200", "IC95": [0.0078, 0.0503]},
            "tranche00": {"global": "0/200 = 0.0000", "IC95": [0.0, 0.0188]},
            "pilote1": {"global": "81/200 = 0.4050", "IC95": [0.3394, 0.4742]},
            "pilote2": {"global": "21/200 = 0.1050", "IC95": [0.0697, 0.1552]},
        },
        "axes": axes,
        "agregats": r["agregats"],
        "verdict": verdict,
        "items": items,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
