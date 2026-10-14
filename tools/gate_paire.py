"""Gate apparié tranche01 (S5, version durcie).

Principe : bootstrap par document APPARIÉ (mêmes tirages pour les deux
modèles), estimateur ratio (Σ bits / Σ caractères sur les docs
rééchantillonnés), 1000 tirages, seed fixée. Δ = BPC(candidat) − BPC(réf).

Entrées : JSON per-doc de tools/eval_valbig.py --dump-per-doc
[{"nll_a","nll_b","nll_c","chars","chars_c","n","n_unk"}] pour ref et new
(mêmes docs, même ordre, même fenêtre — assert vérifié).

Verdict BPC : FAIL si la borne haute de l'IC 95 % du Δ > ε sur (b) OU (c).
(a) affichée pour continuité, sans veto (biais pro-élève, C2-quater).
ε = 0,03 BPC (2× le MDE mesuré ≈ 0,015 sur val_big ; une régression
d'ampleur G1 +0,044 reste détectée, le bruit ±0,015 ne déclenche plus).

MDE (plus petit Δ distinguable de 0 à 95 %) = demi-largeur de l'IC 95 %
du Δ. Si MDE > 0,01 : le gate est FAIBLE (dit explicitement, pas de FAIL
automatique — le verdict reste l'IC vs ε).

Boucles greedy : comparaison par proportion sur >= 30 prompts fixes
(metrics JSON de tools/eval_fixed.py --metrics-out, même version).
pire = boucle greedy new > ref (stricte). FAIL si la borne BASSE de
l'IC 95 % (Wilson) de la proportion > 0,5 (majorité significativement
pire). Jamais un point isolé.

Usage :
  uv run python tools/gate_paire.py --ref docs/results/valbig_tranche00.json \\
      --new docs/results/valbig_X.json [--ref-loops ... --new-loops ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

LN2 = math.log(2)
DRAWS = 1000
SEED = 0
# ε fixé APRÈS mesure du MDE (S5.3) : MDE mesuré ≈ 0,015 sur val_big
# (300 docs) → ε = 0,03 (2× MDE). Justification : avec ε=0,01 (< MDE),
# le gate FAIL sur du bruit (tranche00 vs 135M : Δ_b=+0,0005,
# IC=[-0,0127, +0,0164] → FAIL injustifié) ; avec ε=0,03, le bruit
# (hi≈0,016) passe et une régression d'ampleur G1 (+0,044 en b/c)
# échoue encore. Le gate reste aveugle sous 0,03 BPC (dit en S5.4).
EPS = 0.03


def load_perdoc(path: str) -> list[dict]:
    d = json.load(open(path, encoding="utf-8"))
    docs = d["docs"] if isinstance(d, dict) and "docs" in d else d
    return docs


def check_paired(ref: list[dict], new: list[dict]) -> list[str]:
    avert = []
    assert len(ref) == len(new) > 0, "nb docs différent ou nul"
    for i, (r, n) in enumerate(zip(ref, new)):
        assert r["chars"] == n["chars"], f"doc {i} : chars non appariés"
        if r["n"] != n["n"]:
            avert.append(f"doc {i} : positions {r['n']} vs {n['n']} "
                         "(fenêtrages différents — Δ par sommes, OK)")
    return avert


def paired_delta(ref: list[dict], new: list[dict], variant: str,
                 draws: int = DRAWS, seed: int = SEED) -> dict:
    """Δ BPC apparié (ratio) + IC95 + MDE."""
    key_nll = {"a": "nll_a", "b": "nll_b", "c": "nll_c"}[variant]
    key_den = "chars_c" if variant == "c" else "chars"
    R = np.array([[d[key_nll], d[key_den]] for d in ref], dtype=float)
    N = np.array([[d[key_nll], d[key_den]] for d in new], dtype=float)
    n = len(ref)
    rng = np.random.default_rng(seed)
    samp = rng.integers(0, n, size=(draws, n))
    bpc_r = R[samp, 0].sum(axis=1) / LN2 / np.maximum(
        R[samp, 1].sum(axis=1), 1e-9)
    bpc_n = N[samp, 0].sum(axis=1) / LN2 / np.maximum(
        N[samp, 1].sum(axis=1), 1e-9)
    delta = bpc_n - bpc_r
    full = (N[:, 0].sum() / max(N[:, 1].sum(), 1e-9)
            - R[:, 0].sum() / max(R[:, 1].sum(), 1e-9)) / LN2
    lo, hi = float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))
    return {"delta": float(full), "lo": lo, "hi": hi, "mde": (hi - lo) / 2}


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Borne basse IC95 Wilson d'une proportion."""
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n
            + z * z / (4 * n * n))) / den


# P3 : seuil absolu anti-boucle sur SAMPLING (T=0.7, seed fixe).
# Le greedy ne sépare pas (parents greedy 0.63/3.2 bouclent plus que nos
# tinies) ; le sampling oui. Refs mesurées sampling/32 : tranche00 0.0441,
# R2 0.0144, R2b 0.0098, R3 0.0472, parents HF 0.1074/0.1100 (sampling
# non seedé côté HF, variance inter-runs).
# (R4 dégénéré 0.3263 FAIL, e6 vide exclu).
# Marge +35 % sur le max sain, 2× sous R4.
S_REP_SEUIL = 0.15


def check_repetition(metrics_path: str) -> dict:
    """Gate répétition absolu (P3) : FAIL = tranche boucleuse, tout seul.

    Teste le NOUVEAU modèle seul (pas de ref) sur SAMPLING :
    repet4g moy > S_REP_SEUIL. Câblé dans verdict() quand new_loops
    est fourni.
    """
    m = json.load(open(metrics_path, encoding="utf-8"))
    mm = m["metrics"]
    reps = [v["sampling"]["taux_repetition_4g"] for v in mm.values()]
    bous = [v["sampling"]["plus_longue_boucle"] for v in mm.values()]
    rm = sum(reps) / len(reps)
    bm = sum(bous) / len(bous)
    motifs = []
    if rm > S_REP_SEUIL:
        motifs.append(f"repetition sampling {rm:.4f} > {S_REP_SEUIL}")
    return {"rep_moy": rm, "boucle_moy": bm, "boucle_max": max(bous),
            "n": len(mm), "fail": bool(motifs), "motif": "; ".join(motifs)}


def compare_loops(ref_path: str, new_path: str) -> dict:
    """Proportion de prompts où la boucle greedy empire + borne Wilson."""
    r = json.load(open(ref_path, encoding="utf-8"))
    n_ = json.load(open(new_path, encoding="utf-8"))
    assert r.get("version") == n_.get("version"), "versions prompts ≠"
    assert len(r["metrics"]) >= 30 and len(n_["metrics"]) >= 30, \
        "< 30 prompts"
    assert set(r["metrics"]) == set(n_["metrics"]), "prompts non appariés"
    worse = sum(1 for p in r["metrics"]
                if n_["metrics"][p]["greedy"]["plus_longue_boucle"]
                > r["metrics"][p]["greedy"]["plus_longue_boucle"])
    tot = len(r["metrics"])
    return {"worse": worse, "n": tot, "p": worse / tot,
            "wilson_lo": wilson_lower(worse, tot),
            "fail": wilson_lower(worse, tot) > 0.5}


def verdict(ref_path: str, new_path: str, draws: int = DRAWS,
            seed: int = SEED, eps: float = EPS,
            ref_loops: str = "", new_loops: str = "") -> tuple[bool, dict]:
    ref, new = load_perdoc(ref_path), load_perdoc(new_path)
    avert = check_paired(ref, new)
    out = {v: paired_delta(ref, new, v, draws, seed) for v in ("a", "b", "c")}
    fails = [v for v in ("b", "c") if out[v]["hi"] > eps]
    faible = any(out[v]["mde"] > 0.01 for v in ("a", "b", "c"))
    loops = None
    if ref_loops and new_loops:
        loops = compare_loops(ref_loops, new_loops)
        if loops["fail"]:
            fails.append("loops")
    rep = check_repetition(new_loops) if new_loops else None
    if rep is not None and rep["fail"]:
        fails.append("repetition")
    ok = not fails
    return ok, {"bpc": out, "eps": eps, "fails": fails,
                "gate_faible": faible, "loops": loops, "repetition": rep,
                "avertissements": avert[:5]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate apparié tranche01.")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--ref-loops", default="")
    ap.add_argument("--new-loops", default="")
    a = ap.parse_args()
    ok, res = verdict(a.ref, a.new, a.draws, a.seed, a.eps,
                      a.ref_loops, a.new_loops)
    for w in res.get("avertissements", []):
        print(f"! {w}", flush=True)
    for v in ("a", "b", "c"):
        r = res["bpc"][v]
        veto = "" if v == "a" else (" VETO" if r["hi"] > a.eps else "")
        print(f"[{v}] Δ={r['delta']:+.4f} IC95=[{r['lo']:+.4f}, {r['hi']:+.4f}]"
              f" MDE={r['mde']:.4f}{veto}", flush=True)
    if res["gate_faible"]:
        print("gate FAIBLE : MDE > 0,01 sur au moins une variante",
              flush=True)
    if res["loops"] is not None:
        L = res["loops"]
        print(f"[loops] pire {L['worse']}/{L['n']} "
              f"(Wilson_lo={L['wilson_lo']:.3f})", flush=True)
    if res.get("repetition") is not None:
        R = res["repetition"]
        print(f"[repetition] sampling moy={R['rep_moy']:.4f} "
              f"(seuil {S_REP_SEUIL})"
              + (f" FAIL {R['motif']}" if R["fail"] else ""), flush=True)
    print(("PASS" if ok else "FAIL")
          + ("" if ok else f" : {res['fails']}"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
