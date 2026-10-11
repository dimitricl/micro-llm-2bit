"""Revalidation tranche01 (P6) : gate apparié, ε par paire, critère OOD.

Brique de non-régression PAR PAIRE (pas le décideur P1 complet : la
sélection R1/R2 exige 3 appels — R1−tranche00, R2−tranche00, R1−R2 —
plus la règle in-run, cf. docs/ABLATIONS.md section R1/R2).

Règles (c primaire, a/b affichés sans veto) :
  val_big : veto ⟺ borne basse IC95 du Δc > 0 (régression significative
    à 95 %, unilatéral).
  OOD : veto ⟺ borne basse IC95 du Δc_OOD > 0.
ε n'est PAS une constante : le seuil effectif est le MDE de la paire
(demi-largeur IC95 du Δc, bootstrap apparié ratio, 1000 tirages,
seed 0). Exemple mesuré sur val_big : MDE ~0,015.
Garde-fous : NLL non finies (NaN/inf) → veto ; OOD fourni à moitié →
erreur explicite ; docs non appariés → FAIL propre (pas de traceback).
L'ancien gate (IC non appariés, ε=0,03 fixe, variante a) est remplacé ;
voir tools/gate_paire.py pour l'outil d'analyse interactif.

Entrées : per-doc JSON de tools/eval_valbig.py --dump-per-doc
(val_big ET OOD, mêmes docs appariés ref/new).
Sortie : PASS/FAIL + motifs ; code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.gate_paire import load_perdoc, paired_delta

DRAWS = 1000
SEED = 0


def _finis(docs: list[dict]) -> bool:
    import math
    return all(math.isfinite(d[k])
               for d in docs for k in ("nll_a", "nll_b", "nll_c"))


def verdict_apparie(ref: list[dict], new: list[dict],
                    ood_ref: list[dict] | None = None,
                    ood_new: list[dict] | None = None,
                    draws: int = DRAWS, seed: int = SEED
                    ) -> tuple[bool, dict]:
    from tools.gate_paire import check_paired

    if (ood_ref is None) != (ood_new is None):
        raise ValueError("OOD : fournir --ood-ref ET --ood-new, ou ni l'un "
                         "ni l'autre (demi-OOD = gate contourné)")
    vetos = []
    if not (_finis(ref) and _finis(new)):
        vetos.append("non_fini")
    try:
        avert = check_paired(ref, new)
    except AssertionError as e:
        return False, {"delta_c": float("nan"), "eps_pair": float("nan"),
                       "mde_c": float("nan"), "mde_b": float("nan"),
                       "mde_a": float("nan"), "ic_c": [float("nan")] * 2,
                       "vetos": vetos + ["non_apparie"],
                       "ood": None, "avertissements": [str(e)],
                       "det": {}}
    import numpy as _np
    with _np.errstate(invalid="ignore"):
        det = {v: paired_delta(ref, new, v, draws, seed)
               for v in ("a", "b", "c")}
    eps_pair = det["c"]["mde"]
    if det["c"]["lo"] > 0:
        vetos.append("c_valbig")
    ood = None
    ood_avert: list[str] = []
    if ood_ref is not None and ood_new is not None:
        if not (_finis(ood_ref) and _finis(ood_new)):
            vetos.append("non_fini")
        else:
            try:
                ood_avert = check_paired(ood_ref, ood_new)
            except AssertionError as e:
                return False, {"delta_c": det["c"]["delta"],
                               "eps_pair": eps_pair,
                               "mde_c": det["c"]["mde"],
                               "mde_b": det["b"]["mde"],
                               "mde_a": det["a"]["mde"],
                               "ic_c": [det["c"]["lo"], det["c"]["hi"]],
                               "vetos": vetos + ["non_apparie"],
                               "ood": None,
                               "avertissements": avert[:5] + [str(e)],
                               "det": det}
            ood = {v: paired_delta(ood_ref, ood_new, v, draws, seed)
                   for v in ("a", "b", "c")}
            if ood["c"]["lo"] > 0:
                vetos.append("c_ood")
    ok = not vetos
    return ok, {"delta_c": det["c"]["delta"], "eps_pair": eps_pair,
                "mde_c": det["c"]["mde"], "mde_b": det["b"]["mde"],
                "mde_a": det["a"]["mde"], "ic_c": [det["c"]["lo"],
                                                   det["c"]["hi"]],
                "vetos": vetos, "ood": ood,
                "avertissements": avert[:5] + ood_avert[:5], "det": det}


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate apparié tranche01 (P6).")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--ood-ref", default="")
    ap.add_argument("--ood-new", default="")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    ref, new = load_perdoc(a.ref), load_perdoc(a.new)
    ood_ref = load_perdoc(a.ood_ref) if a.ood_ref else None
    ood_new = load_perdoc(a.ood_new) if a.ood_new else None
    try:
        ok, res = verdict_apparie(ref, new, ood_ref, ood_new, a.draws,
                                  a.seed)
    except ValueError as e:
        print(f"FAIL : {e}", flush=True)
        sys.exit(2)
    for w in res.get("avertissements", []):
        print(f"! {w}", flush=True)
    for v in ("a", "b", "c"):
        d = res["det"][v]
        print(f"[{v}] Δ={d['delta']:+.4f} "
              f"IC95=[{d['lo']:+.4f}, {d['hi']:+.4f}]", flush=True)
    print(f"[c] ε_pair(MDE)={res['eps_pair']:.4f}", flush=True)
    if res["ood"] is not None:
        o = res["ood"]["c"]
        print(f"[c-ood] Δ={o['delta']:+.4f} MDE={o['mde']:.4f} "
              f"IC95=[{o['lo']:+.4f}, {o['hi']:+.4f}]", flush=True)
    print(("PASS" if ok else "FAIL")
          + ("" if ok else f" : {res['vetos']}"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
