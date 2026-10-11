"""Revalidation tranche01 (G4) : applique le critère go/no-go, verdict PASS/FAIL.

Critère (remplace "BPC < 1,55") :
 1. Pas de NaN/divergence : le train.log contient "terminé en" et PAS
    "non finie" (garde-fou _nonfinite). Sinon FAIL.
 2. Pas de régression BPC sur la val élargie : FAIL seulement si l'IC 95 %
    du nouveau BPC est ENTIÈREMENT au-dessus de celui de référence
    (new_lo > ref_hi). Chevauchement ou mieux -> PASS.
 3. Boucles greedy (eval_fixed, moyenne des plus_longue_boucle sur les 20
    prompts) non en hausse : new <= ref, sinon FAIL.

Entrées (produites par les commandes documentées en docstring de
produire_new_json, voir README du script) :
 --ref JSON tranche00 : {"bpc_a": x, "bpc_a_ci": [lo, hi], "loop": r}
 --new JSON tranche01 : même format + {"has_nan": bool}
 --train-log : log du run tranche01 (vérification indépendante du flag)
Sortie : verdict + motifs ; code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys


def verdict(ref: dict, new: dict, log_text: str) -> tuple[bool, list[str]]:
    motifs = []
    ok = True
    termine = "terminé en" in log_text
    nonfinie = "non finie" in log_text
    if not termine or nonfinie or new.get("has_nan"):
        ok = False
        motifs.append(f"train: termine={termine} nonfinie={nonfinie} "
                      f"has_nan={new.get('has_nan')}")
    else:
        motifs.append("train: terminé, pas de NaN/divergence")
    rlo, rhi = ref["bpc_a_ci"]
    nlo, nhi = new["bpc_a_ci"]
    if nlo > rhi:
        ok = False
        motifs.append(f"BPC: régression [{nlo:.4f}, {nhi:.4f}] entièrement "
                      f"au-dessus de réf [{rlo:.4f}, {rhi:.4f}]")
    else:
        motifs.append(f"BPC: pas de régression prouvée "
                      f"(new [{nlo:.4f}, {nhi:.4f}] vs réf [{rlo:.4f}, {rhi:.4f}])")
    if new["loop"] > ref["loop"]:
        ok = False
        motifs.append(f"boucles: hausse {ref['loop']:.2f} -> {new['loop']:.2f}")
    else:
        motifs.append(f"boucles: non en hausse "
                      f"({ref['loop']:.2f} -> {new['loop']:.2f})")
    return ok, motifs


def main() -> None:
    ap = argparse.ArgumentParser(description="Verdict PASS/FAIL tranche01.")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--train-log", required=True)
    a = ap.parse_args()
    ref = json.load(open(a.ref, encoding="utf-8"))
    new = json.load(open(a.new, encoding="utf-8"))
    log_text = open(a.train_log, encoding="utf-8", errors="replace").read()
    ok, motifs = verdict(ref, new, log_text)
    print(("PASS" if ok else "FAIL"), flush=True)
    for m in motifs:
        print(f" - {m}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
