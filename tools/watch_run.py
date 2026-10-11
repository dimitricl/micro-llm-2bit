"""Veilleur de run (H5.2) : surveille un train.log, verdict GO/STOP.

Règles (tranche01 et R1/R2) :
  1. STOP si « loss non finie » (ARRÊT divergence) ou si le log ne se
     termine pas (« terminé en » absent en fin de run — à appeler sur un
     log final ou avec --en-cours pour un contrôle intermédiaire).
  2. STOP si une val-PPL in-run dépasse 2× la PREMIÈRE val-PPL du run.
  3. STOP si, à 25 % de l'epoch (premier quart des steps totaux écoulé
     avec au moins une val), la val-PPL n'est pas repassée SOUS la
     valeur initiale.
Seuils lus dans le log lui-même ([val] step G/TOTAL val_ppl~X) : aucune
constante codée en dur hormis les facteurs 2× et 25 %.

Usage :
  uv run python tools/watch_run.py --log runs/X/train.log [--en-cours]
Sortie : GO ou STOP + motifs ; code 0 = GO, 1 = STOP.
"""

from __future__ import annotations

import argparse
import re
import sys

RE_VAL = re.compile(r"\[val\] step (\d+)/(\d+) val_ce=[\d.]+ val_ppl~([\d.]+)")


def surveiller(texte: str, en_cours: bool = False) -> tuple[bool, list[str]]:
    motifs: list[str] = []
    ok = True
    if "loss non finie" in texte:
        return False, ["divergence : loss non finie"]
    vals = [(int(g), int(t), float(p)) for g, t, p in RE_VAL.findall(texte)]
    termine = "terminé en" in texte
    if not termine and not en_cours:
        ok = False
        motifs.append("run non terminé (pas de « terminé en »)")
    if not vals:
        motifs.append("aucune val-PPL in-run (rien à contrôler)")
        return ok, motifs
    init = vals[0][2]
    motifs.append(f"val initiale : {init:.1f}")
    pires = [(g, p) for g, t, p in vals if p > 2 * init]
    if pires:
        ok = False
        motifs.append(f"val-PPL > 2× initiale : {pires[:3]}")
    else:
        motifs.append("val-PPL jamais > 2× initiale")
    total = vals[0][1]
    quart = [p for g, t, p in vals[1:] if g <= total / 4]
    if quart:
        if min(quart) >= init:
            ok = False
            motifs.append(f"à 25 % ({total // 4} steps) : min {min(quart):.1f} "
                          f"jamais sous initiale {init:.1f}")
        else:
            motifs.append(f"à 25 % : repassée sous initiale ({min(quart):.1f})")
    else:
        motifs.append("aucune val postérieure dans le premier quart "
                      "(contrôle 25 % N/A)")
    return ok, motifs


def main() -> None:
    ap = argparse.ArgumentParser(description="Veilleur de run.")
    ap.add_argument("--log", required=True)
    ap.add_argument("--en-cours", action="store_true")
    a = ap.parse_args()
    texte = open(a.log, encoding="utf-8", errors="replace").read()
    ok, motifs = surveiller(texte, a.en_cours)
    print(("GO" if ok else "STOP"), flush=True)
    for m in motifs:
        print(f" - {m}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
