"""PPL d'un .bin sur un jeu mis de côté, avec le moteur quantifié 2 bits.

- Jeu : ~2 Mo pris dans une tranche JAMAIS entraînée, tokenisé avec le
  vocab du run. Fuite interdite : tout doc présent (à normalisation près)
  dans les docs train/val est exclu, avec compteur.
- PPL calculée avec les logits du .bin (quantification réelle, NEON),
  fenêtres non recouvrantes de seq_max avec reset (approximation
  documentée : pas de contexte inter-fenêtres).
- Compare à la val-PPL de fin d'entraînement : écart relatif affiché,
  > 5 % signalé comme problème de fidélité (export/quantification).

Usage : nice -n 10 .venv/bin/python tools/eval_ppl.py --model ... \\
            --heldout ... --train-docs ... --val-docs ... --val-ppl 60.9
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

import data as D
import engine as E


def normalise(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()[:200]


def sans_fuite(candidats: list[str], train_docs: list[str],
               val_docs: list[str]) -> tuple[list[str], int]:
    """Exclut tout doc candidat déjà vu (normalisé) dans train/val."""
    vus = {normalise(t) for t in train_docs} | {normalise(t) for t in val_docs}
    gardes = [t for t in candidats if normalise(t) not in vus]
    return gardes, len(candidats) - len(gardes)


def charge_textes_jsonl(path: str, field: str = "text") -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line).get(field, ""))
    return [t for t in out if t.strip()]


def ppl_moteur(eng: E.Engine, ids: list[int]) -> tuple[float, int, float]:
    """PPL par fenêtres de seq_max (reset entre fenêtres). Retour (ppl, N, s)."""
    S = eng.meta["seq_max"]
    t0 = time.time()
    logp, n = 0.0, 0
    for b in range(0, len(ids) - 1, S):
        win = ids[b:b + S + 1]
        if len(win) < 2:
            continue
        eng.reset()
        prev = eng.step(win[0])
        for i in range(1, len(win)):
            z = prev.astype(np.float64)
            z -= z.max()
            e = np.exp(z)
            logp += float(np.log(e[win[i]] / e.sum()))
            n += 1
            if i + 1 < len(win):
                prev = eng.step(win[i])
    dt = time.time() - t0
    return float(np.exp(-logp / max(n, 1))), n, dt


def main() -> None:
    ap = argparse.ArgumentParser(description="PPL .bin sur jeu mis de côté.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--heldout", required=True,
                    help=".jsonl candidat (tranche jamais entraînée).")
    ap.add_argument("--heldout-mb", type=float, default=2.0)
    ap.add_argument("--train-docs", default="", help=".jsonl des docs train")
    ap.add_argument("--val-docs", default="", help=".jsonl des docs val")
    ap.add_argument("--val-ppl", type=float, default=0.0,
                    help="val-PPL de fin d'entraînement (comparaison).")
    ap.add_argument("--vocab", default="")
    ap.add_argument("--parent", default="")
    a = ap.parse_args()
    from main import load_tokenizer

    vocab = json.load(open(a.vocab or (a.model + ".vocab.json"), encoding="utf-8"))
    tok = load_tokenizer(a.parent or vocab["parent"])
    cands = charge_textes_jsonl(a.heldout)
    tr = charge_textes_jsonl(a.train_docs) if a.train_docs else []
    va = charge_textes_jsonl(a.val_docs) if a.val_docs else []
    gardes, exclus = sans_fuite(cands, tr, va)
    print(f"[eval-ppl] candidats={len(cands)} exclus(fuite)={exclus} "
          f"gardés={len(gardes)}", flush=True)
    # Tronque à la taille demandée (en octets de texte).
    total, pris = 0, []
    for t in gardes:
        total += len(t.encode("utf-8"))
        pris.append(t)
        if total >= a.heldout_mb * 1024 * 1024:
            break
    ids: list[int] = []
    for t in pris:
        ids += D.encode_mapped(tok, vocab, t, add_eos=True)
    print(f"[eval-ppl] jeu final : {len(pris)} docs, {len(ids)} tokens", flush=True)
    eng = E.Engine(a.model)
    ppl, n, dt = ppl_moteur(eng, ids)
    print(f"[eval-ppl] PPL .bin (quantifié) = {ppl:.1f} sur {n} tokens "
          f"en {dt:.0f}s", flush=True)
    if a.val_ppl > 0:
        ecart = abs(ppl - a.val_ppl) / a.val_ppl
        print(f"[eval-ppl] val-PPL entraînement = {a.val_ppl:.1f} -> "
              f"écart relatif {ecart * 100:.1f} %", flush=True)
        if ecart > 0.05:
            print("[eval-ppl] SIGNALÉ : écart > 5 %, problème de fidélité "
                  "de l'export ou de la quantification des activations.",
                  flush=True)


if __name__ == "__main__":
    main()
