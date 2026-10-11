"""Évaluation fixe et reproductible d'un .bin (CPU, noyau NEON).

32 prompts FR versionnés (v2 : 8 pays, 8 wiki, 8 courants, 8 factuels ;
v1 = les 20 premiers, inchangés et dans le même ordre).
Par prompt : greedy + sampling (T=0.7, seed fixe), 60 tokens.
Métriques AUTOMATIQUES, sans jugement de qualité : taux de répétition
(4-grammes), plus longue boucle d'un même token, mots distincts.
Sorties écrites telles quelles (ni corrigées ni triées).

Usage : nice -n 10 .venv/bin/python tools/eval_fixed.py
            --model exports/mini_test.bin --out eval/mini_test/samples.md
            --metrics-out eval/mini_test/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

import data as D
import engine as E

PROMPTS_VERSION = 2

PROMPTS = [
    ("pays", "Le Japon est un pays"),
    ("pays", "Le Brésil est un pays"),
    ("pays", "L'Égypte est un pays"),
    ("pays", "Le Canada est un pays"),
    ("pays", "L'Italie est un pays"),
    ("wiki", "Paris est la capitale"),
    ("wiki", "La photosynthèse est"),
    ("wiki", "L'Empire romain"),
    ("wiki", "Le cerveau humain"),
    ("wiki", "La Révolution française"),
    ("courant", "Il fait beau aujourd'hui"),
    ("courant", "Je vais au marché"),
    ("courant", "Le chat dort sur"),
    ("courant", "Demain nous irons"),
    ("courant", "J'aime beaucoup le"),
    ("factuel", "L'eau bout à"),
    ("factuel", "La capitale de l'Espagne est"),
    ("factuel", "Deux plus deux égale"),
    ("factuel", "Le plus grand océan est"),
    ("factuel", "Paris se trouve en"),
    # v2 : +12 (8/catégorie, S5.6 : >= 30 prompts fixes).
    ("pays", "La Norvège est un pays"),
    ("pays", "Le Maroc est un pays"),
    ("pays", "L'Argentine est un pays"),
    ("wiki", "La gravité a été décrite"),
    ("wiki", "Le Moyen Âge"),
    ("wiki", "La Grande Muraille"),
    ("courant", "Ce soir nous mangeons"),
    ("courant", "Le train arrive à"),
    ("courant", "Elle lit un livre"),
    ("factuel", "Le soleil se lève à"),
    ("factuel", "L'oxygène est un"),
    ("factuel", "Napoléon est né en"),
]

MAX_NEW = 60


def metriques(ids: list[int], texte: str) -> dict:
    """Taux de 4-grammes répétés, plus longue boucle, mots distincts."""
    n = len(ids)
    grams: dict = {}
    for i in range(max(n - 3, 0)):
        g = tuple(ids[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    total = max(len(grams), 1)
    repetes = sum(1 for c in grams.values() if c > 1)
    boucle = 1
    cur = 1
    for i in range(1, n):
        cur = cur + 1 if ids[i] == ids[i - 1] else 1
        boucle = max(boucle, cur)
    mots = texte.split()
    return {
        "taux_repetition_4g": round(repetes / total, 4),
        "plus_longue_boucle": boucle,
        "mots_distincts": len(set(mots)),
        "n_tokens": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Éval fixe d'un .bin.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default="")
    ap.add_argument("--parent", default="")
    ap.add_argument("--metrics-out", default="",
                    help="JSON {prompt: {greedy: {...}, sampling: {...}}} "
                         "pour le gate boucles (S5.6).")
    a = ap.parse_args()
    from main import load_tokenizer, decode_ids

    vocab = json.load(open(a.vocab or (a.model + ".vocab.json"), encoding="utf-8"))
    tok = load_tokenizer(a.parent or vocab["parent"])
    eng = E.Engine(a.model)
    print(f"[eval] noyau={E.KERNEL_NAME} prompts={len(PROMPTS)}", flush=True)
    lignes = [f"# Éval {a.model}", ""]
    metrics = {}
    for cat, prompt in PROMPTS:
        ids = D.encode_mapped(tok, vocab, prompt, add_eos=False)
        ids = ids[-eng.meta["seq_max"]:]
        out_g = eng.generate(ids, max_new=MAX_NEW, temperature=0.0,
                             stop={vocab["eos_new"]})
        txt_g = decode_ids(tok, vocab, out_g)
        out_s = eng.generate(ids, max_new=MAX_NEW, temperature=0.7,
                             top_k=40, top_p=0.9, stop={vocab["eos_new"]},
                             rng=np.random.default_rng(1234))
        txt_s = decode_ids(tok, vocab, out_s)
        lignes += [f"## [{cat}] {prompt}", "",
                   f"greedy ({metriques(out_g, txt_g)}) :", "", txt_g, "",
                   "sampling T=0.7 "
                   f"({metriques(out_s, txt_s)}) :", "", txt_s, ""]
        metrics[prompt] = {"greedy": metriques(out_g, txt_g),
                           "sampling": metriques(out_s, txt_s)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    print(f"[eval] -> {a.out}")
    if a.metrics_out:
        json.dump({"version": PROMPTS_VERSION,
                   "metrics": metrics},
                  open(a.metrics_out, "w"), indent=1)
        print(f"[eval] metrics -> {a.metrics_out}")


if __name__ == "__main__":
    main()
