"""Phase 0 G — recouvrement de n-grammes val/OOD vs corpus d'entraînement.

Question : une partie du gain in-domain (bras B) vient-elle de quasi-leakage
(gabarits Wikipédia redondants) plutôt que d'un apprentissage général ?

Méthode exacte, sans échantillonnage :
- tokenisation SmolLM native (pas le vocab réduit, pour éviter que les tokens
  hors vocabulaire fusionnent en un même `unk` et créent de faux matches) ;
- pour n = 8, 10, 13 et 32 : hachage polynomial 64 bits de chaque fenêtre
  (jamais à cheval sur deux documents), déduplication exacte par tri (numpy)
  tranche par tranche ;
- l'union des tranches n'est **jamais matérialisée** : pour chaque fenêtre de
  validation on teste l'appartenance dans chaque tranche (recherche
  dichotomique) et on fait un OU logique. C'est exact et le pic mémoire reste
  à une tranche (~300 Mo), au lieu de ~9 Go pour `np.unique(concat)`.

Deux références :
- `train` = tranches 01-09 (ce que la chaîne E a réellement vu) ;
- `t00`   = tranche 00, **jamais entraînée** (val_E_frozen en est tirée) :
  elle donne le taux de gabarits partagés par le seul fait d'être du
  Wikipédia, sans lien avec l'entraînement.

Sorties : `runs/G-diag/ngram/*.npy` (cache), `docs/results/ngram_overlap.json`.
Usage :
  .venv/bin/python tools/ngram_overlap.py [--rebuild]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from transformers import AutoTokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)

CORPUS = "/Volumes/Lexar/clean-corpus/dedup"
CACHE = "runs/G-diag/ngram"
RESULTS = "docs/results/ngram_overlap.json"
TOKENIZER = "HuggingFaceTB/SmolLM-135M"
NS = (8, 10, 13, 32)
BASE = np.uint64(1000003)
TRAIN_TRANCHES = [f"t{i:02d}" for i in range(1, 10)]
REF_TRANCHES = ["t00"]


def load_docs(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if t.strip():
                out.append(t)
    return out


def hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Hachage polynomial 64 bits des n-grammes de `ids` (uint32)."""
    m = len(ids) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    pw = np.uint64(1)
    with np.errstate(over="ignore"):
        for k in range(n):
            h += ids[k:k + m].astype(np.uint64) * pw
            pw = pw * BASE
    return h


def build_train_cache(tok, rebuild: bool = False) -> None:
    os.makedirs(CACHE, exist_ok=True)
    for tag in TRAIN_TRANCHES + REF_TRANCHES:
        path = f"{CORPUS}/tranche_{tag[1:]}.jsonl"
        need = [n for n in NS
                if rebuild or not os.path.exists(f"{CACHE}/{tag}_n{n}.npy")]
        if not need:
            print(f"[ngram] {tag}: cache complet, sauté", flush=True)
            continue
        print(f"[ngram] tokenisation {tag} ...", flush=True)
        t0 = time.time()
        docs = load_docs(path)
        ids_docs: list[np.ndarray] = []
        for i in range(0, len(docs), 500):
            enc = tok(docs[i:i + 500], add_special_tokens=False)["input_ids"]
            ids_docs.extend(np.asarray(x, dtype=np.uint32) for x in enc)
        n_tok = int(sum(len(x) for x in ids_docs))
        print(f"[ngram] {tag}: {len(docs)} docs, {n_tok/1e6:.1f}M tokens "
              f"({time.time()-t0:.0f}s)", flush=True)
        for n in need:
            uniq = []
            total = 0
            for x in ids_docs:
                h = hashes(x, n)
                total += len(h)
                if len(h):
                    uniq.append(np.unique(h))
            arr = np.unique(np.concatenate(uniq)) if uniq else np.zeros(0, np.uint64)
            np.save(f"{CACHE}/{tag}_n{n}.npy", arr)
            print(f"[ngram]   {tag} n={n}: {total/1e6:.1f}M fenêtres, "
                  f"{len(arr)/1e6:.1f}M uniques ({time.time()-t0:.0f}s)",
                  flush=True)
            del uniq, arr


def eval_set(name: str, path: str, tok, n: int,
             files: list[str]) -> dict:
    """Compte les fenêtres du set présentes dans l'UNION des `files`.

    Chaque fichier est chargé seul, testé par recherche dichotomique, puis
    libéré : le pic mémoire est une tranche, pas l'union.
    """
    docs = load_docs(path)
    hs: list[np.ndarray] = []
    total = 0
    for doc in docs:
        ids = np.asarray(tok(doc, add_special_tokens=False)["input_ids"],
                         dtype=np.uint32)
        h = hashes(ids, n)
        hs.append(h)
        total += len(h)
    masks = [np.zeros(len(h), dtype=bool) for h in hs]
    for fp in files:
        arr = np.load(fp)
        if len(arr) == 0:
            del arr
            continue
        for j, h in enumerate(hs):
            if len(h) == 0:
                continue
            pos = np.clip(np.searchsorted(arr, h), 0, len(arr) - 1)
            masks[j] |= arr[pos] == h
        del arr
    counts = [int(m.sum()) for m in masks]
    hit = sum(counts)
    docs_hit = sum(1 for c in counts if c > 0)
    docs_5 = sum(1 for c in counts if c >= 5)
    out = {
        "fenetres": total,
        "fenetres_vues": hit,
        "taux_fenetres": hit / total if total else 0.0,
        "docs_avec_1_match": docs_hit,
        "docs_avec_5_matches": docs_5,
        "taux_docs_1_match": docs_hit / len(docs) if docs else 0.0,
        "matchs_max_par_doc": max(counts) if counts else 0,
    }
    print(f"[ngram] {name} n={n}: {hit}/{total} fenêtres "
          f"({100*out['taux_fenetres']:.2f}%), {docs_hit}/{len(docs)} docs",
          flush=True)
    return out


def files(tag: list[str], n: int) -> list[str]:
    return [f"{CACHE}/{t}_n{n}.npy" for t in tag]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    build_train_cache(tok, a.rebuild)
    sets = {
        "val_E_frozen": "runs/abl_data/val_E_frozen.jsonl",
        "val_big": "runs/abl_data/val_big.jsonl",
        "ood": "runs/abl_data/ood.jsonl",
    }
    out = {"tokenizer": TOKENIZER, "n": list(NS),
           "train": TRAIN_TRANCHES, "reference_non_entrainee": REF_TRANCHES,
           "sets": {name: {"vs_train_01_09": {}, "vs_t00": {}}
                    for name in sets}}
    for n in NS:
        for name, path in sets.items():
            out["sets"][name]["vs_train_01_09"][str(n)] = eval_set(
                name + "/train", path, tok, n, files(TRAIN_TRANCHES, n))
            out["sets"][name]["vs_t00"][str(n)] = eval_set(
                name + "/t00", path, tok, n, files(REF_TRANCHES, n))
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[ngram] -> {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
