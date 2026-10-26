"""Audit OOD vs textes Gutenberg avant G2.

Objectif : vérifier que les 340 docs OOD (3 romans) ne recoupent aucun texte
Gutenberg candidat, et produire la liste d'exclusion que G2 appliquera.

Méthode : 32-grammes de tokens SmolLM (même définition que
`tools/ngram_overlap.py`), zéro échantillonnage. On construit l'ensemble des
32-grammes des docs OOD, puis on teste chaque livre Gutenberg téléchargé.
Tout livre avec >= 1 match est exclu. Les doublons d'édition (Verne 5095/5096,
Madame Bovary anglaise 2413) sont aussi signalés par titre/auteur.

Sortie : docs/results/gutenberg_ood_overlap_G.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from transformers import AutoTokenizer

from tools.make_ood import strip_gutenberg
from tools.ngram_overlap import hashes, load_docs

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)

CATALOG = "/Volumes/Lexar/rag-data/livres/pg_catalog.csv"
GUTENBERG = "runs/G2-data/gutenberg"
OOD = "runs/abl_data/ood.jsonl"
OOD_MANIFEST = "runs/abl_data/ood_manifest.json"
OOD_REF_DIR = "runs/G2-data/ood_refs"
OUT = "docs/results/gutenberg_ood_overlap_G.json"
TOKENIZER = "HuggingFaceTB/SmolLM-135M"
N = 32

# IDs OOD + doublons d'édition connus. L'audit n-grammes reste la source de
# vérité ; cette liste sert à expliquer les exclusions.
OOD_IDS = {14155, 5097, 19657}
KNOWN_DUPLICATES = {
    5095: "Vingt mille lieues, partie 1 (même texte que 5097)",
    5096: "Vingt mille lieues, partie 2 (même texte que 5097)",
    2413: "Madame Bovary, traduction anglaise (même roman que 14155)",
    2610: "Notre-Dame de Paris, traduction anglaise (même roman que 19657)",
}


def norm_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def read_catalog() -> dict[int, dict]:
    with open(CATALOG, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        try:
            rid = int(r["Text#"])
        except Exception:
            continue
        out[rid] = {
            "id": rid,
            "title": r.get("Title", ""),
            "authors": r.get("Authors", ""),
            "language": r.get("Language", ""),
        }
    return out


def strip_or_raw(raw: str) -> str:
    try:
        return strip_gutenberg(raw)
    except Exception:
        return re.sub(r"\s+", " ", raw).strip()


def ood_hash_set(tok) -> tuple[np.ndarray, dict]:
    docs = load_docs(OOD)
    hs = []
    total = 0
    for doc in docs:
        ids = np.asarray(tok(doc, add_special_tokens=False)["input_ids"],
                         dtype=np.uint32)
        h = hashes(ids, N)
        total += len(h)
        if len(h):
            hs.append(np.unique(h))
    arr = np.unique(np.concatenate(hs)) if hs else np.zeros(0, np.uint64)
    print(f"[audit] OOD: {len(docs)} docs, {total/1e6:.3f}M fenêtres, "
          f"{len(arr)/1e6:.3f}M 32-grammes uniques", flush=True)
    return arr, {"n_docs": len(docs), "fenetres": total,
                 "uniques": int(len(arr))}


def scan_book(path: str, tok, ood: np.ndarray) -> dict:
    raw = open(path, encoding="utf-8", errors="replace").read()
    body = strip_or_raw(raw)
    ids = np.asarray(tok(body, add_special_tokens=False)["input_ids"],
                     dtype=np.uint32)
    h = hashes(ids, N)
    if len(h) == 0 or len(ood) == 0:
        return {"chars": len(body), "tokens": int(len(ids)),
                "fenetres": int(len(h)), "matches": 0, "taux": 0.0}
    pos = np.clip(np.searchsorted(ood, h), 0, len(ood) - 1)
    hit = int((ood[pos] == h).sum())
    return {
        "chars": len(body),
        "tokens": int(len(ids)),
        "fenetres": int(len(h)),
        "matches": hit,
        "taux": hit / len(h) if len(h) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    catalog = read_catalog()
    ood, ood_stats = ood_hash_set(tok)
    manifest = json.load(open(OOD_MANIFEST, encoding="utf-8"))
    ood_titles = {s["ebook"]: s["titre"] for s in manifest["sources"]}

    paths = []
    for fn in sorted(os.listdir(GUTENBERG)):
        m = re.fullmatch(r"pg(\d+)\.txt", fn)
        if m:
            paths.append((int(m.group(1)), os.path.join(GUTENBERG, fn)))
    # 19657 n'est pas dans le catalogue local actuel ; on le garde comme
    # contrôle OOD, pas comme candidat G2.
    for fn in sorted(os.listdir(OOD_REF_DIR)):
        m = re.fullmatch(r"pg(\d+)\.txt", fn)
        if m:
            paths.append((int(m.group(1)), os.path.join(OOD_REF_DIR, fn)))
    if a.limit:
        paths = paths[:a.limit]
    print(f"[audit] {len(paths)} textes à scanner", flush=True)

    books = []
    t0 = time.time()
    for i, (rid, path) in enumerate(paths, 1):
        try:
            r = scan_book(path, tok, ood)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}", "matches": 0,
                 "fenetres": 0, "taux": 0.0}
        cat = catalog.get(rid, {})
        title = cat.get("title") or ood_titles.get(rid, "")
        reasons = []
        if rid in OOD_IDS:
            reasons.append("ebook OOD exact")
        if rid in KNOWN_DUPLICATES:
            reasons.append(KNOWN_DUPLICATES[rid])
        if r.get("matches", 0) > 0:
            reasons.append(f"{r['matches']} 32-grammes OOD")
        status = "exclude" if reasons else "clean"
        books.append({
            "id": rid,
            "title": title,
            "authors": cat.get("authors", ""),
            "language": cat.get("language", ""),
            "path": path,
            "status": status,
            "reasons": reasons,
            **r,
        })
        if i % 25 == 0 or i == len(paths):
            n_ex = sum(1 for b in books if b["status"] == "exclude")
            print(f"[audit] {i}/{len(paths)} exclus={n_ex} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Contrôle de cohérence : les 3 OOD doivent être exclus et matcher.
    excluded_ids = {b["id"] for b in books if b["status"] == "exclude"}
    missing = sorted(OOD_IDS - excluded_ids)
    out = {
        "tokenizer": TOKENIZER,
        "n": N,
        "ood": {"path": OOD, "sources": manifest["sources"], **ood_stats},
        "known_duplicates": {str(k): v for k, v in KNOWN_DUPLICATES.items()},
        "n_texts": len(books),
        "n_excluded": sum(1 for b in books if b["status"] == "exclude"),
        "n_clean": sum(1 for b in books if b["status"] == "clean"),
        "excluded_ids": sorted(excluded_ids),
        "missing_ood_ids": missing,
        "books": books,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[audit] -> {OUT}")
    print(f"[audit] exclus={out['n_excluded']} propres={out['n_clean']} "
          f"manquants={missing}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
