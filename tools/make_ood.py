"""Construit le jeu OOD français (H4) depuis Project Gutenberg.

Sources (domaine public : auteurs morts depuis > 70 ans — Flaubert 1880,
Verne 1905, Hugo 1885 ; licence Gutenberg « Public domain in the USA »,
vérifiée sur la page de chaque ebook) :
  14155 Madame Bovary (French), Flaubert
   5097 Vingt mille Lieues Sous Les Mers, complète (French), Verne
  19657 Notre-Dame de Paris (French), Hugo

Traitement déterministe (seed 0) : téléchargement plain-text UTF-8,
retrait en-tête/pied Project Gutenberg (marqueurs *** START/END OF ...),
normalisation espaces, découpage en docs ~8 000 caractères sur frontières
de paragraphes (pas de mélange : ordre du livre). Sorties :
  runs/abl_data/ood.jsonl (docs, ignoré par git, régénérable)
  runs/abl_data/ood_manifest.json (titres, URLs, SHA1 brut/nettoyé par
    fichier, seed, règle, tailles)
ÉCHOUE si < 200 000 caractères ou si un hash normalisé recoupe les
tranches 00-09, slice_100M ou val_big (disjonction par SHA1).
"""

import hashlib
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "runs", "abl_data")
RAW = os.path.join(ROOT, "ood_raw")
SEED = 0
CHUNK = 8000

SOURCES = [
    {"ebook": 14155, "titre": "Madame Bovary", "auteur": "Gustave Flaubert"},
    {"ebook": 5097, "titre": "Vingt mille lieues sous les mers",
     "auteur": "Jules Verne"},
    {"ebook": 19657, "titre": "Notre-Dame de Paris", "auteur": "Victor Hugo"},
]
for s in SOURCES:
    s["url"] = (f"https://www.gutenberg.org/cache/epub/{s['ebook']}/"
                f"pg{s['ebook']}.txt")
    s["page"] = f"https://www.gutenberg.org/ebooks/{s['ebook']}"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def h(t: str) -> str:
    return hashlib.sha1(norm(t).encode("utf-8")).hexdigest()


def strip_gutenberg(raw: str) -> str:
    start = re.search(
        r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*",
        raw, re.S)
    end = re.search(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*",
                    raw, re.S)
    if not start or not end:
        raise ValueError("marqueurs Gutenberg introuvables")
    body = raw[start.end():end.start()]
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def chunk_docs(body: str) -> list[str]:
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    docs, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > CHUNK:
            docs.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur.strip():
        docs.append(cur)
    return [d for d in docs if len(d) >= 1000]


def main() -> None:
    os.makedirs(RAW, exist_ok=True)
    manifest = {"seed": SEED, "chunk_chars": CHUNK, "sources": [], "docs": []}
    all_docs = []
    for s in SOURCES:
        dest = os.path.join(RAW, f"pg{s['ebook']}.txt")
        if not os.path.exists(dest):
            print(f"[ood] téléchargement {s['url']}", flush=True)
            req = urllib.request.Request(s["url"], headers={"User-Agent": "micro-llm-2bit/ood"})
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(dest, "wb") as f:
                f.write(r.read())
        raw = open(dest, encoding="utf-8", errors="replace").read()
        body = strip_gutenberg(raw)
        docs = chunk_docs(body)
        rec = {**s, "sha1_brut": hashlib.sha1(
            raw.encode("utf-8")).hexdigest(),
            "sha1_nettoye": h(body), "chars_nettoye": len(body),
            "n_docs": len(docs)}
        manifest["sources"].append(rec)
        print(f"[ood] {s['titre']}: {len(body)} chars -> {len(docs)} docs",
              flush=True)
        all_docs += docs
    chars = sum(len(d) for d in all_docs)
    print(f"[ood] total {len(all_docs)} docs, {chars} chars", flush=True)
    if chars < 200_000:
        raise SystemExit(f"[ood] ÉCHEC : {chars} < 200 000 chars")
    # Disjonction : tranches 00-09 + slice_100M + val_big + C2 (hash normé).
    inter: set[str] = set()
    for i in range(10):
        p = f"/Volumes/Lexar/clean-corpus/slices/slice_0{i}.jsonl"
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line:
                    try:
                        inter.add(h(json.loads(line).get("text", "")))
                    except Exception:
                        pass
    for p in ("/Volumes/Lexar/clean-corpus/slice_100M.jsonl",
              os.path.join(ROOT, "val_big.jsonl")):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    inter.add(h(json.loads(line).get("text", "")))
                except Exception:
                    pass
    hashes = [h(d) for d in all_docs]
    recoup = [x for x in hashes if x in inter]
    if recoup:
        raise SystemExit(f"[ood] ÉCHEC : {len(recoup)} docs recoupent le "
                         "corpus d'entraînement/val")
    print("[ood] disjonction OK (tranches 00-09, slice_100M, val_big)",
          flush=True)
    with open(os.path.join(ROOT, "ood.jsonl"), "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    manifest["docs"] = hashes
    manifest["n_docs"] = len(all_docs)
    manifest["chars"] = chars
    json.dump(manifest, open(os.path.join(ROOT, "ood_manifest.json"), "w"),
              indent=1)
    print("[ood] TERMINE", flush=True)


if __name__ == "__main__":
    main()
