"""Télécharge les textes Gutenberg français du catalogue local.

Sortie : runs/G2-data/gutenberg/pg<id>.txt + manifeste JSON.
Ne touche jamais aux fichiers OOD/val ; les textes OOD seront exclus par
l'audit n-grammes avant G2.

Usage :
  .venv/bin/python tools/download_gutenberg_G.py [--limit N] [--workers 3]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)

CATALOG = "/Volumes/Lexar/rag-data/livres/pg_catalog.csv"
OUTDIR = "runs/G2-data/gutenberg"
MANIFEST = "runs/G2-data/gutenberg_manifest.json"
UA = "micro-llm-2bit/0.1 (research; contact local)"


def french_ids() -> list[dict]:
    with open(CATALOG, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r.get("Language") != "fr":
            continue
        try:
            rid = int(r["Text#"])
        except Exception:
            continue
        out.append({
            "id": rid,
            "title": r.get("Title", ""),
            "authors": r.get("Authors", ""),
        })
    return out


def url_candidates(rid: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{rid}/pg{rid}.txt",
        f"https://www.gutenberg.org/files/{rid}/{rid}-0.txt",
        f"https://www.gutenberg.org/files/{rid}/{rid}.txt",
    ]


def fetch_one(rid: int) -> tuple[int, str, str]:
    path = f"{OUTDIR}/pg{rid}.txt"
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return rid, "skip", path
    last = ""
    for url in url_candidates(rid):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
            if len(data) < 1000:
                last = f"too small ({len(data)} bytes) {url}"
                continue
            with open(path, "wb") as f:
                f.write(data)
            return rid, "ok", path
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} {url}"
        except Exception as e:
            last = f"{type(e).__name__}: {e} {url}"
        time.sleep(0.2)
    return rid, f"fail: {last}", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    books = french_ids()
    if a.limit:
        books = books[:a.limit]
    print(f"[gutenberg] {len(books)} livres français à traiter", flush=True)
    results = {}
    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(fetch_one, b["id"]): b for b in books}
        for i, fut in enumerate(as_completed(futs), 1):
            b = futs[fut]
            rid, status, path = fut.result()
            results[str(rid)] = {**b, "status": status, "path": path}
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"[gutenberg] ÉCHEC {rid} {b['title'][:50]} : {status}",
                      flush=True)
            if i % 25 == 0 or i == len(books):
                print(f"[gutenberg] {i}/{len(books)} ok={ok} skip={skip} "
                      f"fail={fail}", flush=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"catalog": CATALOG, "outdir": OUTDIR,
                   "n": len(books), "ok": ok, "skip": skip, "fail": fail,
                   "books": results}, f, ensure_ascii=False, indent=1)
    print(f"[gutenberg] -> {MANIFEST} (ok={ok} skip={skip} fail={fail})")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
