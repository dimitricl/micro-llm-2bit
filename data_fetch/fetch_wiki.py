"""Recupere ~1 Go de Wikipedia FR (streaming) vers /Volumes/Lexar/clean-corpus."""

import json
import os

from datasets import load_dataset

OUT_DIR = "/Volumes/Lexar/clean-corpus"
OUT = os.path.join(OUT_DIR, "wiki_fr.jsonl")
TARGET = 1024 ** 3  # 1 Go
MIN_LEN = 500

os.makedirs(OUT_DIR, exist_ok=True)

ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
written = 0
n_docs = 0
next_mark = 100 * 1024 * 1024
with open(OUT, "w", encoding="utf-8") as f:
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < MIN_LEN:
            continue
        line = json.dumps(
            {"title": row.get("title", ""), "text": text}, ensure_ascii=False
        )
        f.write(line + "\n")
        n_docs += 1
        written += len(line.encode("utf-8")) + 1
        if written >= next_mark:
            print(f"[fetch] {written / 1e9:.2f} Go, {n_docs} docs", flush=True)
            next_mark += 100 * 1024 * 1024
        if written >= TARGET:
            break
print(f"[fetch] TERMINE : {written / 1e9:.2f} Go, {n_docs} docs -> {OUT}")
