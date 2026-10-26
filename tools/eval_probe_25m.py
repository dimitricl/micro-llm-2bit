#!/usr/bin/env python3
"""Sonde FR gelée (200 items) pour le modèle 25M from-scratch (MLX).

Même règle de score que tools/eval_probe.py : normalisation sans accents,
ponctuation -> espace, alias accepté avec frontières de tokens.
Génération greedy, pénalité 1.15, no-repeat 3, max_new=32 (pas d'EOS dans
le tokenizer : la génération va toujours au bout).

Usage :
  .venv/bin/python tools/eval_probe_25m.py \
      --out docs/results/g5_25m/probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
from tokenizers import Tokenizer

from train_25m import MicroLLM
from eval_qa_25m import generate, bootstrap_ci


def normalize(text: str) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    out = []
    for ch in stripped.lower():
        out.append(" " if unicodedata.category(ch).startswith("P") else ch)
    return " ".join("".join(out).split())


def is_hit(gen_text: str, accepted: list[str], max_tokens: int | None = None) -> bool:
    hay = normalize(gen_text)
    if not hay:
        return False
    padded = f" {hay} "
    for alias in accepted or []:
        needle = normalize(alias)
        if needle and f" {needle} " in padded:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/probe/probe_fr_v1.jsonl")
    ap.add_argument("--ckpt", default="models/llm_25m_fr/model.safetensors")
    ap.add_argument("--tokenizer", default="models/tokenizer_fr_16k")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
    tok.no_truncation()
    model = MicroLLM(vocab_size=tok.get_vocab_size())
    model.load_weights(args.ckpt)
    model.eval()
    mx.eval(model.parameters())

    rows = []
    for it in items:
        text, n = generate(model, tok, it["prompt"], args.max_new, 1.15, 3)
        rows.append({
            "id": it["id"], "categorie": it["categorie"], "slice": it["slice"],
            "perimetre": it["perimetre"], "prompt": it["prompt"], "gen": text,
            "hit": is_hit(text, it.get("reponses_acceptees") or []),
        })
    hits = [int(r["hit"]) for r in rows]
    summary = {
        "n": len(rows),
        "hit": sum(hits) / len(hits),
        "hit_ci95": bootstrap_ci(hits),
        "by_slice": {
            s: {"n": sum(1 for r in rows if r["slice"] == s),
                "hit": sum(r["hit"] for r in rows if r["slice"] == s)
                      / max(1, sum(1 for r in rows if r["slice"] == s))}
            for s in sorted({r["slice"] for r in rows})
        },
        "by_categorie": {
            c: {"n": sum(1 for r in rows if r["categorie"] == c),
                "hit": sum(r["hit"] for r in rows if r["categorie"] == c)
                      / max(1, sum(1 for r in rows if r["categorie"] == c))}
            for c in sorted({r["categorie"] for r in rows})
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
