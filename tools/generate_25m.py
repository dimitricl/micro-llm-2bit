#!/usr/bin/env python3
"""Générations du modèle 25M from-scratch (MLX).

Règles de déploiement G : pénalité de répétition 1.15 et interdiction des
3-grammes répétés, greedy (T=0) ou T=0,7.

Usage :
  .venv/bin/python tools/generate_25m.py --temp 0 --out docs/results/g5_25m/gen_greedy.md
  .venv/bin/python tools/generate_25m.py --temp 0.7 --seed 0 \
      --out docs/results/g5_25m/gen_t07.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
from tokenizers import Tokenizer

from train_25m import MicroLLM

DEFAULT_PROMPTS = [
    "La France est un pays",
    "Le chat",
    "Bonjour, je m'appelle",
    "La Révolution française a commencé en",
    "La capitale de l'Italie est",
    "L'eau bout à",
    "Le soleil se lève à l'est et",
    "Un triangle a trois",
    "Python est un langage de programmation",
    "La Seconde Guerre mondiale s'est terminée en",
    "Les êtres humains ont besoin de",
    "Le football est un sport où",
    "Dans une recette de crêpes, il faut",
    "Le changement climatique est causé par",
    "Apprendre une langue étrangère",
    "Le système solaire comprend",
    "Une bonne alimentation",
    "L'intelligence artificielle est",
    "Le travail en équipe",
    "Pour résumer,"
]


def banned(ids: list[int], n: int = 3) -> set[int]:
    if len(ids) < n - 1:
        return set()
    prefix = tuple(ids[-(n - 1):])
    out = set()
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n - 1]) == prefix:
            out.add(ids[i + n - 1])
    return out


def generate(model: MicroLLM, tok: Tokenizer, prompt: str, n_new: int,
             temp: float, rep_pen: float, no_rep: int, seed: int) -> tuple[str, float]:
    rng = mx.random.key(seed)
    ids = tok.encode(prompt).ids
    out = list(ids)
    t0 = time.time()
    for _ in range(n_new):
        x = mx.array([out[-1024:]])
        logits = model(x)[0, -1, :].astype(mx.float32)
        if rep_pen != 1.0:
            for t in set(out[-64:]):
                logits[t] = logits[t] / rep_pen if logits[t] > 0 else logits[t] * rep_pen
        ban = banned(out, no_rep) if no_rep >= 2 else set()
        if ban:
            logits[mx.array(sorted(ban))] = -1e9
        if temp <= 0:
            nxt = int(mx.argmax(logits).item())
        else:
            rng, key = mx.random.split(rng)
            nxt = int(mx.random.categorical(logits / temp, key=key).item())
        out.append(nxt)
    return tok.decode(out), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models/llm_25m_fr/model.safetensors"))
    ap.add_argument("--tokenizer", type=Path, default=Path("models/tokenizer_fr_16k"))
    ap.add_argument("--prompts", type=Path, default=None,
                    help="fichier texte, un prompt par ligne")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [l.strip() for l in args.prompts.read_text(encoding="utf-8").splitlines()
                   if l.strip()]

    tok = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    tok.no_truncation()
    model = MicroLLM(vocab_size=tok.get_vocab_size())
    model.load_weights(str(args.ckpt))
    model.eval()
    mx.eval(model.parameters())

    rows = []
    for i, p in enumerate(prompts):
        text, dt = generate(model, tok, p, args.max_new, args.temp,
                            args.rep_penalty, args.no_repeat_ngram,
                            args.seed * 1000 + i)
        rows.append({"prompt": p, "output": text, "seconds": round(dt, 2)})
        print(f"[gen] {p!r} -> {text!r} ({dt:.1f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".json":
        args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    else:
        lines = [f"# Générations 25M — T={args.temp} ", ""]
        for r in rows:
            lines += [f"**{r['prompt']}**", "", "```", r["output"], "```", ""]
        args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[gen] écrit {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
