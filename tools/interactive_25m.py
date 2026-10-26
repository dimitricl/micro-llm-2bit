#!/usr/bin/env python3
"""Interface interactive pour le modèle 25M/46M (MLX).

Usage :
  .venv/bin/python tools/interactive_25m.py --ckpt runs/G5-qa-s1/model.safetensors
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
from tokenizers import Tokenizer

from train_25m import MicroLLM


def banned(ids: list[int], n: int = 3) -> set[int]:
    if len(ids) < n - 1:
        return set()
    prefix = tuple(ids[-(n - 1):])
    out = set()
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n - 1]) == prefix:
            out.add(ids[i + n - 1])
    return out


def generate(model: MicroLLM, tok: Tokenizer, prompt: str, max_new: int,
             temp: float, rep_pen: float, no_rep: int) -> str:
    ids = tok.encode(prompt).ids
    out = list(ids)
    for _ in range(max_new):
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
            nxt = int(mx.random.categorical(logits / temp).item())
        out.append(nxt)
    return tok.decode(out[len(ids):]).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models/llm_25m_fr/model.safetensors"))
    ap.add_argument("--tokenizer", type=Path, default=Path("models/tokenizer_fr_16k"))
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    tok.no_truncation()
    model = MicroLLM(vocab_size=tok.get_vocab_size())
    model.load_weights(str(args.ckpt))
    model.eval()
    mx.eval(model.parameters())

    print(f"Modèle chargé : {args.ckpt}")
    print(f"Max tokens : {args.max_new}, Température : {args.temp}")
    print("Tapez 'quit' pour quitter\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                break

            output = generate(model, tok, prompt, args.max_new,
                            args.temp, args.rep_penalty, args.no_repeat_ngram)
            print(f"\n{output}\n")
        except KeyboardInterrupt:
            print("\nquit")
            break
        except EOFError:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
