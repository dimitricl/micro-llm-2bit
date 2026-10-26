#!/usr/bin/env python3
"""Export quantifié du modèle 25M (MLX) + mesure taille / RSS / tok/s.

Le modèle lie l'embedding et la tête de sortie ; la classe QuantLLM utilise
`tok_emb.as_linear` (compatible fp32 et quantifié) au lieu du `@ weight.T`.

Usage :
  .venv/bin/python tools/export_25m.py export --bits 8 --out exports/llm_25m_fr_int8.safetensors
  .venv/bin/python tools/export_25m.py bench  --bits 8 --model exports/llm_25m_fr_int8.safetensors
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

from train_25m import MicroLLM, _flatten_params
from generate_25m import generate


class QuantLLM(MicroLLM):
    """MicroLLM avec une tête liée compatible quantization."""

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape
        h = self.tok_emb(x)
        mask = self.causal_mask[:T, :T]
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        if self.tie_weights:
            logits = self.tok_emb.as_linear(h)
        else:
            logits = self.lm_head(h)
        return logits


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def build(bits: int | None, group: int, ckpt: str | None = None) -> QuantLLM:
    tok = Tokenizer.from_file("models/tokenizer_fr_16k/tokenizer.json")
    model = QuantLLM(vocab_size=tok.get_vocab_size())
    if bits:
        nn.quantize(model, group_size=group, bits=bits)
    if ckpt:
        model.load_weights(ckpt)
    model.eval()
    mx.eval(model.parameters())
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "bench"])
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--ckpt", default="models/llm_25m_fr/model.safetensors")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "export":
        base = rss_mb()
        model = build(None, args.group_size, args.ckpt)
        fp_rss = rss_mb()
        nn.quantize(model, group_size=args.group_size, bits=args.bits)
        mx.eval(model.parameters())
        flat: dict = {}
        for name, params in model.parameters().items():
            _flatten_params(name, params, flat)
        out = args.out or f"exports/llm_25m_fr_int{args.bits}.safetensors"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        mx.save_safetensors(out, flat)
        size = os.path.getsize(out)
        print(f"[export] {out}: {size/1e6:.2f} Mo | RSS fp32 {fp_rss:.1f} Mo "
              f"(+{fp_rss-base:.1f}) | tensors {len(flat)}")
        return 0

    # bench : chargement dans un process propre
    t0 = time.time()
    base = rss_mb()
    model = build(args.bits, args.group_size, args.model)
    loaded = rss_mb()
    tok = Tokenizer.from_file("models/tokenizer_fr_16k/tokenizer.json")
    generate(model, tok, "Bonjour", 8, 0.0, 1.15, 3, 0)  # warmup
    n = 64
    text, dt = generate(model, tok, "La France est un pays", n, 0.0, 1.15, 3, 0)
    size = os.path.getsize(args.model) if args.model else 0
    print(f"[bench] {args.model} {size/1e6:.2f} Mo | RSS base {base:.1f} Mo "
          f"-> {loaded:.1f} Mo (Δ {loaded-base:+.1f} Mo) | "
          f"{n/dt:.1f} tok/s | {dt:.1f}s | {text[:120]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
