#!/usr/bin/env python3
"""Export et inférence MLX du modèle hybride 640/10 couches."""

from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

from train_hybrid_mlx import HybridLLM


class QuantHybridLLM(HybridLLM):
    """Tête liée compatible avec l'embedding quantifié MLX."""

    def __call__(self, x: mx.array) -> mx.array:
        batch, length = x.shape
        h = self.tok_emb(x)
        mask = self.causal_mask[:length, :length]
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        return self.tok_emb.as_linear(h) if self.tie_weights else self.lm_head(h)


def build(bits: int | None, group_size: int, checkpoint: Path,
          tokenizer_path: Path, quantized_checkpoint: bool = False):
    tok = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    model = QuantHybridLLM(vocab_size=tok.get_vocab_size(), max_seq_len=512)
    if quantized_checkpoint:
        nn.quantize(model, group_size=group_size, bits=bits)
        model.load_weights(str(checkpoint))
    else:
        model.load_weights(str(checkpoint))
        if bits:
            nn.quantize(model, group_size=group_size, bits=bits)
    model.eval()
    mx.eval(model.parameters())
    return model, tok


def flatten(prefix: str, value, result: dict[str, mx.array]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, child, result)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            flatten(f"{prefix}.{index}" if prefix else str(index), child, result)
    elif hasattr(value, "shape"):
        result[prefix] = value


def generate(model, tok: Tokenizer, prompt: str, max_new: int,
             temperature: float, repetition_penalty: float,
             no_repeat_ngram: int) -> tuple[str, float]:
    ids = tok.encode(prompt).ids
    if not ids:
        raise ValueError("Le prompt ne produit aucun token")
    out = list(ids)
    started = time.time()
    for _ in range(max_new):
        x = mx.array([out[-model.max_seq_len:]])
        logits = model(x)[0, -1, :].astype(mx.float32)
        if repetition_penalty != 1.0:
            for token in set(out[-64:]):
                logits[token] = (logits[token] / repetition_penalty
                                 if logits[token] > 0
                                 else logits[token] * repetition_penalty)
        if no_repeat_ngram >= 2 and len(out) >= no_repeat_ngram - 1:
            prefix = tuple(out[-(no_repeat_ngram - 1):])
            banned = {out[i + no_repeat_ngram - 1]
                      for i in range(len(out) - no_repeat_ngram + 1)
                      if tuple(out[i:i + no_repeat_ngram - 1]) == prefix}
            if banned:
                logits[mx.array(sorted(banned))] = -1e9
        if temperature <= 0:
            token = int(mx.argmax(logits).item())
        else:
            token = int(mx.random.categorical(logits / temperature).item())
        out.append(token)
    return tok.decode(out), time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["export", "infer", "bench"])
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/hybrid-mlx/final.safetensors"))
    parser.add_argument("--model", type=Path, default=None,
                        help="Quantized safetensors file for infer/bench")
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("models/tokenizer_fr_16k"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--prompt", default="Bonjour, que peux-tu faire ?")
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "export":
        if not args.out:
            parser.error("export exige --out")
        model, _ = build(args.bits, args.group_size, args.checkpoint, args.tokenizer)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        flat: dict[str, mx.array] = {}
        for name, params in model.parameters().items():
            flatten(name, params, flat)
        mx.save_safetensors(str(args.out), flat)
        print(f"[export] {args.out}: {args.out.stat().st_size / 1e6:.2f} Mo "
              f"({args.bits}-bit)", flush=True)
        return 0

    if not args.model:
        parser.error("infer/bench exige --model")
    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    model, tok = build(args.bits, args.group_size, args.model, args.tokenizer,
                       quantized_checkpoint=True)
    loaded_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    text, seconds = generate(model, tok, args.prompt, args.max_new,
                             args.temperature, args.repetition_penalty,
                             args.no_repeat_ngram)
    generated_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if args.mode == "infer":
        print(text)
    else:
        print(f"[bench] {args.model.stat().st_size / 1e6:.2f} Mo | "
              f"Δ RSS chargement {loaded_rss - base_rss:.1f} Mo | "
              f"après génération {generated_rss - base_rss:.1f} Mo | "
              f"{args.max_new / max(seconds, 1e-6):.1f} tok/s")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
