#!/usr/bin/env python3
"""Évaluation PPL/BPC du modèle 25M from-scratch (MLX), avec bootstrap.

Protocole :
- fenêtres non recouvrantes de S+1 tokens (S=128 par défaut, comme la
  chaîne E/F ; S=1024 = contexte natif d'entraînement en option) ;
- NLL = logsumexp(logits) - logit(cible), sommé par document ;
- BPC = NLL / ln2 / caractères (convention eval_valbig.py) ;
- PPL = exp(NLL / tokens) ;
- IC 95 % par bootstrap 1000 tirages sur les documents (seed 0).

Aucun token spécial (le tokenizer BPE 16k n'en a pas et le corpus
d'entraînement concatène les documents sans EOS).

Usage :
  .venv/bin/python tools/eval_25m_mlx.py \
      --ckpt models/llm_25m_fr/model.safetensors \
      --docs ood=runs/abl_data/ood.jsonl \
      --docs valE=runs/abl_data/val_E_frozen_clean32.jsonl \
      --docs valbig=runs/abl_data/val_big.jsonl \
      --seq-len 128 --out docs/results/g5_25m_s128.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

from train_25m import MicroLLM

LN2 = math.log(2)


def load_docs(path: str) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if t.strip():
                out.append(t)
    return out


def windows(ids: list[int], S: int) -> list[list[int]]:
    return [ids[i:i + S + 1] for i in range(0, len(ids) - 1, S)
            if len(ids[i:i + S + 1]) == S + 1]


def nll_sum(model: MicroLLM, wins: list[list[int]], batch: int) -> tuple[float, int]:
    """Somme des NLL et nombre de tokens cibles sur un lot de fenêtres."""
    total = 0.0
    ntokens = 0
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]
        x = mx.array([w[:-1] for w in chunk])
        y = mx.array([w[1:] for w in chunk])
        logits = model(x).astype(mx.float32)
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
            logits, y[..., None], axis=-1).squeeze(-1)
        total += float(mx.sum(nll).item())
        ntokens += int(y.size)
    return total, ntokens


def bootstrap_bpc_ppl(per_doc: list[dict], draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(per_doc)
    NLL = np.array([d["nll"] for d in per_doc], dtype=np.float64)
    CH = np.array([d["chars"] for d in per_doc], dtype=np.float64)
    CS = np.array([d["chars_scored"] for d in per_doc], dtype=np.float64)
    NT = np.array([d["n"] for d in per_doc], dtype=np.float64)
    idx = rng.integers(0, n, size=(draws, n))
    nll = NLL[idx].sum(axis=1)
    chars = CH[idx].sum(axis=1)
    chars_s = CS[idx].sum(axis=1)
    ntok = NT[idx].sum(axis=1)
    bpc = nll / LN2 / np.maximum(chars, 1e-9)
    bpc_s = nll / LN2 / np.maximum(chars_s, 1e-9)
    ppl = np.exp(nll / np.maximum(ntok, 1e-9))
    return {
        "n_docs": int(n),
        "nll": float(NLL.sum()),
        "chars": float(CH.sum()),
        "chars_scored": float(CS.sum()),
        "tokens": int(NT.sum()),
        # bpc = convention eval_valbig (dénominateur = texte entier) ;
        # bpc_scored = caractères réellement couverts par les fenêtres.
        "bpc": float(NLL.sum() / LN2 / CH.sum()),
        "bpc_ci": [float(np.quantile(bpc, 0.025)), float(np.quantile(bpc, 0.975))],
        "bpc_scored": float(NLL.sum() / LN2 / CS.sum()),
        "bpc_scored_ci": [float(np.quantile(bpc_s, 0.025)),
                          float(np.quantile(bpc_s, 0.975))],
        "ppl": float(math.exp(NLL.sum() / NT.sum())),
        "ppl_ci": [float(np.quantile(ppl, 0.025)), float(np.quantile(ppl, 0.975))],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="PPL/BPC MLX du modèle 25M")
    ap.add_argument("--ckpt", type=Path, default=Path("models/llm_25m_fr/model.safetensors"))
    ap.add_argument("--tokenizer", type=Path, default=Path("models/tokenizer_fr_16k"))
    ap.add_argument("--docs", action="append", default=[],
                    help="nom=chemin.jsonl (répétable)")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-windows", type=int, default=8)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--quant-bits", type=int, default=0,
                    help="0 = fp32 ; 4/8 = modèle quantifié")
    ap.add_argument("--quant-group", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perdoc-dir", type=Path, default=None)
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    tok.no_truncation()
    if args.quant_bits:
        from export_25m import QuantLLM
        model = QuantLLM(vocab_size=tok.get_vocab_size())
        nn.quantize(model, group_size=args.quant_group, bits=args.quant_bits)
    else:
        model = MicroLLM(vocab_size=tok.get_vocab_size())
    model.load_weights(str(args.ckpt))
    model.eval()
    mx.eval(model.parameters())

    report: dict = {"win": args.seq_len, "draws": args.draws,
                    "seed": args.seed, "ckpt": str(args.ckpt), "sets": {}}
    if args.perdoc_dir:
        args.perdoc_dir.mkdir(parents=True, exist_ok=True)

    for spec in args.docs:
        if "=" not in spec:
            print(f"[eval] spec invalide : {spec}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        docs = load_docs(path)
        if args.max_docs:
            docs = docs[:args.max_docs]
        per_doc: list[dict] = []
        t0 = time.time()
        ntok = 0
        for k, text in enumerate(docs):
            ids = tok.encode(text).ids
            # caractères = longueur du texte (convention eval_valbig.py)
            nchars = len(text)
            wins = windows(ids, args.seq_len)
            if not wins:
                continue
            s, n = nll_sum(model, wins, args.batch_windows)
            # caractères réellement couverts par les fenêtres (cibles)
            scored_ids = ids[1:len(wins) * args.seq_len + 1]
            chars_scored = len(tok.decode(scored_ids))
            per_doc.append({"doc": k, "nll": s, "n": n, "chars": nchars,
                            "chars_scored": chars_scored})
            ntok += n
            if (k + 1) % 25 == 0:
                print(f"[eval] {name}: {k+1}/{len(docs)} docs, {ntok:,} tok, "
                      f"{time.time()-t0:.0f}s", flush=True)
        summ = bootstrap_bpc_ppl(per_doc, args.draws, args.seed)
        dt = time.time() - t0
        summ["duration_s"] = round(dt, 1)
        summ["tok_s"] = round(ntok / dt, 1) if dt else 0.0
        report["sets"][name] = summ
        print(f"[eval] {name}: BPC {summ['bpc']:.4f} "
              f"[{summ['bpc_ci'][0]:.4f}, {summ['bpc_ci'][1]:.4f}] "
              f"(scored {summ['bpc_scored']:.4f}) | "
              f"PPL {summ['ppl']:.2f} "
              f"[{summ['ppl_ci'][0]:.2f}, {summ['ppl_ci'][1]:.2f}] | "
              f"{summ['tok_s']:.0f} tok/s", flush=True)
        if args.perdoc_dir:
            (args.perdoc_dir / f"{name}.json").write_text(
                json.dumps({"docs": per_doc, "win": args.seq_len,
                            "source": path}, ensure_ascii=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[eval] écrit {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
