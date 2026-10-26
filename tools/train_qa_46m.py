#!/usr/bin/env python3
"""Fine-tuning QA du backbone 46M (MLX), recette G1.

- initialisation : `models/llm_25m_fr/model.safetensors` (run from-scratch) ;
- données : fenêtres packées par `tools/prepare_qa_mlx.py` (prompt masqué,
  loss sur la réponse + séparateur) ;
- recette G1 : AdamW lr 3e-4, cosine, warmup 5 %, WD 0,01, clip 1,0,
  batch effectif 128 fenêtres × seq 256 (= 32 768 tokens/step), 924 steps.

Usage (calibration) :
  .venv/bin/python tools/train_qa_46m.py --data runs/G5-qa-data/qa_prompt_packed.npz \
      --out runs/G5-qa-cal --steps 8
Usage (run) :
  .venv/bin/python tools/train_qa_46m.py --data ... --out runs/G5-qa-s0 --steps 924 --seed 0
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
import mlx.optimizers as optim
import numpy as np

from train_25m import MicroLLM, CosineScheduler, save_model


def masked_ce(model: MicroLLM, x: mx.array, y: mx.array, m: mx.array) -> mx.array:
    logits = model(x).astype(mx.float32)
    nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
        logits, y[..., None], axis=-1).squeeze(-1)
    return (nll * m).sum() / mx.maximum(m.sum(), 1.0)


def add_trees(a, b):
    if a is None:
        return b
    if isinstance(a, dict):
        return {k: add_trees(a[k], b[k]) for k in a}
    if isinstance(a, (list, tuple)):
        return [add_trees(x, y) for x, y in zip(a, b)]
    return a + b


def div_tree(a, d: float):
    if isinstance(a, dict):
        return {k: div_tree(v, d) for k, v in a.items()}
    if isinstance(a, (list, tuple)):
        return [div_tree(v, d) for v in a]
    return a / d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="models/llm_25m_fr/model.safetensors")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=924)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data)
    ids, mask = d["ids"], d["mask"]
    assert ids.shape[1] == args.seq_len + 1, (ids.shape, args.seq_len)
    N = ids.shape[0]
    print(f"[qa-train] {N} fenêtres × {args.seq_len} tokens, "
          f"{int(mask.sum()):,} tokens de réponse", flush=True)

    model = MicroLLM(vocab_size=16000)
    model.load_weights(args.init)
    model.train()
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    warmup = max(1, int(args.steps * args.warmup_ratio))
    sched = CosineScheduler(args.lr, warmup, args.steps, min_lr=args.lr * 0.1)
    mx.eval(optimizer.state)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    pos = 0

    def next_idx(n: int) -> np.ndarray:
        nonlocal perm, pos
        parts = []
        while sum(len(p) for p in parts) < n:
            if pos >= N:
                perm = rng.permutation(N)
                pos = 0
            take = min(N - pos, n - sum(len(p) for p in parts))
            parts.append(perm[pos:pos + take])
            pos += take
        return np.concatenate(parts)

    t0 = time.time()
    tok_seen = 0
    for step in range(args.steps):
        lr = sched(step)
        optimizer.learning_rate = lr
        acc = None
        losses = []
        for _ in range(args.accum):
            idx = next_idx(args.batch_size)
            x = mx.array(ids[idx, :-1].astype(np.int32))
            y = mx.array(ids[idx, 1:].astype(np.int32))
            m = mx.array(mask[idx, 1:].astype(np.float32))
            loss, grads = mx.value_and_grad(masked_ce)(model, x, y, m)
            acc = add_trees(acc, grads)
            losses.append(float(loss.item()))
            # Matérialise immédiatement : sans ça, les 4 graphes
            # d'accumulation restent vivants et l'OS tue le process (137).
            mx.eval(acc, loss)
        acc = div_tree(acc, args.accum)
        acc, gnorm = optim.clip_grad_norm(acc, args.max_grad_norm)
        optimizer.update(model, acc)
        mx.eval(model.parameters(), optimizer.state)
        tok_seen += args.batch_size * args.accum * args.seq_len

        if step % args.log_every == 0 or step == args.steps - 1:
            dt = time.time() - t0
            print(f"[qa-train] step {step:5d}/{args.steps} | "
                  f"loss {np.mean(losses):.4f} | lr {lr:.2e} | "
                  f"gnorm {float(gnorm.item()):.2f} | "
                  f"{tok_seen/dt:.0f} tok/s | elapsed {dt:.0f}s", flush=True)

        if args.save_every and step and step % args.save_every == 0:
            save_model(model, out / f"checkpoint_{step:06d}.safetensors")

    save_model(model, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps({
        "init": args.init, "data": args.data, "steps": args.steps,
        "batch_size": args.batch_size, "accum": args.accum,
        "seq_len": args.seq_len, "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "duration_s": round(time.time() - t0, 1),
    }, indent=1))
    print(f"[qa-train] terminé : {out/'model.safetensors'} "
          f"({(out/'model.safetensors').stat().st_size/1e6:.1f} Mo)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
