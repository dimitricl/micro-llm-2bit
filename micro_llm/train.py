"""Boucle d'entraînement MLX : CE + QAT ternaire, validation, checkpoints.

Ce fichier est un squelette fonctionnel, pas un script de production : il
privilégie la lisibilité et la reproductibilité. Les runs longs se lancent avec
`--steps` et un journal de bord séparé.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import get_config
from .data import TokenBatches, encode_corpus, iter_jsonl_texts
from .model import Transformer, count_params


def _load_tokenizer(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(path) / "tokenizer.json"))


def _prepare(paths: list[str], tokenizer, seq_len: int, max_tokens: int | None):
    texts = list(iter_jsonl_texts(paths))
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None:
        eos = 0
    return encode_corpus(tokenizer, texts, eos_id=eos, max_tokens=max_tokens)


def _val_loss(model, tokens, seq_len: int, batch_size: int, batches: int) -> float:
    if len(tokens) < seq_len + 1:
        return float("nan")
    loader = TokenBatches(tokens, seq_len, batch_size, seed=1234)
    total = 0.0
    n = 0
    for _ in range(min(batches, len(loader))):
        x, y = loader.batch()
        loss = model.loss(mx.array(x), mx.array(y))
        mx.eval(loss)
        total += float(loss)
        n += 1
    return total / max(n, 1)


def train(args: argparse.Namespace) -> dict:
    cfg = get_config(args.config)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    model = Transformer(cfg)
    n_params = count_params(model)
    print(f"[train] config={args.config} params={n_params/1e6:.2f}M seq={args.seq_len}", flush=True)

    if args.self_test:
        rng = np.random.default_rng(args.seed)
        tokens = rng.integers(0, cfg.vocab_size, size=args.seq_len * 64, dtype=np.uint16)
        val_tokens = rng.integers(0, cfg.vocab_size, size=args.seq_len * 8, dtype=np.uint16)
    else:
        tokenizer = _load_tokenizer(args.tokenizer)
        tokens = _prepare(args.data, tokenizer, args.seq_len, args.max_tokens)
        val_tokens = _prepare(args.val, tokenizer, args.seq_len, args.max_tokens // 20 if args.max_tokens else None)
    if len(tokens) < args.seq_len + 1:
        raise SystemExit("[train] corpus trop court")
    loader = TokenBatches(tokens, args.seq_len, args.batch_size, seed=args.seed)
    print(f"[train] tokens={len(tokens)/1e6:.2f}M batches={len(loader)}", flush=True)

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    loss_and_grad = nn.value_and_grad(model, model.loss)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    history = []
    for step in range(1, args.steps + 1):
        x, y = loader.batch()
        loss, grads = loss_and_grad(mx.array(x), mx.array(y))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            tok_s = step * args.batch_size * args.seq_len / max(dt, 1e-9)
            print(
                f"[train] step {step}/{args.steps} loss {float(loss):.4f} "
                f"ppl {float(mx.exp(loss)):.2f} tok/s {tok_s:.0f}",
                flush=True,
            )
            history.append({"step": step, "loss": float(loss), "tok_s": tok_s})
        if args.val_every and step % args.val_every == 0 and len(val_tokens) > args.seq_len:
            vl = _val_loss(model, val_tokens, args.seq_len, args.batch_size, args.val_batches)
            print(f"[val] step {step} loss {vl:.4f}", flush=True)

    from mlx.utils import tree_flatten

    ckpt = out / "model.safetensors"
    mx.save_safetensors(str(ckpt), dict(tree_flatten(model.parameters())))
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (out / "train.json").write_text(
        json.dumps({"args": vars(args), "params": n_params, "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"[train] checkpoint -> {ckpt}", flush=True)
    return {"params": n_params, "history": history}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Entraînement ternaire MLX (CE + QAT)")
    p.add_argument("--config", default="tiny", choices=["tiny", "base", "target_130m", "target_200m"])
    p.add_argument("--data", nargs="*", default=[])
    p.add_argument("--val", nargs="*", default=[])
    p.add_argument("--tokenizer", default="models/tokenizer_fr_16k")
    p.add_argument("--out", default="runs/qat")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=2_000_000)
    p.add_argument("--val-every", type=int, default=0)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.seq_len > get_config(args.config).seq_max:
        raise SystemExit("seq_len > seq_max")
    if not args.self_test and not args.data:
        raise SystemExit("--data requis (ou --self-test)")
    train(args)


if __name__ == "__main__":
    main()
