#!/usr/bin/env python3
"""Pré-entraînement hybride MLX depuis zéro (CE seule, Mac M4).

Architecture dense hybride ~47M params. Ce chemin n'utilise pas de modèle parent
ni de distillation ; le parent n'est pas nécessaire pour charger ce script.

Usage:
    python tools/train_hybrid_mlx.py \
        --data /Volumes/Lexar/clean-corpus/dedup data/hybrid \
        --tokenizer models/tokenizer_fr_16k \
        --output models/hybrid_mlx \
        --batch-size 16 --seq-len 512 --epochs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tokenizers import Tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Architecture Hybride (adaptée de train_25m.py)
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.weight


def swiglu(x: mx.array) -> mx.array:
    x, gate = mx.split(x, 2, axis=-1)
    return mx.sigmoid(gate) * x


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        cos, sin = self._build_rope_cache(max_seq_len, head_dim)
        self.rope_cos = cos
        self.rope_sin = sin

    def _build_rope_cache(self, max_seq_len: int, dim: int):
        inv_freq = 1.0 / (10000 ** (mx.arange(0, dim, 2) / dim))
        t = mx.arange(max_seq_len)
        freqs = mx.outer(t, inv_freq)
        return mx.cos(freqs), mx.sin(freqs)

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x1, x2 = mx.split(x, 2, axis=-1)
        cos = cos.reshape(1, -1, 1, x.shape[-1] // 2)
        sin = sin.reshape(1, -1, 1, x.shape[-1] // 2)
        return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            k = mx.repeat(k, self.n_heads // self.n_kv_heads, axis=2)
            v = mx.repeat(v, self.n_heads // self.n_kv_heads, axis=2)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 ffn_dim: int, max_seq_len: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, head_dim, max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        h = x + self.attn(self.attn_norm(x), mask)
        h = h + self.w2(swiglu(self.w1(self.ffn_norm(h))))
        return h


class HybridLLM(nn.Module):
    """Modèle hybride 40M params (adapté pour code + conversation)."""
    def __init__(self, vocab_size: int = 16000, dim: int = 640, n_layers: int = 10,
                 n_heads: int = 8, n_kv_heads: int = 4, ffn_dim: int = 1280,
                 max_seq_len: int = 512, tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        head_dim = dim // n_heads

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = [
            TransformerBlock(dim, n_heads, n_kv_heads, head_dim, ffn_dim, max_seq_len)
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(dim)

        if tie_weights:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.causal_mask = self._build_causal_mask(max_seq_len)

    def _build_causal_mask(self, max_seq_len: int) -> mx.array:
        mask = mx.triu(mx.ones((max_seq_len, max_seq_len)), k=1)
        return mask * -1e9

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape
        h = self.tok_emb(x)
        mask = self.causal_mask[:T, :T]
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        if self.tie_weights:
            logits = h @ self.tok_emb.weight.T
        else:
            logits = self.lm_head(h)
        return logits

    def num_params(self) -> int:
        def count(obj):
            if isinstance(obj, dict):
                return sum(count(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return sum(count(v) for v in obj)
            return int(obj.size) if hasattr(obj, "size") else 0
        return count(self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Data Loader
# ──────────────────────────────────────────────────────────────────────────────

def iter_texts(data_paths: list[Path]):
    """Yield useful text fields without holding the corpus in RAM."""
    for source in data_paths:
        files = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
        for path in files:
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSON invalide {path}:{line_no}") from exc
                    text = row.get("text", "")
                    if not text and row.get("instruction") and row.get("response"):
                        text = (f"<user>{row['instruction']}</user>"
                                f"<assistant>{row['response']}</assistant>")
                    if not text and row.get("question") and row.get("response"):
                        text = f"Question: {row['question']}\nRéponse: {row['response']}"
                    if isinstance(text, str) and text.strip():
                        yield text


def token_cache_key(data_paths: list[Path], tokenizer_path: Path) -> str:
    state = {"tokenizer": str(tokenizer_path.resolve()), "inputs": []}
    for source in data_paths:
        files = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
        for path in files:
            stat = path.stat()
            state["inputs"].append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    state["tokenizer_sha256"] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


class DataLoader:
    def __init__(self, data_paths: list[Path], tokenizer: Tokenizer,
                 tokenizer_path: Path, cache_path: Path, batch_size: int,
                 seq_len: int, seed: int = 42, reuse_only: bool = False):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_path.with_suffix(cache_path.suffix + ".json")
        cache_key = token_cache_key(data_paths, tokenizer_path)
        tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get("cache_key") == "external" and cache_path.exists():
            if manifest.get("tokenizer_sha256") != tokenizer_sha256:
                raise ValueError("Le cache tokens et le tokenizer ne correspondent pas")
            self.tokens = np.memmap(cache_path, dtype=np.uint16, mode="r")
            self.total_tokens = len(self.tokens)
            print(f"[dataloader] Cache réutilisé: {self.total_tokens:,} tokens", flush=True)
            return
        sep = tokenizer.encode("\n\n").ids
        if not sep:
            sep = [0]
        cache_valid = (manifest.get("cache_key") == cache_key and cache_path.exists()
                       and cache_path.stat().st_size % 2 == 0)
        if cache_valid and manifest.get("tokens") is not None:
            cache_valid = cache_path.stat().st_size // 2 == manifest["tokens"]
        if not cache_valid:
            if reuse_only:
                raise ValueError("Le cache demandé est absent ou ne correspond plus "
                                 "aux sources/tokenizer. Relance le pré-entraînement.")
            print(f"[dataloader] Tokenisation en deux passes vers {cache_path}", flush=True)
            total = 0
            docs = 0
            for text in iter_texts(data_paths):
                total += len(tokenizer.encode(text).ids) + len(sep)
                docs += 1
            if total <= seq_len + 1:
                raise ValueError(f"Corpus trop court: {total} tokens")
            print(f"[dataloader] {docs} documents, {total:,} tokens", flush=True)
            tokens = np.memmap(cache_path, dtype=np.uint16, mode="w+", shape=(total,))
            cursor = 0
            for text in iter_texts(data_paths):
                ids = tokenizer.encode(text).ids
                if ids:
                    tokens[cursor:cursor + len(ids)] = ids
                    cursor += len(ids)
                tokens[cursor:cursor + len(sep)] = sep
                cursor += len(sep)
            if cursor != total:
                raise RuntimeError(f"Comptage incohérent: {cursor} != {total}")
            tokens.flush()
            manifest_path.write_text(json.dumps({"cache_key": cache_key,
                                                  "tokenizer_sha256": tokenizer_sha256,
                                                  "tokens": total,
                                                  "documents": docs}, indent=2))
        self.tokens = np.memmap(cache_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.tokens)
        print(f"[dataloader] {self.total_tokens:,} tokens", flush=True)

    def __iter__(self):
        return self

    def sample(self, batch_size: int):
        max_start = self.total_tokens - self.seq_len - 1
        if max_start <= 0:
            raise StopIteration

        starts = self.rng.integers(0, max_start, size=batch_size)
        batch_ids = np.array([self.tokens[s:s+self.seq_len+1] for s in starts],
                             dtype=np.uint16)

        # x = tokens[:-1], y = tokens[1:]
        x = mx.array(batch_ids[:, :-1])
        y = mx.array(batch_ids[:, 1:])
        return x, y

    def __next__(self):
        return self.sample(self.batch_size)


class PackedSFTLoader:
    """Échantillonne les exemples SFT où le masque cible vaut 1."""

    def __init__(self, path: Path, seq_len: int, seed: int):
        packed = np.load(path)
        self.ids = packed["ids"]
        self.mask = packed["mask"]
        if self.ids.shape != self.mask.shape or self.ids.shape[1] != seq_len + 1:
            raise ValueError(f"Dimensions SFT incompatibles: {self.ids.shape}, "
                             f"seq_len={seq_len}")
        self.rng = np.random.default_rng(seed)
        self.num_examples = len(self.ids)

    def sample(self, batch_size: int):
        rows = self.rng.integers(0, self.num_examples, size=batch_size)
        batch_ids = self.ids[rows]
        batch_mask = self.mask[rows]
        x = mx.array(batch_ids[:, :-1])
        y = mx.array(batch_ids[:, 1:])
        mask = mx.array(batch_mask[:, 1:])
        return x, y, mask


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(model: HybridLLM, dataloader: DataLoader, optimizer: optim.Optimizer,
          max_iters: int, ckpt_dir: Path, ckpt_every: int = 1000,
          log_every: int = 100, start_step: int = 0, warmup_frac: float = 0.05,
          min_lr_ratio: float = 0.1, stage: str = "pretrain",
          sft_loader: PackedSFTLoader | None = None, replay_ratio: float = 0.1):
    """Boucle MLX CE, avec loss masquée et replay LM pendant le SFT."""
    loss_fn = nn.losses.cross_entropy
    model.train()
    started = time.time()
    tokens_per_step = dataloader.batch_size * dataloader.seq_len
    base_lr = float(optimizer.learning_rate)
    warmup_steps = max(1, int(max_iters * warmup_frac))

    for step in range(start_step, max_iters):
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            progress = ((step - warmup_steps) /
                        max(max_iters - warmup_steps - 1, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
        optimizer.learning_rate = lr
        if stage == "sft":
            assert sft_loader is not None
            n_replay = min(dataloader.batch_size - 1,
                           max(0, int(round(dataloader.batch_size * replay_ratio))))
            n_sft = dataloader.batch_size - n_replay
            x_sft, y_sft, mask_sft = sft_loader.sample(n_sft)
            if n_replay:
                x_lm, y_lm = dataloader.sample(n_replay)
                x = mx.concatenate([x_sft, x_lm], axis=0)
                y = mx.concatenate([y_sft, y_lm], axis=0)
                mask = mx.concatenate([mask_sft, mx.ones_like(y_lm)], axis=0)
            else:
                x, y, mask = x_sft, y_sft, mask_sft
        else:
            x, y = next(dataloader)
            mask = mx.ones_like(y)

        def loss_fn_with_batch(current_model):
            per_token = loss_fn(current_model(x), y, reduction="none")
            return mx.sum(per_token * mask) / mx.maximum(mx.sum(mask), 1)

        loss, grads = mx.value_and_grad(loss_fn_with_batch)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        if (step + 1) % log_every == 0 or step == start_step:
            ppl = mx.exp(loss)
            elapsed = max(time.time() - started, 1e-6)
            rate = (step - start_step + 1) * tokens_per_step / elapsed
            print(f"[train] step {step+1}/{max_iters} | loss {loss.item():.4f} "
                  f"| ppl {ppl.item():.2f} | lr {lr:.2e} | {rate:.0f} tok/s",
                  flush=True)

        if ckpt_every > 0 and (step + 1) % ckpt_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{step+1:07d}.safetensors"
            model.save_weights(str(ckpt_path))
            old_checkpoints = sorted(ckpt_dir.glob("checkpoint_*.safetensors"))
            for stale in old_checkpoints[:-2]:
                stale.unlink(missing_ok=True)
            print(f"[train] Checkpoint sauvegardé: {ckpt_path}", flush=True)

    # Final checkpoint
    final_path = ckpt_dir / "final.safetensors"
    model.save_weights(str(final_path))
    print(f"[train] Modèle final sauvegardé: {final_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement hybride MLX")
    parser.add_argument("--stage", choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="Fichiers ou dossiers JSONL à mélanger")
    parser.add_argument("--sft-data", type=Path, default=None,
                        help="NPZ IDs+masque pour --stage sft")
    parser.add_argument("--token-cache", type=Path, default=None,
                        help="Cache tokens uint16 existant à réutiliser")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Dossier tokenizer")
    parser.add_argument("--output", type=Path, required=True, help="Dossier de sortie")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0,
                        help="Passes équivalentes sur le flux tokenisé")
    parser.add_argument("--max-iters", type=int, default=0,
                        help="Remplace le nombre de steps calculé depuis epochs")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--replay-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Reprendre les poids; l'optimiseur repart à zéro")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Step global du checkpoint repris")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not 0.0 <= args.replay_ratio < 1.0:
        parser.error("--replay-ratio doit être dans [0, 1)")

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # Charger tokenizer
    tokenizer_path = args.tokenizer / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"[main] Vocab size: {vocab_size}", flush=True)

    # Créer modèle hybride
    model = HybridLLM(
        vocab_size=vocab_size,
        dim=640,
        n_layers=10,
        n_heads=8,
        n_kv_heads=4,
        ffn_dim=1280,
        max_seq_len=args.seq_len,
        tie_weights=True,
    )
    num_params = model.num_params()
    print(f"[main] Modèle créé: {num_params:,} params", flush=True)

    # DataLoader
    cache_path = args.token_cache or (args.output / "tokens.uint16.bin")
    dataloader = DataLoader(args.data, tokenizer, tokenizer_path, cache_path,
                            args.batch_size, args.seq_len, args.seed,
                            reuse_only=args.token_cache is not None)

    sft_loader = None
    if args.stage == "sft":
        if not args.sft_data:
            parser.error("--stage sft exige --sft-data")
        sft_loader = PackedSFTLoader(args.sft_data, args.seq_len, args.seed)
        replay_samples = args.batch_size - max(
            0, min(args.batch_size - 1, int(round(args.batch_size * args.replay_ratio))))
        print(f"[sft] {sft_loader.num_examples} exemples | "
              f"{replay_samples} exemples SFT/step + "
              f"{args.batch_size-replay_samples} replay LM", flush=True)
        steps_per_epoch = math.ceil(sft_loader.num_examples / max(replay_samples, 1))
    else:
        steps_per_epoch = math.ceil(dataloader.total_tokens /
                                    (args.batch_size * args.seq_len))
    max_iters = args.max_iters or math.ceil(steps_per_epoch * args.epochs)

    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    # Output dir
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        model.load_weights(str(args.resume))
        if args.resume_step == 0:
            match = re.search(r"(\d+)", args.resume.stem)
            args.resume_step = int(match.group(1)) if match else 0
        print(f"[main] Reprise des poids {args.resume} au step {args.resume_step}; "
              "état AdamW réinitialisé", flush=True)

    config = vars(args).copy()
    config["data"] = [str(p) for p in args.data]
    config["output"] = str(args.output)
    config["tokenizer"] = str(args.tokenizer)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[main] {dataloader.total_tokens:,} tokens | "
          f"étape={args.stage} | {steps_per_epoch} steps/epoch | "
          f"{max_iters} steps prévus", flush=True)
    start_time = time.time()
    train(model, dataloader, optimizer, max_iters, args.output,
          args.ckpt_every, args.log_every, args.resume_step,
          args.warmup_frac, args.min_lr_ratio, args.stage,
          sft_loader, args.replay_ratio)
    elapsed = time.time() - start_time
    print(f"[main] Entraînement terminé en {elapsed/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
