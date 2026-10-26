#!/usr/bin/env python3
"""
Entraînement from-scratch 25M paramètres 100% MLX sur Mac M4.

Architecture MobileBERT/ALBERT-style ~25M params :
- Vocab: 16000, Dim: 512, Layers: 12, Heads: 8 (GQA 8:2), FFN: 1536 (SwiGLU)
- RoPE (context 1024), Tied Embeddings, RMSNorm, SwiGLU
- Tied weights: lm_head.weight = token_embeddings.weight

Usage:
    python tools/train_25m.py \
        --data data/corpus.bin \
        --tokenizer models/tokenizer_fr_16k \
        --output models/llm_25m_fr \
        --batch-size 32 --seq-len 1024 --max-iters 100000 \
        --lr 3e-4 --warmup-iters 2000 --weight-decay 0.1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tokenizers import Tokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Architecture MobileBERT/ALBERT-style ~25M params
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """RMSNorm (plus stable que LayerNorm pour modèles compacts)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.weight


def swiglu(x: mx.array) -> mx.array:
    """SwiGLU activation: swish(x) * x where swish(x) = x * sigmoid(x)"""
    x, gate = mx.split(x, 2, axis=-1)
    return mx.sigmoid(gate) * x


class GroupedQueryAttention(nn.Module):
    """GQA: 8 query heads, 2 key/value heads (ratio 4:1)."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Projections
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        # RoPE cache (pre-computed as arrays)
        cos, sin = self._build_rope_cache(max_seq_len, head_dim)
        self.rope_cos = cos
        self.rope_sin = sin

    def _build_rope_cache(self, max_seq_len: int, dim: int):
        """Precompute RoPE cos/sin."""
        inv_freq = 1.0 / (10000 ** (mx.arange(0, dim, 2) / dim))
        t = mx.arange(max_seq_len)
        freqs = mx.outer(t, inv_freq)
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        return cos, sin

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Apply RoPE to query/key. x shape: (B, T, H, D)"""
        x1, x2 = mx.split(x, 2, axis=-1)
        # cos/sin shape: (T, D/2) -> need to broadcast to (B, T, H, D/2)
        cos = cos.reshape(1, -1, 1, x.shape[-1] // 2)
        sin = sin.reshape(1, -1, 1, x.shape[-1] // 2)
        return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, T, _ = x.shape

        # Projections
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        # RoPE
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # GQA: répète k, v pour matcher n_heads
        if self.n_kv_heads != self.n_heads:
            k = mx.repeat(k, self.n_heads // self.n_kv_heads, axis=2)
            v = mx.repeat(v, self.n_heads // self.n_kv_heads, axis=2)

        # Attention
        q = q.transpose(0, 2, 1, 3)  # (B, H, T, D)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)  # (B, H, T, D)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    """Bloc Transformer avec RMSNorm pré-normalisation, SwiGLU, résiduels."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 ffn_dim: int, max_seq_len: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, head_dim, max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        # SwiGLU: input -> 2 * ffn_dim -> split -> swiglu -> output
        self.w1 = nn.Linear(dim, 2 * ffn_dim, bias=False)  # gate + up
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)      # down

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # Attention block
        h = x + self.attn(self.attn_norm(x), mask)
        # FFN block (SwiGLU)
        h = h + self.w2(swiglu(self.w1(self.ffn_norm(h))))
        return h


class MicroLLM(nn.Module):
    """
    Micro-LLM 25M params : Embedding + 12 Transformer Blocks + Tied LM Head
    """
    def __init__(self, vocab_size: int = 16000, dim: int = 512, n_layers: int = 12,
                 n_heads: int = 8, n_kv_heads: int = 2, ffn_dim: int = 1536,
                 max_seq_len: int = 1024, tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        head_dim = dim // n_heads

        # Embedding partagée (input + output)
        self.tok_emb = nn.Embedding(vocab_size, dim)

        # Blocs Transformer
        self.layers = [
            TransformerBlock(dim, n_heads, n_kv_heads, head_dim, ffn_dim, max_seq_len)
            for _ in range(n_layers)
        ]

        self.norm = RMSNorm(dim)

        # LM Head - TIED WEIGHTS (partage avec tok_emb)
        if tie_weights:
            self.lm_head = None  # Utilise tok_emb.weight transposé
        else:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Masque causal pré-calculé
        self.causal_mask = self._build_causal_mask(max_seq_len)

    def _build_causal_mask(self, max_seq_len: int) -> mx.array:
        """Masque causal upper-triangular (True = masked)."""
        mask = mx.triu(mx.ones((max_seq_len, max_seq_len)), k=1)
        return mask * -1e9

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape

        # Embedding
        h = self.tok_emb(x)

        # Masque causal pour cette longueur
        mask = self.causal_mask[:T, :T]

        # Blocs Transformer
        for layer in self.layers:
            h = layer(h, mask)

        h = self.norm(h)

        # LM Head (tied ou non)
        if self.tie_weights:
            # tok_emb.weight shape: (V, D) -> need (D, V) for matmul
            logits = h @ self.tok_emb.weight.T
        else:
            logits = self.lm_head(h)

        return logits

    def num_params(self) -> int:
        """Compte récursivement tous les tenseurs (dicts, listes, modules)."""
        def count(obj) -> int:
            if isinstance(obj, dict):
                return sum(count(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return sum(count(v) for v in obj)
            return int(obj.size) if hasattr(obj, "size") else 0
        return count(self.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Data Loader (memmap + random sampling)
# ──────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """Générateur de batches depuis memmap uint16."""

    def __init__(self, data_path: Path, batch_size: int, seq_len: int, seed: int = 42):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

        # Charge memmap
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        print(f"[dataloader] {self.total_tokens:,} tokens chargés depuis {data_path}", flush=True)

    def __iter__(self):
        return self

    def __next__(self):
        # Échantillonnage aléatoire de positions de départ
        max_start = self.total_tokens - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(f"Corpus trop petit ({self.total_tokens}) pour seq_len={self.seq_len}")

        starts = self.rng.integers(0, max_start, size=self.batch_size)

        # Construit batch
        x_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.uint16)
        y_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.uint16)

        for i, start in enumerate(starts):
            seq = self.data[start:start + self.seq_len + 1]
            x_batch[i] = seq[:-1]
            y_batch[i] = seq[1:]

        return mx.array(x_batch), mx.array(y_batch)


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler Warmup + Cosine Decay
# ──────────────────────────────────────────────────────────────────────────────

class CosineScheduler:
    """Learning rate scheduler: Warmup linéaire + Cosine Decay."""
    def __init__(self, lr: float, warmup_iters: int, max_iters: int, min_lr: float = 0.0):
        self.lr = lr
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.min_lr = min_lr

    def __call__(self, step: int) -> float:
        if step < self.warmup_iters:
            return self.lr * (step + 1) / self.warmup_iters
        if step >= self.max_iters:
            return self.min_lr
        # Cosine decay
        progress = (step - self.warmup_iters) / (self.max_iters - self.warmup_iters)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────────────────
# Loss & Training Step
# ──────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(logits: mx.array, targets: mx.array) -> mx.array:
    """Cross Entropy avec logits (B, T, V) et targets (B, T)."""
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def train_step(model: nn.Module, x: mx.array, y: mx.array):
    """Un step d'entraînement : forward + loss + gradients."""
    def loss_fn(model):
        logits = model(x)
        return cross_entropy_loss(logits, y)

    loss, grads = mx.value_and_grad(loss_fn)(model)
    return loss, grads


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def _flatten_params(prefix: str, obj, flat_params: dict):
    """Récursivement aplatit les paramètres."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_params(f"{prefix}.{k}" if prefix else k, v, flat_params)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten_params(f"{prefix}.{i}", v, flat_params)
    elif hasattr(obj, 'size') and hasattr(obj, 'shape'):
                flat_params[prefix] = obj


def save_model(model: nn.Module, path: Path):
    """Sauvegarde le modèle en safetensors."""
    flat_params = {}
    for module_name, module_params in model.parameters().items():
        _flatten_params(module_name, module_params, flat_params)
    mx.save_safetensors(str(path), flat_params)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train 25M LLM from scratch with MLX")
    parser.add_argument("--data", type=Path, default=Path("data/corpus.bin"),
                        help="Fichier memmap uint16")
    parser.add_argument("--tokenizer", type=Path, default=Path("models/tokenizer_fr_16k"),
                        help="Dossier tokenizer")
    parser.add_argument("--output", type=Path, default=Path("models/llm_25m_fr"),
                        help="Dossier de sortie")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    # Charge tokenizer pour vocab_size
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    print(f"[train] Vocab size: {vocab_size}", flush=True)

    # Modèle
    model = MicroLLM(vocab_size=vocab_size)
    print(f"[train] Modèle créé : {model.num_params():,} paramètres", flush=True)
    print(f"[train] Tied weights: True (embedding = lm_head)", flush=True)

    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Scheduler
    scheduler = CosineScheduler(args.lr, args.warmup_iters, args.max_iters, min_lr=args.lr * 0.1)

    # Data
    train_loader = DataLoader(args.data, args.batch_size, args.seq_len, seed=args.seed)

    # État pour sauvegarde
    step = 0
    tokens_processed = 0
    start_time = time.time()

    print(f"[train] Début entraînement : {args.max_iters} itérations, batch={args.batch_size}, seq={args.seq_len}", flush=True)

    # Boucle d'entraînement
    for step in range(args.max_iters):
        iter_start = time.time()

        # Update LR
        lr = scheduler(step)
        optimizer.learning_rate = lr

        # Batch
        x, y = next(train_loader)

        # Forward + Backward
        loss, grads = train_step(model, x, y)

        # Update
        optimizer.update(model, grads)

        # IMPORTANT : Force l'évaluation du graphe asynchrone (libère mémoire)
        mx.eval(model.parameters(), optimizer.state)

        # Stats
        tokens_processed += args.batch_size * args.seq_len
        iter_time = time.time() - iter_start
        tok_per_sec = args.batch_size * args.seq_len / iter_time

        # Log
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            print(f"[train] step {step:6d}/{args.max_iters} | loss: {loss.item():.4f} | "
                  f"lr: {lr:.2e} | {tok_per_sec:,.0f} tok/s | elapsed: {elapsed:.0f}s", flush=True)

        # Checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = args.output / f"checkpoint_{step + 1:07d}.safetensors"
            save_model(model, ckpt_path)
            print(f"[train] 💾 Checkpoint sauvé : {ckpt_path}", flush=True)

    # Sauvegarde finale
    final_path = args.output / "model.safetensors"
    save_model(model, final_path)
    print(f"[train] ✅ Entraînement terminé. Modèle final : {final_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())