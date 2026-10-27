"""Transformer décodeur ternaire en MLX (QAT + STE).

Architecture : RMSNorm pré-norm, GQA, RoPE, SwiGLU, embeddings liés.
Les seuls poids non ternaires sont les embeddings (int8 à l'export), les normes
et les échelles absmean (apprises implicitement via les poids latents).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .quant import ternary_ste_mlx


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps) * self.weight


class TernaryLinear(nn.Module):
    """Linéaire dont le forward utilise des poids ternaires (STE)."""

    def __init__(self, in_dim: int, out_dim: int, group_size: int):
        super().__init__()
        self.weight = mx.random.normal((out_dim, in_dim)) * (1.0 / math.sqrt(in_dim))
        self.group_size = group_size

    def __call__(self, x):
        w = ternary_ste_mlx(self.weight, self.group_size)
        return x @ w.T


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.wq = TernaryLinear(d, cfg.n_heads * cfg.head_dim, cfg.group_size)
        self.wk = TernaryLinear(d, cfg.n_kv_heads * cfg.head_dim, cfg.group_size)
        self.wv = TernaryLinear(d, cfg.n_kv_heads * cfg.head_dim, cfg.group_size)
        self.wo = TernaryLinear(cfg.n_heads * cfg.head_dim, d, cfg.group_size)

    @staticmethod
    def _rope(x, cos, sin):
        # x : (B, H, S, Dh), cos/sin : (S, Dh/2)
        b, h, s, dh = x.shape
        pairs = x.reshape(b, h, s, dh // 2, 2)
        x1, x2 = pairs[..., 0], pairs[..., 1]
        c, sn = cos[None, None, :, :], sin[None, None, :, :]
        o1 = x1 * c - x2 * sn
        o2 = x1 * sn + x2 * c
        return mx.stack([o1, o2], axis=-1).reshape(b, h, s, dh)

    def __call__(self, x, mask, cos, sin):
        cfg = self.cfg
        b, s, _ = x.shape
        hq, hkv, dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        q = self.wq(x).reshape(b, s, hq, dh).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(b, s, hkv, dh).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(b, s, hkv, dh).transpose(0, 2, 1, 3)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        rep = hq // hkv
        k = mx.broadcast_to(k[:, :, None], (b, hkv, rep, s, dh)).reshape(b, hq, s, dh)
        v = mx.broadcast_to(v[:, :, None], (b, hkv, rep, s, dh)).reshape(b, hq, s, dh)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dh)
        scores = scores + mask
        probs = mx.softmax(scores, axis=-1)
        out = probs @ v
        return self.wo(out.transpose(0, 2, 1, 3).reshape(b, s, hq * dh))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = TernaryLinear(cfg.d_model, cfg.ffn_dim, cfg.group_size)
        self.up = TernaryLinear(cfg.d_model, cfg.ffn_dim, cfg.group_size)
        self.down = TernaryLinear(cfg.ffn_dim, cfg.d_model, cfg.group_size)

    def __call__(self, x):
        g = self.gate(x)
        return self.down(g * mx.sigmoid(g) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def __call__(self, x, mask, cos, sin):
        x = x + self.attn(self.attn_norm(x), mask, cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def _rope_cache(self, seq_len: int):
        cfg = self.cfg
        dh = cfg.head_dim
        pos = mx.arange(seq_len).astype(mx.float32)
        inv = mx.exp(
            -math.log(cfg.rope_theta) * mx.arange(0, dh, 2).astype(mx.float32) / dh
        )
        angles = pos[:, None] * inv[None, :]
        return mx.cos(angles), mx.sin(angles)

    def __call__(self, ids):
        cfg = self.cfg
        b, s = ids.shape
        if s > cfg.seq_max:
            raise ValueError(f"séquence {s} > seq_max {cfg.seq_max}")
        x = self.tok_emb(ids)
        cos, sin = self._rope_cache(s)
        mask = mx.triu(mx.full((s, s), -1e9, dtype=mx.float32), k=1)
        for layer in self.layers:
            x = layer(x, mask, cos, sin)
        x = self.final_norm(x)
        # Tête liée : logits = x @ E^T.
        return x @ self.tok_emb.weight.T

    def loss(self, ids, targets, loss_mask=None):
        logits = self(ids)
        flat_logits = logits.reshape(-1, self.cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        per_token = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="none")
        if loss_mask is None:
            return per_token.mean()
        w = loss_mask.reshape(-1).astype(mx.float32)
        return (per_token * w).sum() / mx.maximum(w.sum(), 1.0)


def count_params(model) -> int:
    from mlx.utils import tree_flatten

    return int(sum(p.size for _, p in tree_flatten(model.parameters())))
