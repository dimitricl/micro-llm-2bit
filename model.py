"""Modèle enfant decoder-only avec linéaires ternaires 2 bits (QAT/STE).

Architecture : RMSNorm, attention multi-têtes GQA avec RoPE, MLP SwiGLU,
embeddings partagés avec la tête de sortie (weight tying).
Précisions : embeddings int8, normes fp16/fp32, linéaires 2 bits (poids
latents fp32 entraînables, quantifiés au forward via quant.py).

Deux configs :
- tiny (défaut) : d=512, L=8, Hq=8, Hkv=4, ffn=1024, vocab=8000  -> ~23M params
- base          : d=768, L=12, Hq=8, Hkv=4, ffn=1536, vocab=12000 -> ~73M params
Toutes deux < 100 Mo une fois packées (voir memory_report()).
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import absmean_scales, quantized_linear_forward, quantize_activation_int8


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelConfig:
    """Hyperparamètres du modèle enfant."""

    vocab_size: int = 8000
    d_model: int = 512
    n_layers: int = 8
    n_heads_q: int = 8
    n_heads_kv: int = 4
    ffn_dim: int = 1024
    seq_max: int = 2048
    group_size: int = 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    subln: bool = False  # V4 (ablation) : RMSNorm avant o_proj et down_proj


TINY = ModelConfig()
BASE = ModelConfig(
    vocab_size=12000,
    d_model=768,
    n_layers=12,
    n_heads_q=8,
    n_heads_kv=4,
    ffn_dim=1536,
    seq_max=2048,
)

CONFIGS = {"tiny": TINY, "base": BASE}


# ---------------------------------------------------------------------------
# Briques
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm sans biais, poids fp32 (négligeables en mémoire)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x.float() * torch.rsqrt(var + self.eps)
        return (x_norm * self.weight).to(x.dtype)


class TernaryLinear(nn.Module):
    """Linéaire (in, out) : poids latents fp32, forward quantifié ternaire.

    En entraînement : QAT avec STE (voir quant.quantized_linear_forward).
    En inférence exportée : les poids sont remplacés par leur version packée
    (voir export.py via main.py export) et le calcul passe par le noyau C.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 64):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError("in_features doit être multiple de group_size.")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        # Init type Xavier uniform resserrée (bon point de départ ternaire).
        bound = math.sqrt(6.0 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quantized_linear_forward(x, self.weight, self.group_size)

    def ternary_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne (q int8 (out, in), scales fp32 (out, num_groups))."""
        from quant import quantize_ternary_infer

        with torch.no_grad():
            n, k = self.weight.shape
            g = k // self.group_size
            scales = self.weight.view(n, g, self.group_size).abs().mean(dim=-1)
            scale_exp = scales.repeat_interleave(self.group_size, dim=1)
            q = (
                torch.round(self.weight / scale_exp.clamp(min=1e-8))
                .clamp(-1, 1)
                .to(torch.int8)
            )
        return q, scales


def _rope_freqs(head_dim: int, seq_len: int, theta: float, device) -> torch.Tensor:
    """Fréquences RoPE : (seq_len, head_dim // 2) en complexe (cos + i sin)."""
    inv = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    pos = torch.arange(seq_len, device=device).float()
    return torch.einsum("i,j->ij", pos, inv)  # angles (S, Dh/2)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Applique RoPE à x (B, S, H, Dh). freqs : (S, Dh/2) angles."""
    x1, x2 = x[..., ::2], x[..., 1::2]  # paires
    cos = freqs.cos()[None, :, None, :]  # (1, S, 1, Dh/2)
    sin = freqs.sin()[None, :, None, :]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    """Attention GQA avec RoPE et KV cache externe (inférence)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads_q != 0:
            raise ValueError("d_model doit être multiple de n_heads_q.")
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads_q
        self.q_proj = TernaryLinear(
            cfg.d_model, cfg.n_heads_q * self.head_dim, cfg.group_size
        )
        self.k_proj = TernaryLinear(
            cfg.d_model, cfg.n_heads_kv * self.head_dim, cfg.group_size
        )
        self.v_proj = TernaryLinear(
            cfg.d_model, cfg.n_heads_kv * self.head_dim, cfg.group_size
        )
        self.o_proj = TernaryLinear(
            cfg.n_heads_q * self.head_dim, cfg.d_model, cfg.group_size
        )
        # V4 (ablation, défaut désactivé) : norme avant o_proj.
        self.o_norm = None
        if getattr(cfg, "subln", False):
            self.o_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.cfg.n_heads_q, self.head_dim)
        k = self.k_proj(x).view(b, s, self.cfg.n_heads_kv, self.head_dim)
        v = self.v_proj(x).view(b, s, self.cfg.n_heads_kv, self.head_dim)
        q, k = _apply_rope(q, freqs), _apply_rope(k, freqs)
        # Répète les têtes KV pour matcher les têtes Q (GQA -> MHA).
        repeat = self.cfg.n_heads_q // self.cfg.n_heads_kv
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            is_causal=(mask is None),
        )
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        if self.o_norm is not None:
            attn = self.o_norm(attn)
        return self.o_proj(attn)


class SwiGLUMlp(nn.Module):
    """MLP SwiGLU : down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = TernaryLinear(cfg.d_model, cfg.ffn_dim, cfg.group_size)
        self.up = TernaryLinear(cfg.d_model, cfg.ffn_dim, cfg.group_size)
        self.down = TernaryLinear(cfg.ffn_dim, cfg.d_model, cfg.group_size)
        # V4 (ablation, défaut désactivé) : norme avant down_proj.
        self.down_norm = None
        if getattr(cfg, "subln", False):
            self.down_norm = RMSNorm(cfg.ffn_dim, cfg.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate(x)) * self.up(x)
        if self.down_norm is not None:
            h = self.down_norm(h)
        return self.down(h)


class DecoderBlock(nn.Module):
    """Un bloc : x + attn(RMSNorm(x)) puis x + mlp(RMSNorm(x)) (pre-norm)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLUMlp(cfg)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TinyTransformer(nn.Module):
    """Transformer decoder-only complet, tête liée aux embeddings."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Embeddings int8 quantifiés à la volée (absmax par ligne) : on stocke
        # des latents fp32 à l'entraînement, quantifiés au forward (STE).
        self.embed_latent = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
        nn.init.normal_(self.embed_latent, mean=0.0, std=1.0 / math.sqrt(cfg.d_model))
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.register_buffer(
            "embed_scale", torch.ones(cfg.vocab_size, 1), persistent=False
        )

    def forward_with_hidden(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ids : (B, S) -> (logits (B, S, V), dernier caché (B, S, D))."""
        cfg = self.cfg
        # Embeddings : quantification int8 par ligne + STE.
        with torch.no_grad():
            amax = self.embed_latent.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            self.embed_scale.copy_(amax / 127.0)
        eq = (self.embed_latent / self.embed_scale).round().clamp(-128, 127)
        emb_table = (
            self.embed_latent + ((eq * self.embed_scale) - self.embed_latent).detach()
        )
        x = emb_table[ids]  # (B, S, D)
        freqs = _rope_freqs(
            cfg.d_model // cfg.n_heads_q, ids.shape[1], cfg.rope_theta, ids.device
        )
        for layer in self.layers:
            x = layer(x, freqs)
        h = self.final_norm(x)
        # Tête liée : logits = x @ E^T (embeddings quantifiés, STE déjà appliqué).
        return h @ emb_table.t(), h

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids : (B, S) -> logits (B, S, V)."""
        logits, _ = self.forward_with_hidden(ids)
        return logits

    # ------------------------------------------------------------------
    # Comptabilité paramètres / mémoire
    # ------------------------------------------------------------------

    def count_params(self) -> dict:
        """Compte les paramètres par catégorie (latents = fp32 entraînement)."""
        n_embed = self.cfg.vocab_size * self.cfg.d_model
        n_linear = sum(
            m.weight.numel() for m in self.modules() if isinstance(m, TernaryLinear)
        )
        n_norm = sum(m.weight.numel() for m in self.modules() if isinstance(m, RMSNorm))
        return {
            "embed": n_embed,
            "linear_2bit": n_linear,
            "norm": n_norm,
            "total": n_embed + n_linear + n_norm,
        }

    def memory_report(self, seq_kv: int | None = None, kv_bytes: int = 1) -> dict:
        """Taille mémoire exacte : poids packés + KV cache + activations.

        seq_kv : longueur du KV cache pré-alloué (défaut : cfg.seq_max).
        kv_bytes : 1 (int8) ou 2 (fp16).
        Retour en octets + check du budget 100 Mo.
        """
        cfg = self.cfg
        p = self.count_params()
        s = seq_kv or cfg.seq_max
        # Poids packés : linéaires 0.25 o/param + scales fp32 par groupe
        # (fp32 : simplicité du noyau C et de l'export ; surcoût < 1 Mo).
        lin_packed = p["linear_2bit"] // 4
        n_scales = p["linear_2bit"] // cfg.group_size
        lin_scales = n_scales * 4  # fp32
        # Embeddings int8 : 1 o/param + 1 scale fp32 par ligne (token).
        emb = p["embed"] * 1 + cfg.vocab_size * 4
        # Normes fp32.
        norms = p["norm"] * 4
        weights = lin_packed + lin_scales + emb + norms
        # KV cache pré-alloué : 2 (K+V) * L * S * Hkv * Dh * bytes.
        kv_dim = cfg.n_heads_kv * (cfg.d_model // cfg.n_heads_q)
        kv_cache = 2 * cfg.n_layers * s * kv_dim * kv_bytes
        # Activations token-par-token (pire cas, fp32) : hidden + ffn + attn.
        d, f = cfg.d_model, cfg.ffn_dim
        hq = cfg.n_heads_q * (cfg.d_model // cfg.n_heads_q)
        activations = (d + 2 * f + 2 * hq + cfg.vocab_size) * 4
        total = weights + kv_cache + activations
        budget = 100 * 1024 * 1024
        return {
            "poids_packed_o": weights,
            "kv_cache_o": kv_cache,
            "activations_o": activations,
            "total_o": total,
            "total_Mo": total / 1024 / 1024,
            "budget_o": budget,
            "sous_budget": total < budget,
            "params": p,
        }
