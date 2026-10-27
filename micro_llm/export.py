"""Export d'un checkpoint MLX vers le format `.bin` LLM2 (v1 fp32 / v2 fp16)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .quant import pack_ternary, quantize_ternary
from .reference import HEADER_SIZE, write_header


def _linear_names(i: int) -> list[str]:
    return [
        f"layers.{i}.attn.wq.weight",
        f"layers.{i}.attn.wk.weight",
        f"layers.{i}.attn.wv.weight",
        f"layers.{i}.attn.wo.weight",
        f"layers.{i}.mlp.gate.weight",
        f"layers.{i}.mlp.up.weight",
        f"layers.{i}.mlp.down.weight",
    ]


def _quantize_embedding(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    emb = np.asarray(emb, dtype=np.float32)
    amax = np.abs(emb).max(axis=-1, keepdims=True)
    scales = (amax / 127.0).astype(np.float32)
    scales = np.maximum(scales, 1e-8)
    q = np.clip(np.round(emb / scales), -128, 127).astype(np.int8)
    return q, scales[:, 0]


def _write_tensors(fh, cfg: ModelConfig, tensors: dict, version: int) -> None:
    scale_dtype = np.float32 if version == 1 else np.float16
    fh.write(np.ascontiguousarray(tensors["embed_q"], dtype=np.int8).tobytes())
    fh.write(np.ascontiguousarray(tensors["embed_s"], dtype=np.float32).tobytes())
    for i in range(cfg.n_layers):
        for name in ("wq", "wk", "wv", "wo", "gate", "up", "down"):
            packed, scales = tensors[f"layers.{i}.{name}"]
            fh.write(np.ascontiguousarray(packed, dtype=np.uint8).tobytes())
            fh.write(np.ascontiguousarray(scales, dtype=scale_dtype).tobytes())
        fh.write(np.ascontiguousarray(tensors[f"layers.{i}.attn_norm"], dtype=np.float32).tobytes())
        fh.write(np.ascontiguousarray(tensors[f"layers.{i}.mlp_norm"], dtype=np.float32).tobytes())
    fh.write(np.ascontiguousarray(tensors["final_norm"], dtype=np.float32).tobytes())


def write_bin(
    path: str | Path,
    cfg: ModelConfig,
    tensors: dict,
    version: int = 2,
) -> int:
    """Écrit un `.bin` LLM2 et renvoie sa taille en octets."""
    meta = {
        "version": version,
        "vocab": cfg.vocab_size,
        "d": cfg.d_model,
        "L": cfg.n_layers,
        "Hq": cfg.n_heads,
        "Hkv": cfg.n_kv_heads,
        "ffn": cfg.ffn_dim,
        "seq_max": cfg.seq_max,
        "group": cfg.group_size,
        "head_dim": cfg.head_dim,
    }
    path = Path(path)
    with path.open("wb") as fh:
        write_header(fh, meta)
        _write_tensors(fh, cfg, tensors, version)
    return path.stat().st_size


def export_checkpoint(
    checkpoint: str | Path,
    out: str | Path,
    cfg: ModelConfig,
    version: int = 2,
) -> int:
    """Convertit un `.safetensors` MLX (poids latents fp32) en `.bin` ternaire."""
    import mlx.core as mx

    weights = mx.load(str(checkpoint))
    weights = {k: np.asarray(v) for k, v in weights.items()}
    tensors: dict = {}
    tokens_key = "tok_emb.weight" if "tok_emb.weight" in weights else "model.tok_emb.weight"
    emb_q, emb_s = _quantize_embedding(weights[tokens_key])
    tensors["embed_q"] = emb_q
    tensors["embed_s"] = emb_s
    for i in range(cfg.n_layers):
        for name, key in zip(
            ("wq", "wk", "wv", "wo", "gate", "up", "down"), _linear_names(i)
        ):
            if key not in weights:
                raise KeyError(f"poids manquant : {key}")
            q, scales = quantize_ternary(weights[key], cfg.group_size)
            packed, _ = pack_ternary(q)
            tensors[f"layers.{i}.{name}"] = (packed, scales)
        tensors[f"layers.{i}.attn_norm"] = weights[f"layers.{i}.attn_norm.weight"]
        tensors[f"layers.{i}.mlp_norm"] = weights[f"layers.{i}.mlp_norm.weight"]
    tensors["final_norm"] = weights["final_norm.weight"]
    size = write_bin(out, cfg, tensors, version=version)
    meta_path = Path(str(out) + ".json")
    meta_path.write_text(
        json.dumps({"config": cfg.to_dict(), "version": version}, indent=2),
        encoding="utf-8",
    )
    return size


def random_model(cfg: ModelConfig, seed: int = 0) -> dict:
    """Génère un modèle aléatoire valide (tests, benchmarks de RSS)."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 0.02, size=(cfg.vocab_size, cfg.d_model)).astype(np.float32)
    emb_q, emb_s = _quantize_embedding(emb)
    tensors: dict = {"embed_q": emb_q, "embed_s": emb_s}
    dims = {
        "wq": (cfg.d_model, cfg.d_model),
        "wk": (cfg.n_kv_heads * cfg.head_dim, cfg.d_model),
        "wv": (cfg.n_kv_heads * cfg.head_dim, cfg.d_model),
        "wo": (cfg.d_model, cfg.d_model),
        "gate": (cfg.ffn_dim, cfg.d_model),
        "up": (cfg.ffn_dim, cfg.d_model),
        "down": (cfg.d_model, cfg.ffn_dim),
    }
    for i in range(cfg.n_layers):
        for name, (rows, cols) in dims.items():
            q = rng.integers(-1, 2, size=(rows, cols), dtype=np.int8)
            scales = rng.uniform(0.01, 0.05, size=(rows, cols // cfg.group_size)).astype(
                np.float32
            )
            packed, _ = pack_ternary(q)
            tensors[f"layers.{i}.{name}"] = (packed, scales)
        tensors[f"layers.{i}.attn_norm"] = rng.uniform(
            0.5, 1.5, size=(cfg.d_model,)
        ).astype(np.float32)
        tensors[f"layers.{i}.mlp_norm"] = rng.uniform(
            0.5, 1.5, size=(cfg.d_model,)
        ).astype(np.float32)
    tensors["final_norm"] = rng.uniform(0.5, 1.5, size=(cfg.d_model,)).astype(np.float32)
    return tensors


def write_random_bin(
    path: str | Path, cfg: ModelConfig, seed: int = 0, version: int = 2
) -> int:
    return write_bin(path, cfg, random_model(cfg, seed), version=version)


__all__ = [
    "HEADER_SIZE",
    "export_checkpoint",
    "random_model",
    "write_bin",
    "write_random_bin",
]
