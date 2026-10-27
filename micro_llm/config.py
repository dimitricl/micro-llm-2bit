"""Configurations du Transformer ternaire et comptage exact des paramètres."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparamètres d'un Transformer décodeur de type Llama."""

    vocab_size: int = 16_384
    d_model: int = 1_024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_dim: int = 2_816
    seq_max: int = 2_048
    group_size: int = 64
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model doit être divisible par n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads doit être divisible par n_kv_heads")
        if self.d_model % self.group_size != 0:
            raise ValueError("d_model doit être divisible par group_size")
        if self.ffn_dim % self.group_size != 0:
            raise ValueError("ffn_dim doit être divisible par group_size")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def n_params(self) -> int:
        """Nombre exact de paramètres (embeddings liés, sans doublon de tête)."""
        emb = self.vocab_size * self.d_model
        per_layer = (
            2 * self.d_model * self.d_model                       # q, o
            + 2 * self.n_kv_heads * self.head_dim * self.d_model  # k, v
            + 3 * self.ffn_dim * self.d_model                     # gate, up, down
            + 2 * self.d_model                                    # 2 RMSNorm
        )
        return emb + self.n_layers * per_layer + self.d_model

    def weight_bytes(self, scale_bytes: int = 2) -> int:
        """Taille estimée du `.bin` : embeddings int8 + linéaires 2 bits."""
        emb = self.vocab_size * self.d_model + self.vocab_size * 4
        linear_params = self.n_params() - self.vocab_size * self.d_model - self.d_model
        linear = int(linear_params * (0.25 + scale_bytes / self.group_size))
        norms = (2 * self.n_layers + 1) * self.d_model * 4
        return emb + linear + norms + 44

    def kv_bytes(self, seq_len: int | None = None) -> int:
        """KV cache int8 + échelles fp32 par tête."""
        s = self.seq_max if seq_len is None else seq_len
        dims = self.n_kv_heads * self.head_dim
        return 2 * self.n_layers * s * dims + 2 * self.n_layers * s * self.n_kv_heads * 4

    def to_dict(self) -> dict:
        return asdict(self)


# E-t08 : référence historique, 25,0 M paramètres, export 12,13 Mo.
TINY = ModelConfig(
    vocab_size=12_000,
    d_model=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=4,
    ffn_dim=1_024,
    seq_max=128,
)

# Base historique : 72,9 M paramètres.
BASE = ModelConfig(
    vocab_size=12_000,
    d_model=768,
    n_layers=12,
    n_heads=8,
    n_kv_heads=4,
    ffn_dim=1_536,
    seq_max=256,
)

# Cible sûre : 130,3 M paramètres, ~64 Mo au total (contexte 1536).
TARGET_130M = ModelConfig(
    vocab_size=16_384,
    d_model=896,
    n_layers=14,
    n_heads=14,
    n_kv_heads=4,
    ffn_dim=2_304,
    seq_max=1_536,
)

# Cible haute : 197,2 M paramètres, ~90 Mo au total (contexte 2048).
TARGET_200M = ModelConfig(
    vocab_size=16_384,
    d_model=1_024,
    n_layers=16,
    n_heads=16,
    n_kv_heads=4,
    ffn_dim=2_816,
    seq_max=2_048,
)

PRESETS = {
    "tiny": TINY,
    "base": BASE,
    "target_130m": TARGET_130M,
    "target_200m": TARGET_200M,
}


def get_config(name: str) -> ModelConfig:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"config inconnue : {name} (choix : {sorted(PRESETS)})") from exc
