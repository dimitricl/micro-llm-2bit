"""Quantification ternaire façon BitNet b1.58 et packing 2 bits.

Conventions (utilisées par le runtime C, ne pas changer sans bump de version) :

    0  -> 0b00
    +1 -> 0b01
    -1 -> 0b11   (0b10 réservé, décodé comme 0)

Le premier poids d'un octet occupe les bits de poids faible.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Ternaire (NumPy) — utilisé à l'export et dans les tests
# ---------------------------------------------------------------------------


def absmean_scales(w: np.ndarray, group_size: int = 64) -> np.ndarray:
    """Échelle absmean par groupe sur la dernière dimension.

    Args:
        w: tableau (..., cols), cols multiple de group_size.
        group_size: taille du groupe.

    Returns:
        Échelles (..., cols // group_size) en float32.
    """
    w = np.asarray(w, dtype=np.float32)
    if w.shape[-1] % group_size != 0:
        raise ValueError(
            f"group_size={group_size} doit diviser {w.shape[-1]}"
        )
    groups = w.reshape(*w.shape[:-1], -1, group_size)
    return np.abs(groups).mean(axis=-1, dtype=np.float32)


def quantize_ternary(
    w: np.ndarray, group_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Quantifie `w` en {-1, 0, +1} avec une échelle absmean par groupe."""
    w = np.asarray(w, dtype=np.float32)
    scales = absmean_scales(w, group_size)
    expanded = np.repeat(scales, group_size, axis=-1)[..., : w.shape[-1]]
    q = np.clip(np.round(w / np.maximum(expanded, 1e-8)), -1, 1).astype(np.int8)
    return q, scales


def dequantize_ternary(
    q: np.ndarray, scales: np.ndarray, group_size: int = 64
) -> np.ndarray:
    """Inverse de `quantize_ternary` (q int8 -> fp32)."""
    q = np.asarray(q)
    scales = np.asarray(scales, dtype=np.float32)
    expanded = np.repeat(scales, group_size, axis=-1)[..., : q.shape[-1]]
    return q.astype(np.float32) * expanded


def quantize_activation_int8(x: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """Quantifie une activation en int8, échelle absmax (une par ligne)."""
    x = np.asarray(x, dtype=np.float32)
    amax = float(np.abs(x).max()) if x.size else 0.0
    scale = np.float32(amax / 127.0) if amax > 1e-12 else np.float32(1e-8)
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale


# ---------------------------------------------------------------------------
# Packing 2 bits : 4 poids par octet
# ---------------------------------------------------------------------------


def pack_ternary(q: np.ndarray) -> tuple[np.ndarray, int]:
    """Packe un tenseur ternaire en uint8 (4 poids par octet).

    Returns:
        (packed uint8 1D, n) où n est le nombre de poids d'origine.
    """
    flat = np.asarray(q, dtype=np.int8).reshape(-1)
    n = int(flat.size)
    if not np.all((flat == -1) | (flat == 0) | (flat == 1)):
        raise ValueError("pack_ternary n'accepte que -1, 0, +1")
    codes = np.zeros(n, dtype=np.uint8)
    codes[flat == 1] = 0b01
    codes[flat == -1] = 0b11
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    c = codes.reshape(-1, 4).astype(np.uint32)
    packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).astype(
        np.uint8
    )
    return packed, n


def unpack_ternary(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse exact de `pack_ternary`."""
    b = np.asarray(packed, dtype=np.uint8).reshape(-1)
    codes = np.stack([(b >> s) & 0x3 for s in (0, 2, 4, 6)], axis=1).reshape(-1)[:n]
    out = np.zeros(n, dtype=np.int8)
    out[codes == 0b01] = 1
    out[codes == 0b11] = -1
    return out


# ---------------------------------------------------------------------------
# Ternaire (MLX) — utilisé à l'entraînement (QAT + STE)
# ---------------------------------------------------------------------------


def ternary_dequant_mlx(w, group_size: int = 64):
    """Quantifie `w` (out, in) et renvoie les poids déquantifiés (fp32).

    Forward : valeurs dans {-s, 0, +s}. Le gradient est coupé ici ; l'appelant
    ajoute `w + stop_gradient(wq - w)` pour le STE.
    """
    import mlx.core as mx

    out_dim, in_dim = w.shape
    if in_dim % group_size != 0:
        raise ValueError("in_dim doit être multiple de group_size")
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    scales = mx.abs(groups).mean(axis=-1, keepdims=True)
    q = mx.clip(mx.round(groups / mx.maximum(scales, 1e-8)), -1.0, 1.0)
    return (q * scales).reshape(out_dim, in_dim)


def ternary_ste_mlx(w, group_size: int = 64):
    """Poids effectifs pour le forward QAT : `w + stop_gradient(wq - w)`."""
    import mlx.core as mx

    wq = ternary_dequant_mlx(w, group_size)
    return w + mx.stop_gradient(wq - w)


def quantize_ternary_mlx(w, group_size: int = 64):
    """Quantifie un poids MLX en (q int8, scales) pour l'export."""
    import mlx.core as mx

    out_dim, in_dim = w.shape
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    scales = mx.abs(groups).mean(axis=-1)
    q = mx.clip(mx.round(groups / mx.maximum(scales[..., None], 1e-8)), -1, 1)
    return q.astype(mx.int8).reshape(out_dim, in_dim), scales.astype(mx.float32)


def pack_ternary_mlx(q, group_size: int = 64) -> np.ndarray:
    """Packe un tenseur ternaire MLX en uint8 via NumPy."""
    return pack_ternary(np.asarray(q))[0]


# ---------------------------------------------------------------------------
# Référence de matmul (tests de parité)
# ---------------------------------------------------------------------------


def ternary_matvec_ref(
    x_q: np.ndarray,
    x_scale: float,
    q: np.ndarray,
    scales: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    """y[n] = x_scale * Σ_g scales[n,g] * Σ_i sign(w[n,i]) * x_q[i].

    `q` est (rows, cols) dans {-1,0,+1}, `scales` (rows, cols/group).
    """
    q = np.asarray(q, dtype=np.float32)
    x = np.asarray(x_q, dtype=np.float32)
    rows, cols = q.shape
    out = np.empty(rows, dtype=np.float32)
    for n in range(rows):
        acc = 0.0
        for g in range(cols // group_size):
            sl = slice(g * group_size, (g + 1) * group_size)
            seg = x[sl]
            signs = q[n, sl]
            acc += float(np.where(signs == 1, seg, np.where(signs == -1, -seg, 0.0)).sum()) * float(
                scales[n, g]
            )
        out[n] = np.float32(acc * x_scale)
    return out
