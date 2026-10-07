"""Moteur d'inférence CPU : charge le .bin packé, génère token par token.

- Linéaires via le noyau C (.dylib, ctypes) avec fallback NumPy si la
  compilation/la lib est absente.
- KV cache pré-alloué à taille fixe (int8 + scales fp32 par tête).
- Sampling : glouton, température, top-k, top-p.
"""

from __future__ import annotations

import ctypes
import os
import struct

import numpy as np

LIB_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "build", "libmicro2bit.dylib"),
    os.path.join(os.getcwd(), "build", "libmicro2bit.dylib"),
]

MAGIC = b"LLM2"
VERSION = 1
# magic, version, vocab, d, L, Hq, Hkv, ffn, seq_max, group, head_dim
HEADER_FMT = "<4s10I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def _load_lib():
    """Charge la .dylib, ou None si introuvable (fallback NumPy)."""
    for path in LIB_CANDIDATES:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                lib.ternary_matvec.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.ternary_matvec.restype = None
                lib.kernel_variant.restype = ctypes.c_char_p
                return lib, lib.kernel_variant().decode()
            except OSError:
                continue
    return None, "numpy-fallback"


_LIB, KERNEL_NAME = _load_lib()


def _unpack_row(packed: np.ndarray, cols: int) -> np.ndarray:
    """Décode une ligne packée (cols/4 octets) en signes int8 (cols,)."""
    b = packed.astype(np.int32)
    codes = np.stack([(b >> s) & 0x3 for s in (0, 2, 4, 6)], axis=1).reshape(-1)[:cols]
    out = np.zeros(cols, dtype=np.int8)
    out[codes == 0x01] = 1
    out[codes == 0x03] = -1
    return out


class PackedLinear:
    """Linéaire ternaire packée : y = act_scale * Σ_g s_g Σ sign*w * act."""

    def __init__(
        self, packed: np.ndarray, scales: np.ndarray, rows: int, cols: int, group: int
    ):
        self.packed = packed.reshape(rows, cols // 4)
        self.scales = scales.reshape(rows, cols // group).astype(np.float32)
        self.rows, self.cols, self.group = rows, cols, group

    def forward(self, x_q: np.ndarray, x_scale: float) -> np.ndarray:
        x_q = np.ascontiguousarray(x_q, dtype=np.int8)
        out = np.empty(self.rows, dtype=np.float32)
        if _LIB is not None:
            packed_c = np.ascontiguousarray(self.packed)
            scales_c = np.ascontiguousarray(self.scales, dtype=np.float32)
            _LIB.ternary_matvec(
                x_q.ctypes.data_as(ctypes.c_char_p),
                packed_c.ctypes.data_as(ctypes.c_char_p),
                scales_c.ctypes.data_as(ctypes.c_char_p),
                ctypes.c_float(float(x_scale)),
                out.ctypes.data_as(ctypes.c_char_p),
                self.rows,
                self.cols,
                self.group,
            )
        else:
            # Fallback NumPy : additions/soustractions pures.
            for n in range(self.rows):
                signs = _unpack_row(self.packed[n], self.cols).astype(np.float32)
                acc = 0.0
                for g in range(self.cols // self.group):
                    sl = slice(g * self.group, (g + 1) * self.group)
                    seg = np.where(
                        signs[sl] == 1,
                        x_q[sl].astype(np.float32),
                        np.where(signs[sl] == -1, -x_q[sl].astype(np.float32), 0.0),
                    )
                    acc += float(seg.sum()) * float(self.scales[n, g])
                out[n] = acc * x_scale
        return out


def _quant_act(x: np.ndarray) -> tuple[np.ndarray, float]:
    amax = float(np.abs(x).max()) if x.size else 0.0
    scale = amax / 127.0 if amax > 1e-12 else 1e-8
    return np.clip(np.round(x / scale), -128, 127).astype(np.int8), scale


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x / np.sqrt((x * x).mean() + eps) * w).astype(np.float32)


# ---------------------------------------------------------------------------
# Format .bin maison
# ---------------------------------------------------------------------------


def write_bin(path: str, meta: dict, tensors: list[np.ndarray]) -> None:
    """Écrit header + tenseurs bruts (C-order). meta : 10 entiers du header."""
    with open(path, "wb") as f:
        f.write(
            struct.pack(
                HEADER_FMT,
                MAGIC,
                VERSION,
                meta["vocab"],
                meta["d"],
                meta["L"],
                meta["Hq"],
                meta["Hkv"],
                meta["ffn"],
                meta["seq_max"],
                meta["group"],
                meta["head_dim"],
            )
        )
        for t in tensors:
            f.write(np.ascontiguousarray(t).tobytes())


def read_bin(path: str) -> tuple[dict, list[np.ndarray]]:
    """Lit un .bin. Retourne (meta, [tenseurs...]) sans les shapes (voir Engine)."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
        parts = struct.unpack(HEADER_FMT, raw)
        magic, ver = parts[0], parts[1]
        if magic != MAGIC:
            raise ValueError("Mauvais magic : pas un .bin LLM2.")
        if ver != VERSION:
            raise ValueError(f"Version {ver} non supportée (attendue {VERSION}).")
        keys = ["vocab", "d", "L", "Hq", "Hkv", "ffn", "seq_max", "group", "head_dim"]
        meta = dict(zip(keys, parts[2:]))
        blob = f.read()
    # Les shapes sont reconstruites par l'Engine (ordre fixe, voir save order).
    return meta, [blob]  # blob brut, découpé dans Engine._parse


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """Charge un modèle packé et génère du texte (ids)."""

    def __init__(self, path: str):
        meta, (blob,) = read_bin(path)
        self.meta = meta
        V, D, L = meta["vocab"], meta["d"], meta["L"]
        Hq, Hkv, Dh = meta["Hq"], meta["Hkv"], meta["head_dim"]
        F, G = meta["ffn"], meta["group"]
        off = 0

        def take(n: int, dt: np.dtype) -> np.ndarray:
            nonlocal off
            arr = np.frombuffer(blob, dtype=dt, count=n, offset=off).copy()
            off += n * np.dtype(dt).itemsize
            return arr

        self.embed_q = take(V * D, np.int8).reshape(V, D)
        self.embed_s = take(V, np.float32)
        self.linears: list[
            list[PackedLinear]
        ] = []  # par couche : [q,k,v,o,gate,up,down]
        # dims : q (Hq*Dh, D), k/v (Hkv*Dh, D), o (D, Hq*Dh),
        #        gate/up (F, D), down (D, F)
        dims = [
            (Hq * Dh, D),
            (Hkv * Dh, D),
            (Hkv * Dh, D),
            (D, Hq * Dh),
            (F, D),
            (F, D),
            (D, F),
        ]
        self.attn_norms, self.mlp_norms = [], []
        for _ in range(L):
            layer = []
            for rows, cols in dims:
                packed = take(rows * cols // 4, np.uint8)
                scales = take(rows * cols // G, np.float32)
                layer.append(PackedLinear(packed, scales, rows, cols, G))
            self.linears.append(layer)
            self.attn_norms.append(take(D, np.float32))
            self.mlp_norms.append(take(D, np.float32))
        self.final_norm = take(D, np.float32)
        assert off == len(blob), f"Blob mal découpé : {off} vs {len(blob)}"

        # KV cache pré-alloué : int8 [L, S, Hkv*Dh] + scales [L, S, Hkv].
        S = meta["seq_max"]
        self.k_cache = np.zeros((L, S, Hkv * Dh), dtype=np.int8)
        self.v_cache = np.zeros((L, S, Hkv * Dh), dtype=np.int8)
        self.k_scales = np.ones((L, S, Hkv), dtype=np.float32)
        self.v_scales = np.ones((L, S, Hkv), dtype=np.float32)
        self.pos = 0
        # Fréquences RoPE pré-calculées.
        inv = 1.0 / (10000.0 ** (np.arange(0, Dh, 2) / Dh))
        self.rope_angles = np.outer(np.arange(S), inv)  # (S, Dh/2)

    # -- utils --
    def _rope(self, vec: np.ndarray, pos: int) -> np.ndarray:
        ang = self.rope_angles[pos]
        c, s = np.cos(ang), np.sin(ang)
        x1, x2 = vec[..., ::2], vec[..., 1::2]
        out = np.empty_like(vec)
        out[..., ::2] = x1 * c - x2 * s
        out[..., 1::2] = x1 * s + x2 * c
        return out

    def reset(self) -> None:
        self.pos = 0

    # -- forward un pas --
    def step(self, tok: int) -> np.ndarray:
        """Avance d'un token, retourne les logits (V,) fp32."""
        m = self.meta
        D, Dh, Hq, Hkv = m["d"], m["head_dim"], m["Hq"], m["Hkv"]
        x = self.embed_q[tok].astype(np.float32) * float(self.embed_s[tok])
        p = self.pos
        for li in range(m["L"]):
            q_l, k_l, v_l, o_l, g_l, u_l, d_l = self.linears[li]
            h = rmsnorm(x, self.attn_norms[li])
            hq_, hs_ = _quant_act(h)
            q = self._rope(q_l.forward(hq_, hs_).reshape(Hq, Dh), p)
            k = self._rope(k_l.forward(hq_, hs_).reshape(Hkv, Dh), p)
            v = v_l.forward(hq_, hs_).reshape(Hkv, Dh)
            # Stocke K/V quantifiés (1 scale par tête).
            for hh in range(Hkv):
                for name, vec in (("k", k[hh]), ("v", v[hh])):
                    amax = float(np.abs(vec).max())
                    sc = amax / 127.0 if amax > 1e-12 else 1e-8
                    qq = np.clip(np.round(vec / sc), -128, 127).astype(np.int8)
                    if name == "k":
                        self.k_cache[li, p, hh * Dh : (hh + 1) * Dh] = qq
                        self.k_scales[li, p, hh] = sc
                    else:
                        self.v_cache[li, p, hh * Dh : (hh + 1) * Dh] = qq
                        self.v_scales[li, p, hh] = sc
            # Attention causale sur [0..p].
            rep = Hq // Hkv
            outs = []
            for hh in range(Hq):
                kv_h = hh // rep
                K = self.k_cache[li, : p + 1, kv_h * Dh : (kv_h + 1) * Dh].astype(
                    np.float32
                )
                K *= self.k_scales[li, : p + 1, kv_h][:, None]
                Vv = self.v_cache[li, : p + 1, kv_h * Dh : (kv_h + 1) * Dh].astype(
                    np.float32
                )
                Vv *= self.v_scales[li, : p + 1, kv_h][:, None]
                scores = (K @ q[hh]) / np.sqrt(Dh)
                scores -= scores.max()
                e = np.exp(scores)
                prob = e / e.sum()
                outs.append(prob @ Vv)
            a = np.concatenate(outs)
            aq, asc = _quant_act(a)
            x = x + o_l.forward(aq, asc)
            h2 = rmsnorm(x, self.mlp_norms[li])
            h2q, hs2 = _quant_act(h2)
            gate, up = g_l.forward(h2q, hs2), u_l.forward(h2q, hs2)
            down_in = (gate / (1.0 + np.exp(-gate))) * up  # silu(gate)*up
            dq, ds = _quant_act(down_in.astype(np.float32))
            x = x + d_l.forward(dq, ds)
        x = rmsnorm(x, self.final_norm)
        E = self.embed_q.astype(np.float32) * self.embed_s[:, None]
        self.pos += 1
        return x @ E.T

    # -- génération --
    def generate(
        self,
        ids: list[int],
        max_new: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop: set[int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """Préfill puis génération. temperature=0 -> glouton."""
        self.reset()
        logits = None
        for t in ids:
            logits = self.step(t)
        out = []
        rng = rng or np.random.default_rng(0)
        for _ in range(max_new):
            nxt = sample(logits, temperature, top_k, top_p, rng)
            out.append(nxt)
            if stop and nxt in stop:
                break
            logits = self.step(nxt)
        return out


def sample(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Échantillonne un token : glouton si temperature<=0."""
    z = logits.astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(z))
    z = z / temperature
    if top_k > 0:
        idx = np.argpartition(z, -top_k)[-top_k:]
        mask = np.full_like(z, -np.inf)
        mask[idx] = z[idx]
        z = mask
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[cum <= top_p]
        keep = np.concatenate([keep, order[len(keep) : len(keep) + 1]])
        mask = np.zeros_like(p)
        mask[keep] = p[keep]
        p = mask / mask.sum()
    rng = rng or np.random.default_rng()
    return int(rng.choice(len(p), p=p))
