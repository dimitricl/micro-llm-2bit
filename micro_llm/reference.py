"""Forward NumPy de référence pour le format `.bin` LLM2.

Sert d'oracle de parité pour le runtime C. Volontairement lent et lisible :
il reproduit pas à pas le moteur Python historique (KV cache int8, quantif
d'activations absmax, RoPE interleavé).
"""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"LLM2"
HEADER_FMT = "<4s10I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
FIELDS = ["version", "vocab", "d", "L", "Hq", "Hkv", "ffn", "seq_max", "group", "head_dim"]


def read_header(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read(HEADER_SIZE)
    parts = struct.unpack(HEADER_FMT, raw)
    if parts[0] != MAGIC:
        raise ValueError("magic LLM2 absent")
    meta = dict(zip(FIELDS, parts[1:]))
    if meta["version"] not in (1, 2):
        raise ValueError(f"version {meta['version']} non supportée")
    return meta


def write_header(fh, meta: dict) -> None:
    fh.write(
        struct.pack(
            HEADER_FMT,
            MAGIC,
            meta["version"],
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


def load_model(path: str) -> dict:
    meta = read_header(path)
    V, D, L = meta["vocab"], meta["d"], meta["L"]
    Hq, Hkv, Dh = meta["Hq"], meta["Hkv"], meta["head_dim"]
    F, G = meta["ffn"], meta["group"]
    scale_dtype = np.float32 if meta["version"] == 1 else np.float16
    blob = np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER_SIZE)
    off = 0

    def take(n: int, dtype) -> np.ndarray:
        nonlocal off
        arr = np.frombuffer(blob, dtype=dtype, count=n, offset=off)
        off += n * np.dtype(dtype).itemsize
        return arr

    model = {"meta": meta, "_blob": blob}
    model["embed_q"] = take(V * D, np.int8).reshape(V, D)
    model["embed_s"] = take(V, np.float32).astype(np.float32)
    dims = [
        (Hq * Dh, D),
        (Hkv * Dh, D),
        (Hkv * Dh, D),
        (D, Hq * Dh),
        (F, D),
        (F, D),
        (D, F),
    ]
    model["linears"] = []
    for _ in range(L):
        layer = []
        for rows, cols in dims:
            packed = take(rows * cols // 4, np.uint8).reshape(rows, cols // 4)
            scales = take(rows * cols // G, scale_dtype).reshape(rows, cols // G)
            layer.append((packed, scales.astype(np.float32), rows, cols, G))
        model["linears"].append(layer)
        model["attn_norms"] = model.get("attn_norms", []) + [
            take(D, np.float32).astype(np.float32)
        ]
        model["mlp_norms"] = model.get("mlp_norms", []) + [
            take(D, np.float32).astype(np.float32)
        ]
    model["final_norm"] = take(D, np.float32).astype(np.float32)
    if off != len(blob):
        raise ValueError(f"blob mal découpé : {off} != {len(blob)}")
    return model


def _unpack_row(packed: np.ndarray, cols: int) -> np.ndarray:
    b = packed.astype(np.int32)
    codes = np.stack([(b >> s) & 0x3 for s in (0, 2, 4, 6)], axis=1).reshape(-1)[:cols]
    out = np.zeros(cols, dtype=np.float32)
    out[codes == 0x01] = 1.0
    out[codes == 0x03] = -1.0
    return out


def _quant_act(x: np.ndarray) -> tuple[np.ndarray, float]:
    amax = float(np.abs(x).max()) if x.size else 0.0
    scale = amax / 127.0 if amax > 1e-12 else 1e-8
    return np.clip(np.round(x / scale), -128, 127).astype(np.int8), float(scale)


def _rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x / np.sqrt((x * x).mean() + eps) * w).astype(np.float32)


def _matvec(x_q, x_scale, linear) -> np.ndarray:
    packed, scales, rows, cols, group = linear
    signs = np.stack([_unpack_row(packed[n], cols) for n in range(rows)])
    x = x_q.astype(np.float32)
    out = np.zeros(rows, dtype=np.float32)
    for g in range(cols // group):
        sl = slice(g * group, (g + 1) * group)
        contrib = (signs[:, sl] * x[sl][None, :]).sum(axis=1)
        out += contrib * scales[:, g]
    return (out * x_scale).astype(np.float32)


def _rope(vec: np.ndarray, pos: int, dh: int, theta: float = 10_000.0) -> np.ndarray:
    inv = 1.0 / (theta ** (np.arange(0, dh, 2) / dh))
    ang = pos * inv
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(vec)
    out[..., ::2] = vec[..., ::2] * c - vec[..., 1::2] * s
    out[..., 1::2] = vec[..., ::2] * s + vec[..., 1::2] * c
    return out


class ReferenceEngine:
    def __init__(self, path: str):
        self.m = load_model(path)
        meta = self.m["meta"]
        self.pos = 0
        L, S, Hkv, Dh = meta["L"], meta["seq_max"], meta["Hkv"], meta["head_dim"]
        self.k_cache = np.zeros((L, S, Hkv * Dh), dtype=np.int8)
        self.v_cache = np.zeros((L, S, Hkv * Dh), dtype=np.int8)
        self.k_scales = np.ones((L, S, Hkv), dtype=np.float32)
        self.v_scales = np.ones((L, S, Hkv), dtype=np.float32)

    def reset(self) -> None:
        self.pos = 0

    def step(self, tok: int) -> np.ndarray:
        m = self.m
        meta = m["meta"]
        D, Dh, Hq, Hkv = meta["d"], meta["head_dim"], meta["Hq"], meta["Hkv"]
        V = meta["vocab"]
        if self.pos >= meta["seq_max"]:
            raise IndexError("contexte plein")
        x = m["embed_q"][tok].astype(np.float32) * float(m["embed_s"][tok])
        p = self.pos
        for li in range(meta["L"]):
            q_l, k_l, v_l, o_l, g_l, u_l, d_l = m["linears"][li]
            h = _rmsnorm(x, m["attn_norms"][li])
            hq, hs = _quant_act(h)
            q = _rope(_matvec(hq, hs, q_l).reshape(Hq, Dh), p, Dh)
            k = _rope(_matvec(hq, hs, k_l).reshape(Hkv, Dh), p, Dh)
            v = _matvec(hq, hs, v_l).reshape(Hkv, Dh)
            for hh in range(Hkv):
                for vec, cache, scs, name in (
                    (k[hh], self.k_cache, self.k_scales, "k"),
                    (v[hh], self.v_cache, self.v_scales, "v"),
                ):
                    amax = float(np.abs(vec).max())
                    sc = amax / 127.0 if amax > 1e-12 else 1e-8
                    cache[li, p, hh * Dh : (hh + 1) * Dh] = np.clip(
                        np.round(vec / sc), -128, 127
                    ).astype(np.int8)
                    scs[li, p, hh] = sc
            rep = Hq // Hkv
            outs = []
            for hh in range(Hq):
                kv_h = hh // rep
                K = self.k_cache[li, : p + 1, kv_h * Dh : (kv_h + 1) * Dh].astype(
                    np.float32
                ) * self.k_scales[li, : p + 1, kv_h][:, None]
                Vv = self.v_cache[li, : p + 1, kv_h * Dh : (kv_h + 1) * Dh].astype(
                    np.float32
                ) * self.v_scales[li, : p + 1, kv_h][:, None]
                scores = (K @ q[hh]) / np.sqrt(Dh)
                scores -= scores.max()
                e = np.exp(scores)
                prob = e / e.sum()
                outs.append(prob @ Vv)
            a = np.concatenate(outs)
            aq, asc = _quant_act(a)
            x = x + _matvec(aq, asc, o_l)
            h2 = _rmsnorm(x, m["mlp_norms"][li])
            h2q, hs2 = _quant_act(h2)
            gate = _matvec(h2q, hs2, g_l)
            up = _matvec(h2q, hs2, u_l)
            down_in = (gate / (1.0 + np.exp(-gate))) * up
            dq, ds = _quant_act(down_in.astype(np.float32))
            x = x + _matvec(dq, ds, d_l)
        x = _rmsnorm(x, m["final_norm"])
        self.pos += 1
        eq = m["embed_q"].astype(np.float32) * m["embed_s"][:, None]
        return (eq @ x).astype(np.float32)

    def generate(self, ids: list[int], max_new: int, greedy: bool = True) -> list[int]:
        self.reset()
        logits = None
        for t in ids[-self.m["meta"]["seq_max"] :]:
            logits = self.step(t)
        out: list[int] = []
        for _ in range(max_new):
            if self.pos >= self.m["meta"]["seq_max"]:
                break
            nxt = int(np.argmax(logits))
            out.append(nxt)
            logits = self.step(nxt)
        return out
