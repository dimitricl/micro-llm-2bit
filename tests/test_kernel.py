"""Tests d'équivalence noyau C vs PyTorch (tolérance float)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

import engine as E
from quant import pack_ternary, quantize_activation_int8, ternary_matmul_ref


def _random_case(k=128, n=64, group=64, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(n, k)
    x = torch.randn(3, k)
    return w, x, group


def _packed_linear(w, group):
    n, k = w.shape
    scales_n_g = w.view(n, k // group, group).abs().mean(dim=-1)
    exp = scales_n_g.repeat_interleave(group, dim=1).clamp(min=1e-8)
    q_nk = torch.round(w / exp).clamp(-1, 1).to(torch.int8)
    packed, m = pack_ternary(q_nk)
    assert m == q_nk.numel()
    q_t = q_nk.t().float()  # (K, N) pour la référence
    pl = E.PackedLinear(
        packed.numpy(), scales_n_g.numpy().astype(np.float32), n, k, group
    )
    return pl, q_t, scales_n_g


def test_c_vs_torch():
    if E._LIB is None:
        pytest.skip("lib C absente")
    w, x, group = _random_case()
    pl, q_t, scales_n_g = _packed_linear(w, group)
    x_q, x_s = quantize_activation_int8(x)
    ref = ternary_matmul_ref(x_q, x_s, q_t.float(), scales_n_g, group).numpy()
    got = np.stack(
        [pl.forward(x_q[i].numpy(), float(x_s[i])) for i in range(x.shape[0])]
    )
    denom = np.abs(ref).mean()
    assert np.allclose(got, ref, atol=1e-3 * max(denom, 1.0)), (
        f"écart max {(np.abs(got - ref)).max()} (moy |ref|={denom:.4f})"
    )


def test_fallback_numpy_coherent():
    w, x, group = _random_case(k=64, n=16, seed=1)
    pl, q_t, scales_n_g = _packed_linear(w, group)
    x_q, x_s = quantize_activation_int8(x)
    ref = ternary_matmul_ref(x_q, x_s, q_t.float(), scales_n_g, group).numpy()
    old, E._LIB = E._LIB, None
    try:
        got = np.stack(
            [pl.forward(x_q[i].numpy(), float(x_s[i])) for i in range(x.shape[0])]
        )
    finally:
        E._LIB = old
    assert np.allclose(got, ref, atol=1e-4)


def test_zéros():
    if E._LIB is None:
        pytest.skip("lib C absente")
    k, n, group = 64, 8, 64
    pl = E.PackedLinear(
        np.zeros(n * k // 4, dtype=np.uint8),
        np.ones((n, k // group), dtype=np.float32),
        n,
        k,
        group,
    )
    out = pl.forward(np.zeros(k, dtype=np.int8), 1.0)
    assert np.all(out == 0.0)
