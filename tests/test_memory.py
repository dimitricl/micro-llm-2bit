"""Tests budget mémoire : poids packés + KV cache + activations < 100 Mo."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import CONFIGS, TinyTransformer

BUDGET = 100 * 1024 * 1024


def test_tiny_sous_budget():
    rep = TinyTransformer(CONFIGS["tiny"]).memory_report()
    assert rep["sous_budget"], f"{rep['total_Mo']:.1f} Mo >= 100 Mo"
    print(
        f"\ntiny : {rep['total_Mo']:.2f} Mo "
        f"(poids {rep['poids_packed_o'] / 1e6:.2f} Mo, "
        f"KV {rep['kv_cache_o'] / 1e6:.2f} Mo)"
    )


def test_base_sous_budget():
    rep = TinyTransformer(CONFIGS["base"]).memory_report()
    assert rep["sous_budget"], f"{rep['total_Mo']:.1f} Mo >= 100 Mo"
    print(f"\nbase : {rep['total_Mo']:.2f} Mo")


def test_tailles_exactes_tiny():
    m = TinyTransformer(CONFIGS["tiny"])
    p = m.count_params()
    # Linéaires : 8 couches x (attn 786432 + mlp 1572864) = 18 874 368.
    assert p["linear_2bit"] == 8 * (786432 + 1572864), p
    assert p["embed"] == 8000 * 512
    rep = m.memory_report()
    assert rep["poids_packed_o"] == (
        p["linear_2bit"] // 4
        + (p["linear_2bit"] // 64) * 4
        + p["embed"]
        + 8000 * 4
        + p["norm"] * 4
    )
    assert rep["kv_cache_o"] == 2 * 8 * 2048 * 4 * 64 * 1
    assert rep["total_o"] < BUDGET
