"""Tests mémoire réelle + déterminisme greedy + équivalence C/NumPy."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import engine as E

BASE_BIN = os.path.join(os.path.dirname(__file__), "..", "exports", "enfant_base_fr.bin")
TINY_BIN = os.path.join(os.path.dirname(__file__), "..", "exports", "enfant.bin")

# Golden greedy (ids range(16), temperature=0.0), générés avec le moteur C.
GOLD_BASE_FR_8 = [16, 4, 48, 22, 277, 118, 675, 148]
GOLD_TINY_8 = [3, 66, 15, 3, 66, 15, 3, 8]

DELTA_BUDGET_MO = 100.0


def _rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


@pytest.mark.skipif(not os.path.exists(BASE_BIN), reason="export base_fr absent")
def test_delta_rss_sous_100mo():
    """Le DELTA RSS (après génération moins baseline post-imports) < 100 Mo."""
    baseline = _rss_mb()
    eng = E.Engine(BASE_BIN)
    eng.generate(list(range(16)), max_new=64, temperature=0.0)
    delta = _rss_mb() - baseline
    print(f"\ndelta RSS base_fr : +{delta:.1f} Mo / {DELTA_BUDGET_MO:.0f} Mo")
    assert delta < DELTA_BUDGET_MO, f"delta RSS {delta:.1f} Mo >= 100 Mo"


@pytest.mark.skipif(not os.path.exists(BASE_BIN), reason="export base_fr absent")
def test_greedy_deterministe_et_reference():
    """Deux Engines donnent les mêmes tokens + golden anti-régression."""
    ids = list(range(16))
    out1 = E.Engine(BASE_BIN).generate(ids, max_new=32, temperature=0.0)
    out2 = E.Engine(BASE_BIN).generate(ids, max_new=32, temperature=0.0)
    assert out1 == out2, "greedy non déterministe"
    assert out1[:8] == GOLD_BASE_FR_8, f"sortie C modifiée : {out1[:8]}"


@pytest.mark.skipif(not os.path.exists(TINY_BIN), reason="export tiny absent")
def test_c_vs_numpy_equivalence(monkeypatch):
    """Le fallback NumPy donne les mêmes tokens que le noyau C (tiny)."""
    ids = list(range(16))
    out_c = E.Engine(TINY_BIN).generate(ids, max_new=4, temperature=0.0)
    assert out_c == GOLD_TINY_8[:4]
    monkeypatch.setattr(E, "_LIB", None)
    out_np = E.Engine(TINY_BIN).generate(ids, max_new=4, temperature=0.0)
    assert out_np == out_c, f"C={out_c} vs NumPy={out_np}"
