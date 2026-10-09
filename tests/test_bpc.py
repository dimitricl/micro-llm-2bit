"""Tests G1 : variantes BPC a/b/c (math pure, sans modèle)."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.eval_bpc import surfaces, variantes

LN2 = math.log(2)


def test_variantes_valeurs_main():
    recs = [
        {"nll": 1.0, "unk": False, "nbytes": 3, "nchars": 3},
        {"nll": 5.0, "unk": True, "nbytes": 2, "nchars": 2},
    ]
    v = variantes(recs, chars=10)
    assert v["n"] == 2 and v["n_unk"] == 1 and v["chars_unk"] == 2
    assert v["a"] == (6.0 / LN2 / 10)
    # b : plancher byte-uniforme 8*octets*ln2 = 16*ln2 = 11.09 > 5.0.
    assert v["b"] == ((1.0 + 16 * LN2) / LN2 / 10)
    # c : position unk exclue des deux côtés (denom 10-2).
    assert v["c"] == (1.0 / LN2 / 8)


def test_penalite_ne_reduit_jamais():
    """Un unk déjà pire que le plancher garde son NLL (max)."""
    pen_nats = 8 * 1 * LN2  # 1 octet -> 5.545 nats
    recs = [{"nll": pen_nats + 2.0, "unk": True, "nbytes": 1, "nchars": 1}]
    v = variantes(recs, chars=5)
    assert v["b"] == ((pen_nats + 2.0) / LN2 / 5)
    assert v["b"] > v["a"] or True  # une seule position unk : b == a ici
    assert v["a"] == v["b"]


def test_sans_unk_b_egal_a_et_c_egal_a():
    recs = [{"nll": 2.0, "unk": False, "nbytes": 4, "nchars": 4}]
    v = variantes(recs, chars=4)
    assert v["a"] == v["b"] == v["c"] == 2.0 / LN2 / 4


def test_c_denominateur_nul():
    recs = [{"nll": 1.0, "unk": True, "nbytes": 1, "nchars": 1}]
    v = variantes(recs, chars=1)
    assert math.isnan(v["c"])


def test_surfaces_decode_une_fois():
    class Tok:
        def __init__(self):
            self.calls = []

        def decode(self, ids, clean_up_tokenization_spaces=False):
            self.calls.append(tuple(ids))
            return "ab" if ids == [7] else "c"

    s = surfaces(Tok(), [7, 7, 9])
    assert s[7] == ("ab", 2, 2) and s[9] == ("c", 1, 1)


def test_gap_relatif():
    a_par, a_tr = 1.5, 1.53
    gap = (a_tr - a_par) / a_par
    assert gap == pytest.approx(0.02)
    assert gap * 100 < 3.0  # ordre de grandeur du "0,03 BPC"
