"""Tests S3 : fenêtrage unique + parité .bin/.pt (logique pure, CPU)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.check_parity_bin_pt import make_windows, TOL_DBPC
from tools.eval_bpc import variantes, load_c2_texts


def test_make_windows_couverture_sans_recouvrement():
    ids = list(range(300))  # S=128 : fenêtres [0:129], [128:257], [256:300]
    w = make_windows(ids)
    assert len(w) == 3
    assert w[0] == ids[0:129] and w[1] == ids[128:257]
    assert w[2] == ids[256:300]
    # Positions cibles couvertes exactement une fois : 1..299.
    targets = [x for win in w for x in win[1:]]
    assert targets == ids[1:]


def test_make_windows_ignore_incompletes():
    assert make_windows([5]) == []
    assert make_windows([5, 6]) == [[5, 6]]


def test_make_windows_meme_decoupage_des_deux_cotes():
    doc = list(range(200))
    assert make_windows(doc) == make_windows(list(doc))


def test_delta_bpc_nul_sur_records_identiques():
    recs = [
        {"nll": 1.0, "unk": False, "nbytes": 3, "nchars": 3},
        {"nll": 5.0, "unk": True, "nbytes": 2, "nchars": 2},
    ]
    vb, vp = variantes(recs, 10), variantes([dict(r) for r in recs], 10)
    for v in ("a", "b", "c"):
        assert abs(vb[v] - vp[v]) < TOL_DBPC


def test_c2_manifeste_sha1_verifies():
    texts = load_c2_texts()
    assert len(texts) == 6 and sum(len(t) for t in texts) == 354234
