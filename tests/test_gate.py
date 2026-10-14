"""Tests S5 : gate apparié (logique pure, données synthétiques)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.gate_paire import (paired_delta, check_paired, wilson_lower,
                              compare_loops, verdict)


def _docs(shift: float = 0.0, n: int = 60, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        chars = float(rng.integers(5000, 40000))
        nll = float(rng.uniform(8000, 60000))
        out.append({"nll_a": nll + shift * chars * 0.6931,
                    "nll_b": nll + shift * chars * 0.6931,
                    "nll_c": nll + shift * chars * 0.6931,
                    "chars": chars, "chars_c": chars * 0.99,
                    "n": 1000, "n_unk": 5})
    return out


def test_negatif_identique_pass():
    d = _docs()
    for v in ("a", "b", "c"):
        r = paired_delta(d, [dict(x) for x in d], v)
        assert abs(r["delta"]) < 1e-9
        assert r["lo"] <= 0 <= r["hi"]
        assert r["mde"] >= 0


def test_positif_shift_006_fail_sur_b_et_c():
    import json, tempfile
    ref, new = _docs(), _docs(shift=0.06)
    for v in ("b", "c"):
        r = paired_delta(ref, new, v)
        assert r["lo"] > 0.03, r  # régression nette (ε=0,03), MDE << shift
    with tempfile.TemporaryDirectory() as td:
        pr, pn = f"{td}/r.json", f"{td}/n.json"
        json.dump({"docs": ref}, open(pr, "w"))
        json.dump({"docs": new}, open(pn, "w"))
        ok, res = verdict(pr, pn)
        assert not ok and set(("b", "c")) <= set(res["fails"])


def test_negatif_verdict_pass():
    import json, tempfile
    d = _docs()
    with tempfile.TemporaryDirectory() as td:
        pr = f"{td}/r.json"
        json.dump({"docs": d}, open(pr, "w"))
        ok, res = verdict(pr, pr)
        assert ok and res["fails"] == []


def test_deterministe_seed():
    d = _docs()
    r1 = paired_delta(d, _docs(shift=0.005, seed=1), "b", seed=7)
    r2 = paired_delta(d, _docs(shift=0.005, seed=1), "b", seed=7)
    assert r1 == r2


def test_check_paired_refuse_desapparie():
    import pytest
    d = _docs()
    bad = [dict(x, chars=x["chars"] + 1) for x in d]
    with pytest.raises(AssertionError):
        check_paired(d, bad)
    with pytest.raises(AssertionError):
        check_paired(d, d[:-1])


def test_wilson_borne():
    assert wilson_lower(0, 30) < 0.12
    assert wilson_lower(25, 30) > 0.5
    assert wilson_lower(15, 30) <= 0.5  # 50 % pile : pas significatif


def test_compare_loops_proportion():
    import json, tempfile
    # ref boucle=2 partout, new pire (9) sur `k` prompts, 32 prompts v2.
    ref = {"version": 2, "metrics": {
        f"p{i}": {"greedy": {"plus_longue_boucle": 2}, "sampling": {}}
        for i in range(32)}}
    for k, fail in ((0, False), (16, False), (26, True)):
        new = {"version": 2, "metrics": {
            f"p{i}": {"greedy": {"plus_longue_boucle": 9 if i < k else 2},
                      "sampling": {}} for i in range(32)}}
        with tempfile.TemporaryDirectory() as td:
            pr, pn = f"{td}/r.json", f"{td}/n.json"
            json.dump(ref, open(pr, "w"))
            json.dump(new, open(pn, "w"))
            r = compare_loops(pr, pn)
            assert r["worse"] == k and r["n"] == 32
            assert r["fail"] == fail


def _mk_metrics(rep, bou, n=32):
    return {"version": 2,
            "metrics": {f"p{i}": {"greedy": {"plus_longue_boucle": 1,
                                             "taux_repetition_4g": 0.3,
                                             "mots_distincts": 10},
                                  "sampling": {"plus_longue_boucle": bou,
                                               "taux_repetition_4g": rep,
                                               "mots_distincts": 10}}
                          for i in range(n)}}


def test_repetition_gate_absolu(tmp_path):
    """P3 : seuil absolu calibré — sain passe, dégénéré FAIL."""
    import json
    from tools.gate_paire import check_repetition, S_REP_SEUIL
    sain = os.path.join(tmp_path, "sain.json")
    deg = os.path.join(tmp_path, "deg.json")
    json.dump(_mk_metrics(0.03, 1), open(sain, "w"))
    json.dump(_mk_metrics(0.33, 28), open(deg, "w"))
    r1 = check_repetition(sain)
    assert r1["fail"] is False
    r2 = check_repetition(deg)
    assert r2["fail"] is True and "repetition" in r2["motif"]
    assert S_REP_SEUIL == 0.15


def test_repetition_gate_borne(tmp_path):
    """Juste sous le seuil : PASS (pas de faux positif limite)."""
    import json
    from tools.gate_paire import check_repetition
    f = os.path.join(tmp_path, "lim.json")
    json.dump(_mk_metrics(0.09, 1), open(f, "w"))
    assert check_repetition(f)["fail"] is False
