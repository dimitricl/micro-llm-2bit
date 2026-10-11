"""Tests P6 : revalidate_tranche01 — gate apparié, ε par paire, OOD (TDD).

Règles : veto ⟺ Δc > MDE_c de la paire (val_big) ; OOD : veto ⟺
Δc_OOD > MDE_OOD de la paire. c primaire (a/b affichés, sans veto).
"""

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.revalidate_tranche01 import verdict_apparie


def _docs(shift_c: float = 0.0, n: int = 60, seed: int = 0,
          noise_seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    nz = np.random.default_rng(noise_seed)
    out = []
    for _ in range(n):
        chars = float(rng.integers(5000, 40000))
        base = float(rng.uniform(8000, 60000))
        noise = float(nz.normal(0, 1500))  # indépendant par modèle
        out.append({"nll_a": base, "nll_b": base + 100.0,
                    "nll_c": base + noise + shift_c * chars * 0.6931,
                    "chars": chars, "chars_c": chars * 0.99,
                    "n": 1000, "n_unk": 5})
    return out


def test_pass_identique():
    d = _docs()
    ok, res = verdict_apparie(d, [dict(x) for x in d])
    assert ok and res["vetos"] == []
    assert res["eps_pair"] >= 0
    _, res2 = verdict_apparie(_docs(noise_seed=1),
                                _docs(shift_c=0.001, noise_seed=2))
    assert res2["eps_pair"] > 0  # ε dérivé de la paire, pas une constante


def test_fail_regression_significative_c():
    ref, new = _docs(noise_seed=1), _docs(shift_c=0.06, noise_seed=2)
    ok, res = verdict_apparie(ref, new)
    assert not ok and "c_valbig" in res["vetos"]


def test_pass_bruit_sous_mde():
    # Δc petit devant le MDE de la paire → PASS (pas de veto).
    ref, new = _docs(noise_seed=1), _docs(shift_c=0.001, noise_seed=2)
    ok, res = verdict_apparie(ref, new)
    assert abs(res["delta_c"]) < res["eps_pair"]
    assert ok


def _shift_to(ref: list[dict], delta_cible: float,
              noise_seed: int = 21) -> list[dict]:
    import math
    nz = np.random.default_rng(noise_seed)
    out = []
    for x in ref:
        y = dict(x)
        # bruit indépendant (comme un autre modèle) + shift calibré
        y["nll_c"] = (x["nll_c"] + float(nz.normal(0, 1500))
                      + delta_cible * x["chars_c"] * math.log(2))
        out.append(y)
    return out


def test_controle_gradue_proche_seuil():
    # Dégradation graduée autour du seuil (MDE mesuré sur la paire) :
    # 0,5× MDE → PASS ; 1,5× MDE → FAIL. MDE quasi identique entre les
    # deux appels (mêmes docs, bruit indépendant de même amplitude).
    ref = _docs(noise_seed=1)
    import tools.revalidate_tranche01 as R
    mde = R.verdict_apparie(ref, _docs(shift_c=0.02, noise_seed=2))[1][
        "eps_pair"]
    ok_lo, r_lo = verdict_apparie(ref, _shift_to(ref, 0.5 * mde))
    ok_hi, r_hi = verdict_apparie(ref, _shift_to(ref, 1.5 * mde))
    assert ok_lo and r_lo["vetos"] == []
    assert not ok_hi and "c_valbig" in r_hi["vetos"]


def test_critere_ood():
    ref, new = _docs(noise_seed=1), _docs(shift_c=0.06, noise_seed=2)
    ood_ref = _docs(seed=9, noise_seed=11)
    ood_new = _docs(shift_c=0.06, seed=9, noise_seed=12)
    ok, res = verdict_apparie(ref, new, ood_ref, ood_new)
    assert not ok and "c_ood" in res["vetos"]
    ok2, res2 = verdict_apparie(ref, new, ood_ref, _docs(seed=9,
                                                        noise_seed=13))
    assert "c_ood" not in res2["vetos"]


def test_ood_bloque_meme_si_valbig_ok():
    d, ood = _docs(noise_seed=1), _docs(seed=9, noise_seed=11)
    ok, res = verdict_apparie(d, [dict(x) for x in d],
                              ood, _docs(shift_c=0.06, seed=9,
                                         noise_seed=12))
    assert not ok and res["vetos"] == ["c_ood"]


def test_cli_exit_codes(tmp_path):
    import json
    ref = os.path.join(tmp_path, "ref.json")
    new = os.path.join(tmp_path, "new.json")
    json.dump({"docs": _docs()}, open(ref, "w"))
    json.dump({"docs": _docs()}, open(new, "w"))
    r = subprocess.run(
        [sys.executable, "tools/revalidate_tranche01.py", "--ref", ref,
         "--new", new],
        capture_output=True, text=True, cwd=".")
    assert r.returncode == 0 and "PASS" in r.stdout


def test_cli_fail_code(tmp_path):
    import json
    ref = os.path.join(tmp_path, "ref.json")
    new = os.path.join(tmp_path, "new.json")
    json.dump({"docs": _docs(noise_seed=1)}, open(ref, "w"))
    json.dump({"docs": _docs(shift_c=0.06, noise_seed=2)}, open(new, "w"))
    r = subprocess.run(
        [sys.executable, "tools/revalidate_tranche01.py", "--ref", ref,
         "--new", new],
        capture_output=True, text=True, cwd=".")
    assert r.returncode == 1 and "FAIL" in r.stdout


def test_nan_inf_veto():
    bad = _docs(noise_seed=2)
    bad[3]["nll_c"] = float("nan")
    ok, res = verdict_apparie(_docs(noise_seed=1), bad)
    assert not ok and "non_fini" in res["vetos"]
    bad2 = _docs(noise_seed=2)
    bad2[5]["nll_a"] = float("inf")
    ok2, res2 = verdict_apparie(_docs(noise_seed=1), bad2)
    assert not ok2 and "non_fini" in res2["vetos"]


def test_ood_partiel_refuse():
    import pytest
    d = _docs(noise_seed=1)
    with pytest.raises(ValueError):
        verdict_apparie(d, [dict(x) for x in d], _docs(seed=9), None)
    with pytest.raises(ValueError):
        verdict_apparie(d, [dict(x) for x in d], None, _docs(seed=9))


def test_non_apparie_fail_propre():
    ref = _docs(noise_seed=1)
    bad = [dict(x, chars=x["chars"] + 1) for x in _docs(noise_seed=2)]
    ok, res = verdict_apparie(ref, bad)
    assert not ok and "non_apparie" in res["vetos"]
