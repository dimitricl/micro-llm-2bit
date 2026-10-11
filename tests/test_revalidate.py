"""Tests G4 : verdict PASS/FAIL de revalidation (logique pure)."""

import os
import subprocess
import sys
import tempfile

import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.revalidate_tranche01 import verdict

REF = {"bpc_a": 2.0, "bpc_a_ci": [1.9, 2.1], "loop": 1.5}
GOODLOG = "[train] terminé en 8000s -> x.pt"


def test_pass_nominal():
    new = {"bpc_a": 1.95, "bpc_a_ci": [1.85, 2.05], "loop": 1.2,
           "has_nan": False}
    ok, motifs = verdict(REF, new, GOODLOG)
    assert ok and len(motifs) == 3


def test_fail_regression_bpc_stricte():
    """IC nouveau entièrement au-dessus : FAIL."""
    new = {"bpc_a": 2.3, "bpc_a_ci": [2.11, 2.5], "loop": 1.0,
           "has_nan": False}
    ok, motifs = verdict(REF, new, GOODLOG)
    assert not ok and any("régression" in m for m in motifs)


def test_pass_chevauchement():
    """Chevauchement d'IC : pas de régression prouvée -> PASS."""
    new = {"bpc_a": 2.05, "bpc_a_ci": [1.95, 2.2], "loop": 1.5,
           "has_nan": False}
    ok, _ = verdict(REF, new, GOODLOG)
    assert ok


def test_fail_nan_et_log():
    new = {"bpc_a": 1.9, "bpc_a_ci": [1.8, 2.0], "loop": 1.0, "has_nan": True}
    ok, _ = verdict(REF, new, GOODLOG)
    assert not ok
    ok2, _ = verdict(REF, {**new, "has_nan": False}, "ARRÊT : loss non finie")
    assert not ok2


def test_fail_boucles_en_hausse():
    new = {"bpc_a": 1.9, "bpc_a_ci": [1.8, 2.0], "loop": 2.5,
           "has_nan": False}
    ok, motifs = verdict(REF, new, GOODLOG)
    assert not ok and any("hausse" in m for m in motifs)


def test_cli_exit_codes(tmp_path):
    """Le script rend PASS/0 et FAIL/1 via CLI."""
    ref = os.path.join(tmp_path, "ref.json")
    new = os.path.join(tmp_path, "new.json")
    log = os.path.join(tmp_path, "train.log")
    json.dump(REF, open(ref, "w"))
    json.dump({"bpc_a": 1.9, "bpc_a_ci": [1.8, 2.0], "loop": 1.0,
               "has_nan": False}, open(new, "w"))
    open(log, "w").write(GOODLOG)
    r = subprocess.run(
        [sys.executable, "tools/revalidate_tranche01.py", "--ref", ref,
         "--new", new, "--train-log", log],
        capture_output=True, text=True, cwd=".")
    assert r.returncode == 0 and "PASS" in r.stdout
