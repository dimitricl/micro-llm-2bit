"""Tests H5.2 : veilleur sur logs synthétiques + un vrai log."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.watch_run import surveiller

BASE = "[train] démarrage\n"
VAL = "[val] step {g}/400 val_ce=5.0 val_ppl~{p}\n"
END = "[train] terminé en 1000s -> x.pt\n"


def test_go_nominal():
    log = BASE + VAL.format(g=50, p=800.0) + VAL.format(g=100, p=300.0) \
        + VAL.format(g=200, p=200.0) + END
    ok, motifs = surveiller(log)
    assert ok, motifs


def test_stop_divergence():
    ok, motifs = surveiller(BASE + "[train] ARRÊT : loss non finie à step 3\n")
    assert not ok and any("non finie" in m for m in motifs)


def test_stop_2x():
    log = BASE + VAL.format(g=50, p=800.0) + VAL.format(g=100, p=300.0) \
        + VAL.format(g=200, p=1700.0) + END
    ok, motifs = surveiller(log)
    assert not ok and any("2×" in m for m in motifs)


def test_stop_25pc_jamais_sous_initiale():
    log = BASE + VAL.format(g=50, p=800.0) + VAL.format(g=100, p=900.0) \
        + VAL.format(g=300, p=100.0) + END
    ok, motifs = surveiller(log)
    assert not ok and any("25 %" in m for m in motifs)


def test_go_vrai_log_v7():
    p = "runs/2026-10-09-abl-V7-ce-s0/train.log"
    if not os.path.exists(p):
        import pytest
        pytest.skip("log V7 absent")
    ok, _ = surveiller(open(p, encoding="utf-8", errors="replace").read())
    assert ok


def test_cli_codes(tmp_path):
    log = os.path.join(tmp_path, "train.log")
    open(log, "w").write(BASE + VAL.format(g=50, p=800.0)
                         + VAL.format(g=100, p=300.0) + END)
    r = subprocess.run([sys.executable, "tools/watch_run.py", "--log", log],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0 and "GO" in r.stdout
