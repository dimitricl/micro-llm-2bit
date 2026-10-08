"""Tests parsing de logs + anti-fuite du jeu mis de côté (CPU)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.eval_ppl import normalise, sans_fuite
from tools.plot_run import parse_log

EXEMPLE = """[train] device=mps parent=X
[train] enfant: 19.9M params latents, 100 batchs, 50 steps
[train] step 20/50 loss=7.6200 kl=7.7858 ce=7.2332 ppl~1384.6 212 tok/s
[val] step 20/50 val_ce=6.9689 val_ppl~1063.1 [best]
[train] step 40/50 loss=6.6244 kl=7.0115 ce=5.7210 ppl~305.2 167 tok/s
[val] step 40/50 val_ce=6.2849 val_ppl~536.4
[eval] train-PPL ~55.7 (50 batchs) vs val-PPL ~60.9 (212 batchs)
"""


def test_parse_log(tmp_path):
    p = tmp_path / "train.log"
    p.write_text(EXEMPLE, encoding="utf-8")
    d = parse_log(str(p))
    assert d["steps"] == [20, 40]
    assert d["ppl_tr"] == [1384.6, 305.2]
    assert d["toks"] == [212.0, 167.0]
    assert d["vsteps"] == [20, 40]
    assert d["ppl_va"] == [1063.1, 536.4]
    assert d["eval_fin"] == (55.7, 60.9)


def test_parse_log_vide(tmp_path):
    p = tmp_path / "vide.log"
    p.write_text("rien ici\n", encoding="utf-8")
    d = parse_log(str(p))
    assert d["steps"] == [] and d["vsteps"] == [] and d["eval_fin"] is None


def test_sans_fuite():
    train = ["Paris est la capitale de la France.", "Le chat dort."]
    val = ["L'eau bout à cent degrés."]
    cands = ["Paris   est la capitale de la France.",  # même normalisé
             "Un texte jamais vu du tout.",
             "l'eau bout à cent degrés."]  # casse différente
    gardes, exclus = sans_fuite(cands, train, val)
    assert exclus == 2  # doublon normalisé + même texte en minuscules
    assert gardes == ["Un texte jamais vu du tout."]


def test_normalise():
    assert normalise("a   b\nc") == "a b c"
    assert normalise("L'Eau") == normalise("l'eau")
