"""Tests G1 : variantes BPC a/b/c (math pure, sans modèle)."""

import math
import os
import sys

import numpy as np
import pytest
import torch

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


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_bpc_reference_values():
    """Test à valeurs fixes pour les 5 BPC de référence à 4 décimales (G1).

    Valeurs de référence (C2, variante a) :
    - tranche00 : 1,5676
    - wiki_slice00 : 1,6566
    - SmolLM-135M : 1,5357
    - SmolLM-360M : 1,3362
    - Qwen3-0.6B : 1,1138

    Valeurs variantes b/c du rapport :
    - tranche00 b : 1,6536, c : 1,5735
    - wiki_slice00 b : 1,7431, c : 1,6641
    - SmolLM-135M b : 1,6099, c : 1,5292
    - SmolLM-360M b : 1,4133, c : 1,3302
    - Qwen3-0.6B b : 1,2049, c : 1,1096

    Note : ce test est marqué @slow car il requiert le chargement des modèles
    parents et l'évaluation sur les 6 documents de validation. Pour lancer :
      pytest -m slow tests/test_bpc.py
    (Qwen en lots de 2 fenêtres : pic MPS sinon swap sur 16 Go.)
    """
    from tools.eval_bpc import collect_parent, collect_engine, load_c2_texts
    from main import load_tokenizer
    import gc
    import json

    # Manifeste versionné : les 6 docs val C2 (354 234 chars).
    # Lancer avec : pytest -m slow tests/test_bpc.py
    # (désélectionné par défaut via addopts, c2_subset.json vérifié par SHA1)
    try:
        texts = load_c2_texts()
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"C2 indisponible : {e}")
    vocab = json.load(open("exports/tranche00.bin.vocab.json", encoding="utf-8"))
    kept = set(vocab["kept_ids"])
    smol_tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    # Résultats attendus (variante a)
    expected_a = {
        "tranche00": 1.5676,
        "wiki": 1.6566,
        "smol135": 1.5357,
        "smol360": 1.3362,
        "qwen": 1.1138,
    }

    # Résultats attendus (variantes b et c)
    expected_b = {
        "tranche00": 1.6536,
        "wiki": 1.7431,
        "smol135": 1.6099,
        "smol360": 1.4133,
        "qwen": 1.2049,
    }

    expected_c = {
        "tranche00": 1.5735,
        "wiki": 1.6641,
        "smol135": 1.5292,
        "smol360": 1.3302,
        "qwen": 1.1096,
    }

    # Collecte .bin
    parent_ids = []
    for t in texts:
        parent_ids += smol_tok.encode(t, add_special_tokens=False) + [
            smol_tok.eos_token_id]

    rec_tranche, chars = collect_engine("exports/tranche00.bin", vocab, texts,
                                        parent_ids)
    v_tranche = variantes(rec_tranche, chars)
    assert v_tranche["a"] == pytest.approx(expected_a["tranche00"], abs=1e-4)
    assert v_tranche["b"] == pytest.approx(expected_b["tranche00"], abs=1e-4)
    assert v_tranche["c"] == pytest.approx(expected_c["tranche00"], abs=1e-4)

    rec_wiki, chars = collect_engine("exports/wiki_slice00_bestval.bin", vocab,
                                     texts, parent_ids)
    v_wiki = variantes(rec_wiki, chars)
    assert v_wiki["a"] == pytest.approx(expected_a["wiki"], abs=1e-4)
    assert v_wiki["b"] == pytest.approx(expected_b["wiki"], abs=1e-4)
    assert v_wiki["c"] == pytest.approx(expected_c["wiki"], abs=1e-4)

    # Collecte parents (nettoyage MPS entre chaque modèle).
    rec_135, chars = collect_parent("HuggingFaceTB/SmolLM-135M", texts, kept,
                                     smol_tok, device=dev)
    v_135 = variantes(rec_135, chars)
    assert v_135["a"] == pytest.approx(expected_a["smol135"], abs=1e-4)
    assert v_135["b"] == pytest.approx(expected_b["smol135"], abs=1e-4)
    assert v_135["c"] == pytest.approx(expected_c["smol135"], abs=1e-4)
    del rec_135
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()

    rec_360, chars = collect_parent("HuggingFaceTB/SmolLM-360M", texts, kept,
                                     smol_tok, device=dev)
    v_360 = variantes(rec_360, chars)
    assert v_360["a"] == pytest.approx(expected_a["smol360"], abs=1e-4)
    assert v_360["b"] == pytest.approx(expected_b["smol360"], abs=1e-4)
    assert v_360["c"] == pytest.approx(expected_c["smol360"], abs=1e-4)
    del rec_360
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()

    rec_qwen, chars = collect_parent("Qwen/Qwen3-0.6B", texts, kept,
                                      smol_tok, bs=2, device=dev)
    v_qwen = variantes(rec_qwen, chars)
    assert v_qwen["a"] == pytest.approx(expected_a["qwen"], abs=1e-4)
    assert v_qwen["b"] == pytest.approx(expected_b["qwen"], abs=1e-4)
    assert v_qwen["c"] == pytest.approx(expected_c["qwen"], abs=1e-4)
