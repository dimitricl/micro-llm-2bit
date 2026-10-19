"""Tests du harnais eval_fixed : leviers de décodage (P3).

(a) Avec les défauts (--repetition-penalty 1.0, --no-repeat-ngram 0), les
    appels à eng.generate sont EXACTEMENT ceux d'avant l'ajout des options
    (aucun kwarg supplémentaire) -> sortie bit-à-bit identique.
(b) Les deux valeurs sont propagées aux DEUX appels (greedy ET sampling).
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data as D
import engine as E
import main as M
import tools.eval_fixed as F


class FakeEngine:
    instances: list["FakeEngine"] = []

    def __init__(self, path: str):
        self.path = path
        self.meta = {"seq_max": 128}
        self.calls: list[dict] = []
        FakeEngine.instances.append(self)

    def generate(self, ids, **kw):
        self.calls.append(kw)
        return [1, 2, 3, 4, 5]


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    FakeEngine.instances = []
    vocab_path = tmp_path / "m.bin.vocab.json"
    vocab_path.write_text(json.dumps({"parent": "fake", "eos_new": 0}))
    monkeypatch.setattr(E, "Engine", FakeEngine)
    monkeypatch.setattr(M, "load_tokenizer", lambda name: object())
    monkeypatch.setattr(M, "decode_ids", lambda tok, vocab, ids: "a b c d e")
    monkeypatch.setattr(
        D, "encode_mapped", lambda tok, vocab, text, add_eos=False: [1, 2, 3]
    )

    def run(extra: list[str]):
        out = tmp_path / "samples.md"
        met = tmp_path / "metrics.json"
        F.main(
            [
                "--model", str(tmp_path / "m.bin"),
                "--vocab", str(vocab_path),
                "--out", str(out),
                "--metrics-out", str(met),
            ]
            + extra
        )
        return met

    return run


def test_defauts_appels_identiques_a_avant(harness):
    """(a) Défauts : aucun kwarg de décodage ajouté -> identité stricte."""
    met = harness([])
    assert len(FakeEngine.instances) == 1
    calls = FakeEngine.instances[0].calls
    assert len(calls) == 2 * len(F.PROMPTS)

    greedy, sampling = calls[0], calls[1]
    assert set(greedy) == {"max_new", "temperature", "stop"}
    assert greedy["temperature"] == 0.0
    assert set(sampling) == {"max_new", "temperature", "top_k", "top_p",
                             "stop", "rng"}
    assert sampling["temperature"] == 0.7

    # Aucun appel ne porte de levier de décodage.
    for c in calls:
        assert "repetition_penalty" not in c
        assert "no_repeat_ngram_size" not in c

    data = json.loads(met.read_text())
    assert len(data["metrics"]) == len(F.PROMPTS)


def test_leviers_propages_aux_deux_appels(harness):
    """(b) --repetition-penalty / --no-repeat-ngram -> greedy ET sampling."""
    harness(["--repetition-penalty", "1.15", "--no-repeat-ngram", "3"])
    calls = FakeEngine.instances[0].calls
    assert len(calls) == 2 * len(F.PROMPTS)
    for c in calls:
        assert c["repetition_penalty"] == 1.15
        assert c["no_repeat_ngram_size"] == 3
    # Le sampling garde ses réglages d'origine.
    assert calls[1]["temperature"] == 0.7
    assert calls[1]["top_k"] == 40


def test_leviers_partiels(harness):
    """Une seule option activée : l'autre reste au défaut OFF."""
    harness(["--no-repeat-ngram", "3"])
    for c in FakeEngine.instances[0].calls:
        assert c["repetition_penalty"] == 1.0
        assert c["no_repeat_ngram_size"] == 3


@pytest.mark.slow
def test_identite_bit_a_bit_sur_et08(tmp_path):
    """(a) preuve réelle : C0 (défauts) == eval/Et08/metrics.json à l'octet.

    Lancer : pytest -m slow tests/test_eval_fixed_decode.py -k bit_a_bit
    """
    root = os.path.join(os.path.dirname(__file__), "..")
    model = os.path.join(root, "runs/E-t08/model.bin")
    ref = os.path.join(root, "eval/Et08/metrics.json")
    out = tmp_path / "metrics.json"
    F.main(["--model", model,
            "--out", str(tmp_path / "samples.md"),
            "--metrics-out", str(out)])
    assert out.read_text() == open(ref, encoding="utf-8").read()
