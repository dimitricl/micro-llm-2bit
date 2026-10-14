"""P3 : anti-répétition — frequency penalty + no-repeat-ngram (TDD).

Défaut OFF : repetition_penalty=1.0, no_repeat_ngram_size=0 ne changent
rien (goldens greedy bit-identiques préservés).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import sample


def test_sample_defaut_inchange():
    z = np.array([1.0, 5.0, 3.0])
    assert sample(z, temperature=0.0) == 1
    assert sample(z, temperature=0.0, repetition_penalty=1.0,
                  seen=None) == 1


def test_frequency_penalty_casse_boucle():
    # Token 1 vu 50× : pénalité forte -> argmax bascule sur 2.
    z = np.array([0.0, 10.0, 9.0])
    assert sample(z, temperature=0.0, repetition_penalty=1.0,
                  seen={1: 50}) == 1
    assert sample(z, temperature=0.0, repetition_penalty=1.5,
                  seen={1: 50}) == 2


def test_frequency_penalty_neutre_sans_seen():
    z = np.array([0.0, 10.0, 9.0])
    assert sample(z, temperature=0.0, repetition_penalty=2.0,
                  seen={}) == 1
    assert sample(z, temperature=0.0, repetition_penalty=2.0,
                  seen=None) == 1


def test_no_repeat_ngram_bannit():
    # n-gramme (7,8) déjà vu : le token 8 est banni après préfixe 7.
    from engine import forbidden_by_ngram
    seen = {(7, 8), (8, 9)}
    assert forbidden_by_ngram([7], seen, 2) == {8}
    assert forbidden_by_ngram([5], seen, 2) == set()
    assert forbidden_by_ngram([7], seen, 0) == set()
