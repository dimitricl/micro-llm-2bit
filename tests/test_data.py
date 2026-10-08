"""Tests data.py avec un tokenizer factice (sans réseau)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data as D


class StubTok:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3
    name_or_path = "stub"

    def encode(self, text, add_special_tokens=False):
        # Tokenisation jouet : 1 id par mot (hash stable), modulo 500.
        return [abs(hash(w)) % 500 for w in text.split()]


def test_vocab_reduit():
    tok = StubTok()
    texts = ["bonjour le monde", "le monde est vaste", "bonjour vaste monde"]
    vocab = D.build_reduced_vocab(tok, texts, vocab_size=32)
    assert vocab["vocab_size"] == 32
    # Spéciaux présents.
    for s in (0, 1, 2, 3):
        assert s in vocab["kept_ids"]
    # Mots fréquents présents.
    frequent_old = abs(hash("monde")) % 500
    assert frequent_old in vocab["kept_ids"]
    # Encodage : inconnu -> unk, eos ajouté.
    ids = D.encode_mapped(tok, vocab, "monde xyzzyplugh")
    assert ids[-1] == vocab["eos_new"]
    assert vocab["unk_new"] in ids


def test_batches_alignes():
    stream = list(range(100))
    b1 = D.make_batches(stream, 16, 4, shuffle=True, seed=7)
    b2 = D.make_batches(stream, 16, 4, shuffle=True, seed=7)
    assert len(b1) == len(b2)
    for x, y in zip(b1, b2):
        assert (x == y).all()
    # Forme (B, seq+1) pour LM causale.
    assert b1[0].shape[1] == 17


def test_split_sans_fuite():
    texts = [f"doc {i} " + "mot " * (i % 7) for i in range(40)]
    tr1, va1 = D.train_val_split(texts, 0.05, seed=3)
    tr2, va2 = D.train_val_split(texts, 0.05, seed=3)
    # Déterministe, sans chevauchement, ratio respecté.
    assert (tr1, va1) == (tr2, va2)
    assert len(va1) == 2 and len(tr1) == 38
    assert not set(map(id, tr1)) & set(map(id, va1))
    assert sorted(tr1 + va1, key=texts.index) == texts
    # Seed différente -> split différent (mélange réel).
    _, va3 = D.train_val_split(texts, 0.05, seed=4)
    assert va3 != va1
