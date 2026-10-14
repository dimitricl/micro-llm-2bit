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


def test_check_parent_vocab_match():
    """F0a : refuse tokenizer parent ≠ source du vocab sauf from-scratch."""
    from data import check_parent_vocab_match
    import pytest
    vocab = {"parent": "HuggingFaceTB/SmolLM-135M", "kept_ids": [0, 1, 2]}
    # Match exact : OK.
    check_parent_vocab_match("HuggingFaceTB/SmolLM-135M", vocab, False)
    # Mismatch (cas R3 : parent Qwen + vocab SmolLM) : refus.
    with pytest.raises(ValueError, match="tokenizer"):
        check_parent_vocab_match("Qwen/Qwen3-0.6B-Base", vocab, False)
    # Flag explicite from-scratch : bypass documenté.
    check_parent_vocab_match("Qwen/Qwen3-0.6B-Base", vocab, True)
    # Vocab sans champ parent (ancien format) : on ne peut pas vérifier.
    check_parent_vocab_match("Qwen/Qwen3-0.6B-Base", {"kept_ids": [0]}, False)


def test_prepare_corpus_refuse_mismatch_parent(tmp_path):
    """F0a intégration : cas R3 (parent Qwen + vocab SmolLM) refusé vite."""
    import json
    import time
    from main import prepare_corpus

    d = tmp_path / "mini.jsonl"
    d.write_text('{"text": "' + "bonjour le monde " * 40 + '"}\n'
                   '{"text": "' + "au revoir " * 40 + '"}\n')
    v = tmp_path / "v.json"
    v.write_text(json.dumps({"parent": "HuggingFaceTB/SmolLM-135M",
                             "kept_ids": [0, 1, 2],
                             "unk_new": 0, "eos_new": 1}))

    class FakeTok:
        eos_token_id = 1
        unk_token_id = 0

        def encode(self, t, add_special_tokens=False):
            return list(range(64))

    import argparse
    a = argparse.Namespace(data=str(d), val_ratio=0.0, seed=0,
                           vocab_from=str(v), vocab_size=3, save_batch=2,
                           seq_len=8, parent="Qwen/Qwen3-0.6B-Base",
                           from_scratch=False)
    t0 = time.time()
    try:
        prepare_corpus(FakeTok(), a)
    except ValueError as e:
        assert "Mismatch" in str(e) or "mismatch" in str(e).lower()
        assert time.time() - t0 < 10.0
    else:
        raise AssertionError("mismatch parent/vocab non refusé")
    # Avec le flag : passe.
    a.from_scratch = True
    pb, sb, vpb, vsb, vocab = prepare_corpus(FakeTok(), a)
    assert len(pb) == len(sb) > 0
