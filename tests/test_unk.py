"""Tests B2 : vocab à unk distinct (ligne dédiée, logsumexp parent).

Couvre : build_reduced_vocab (sentinelle -1, taille inchangée, eos intact),
teacher_small_logits (direct + cache : colonne unk = logsumexp hors-vocab),
check_vocab_compatible (refus de mélange), decode_ids (marqueur visible),
sample/generate (masquage unk), tools/convert_vocab.py.
"""

import dataclasses
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data as D
import distill as dist
import engine as E
import main as M


class FakeTok:
    """Tokenizer jouet : spéciaux 0-3, lettres -> 10+, 'z' -> 30 (rare)."""

    pad_token_id = 3
    eos_token_id = 2
    bos_token_id = 1
    unk_token_id = 0
    name_or_path = "fake-tok"

    def encode(self, text, add_special_tokens=False):
        out = []
        for c in text:
            if c == "z":
                out.append(30)
            elif c.isalpha():
                out.append(10 + (ord(c.lower()) - ord("a")) % 16)
        return out

    def decode(self, ids, skip_special_tokens=True):
        return "<" + ",".join(str(i) for i in ids) + ">"


class FakeTokZero:
    """Spéciaux à la SmolLM : bos=eos=unk=0, pas de pad."""

    pad_token_id = None
    eos_token_id = 0
    bos_token_id = 0
    unk_token_id = 0
    name_or_path = "fake-zero"

    def encode(self, text, add_special_tokens=False):
        return [10 + (ord(c.lower()) - ord("a")) % 16 for c in text if c.isalpha()]

    def decode(self, ids, skip_special_tokens=True):
        return "<" + ",".join(str(i) for i in ids) + ">"


def test_vocab_unk_dedie_taille_inchangee():
    tok = FakeTok()
    texts = ["a" * 100 + "b" * 50 + "c" * 10 + "z"]
    vocab = D.build_reduced_vocab(tok, texts, vocab_size=8)
    kept = vocab["kept_ids"]
    assert len(kept) == 8 and vocab["vocab_size"] == 8
    assert vocab["unk_distinct"] is True and vocab["vocab_version"] == 2
    # Spéciaux intacts, eos garde son id.
    for s in (0, 1, 2, 3):
        assert s in kept
    assert kept[vocab["eos_new"]] == 2
    # Une seule sentinelle, à la place du moins fréquent (le 'z' -> 30,
    # ou le dernier filler) : jamais un spécial.
    assert kept.count(-1) == 1
    assert kept[vocab["unk_new"]] == -1
    assert vocab["unk_new"] != vocab["eos_new"]
    # Fréquents gardés.
    assert 10 in kept  # 'a'


def test_vocab_zero_a_la_smollm():
    """bos=eos=unk=0 : eos reste ligne 0, unk devient une ligne dédiée."""
    tok = FakeTokZero()
    vocab = D.build_reduced_vocab(tok, ["a" * 50 + "b" * 5], vocab_size=6)
    assert vocab["eos_new"] == 0 and vocab["kept_ids"][0] == 0
    assert vocab["unk_new"] != 0 and vocab["kept_ids"][vocab["unk_new"]] == -1
    assert vocab["unk_distinct"] is True


def test_encode_mapped_rabat_sur_unk_dedie():
    tok = FakeTok()
    texts = ["a" * 100 + "b" * 50 + "c" * 10]
    vocab = D.build_reduced_vocab(tok, texts, vocab_size=8)
    ids = D.encode_mapped(tok, vocab, "a zzz")
    assert vocab["unk_new"] in ids  # 'z' (30) hors vocab
    assert ids[-1] == vocab["eos_new"]


def test_teacher_small_logits_unk_logsumexp():
    torch.manual_seed(0)
    P, V = 8, 4
    kept = [0, 2, -1, 5]
    full = torch.randn(2, 3, P, dtype=torch.float32)
    small = dist.teacher_small_logits(full, kept)
    assert small.shape == (2, 3, V)
    assert torch.equal(small[..., 0], full[..., 0])
    assert torch.equal(small[..., 1], full[..., 2])
    assert torch.equal(small[..., 3], full[..., 5])
    ref = full[..., [1, 3, 4, 6, 7]].logsumexp(dim=-1)
    assert torch.allclose(small[..., 2], ref)


def test_teacher_small_logits_vocab_ancien_inchange():
    """Sans sentinelle : strictement le gather d'avant (compat)."""
    torch.manual_seed(0)
    full = torch.randn(2, 3, 8)
    kept = [0, 2, 5]
    got = dist.teacher_small_logits(full, kept)
    ref = full.index_select(-1, torch.tensor(kept))
    assert torch.equal(got, ref)


def test_compat_refuse_melange():
    old = {"kept_ids": [0, 1, 2], "unk_new": 0, "eos_new": 0}
    new = {"kept_ids": [0, 1, -1], "unk_new": 2, "eos_new": 0,
           "unk_distinct": True, "vocab_version": 2}
    with pytest.raises(ValueError, match="unk distinct"):
        D.check_vocab_compatible(new, old)
    with pytest.raises(ValueError, match="unk distinct"):
        D.check_vocab_compatible(old, new)
    D.check_vocab_compatible(new, dict(new))  # même format : ok
    D.check_vocab_compatible(old, dict(old))


def test_decode_ids_marqueur_unk():
    tok = FakeTok()
    vocab = {"kept_ids": [0, 1, 2, 10, -1], "unk_new": 4, "eos_new": 2,
             "unk_distinct": True}
    text = M.decode_ids(tok, vocab, [3, 4, 3, 99])
    assert text.count(M.UNK_MARKER) == 2  # unk + id invalide
    assert "<10>" in text  # tronçons décodés normalement


def test_decode_ids_vocab_ancien_inchange():
    """Ancien vocab : decode direct, unk/eos supprimés comme avant."""
    tok = FakeTok()
    vocab = {"kept_ids": [0, 1, 2, 10], "unk_new": 0, "eos_new": 0}
    assert M.UNK_MARKER not in M.decode_ids(tok, vocab, [3, 0])
    assert M.decode_ids(tok, vocab, [3]) == "<10>"


def test_sample_masque_unk():
    """exclude : glouton évite le max masqué ; tirage : proba nulle."""
    z = __import__("numpy").array([1.0, 5.0, 3.0])
    assert E.sample(z, 0.0, exclude={1}) == 2
    assert E.sample(z, 0.0, exclude={9}) == 1  # hors limites : ignoré
    assert E.sample(z, 0.0) == 1  # sans masque : inchangé
    rng = __import__("numpy").random.default_rng(0)
    tirages = [E.sample(__import__("numpy").array([0.0, 0.0, 0.0]),
                        1.0, rng=rng, exclude={0}) for _ in range(200)]
    assert 0 not in tirages and set(tirages) == {1, 2}


class FakeParent(nn.Module):
    def __init__(self, p_full=32, h=16):
        super().__init__()
        self.emb = nn.Embedding(p_full, h)
        self.lm = nn.Linear(h, p_full)
        self.config = types.SimpleNamespace(hidden_size=h)
        self.name_or_path = "fake-parent"

    def forward(self, input_ids, output_hidden_states=False):
        h = self.emb(input_ids).float()
        out = types.SimpleNamespace(logits=self.lm(h))
        if output_hidden_states:
            out.hidden_states = [h]
        return out


def _fake_from_pretrained(p_full):
    import transformers
    real = transformers.AutoModelForCausalLM.from_pretrained

    @classmethod
    def fake(cls, *a, **k):
        return FakeParent(p_full)

    transformers.AutoModelForCausalLM.from_pretrained = fake
    return real


def test_cache_avec_unk_dedie(tmp_path):
    """build/load cache : colonne unk = logsumexp, top-k cohérent."""
    torch.manual_seed(0)
    kept = [0, 1, 2, 10, 11, -1]
    vocab = {"parent": "fake", "vocab_size": 6, "kept_ids": kept,
             "unk_new": 5, "eos_new": 2, "unk_distinct": True,
             "vocab_version": 2}
    stream = [10, 11, 12, 30, 10, 2] * 20
    from data import make_batches
    pb = make_batches(stream, 8, 4, shuffle=False, seed=0)
    sb = make_batches(stream, 8, 4, shuffle=False, seed=0)
    parent = FakeParent(32)
    prefix = str(tmp_path / "cache_unk")
    cache = dist.build_teacher_cache(parent, pb, sb, kept, 4,
                                     torch.device("cpu"), 2, prefix)
    assert cache["manifest"]["unk_distinct"] is True
    got = dist.teacher_logits_from_cache(cache, 0, 0, 6, torch.device("cpu"))
    assert got.shape == (2, 8, 6)
    with torch.no_grad():
        ref = dist.teacher_small_logits(
            parent(input_ids=pb[0][:2][:, :-1]).logits.float(), kept)
        rv, ri = ref.topk(4, dim=-1)
        dense = torch.full((2 * 8, 6), float("-inf"))
        dense.scatter_(-1, ri.view(-1, 4), rv.view(-1, 4))
    assert torch.equal(torch.isneginf(got), torch.isneginf(dense.view(2, 8, 6)))
    assert torch.allclose(got[~torch.isneginf(got)],
                          dense.view(2, 8, 6)[~torch.isneginf(got)], atol=1e-3)
    # Rejet si le vocab change (hash inclut la sentinelle).
    import pytest as _pt
    with _pt.raises(ValueError):
        dist.load_teacher_cache(prefix, "fake-parent", [0, 1, 2, 10, 11, 12],
                                8, 4)


def test_train_avec_unk_dedie(tmp_path):
    """Entraînement direct + cache avec vocab unk-distinct, de bout en bout."""
    import transformers

    from data import make_batches
    from model import TINY, TinyTransformer

    torch.manual_seed(0)
    tok = FakeTok()
    texts = ["a" * 200 + "b" * 100 + "c" * 20 + "z" * 3]
    # vocab_size=8 sature : [0,1,2,3,10,11,12,30] -> le 'z' (30, le moins
    # fréquent) cède sa place à la ligne unk.
    vocab = D.build_reduced_vocab(tok, texts, vocab_size=8)
    assert vocab["unk_distinct"]
    parent_stream = []
    for t in texts:
        parent_stream += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
    new_of = {o: n for n, o in enumerate(vocab["kept_ids"]) if o >= 0}
    student_stream = [new_of.get(t, vocab["unk_new"]) for t in parent_stream]
    assert vocab["unk_new"] in student_stream  # le 'z' est hors vocab
    pb = make_batches(parent_stream, 8, 4, shuffle=False, seed=0)
    sb = make_batches(student_stream, 8, 4, shuffle=False, seed=0)
    cfg = dataclasses.replace(TINY, vocab_size=8, d_model=64, n_layers=2,
                              n_heads_q=4, n_heads_kv=2, ffn_dim=128,
                              seq_max=32, group_size=32)
    real = _fake_from_pretrained(32)
    try:
        out = str(tmp_path / "e_unk.pt")
        final = dist.train(TinyTransformer(cfg), pb, sb, vocab, "fake-parent",
                           out, epochs=1, batch_size=2, accum=2, log_every=1000,
                           top_k_cache=1, cache_path=str(tmp_path / "c_unk"),
                           cache_k=4, device=torch.device("cpu"))
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real
    ckpt = torch.load(final, map_location="cpu", weights_only=False)
    assert ckpt["step"] > 0


def test_convert_vocab(tmp_path):
    """Conversion ancien -> unk distinct : taille, moyenne, sentinelle."""
    from tools.convert_vocab import convert_vocab_in_ckpt

    torch.manual_seed(0)
    V, d = 16, 8
    emb = torch.randn(V, d)
    ckpt = {"model": {"embed_latent": emb.clone()},
            "vocab": {"parent": "x", "vocab_size": V,
                      "kept_ids": list(range(V)),
                      "unk_new": 0, "eos_new": 0}}
    ref_mean = emb.float().mean(dim=0)
    out, new_vocab = convert_vocab_in_ckpt(ckpt)
    assert len(new_vocab["kept_ids"]) == V
    assert new_vocab["kept_ids"][-1] == -1
    assert 15 not in new_vocab["kept_ids"]
    assert new_vocab["unk_new"] == V - 1 and new_vocab["eos_new"] == 0
    assert new_vocab["unk_distinct"] is True
    assert torch.allclose(out["model"]["embed_latent"][-1],
                          ref_mean.to(emb.dtype))
    assert torch.equal(out["model"]["embed_latent"][:-1], emb[:-1])
    with pytest.raises(ValueError):
        convert_vocab_in_ckpt(out)  # déjà converti : refus
