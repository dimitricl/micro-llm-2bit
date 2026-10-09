"""Tests B6 : alpha=0 = vraie baseline CE seule (parent jamais touché)."""

import dataclasses
import glob
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import distill as dist
from data import make_batches
from model import TINY, TinyTransformer


def test_loss_alpha_zero_ignore_parent():
    """alpha=0 : loss == CE, kl == 0, même avec teacher=None (pas de KL)."""
    torch.manual_seed(0)
    crit = dist.DistillationLoss(alpha=0.0, temperature=2.0)
    s = torch.randn(2, 4, 16)
    y = torch.randint(0, 16, (2, 4))
    loss_none, parts_none = crit(s, None, y)
    assert parts_none["kl"].item() == 0.0
    assert torch.allclose(
        loss_none, F.cross_entropy(s.reshape(-1, 16), y.reshape(-1)))
    # Teacher fourni mais alpha=0 : même résultat (KL non calculée).
    loss_t, parts_t = crit(s, torch.randn(2, 4, 16), y)
    assert torch.equal(loss_t, loss_none)
    assert parts_t["kl"].item() == 0.0


def _run_alpha_zero(tmp_path, **kw):
    torch.manual_seed(0)
    v = 64
    vocab = {"parent": "fake", "vocab_size": v, "kept_ids": list(range(v)),
             "unk_new": 0, "eos_new": 1}
    stream = torch.randint(0, 256, (257,)).tolist()
    pb = make_batches(stream, 16, 4, shuffle=False, seed=0)
    sb = make_batches([t % v for t in stream], 16, 4, shuffle=False, seed=0)
    cfg = dataclasses.replace(TINY, vocab_size=v, d_model=64, n_layers=2,
                              n_heads_q=4, n_heads_kv=2, ffn_dim=128,
                              seq_max=32, group_size=32)
    student = TinyTransformer(cfg)

    import transformers
    real_pre = transformers.AutoModelForCausalLM.from_pretrained
    appels = {"n": 0}

    @classmethod
    def no_parent(cls, *a, **k):
        appels["n"] += 1
        raise AssertionError("le parent ne doit pas être chargé (alpha=0)")

    transformers.AutoModelForCausalLM.from_pretrained = no_parent
    try:
        out = str(tmp_path / "enfant_alpha0.pt")
        final = dist.train(student, pb, sb, vocab, "fake-parent", out,
                           epochs=1, batch_size=2, accum=2, log_every=1000,
                           alpha=0.0, device=torch.device("cpu"), **kw)
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real_pre
    ckpt = torch.load(final, map_location="cpu", weights_only=False)
    return final, ckpt, appels


def test_train_alpha_zero_sans_parent(tmp_path):
    """train(alpha=0) : aucun chargement ni forward parent, step>0."""
    final, ckpt, appels = _run_alpha_zero(tmp_path)
    assert appels["n"] == 0
    assert ckpt["step"] > 0
    assert os.path.exists(final)


def test_train_alpha_zero_ignore_cache(tmp_path):
    """alpha=0 + top_k_cache : aucun cache construit (inutile en CE seule)."""
    prefix = str(tmp_path / "cache_alpha0")
    final, ckpt, appels = _run_alpha_zero(
        tmp_path, top_k_cache=1, cache_path=prefix, cache_k=8)
    assert appels["n"] == 0
    assert ckpt["step"] > 0
    assert glob.glob(prefix + "*") == []
