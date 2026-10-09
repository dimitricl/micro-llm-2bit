"""Tests B1 : décompte exact steps/micros/tokens par epoch.

Couvre plan_epoch() (pur) + intégration train() : steps optimiseur
exécutés == steps planifiés, tokens vus == tokens de l'epoch, reste de fin
d'epoch appliqué (pas jeté), refus clair si batch_size > save_batch.
Parent factice, CPU.
"""

import dataclasses
import os
import sys
import types
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import distill as dist
from data import make_batches
from model import TINY, TinyTransformer


class FakeParent(nn.Module):
    def __init__(self, p_full=256, h=32):
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


def _batches(n_windows, seq=16, save=4, seed=0):
    """Batchs stockés avec EXACTEMENT n_windows fenêtres pleines."""
    torch.manual_seed(seed)
    stream = torch.randint(0, 256, (n_windows * seq + 1,)).tolist()
    pb = make_batches(stream, seq, save, shuffle=False, seed=seed)
    sb = make_batches([t % 64 for t in stream], seq, save, shuffle=False, seed=seed)
    assert sum(b.shape[0] for b in sb) == n_windows
    return pb, sb


def _expected(pb, sb, bs, accum):
    seq = sb[0].shape[1] - 1
    micros = sum((b.shape[0] + bs - 1) // bs for b in sb)
    steps = max((micros + accum - 1) // accum, 1)
    tokens = sum(b.shape[0] for b in sb) * seq
    return micros, steps, tokens


@pytest.mark.parametrize(
    "n_windows,save,bs,accum",
    [
        (32, 4, 2, 2),  # cas nominal : 16 micros, 8 steps
        (32, 4, 2, 3),  # reste 16 % 3 = 1 micro : appliqué -> 6 steps
        (32, 8, 4, 4),  # 8 micros, 2 steps
        (30, 4, 2, 4),  # dernier batch partiel (2 lignes) : 15 micros, 4 steps
        (31, 8, 8, 8),  # bs == save, reste 4 micros : 1 step... vérifié par calcul
    ],
)
def test_plan_epoch_pur(n_windows, save, bs, accum):
    pb, sb = _batches(n_windows, save=save)
    plan = dist.plan_epoch(sb, bs, accum)
    micros, steps, tokens = _expected(pb, sb, bs, accum)
    assert plan["micros"] == micros
    assert plan["steps"] == steps
    assert plan["tokens"] == tokens
    assert plan["eff_micro"] == bs


def test_plan_epoch_refuse_bs_superieur_save():
    _, sb = _batches(16, save=4)
    with pytest.raises(ValueError, match="--batch-size"):
        dist.plan_epoch(sb, batch_size=8, accum=2)


def _tiny_student(v=64):
    cfg = dataclasses.replace(
        TINY, vocab_size=v, d_model=64, n_layers=2, n_heads_q=4,
        n_heads_kv=2, ffn_dim=128, seq_max=32, group_size=32,
    )
    return TinyTransformer(cfg)


def _vocab(v=64):
    return {"parent": "fake", "vocab_size": v, "kept_ids": list(range(v)),
            "unk_new": 0, "eos_new": 1}


def _run_train(tmp_path, pb, sb, bs, accum, tag):
    import transformers

    torch.manual_seed(0)
    student = _tiny_student()
    seen = {"micros": 0, "tokens": 0}
    orig_fwd = student.forward

    def wrap(x):
        seen["micros"] += 1
        seen["tokens"] += x.numel()
        return orig_fwd(x)

    student.forward = wrap
    opt_calls = {"n": 0}
    real_step = torch.optim.AdamW.step

    def counting(self):
        opt_calls["n"] += 1
        return real_step(self)

    real_pre = transformers.AutoModelForCausalLM.from_pretrained

    @classmethod
    def fake_from_pretrained(cls, *a, **k):
        return FakeParent(256)

    transformers.AutoModelForCausalLM.from_pretrained = fake_from_pretrained
    try:
        out = str(tmp_path / f"enfant_{tag}.pt")
        with patch.object(torch.optim.AdamW, "step", counting):
            final = dist.train(
                student, pb, sb, _vocab(), "fake-parent", out,
                epochs=1, batch_size=bs, accum=accum, log_every=10**9,
                device=torch.device("cpu"),
            )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real_pre
    ckpt = torch.load(final, map_location="cpu", weights_only=False)
    return ckpt, seen, opt_calls


@pytest.mark.parametrize(
    "n_windows,save,bs,accum",
    [
        (32, 4, 2, 2),
        (32, 4, 2, 3),  # reste appliqué en fin d'epoch
        (30, 4, 2, 4),  # batch partiel + reste
    ],
)
def test_train_steps_et_tokens_par_epoch(tmp_path, n_windows, save, bs, accum):
    """Steps optimiseur exécutés == steps planifiés, tokens vus == epoch."""
    pb, sb = _batches(n_windows, save=save)
    micros, steps, tokens = _expected(pb, sb, bs, accum)
    ckpt, seen, opt_calls = _run_train(
        tmp_path, pb, sb, bs, accum, f"{save}_{bs}_{accum}")
    assert opt_calls["n"] == steps, (opt_calls, steps)
    assert ckpt["step"] == steps
    assert seen["micros"] == micros, (seen, micros)
    assert seen["tokens"] == tokens, (seen, tokens)


def test_train_refuse_bs_superieur_save_sans_charger_parent(tmp_path):
    """bs > save : ValueError claire AVANT tout chargement du parent."""
    import transformers

    pb, sb = _batches(16, save=4)
    student = _tiny_student()
    real_pre = transformers.AutoModelForCausalLM.from_pretrained
    appels = {"n": 0}

    @classmethod
    def no_load(cls, *a, **k):
        appels["n"] += 1
        raise AssertionError("le parent n'aurait pas dû être chargé")

    transformers.AutoModelForCausalLM.from_pretrained = no_load
    try:
        with pytest.raises(ValueError, match="--batch-size"):
            dist.train(
                student, pb, sb, _vocab(), "fake-parent",
                str(tmp_path / "x.pt"), epochs=1, batch_size=8, accum=2,
                device=torch.device("cpu"),
            )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real_pre
    assert appels["n"] == 0
