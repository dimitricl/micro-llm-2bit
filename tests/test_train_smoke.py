"""Smoke test de la boucle de distillation (parent factice, CPU)."""

import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dataclasses

import torch
import torch.nn as nn

import distill as dist
from data import make_batches
from model import TINY, TinyTransformer


class FakeParent(nn.Module):
    """Parent HF-like minimal : logits + hidden_states, dim cachée 32."""

    def __init__(self, p_full=256, h=32):
        super().__init__()
        self.emb = nn.Embedding(p_full, h)
        self.lm = nn.Linear(h, p_full)
        self.config = types.SimpleNamespace(hidden_size=h)

    def forward(self, input_ids, output_hidden_states=False):
        h = self.emb(input_ids).float()
        out = types.SimpleNamespace(logits=self.lm(h))
        if output_hidden_states:
            out.hidden_states = [h]
        return out


def _run(tmp_path, hidden_weight):
    torch.manual_seed(0)
    p_full, v = 256, 64
    kept = list(range(v))
    vocab = {
        "parent": "fake",
        "vocab_size": v,
        "kept_ids": kept,
        "unk_new": 0,
        "eos_new": 1,
    }
    stream = torch.randint(0, p_full, (512,)).tolist()
    pb = make_batches(stream, 16, 4, shuffle=True, seed=0)
    sb = make_batches([t % v for t in stream], 16, 4, shuffle=True, seed=0)
    cfg = dataclasses.replace(
        TINY,
        vocab_size=v,
        d_model=64,
        n_layers=2,
        n_heads_q=4,
        n_heads_kv=2,
        ffn_dim=128,
        seq_max=32,
        group_size=32,
    )
    student = TinyTransformer(cfg)
    import transformers

    real = transformers.AutoModelForCausalLM.from_pretrained

    @classmethod
    def fake_from_pretrained(cls, *a, **k):
        return FakeParent(p_full)

    transformers.AutoModelForCausalLM.from_pretrained = fake_from_pretrained
    try:
        out = str(tmp_path / "enfant.pt")
        final = dist.train(
            student,
            pb,
            sb,
            vocab,
            "fake-parent",
            out,
            epochs=1,
            batch_size=2,
            accum=2,
            hidden_weight=hidden_weight,
            log_every=1000,
            device=torch.device("cpu"),
        )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real
    assert os.path.exists(final)
    ckpt = torch.load(final, map_location="cpu", weights_only=False)
    assert ckpt["step"] > 0
    return final


def test_train_smoke(tmp_path):
    _run(tmp_path, hidden_weight=0.0)


def test_train_smoke_hidden(tmp_path):
    _run(tmp_path, hidden_weight=0.1)


def test_early_stopping_et_bestval(tmp_path):
    """Avec patience=1 sur données aléatoires : arrêt précoce + _bestval."""
    torch.manual_seed(0)
    p_full, v = 256, 64
    kept = list(range(v))
    vocab = {
        "parent": "fake",
        "vocab_size": v,
        "kept_ids": kept,
        "unk_new": 0,
        "eos_new": 1,
    }
    stream = torch.randint(0, p_full, (512,)).tolist()
    pb = make_batches(stream, 16, 4, shuffle=True, seed=0)
    sb = make_batches([t % v for t in stream], 16, 4, shuffle=True, seed=0)
    vstream = torch.randint(0, p_full, (256,)).tolist()
    vpb = make_batches(vstream, 16, 4, shuffle=False, seed=0)
    vsb = make_batches([t % v for t in vstream], 16, 4, shuffle=False, seed=0)
    cfg = dataclasses.replace(
        TINY,
        vocab_size=v,
        d_model=64,
        n_layers=2,
        n_heads_q=4,
        n_heads_kv=2,
        ffn_dim=128,
        seq_max=32,
        group_size=32,
    )
    student = TinyTransformer(cfg)
    import transformers

    real = transformers.AutoModelForCausalLM.from_pretrained

    @classmethod
    def fake_from_pretrained(cls, *a, **k):
        return FakeParent(p_full)

    transformers.AutoModelForCausalLM.from_pretrained = fake_from_pretrained
    try:
        out = str(tmp_path / "enfant.pt")
        final = dist.train(
            student,
            pb,
            sb,
            vocab,
            "fake-parent",
            out,
            epochs=10,
            batch_size=2,
            accum=2,
            log_every=1000,
            eval_every=1,
            patience=1,
            val_parent_batches=vpb,
            val_student_batches=vsb,
            device=torch.device("cpu"),
        )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real
    ckpt = torch.load(final, map_location="cpu", weights_only=False)
    # Arrêt bien avant les 10 epochs (données aléatoires -> pas de progrès).
    assert ckpt["step"] < 10 * max(len(sb) // 2, 1), ckpt["step"]
    # Checkpoint best-val (poids seuls, sans optimiseur).
    best = torch.load(out.replace(".pt", "_bestval.pt"), map_location="cpu",
                      weights_only=False)
    assert "model" in best and "opt" not in best
    assert set(best.keys()) == {"model", "cfg", "vocab", "parent", "val_ce", "step"}


def _petit_setup():
    """Jeu minimal (même config que _run) pour les tests de reprise."""
    torch.manual_seed(0)
    p_full, v = 256, 64
    kept = list(range(v))
    vocab = {
        "parent": "fake",
        "vocab_size": v,
        "kept_ids": kept,
        "unk_new": 0,
        "eos_new": 1,
    }
    stream = torch.randint(0, p_full, (512,)).tolist()
    pb = make_batches(stream, 16, 4, shuffle=True, seed=0)
    sb = make_batches([t % v for t in stream], 16, 4, shuffle=True, seed=0)
    cfg = dataclasses.replace(
        TINY,
        vocab_size=v,
        d_model=64,
        n_layers=2,
        n_heads_q=4,
        n_heads_kv=2,
        ffn_dim=128,
        seq_max=32,
        group_size=32,
    )
    return p_full, vocab, pb, sb, cfg


def test_resume_vocab_incompatible(tmp_path):
    """Un vocab différent du checkpoint est refusé avec un message clair."""
    import copy

    import transformers

    p_full, vocab, pb, sb, cfg = _petit_setup()
    student = TinyTransformer(cfg)
    real = transformers.AutoModelForCausalLM.from_pretrained

    @classmethod
    def fake_from_pretrained(cls, *a, **k):
        return FakeParent(p_full)

    transformers.AutoModelForCausalLM.from_pretrained = fake_from_pretrained
    try:
        out = str(tmp_path / "enfant.pt")
        final = dist.train(
            student, pb, sb, vocab, "fake-parent", out,
            epochs=1, batch_size=2, accum=2, log_every=1000,
            device=torch.device("cpu"),
        )
        vocab2 = copy.deepcopy(vocab)
        vocab2["kept_ids"] = list(reversed(vocab["kept_ids"]))
        student2 = TinyTransformer(cfg)
        try:
            dist.train(
                student2, pb, sb, vocab2, "fake-parent", out,
                epochs=2, batch_size=2, accum=2, log_every=1000,
                resume=final, device=torch.device("cpu"),
            )
            raise AssertionError("aurait dû refuser le vocab incompatible")
        except ValueError as e:
            assert "incompatible" in str(e), e
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real


def test_lr_resume_applique(tmp_path):
    """--lr-resume remplace bien le LR de l'optimiseur à la reprise."""
    import transformers

    p_full, vocab, pb, sb, cfg = _petit_setup()
    student = TinyTransformer(cfg)
    real = transformers.AutoModelForCausalLM.from_pretrained

    @classmethod
    def fake_from_pretrained(cls, *a, **k):
        return FakeParent(p_full)

    transformers.AutoModelForCausalLM.from_pretrained = fake_from_pretrained
    try:
        out = str(tmp_path / "enfant.pt")
        final = dist.train(
            student, pb, sb, vocab, "fake-parent", out,
            epochs=1, batch_size=2, accum=2, log_every=1000,
            device=torch.device("cpu"),
        )
        student2 = TinyTransformer(cfg)
        final2 = dist.train(
            student2, pb, sb, vocab, "fake-parent", out,
            epochs=1, batch_size=2, accum=2, log_every=1000,
            resume=final, lr_resume=1e-5, device=torch.device("cpu"),
        )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real
    ckpt = torch.load(final2, map_location="cpu", weights_only=False)
    assert ckpt["opt"]["param_groups"][0]["lr"] == 1e-5


def test_resume_depuis_bestval(tmp_path):
    """Reprise depuis un _bestval (poids seuls) : nouvel optimiseur, pas de KeyError."""
    import transformers

    p_full, vocab, pb, sb, cfg = _petit_setup()
    student = TinyTransformer(cfg)
    real = transformers.AutoModelForCausalLM.from_pretrained

    @classmethod
    def fake_from_pretrained(cls, *a, **k):
        return FakeParent(p_full)

    transformers.AutoModelForCausalLM.from_pretrained = fake_from_pretrained
    try:
        out = str(tmp_path / "enfant.pt")
        vstream = torch.randint(0, p_full, (256,)).tolist()
        vpb = make_batches(vstream, 16, 4, shuffle=False, seed=0)
        vsb = make_batches([t % 64 for t in vstream], 16, 4, shuffle=False, seed=0)
        dist.train(
            student, pb, sb, vocab, "fake-parent", out,
            epochs=2, batch_size=2, accum=2, log_every=1000,
            eval_every=1, patience=0,
            val_parent_batches=vpb, val_student_batches=vsb,
            device=torch.device("cpu"),
        )
        best = out.replace(".pt", "_bestval.pt")
        assert os.path.exists(best)
        student2 = TinyTransformer(cfg)
        final = dist.train(
            student2, pb, sb, vocab, "fake-parent", out,
            epochs=10, batch_size=2, accum=2, log_every=1000,
            resume=best, lr_resume=1e-4, rewarmup_ratio=0.05,
            device=torch.device("cpu"),
        )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real
    assert os.path.exists(final)
