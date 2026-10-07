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
