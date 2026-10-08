"""Tests cache top-k mmap, parent 135M et garde-fou loss non finie."""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import distill as dist
from data import make_batches


def _fake_parent(p_full=256, h=32):
    import types
    import torch.nn as nn

    class FakeParent(nn.Module):
        def __init__(self):
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

    return FakeParent()


def test_cache_roundtrip(tmp_path):
    """Build -> mmap -> reload : mêmes logits, refus si incompatible."""
    torch.manual_seed(1)
    p_full, v, k = 256, 64, 8
    kept = list(range(v))
    stream = torch.randint(0, p_full, (256,)).tolist()
    pb = make_batches(stream, 16, 4, shuffle=False, seed=0)
    sb = make_batches([t % v for t in stream], 16, 4, shuffle=False, seed=0)
    parent = _fake_parent(p_full)
    dev = torch.device("cpu")
    prefix = str(tmp_path / "cache")
    cache = dist.build_teacher_cache(parent, pb, sb, kept, k, dev,
                                     micro_batch=2, prefix=prefix)
    for ext in (".manifest.json", ".vals.fp16", ".idx.u16", ".tgt.i64"):
        assert os.path.exists(prefix + ext), ext
    m = cache["manifest"]
    assert m["k"] == k and m["micro_batch"] == 2
    assert m["parent"] == "fake-parent"
    # Lookup chunk (1, 0) == reconstruction manuelle directe.
    got = dist.teacher_logits_from_cache(cache, 1, 0, v, dev)
    assert got.shape == (2, 16, v)
    with torch.no_grad():
        ref = parent(input_ids=pb[1][:2][:, :-1]).logits.float()
        ref = ref.index_select(-1, torch.tensor(kept))
        rv, ri = ref.topk(k, dim=-1)
        dense = torch.full((2 * 16, v), float("-inf"))
        dense.scatter_(-1, ri.view(-1, k), rv.view(-1, k))
        dense = dense.view(2, 16, v)
    assert torch.equal(torch.isneginf(got), torch.isneginf(dense))
    assert torch.allclose(got[~torch.isneginf(got)],
                          dense[~torch.isneginf(dense)], atol=1e-3)
    # Rechargement + refus.
    cache2 = dist.load_teacher_cache(prefix, "fake-parent", kept, 16, k)
    assert cache2["manifest"]["n_chunks"] == m["n_chunks"]
    with pytest.raises(ValueError):
        dist.load_teacher_cache(prefix, "autre-parent", kept, 16, k)
    with pytest.raises(ValueError):
        dist.load_teacher_cache(prefix, "fake-parent", kept[::-1], 16, k)
    with pytest.raises(ValueError):
        dist.load_teacher_cache(prefix, "fake-parent", kept, 32, k)
    with pytest.raises(ValueError):
        dist.load_teacher_cache(prefix, "fake-parent", kept, 16, k + 1)
    with pytest.raises(KeyError):
        dist.teacher_logits_from_cache(cache, 999, 0, v, dev)


def test_tokenizer_135m_compatible():
    """SmolLM-135M partage le tokenizer du 360M (kept_ids transférables)."""
    try:
        from transformers import AutoTokenizer
        t135 = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M",
                                             trust_remote_code=True)
        t360 = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M",
                                             trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"tokenizers HF inaccessibles : {e}")
    assert t135.vocab_size == t360.vocab_size == 49152
    assert (t135.pad_token_id, t135.eos_token_id, t135.unk_token_id) == \
           (t360.pad_token_id, t360.eos_token_id, t360.unk_token_id)
    s = "Paris est la capitale du Japon."
    assert (t135.encode(s, add_special_tokens=False)
            == t360.encode(s, add_special_tokens=False))


def test_garde_fou_nonfinite(tmp_path):
    """Une loss NaN arrête proprement avec checkpoint de sauvegarde."""
    import dataclasses
    import torch.nn as nn

    from model import TINY, TinyTransformer

    torch.manual_seed(0)
    p_full, v = 256, 64
    kept = list(range(v))
    vocab = {"parent": "fake", "vocab_size": v, "kept_ids": kept,
             "unk_new": 0, "eos_new": 1}
    stream = torch.randint(0, p_full, (256,)).tolist()
    pb = make_batches(stream, 16, 4, shuffle=True, seed=0)
    sb = make_batches([t % v for t in stream], 16, 4, shuffle=True, seed=0)
    cfg = dataclasses.replace(TINY, vocab_size=v, d_model=64, n_layers=2,
                              n_heads_q=4, n_heads_kv=2, ffn_dim=128,
                              seq_max=32, group_size=32)
    student = TinyTransformer(cfg)

    import transformers
    real_loss, real_pre = (dist.DistillationLoss,
                           transformers.AutoModelForCausalLM.from_pretrained)

    class NanLoss(dist.DistillationLoss):
        def forward(self, s, t, y):
            nan = torch.tensor(float("nan"))
            return nan, {"kl": nan.detach(), "ce": nan.detach()}

    @classmethod
    def fake_from_pretrained(cls, *a, **k):
        parent = _fake_parent(p_full)
        return parent

    dist.DistillationLoss = NanLoss
    transformers.AutoModelForCausalLM.from_pretrained = fake_from_pretrained
    try:
        out = str(tmp_path / "enfant.pt")
        dist.train(student, pb, sb, vocab, "fake-parent", out,
                   epochs=1, batch_size=2, accum=2, log_every=1000,
                   device=torch.device("cpu"))
    finally:
        dist.DistillationLoss = real_loss
        transformers.AutoModelForCausalLM.from_pretrained = real_pre
    bad = out.replace(".pt", "_nonfinite.pt")
    assert os.path.exists(bad), "checkpoint de sauvegarde manquant"
    ckpt = torch.load(bad, map_location="cpu", weights_only=False)
    assert ckpt["reason"] == "loss non finie" and "model" in ckpt
