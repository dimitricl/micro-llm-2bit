"""Tests B4 : reprise robuste mid-epoch à trajectoire bit-identique.

Protocole : run continu (1 epoch) vs run interrompu (reprise depuis le
checkpoint périodique du step 6/8). Même total, même seed, mêmes données :
la séquence des loss et les poids finaux doivent être bit-identiques
(calcul CPU déterministe à ordre d'ops égal).
NOTE : la garantie ne vaut qu'à hyperparamètres identiques (même total de
steps -> même scheduler). Une reprise avec epochs différents (enchaînement
de tranches + --rewarmup-ratio) change volontairement le scheduler.
Parent factice, CPU.
"""

import dataclasses
import glob
import os
import re
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import distill as dist
from data import make_batches
from model import TINY, TinyTransformer

LOSS_RE = re.compile(r"\[train\] step \d+/\d+ loss=([0-9.]+)")


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


def _data():
    torch.manual_seed(7)
    stream = torch.randint(0, 256, (32 * 16 + 1,)).tolist()
    pb = make_batches(stream, 16, 4, shuffle=False, seed=7)
    sb = make_batches([t % 64 for t in stream], 16, 4, shuffle=False, seed=7)
    vocab = {"parent": "fake", "vocab_size": 64, "kept_ids": list(range(64)),
             "unk_new": 0, "eos_new": 1}
    return pb, sb, vocab


def _student():
    cfg = dataclasses.replace(
        TINY, vocab_size=64, d_model=64, n_layers=2, n_heads_q=4,
        n_heads_kv=2, ffn_dim=128, seq_max=32, group_size=32,
    )
    return TinyTransformer(cfg)


def _run(out, pb, sb, vocab, epochs, seed, ckpt_every, resume="", capsys=None):
    import transformers

    torch.manual_seed(0)  # même init enfant + parent à chaque run
    student = _student()
    real_pre = transformers.AutoModelForCausalLM.from_pretrained

    @classmethod
    def fake_from_pretrained(cls, *a, **k):
        return FakeParent(256)

    transformers.AutoModelForCausalLM.from_pretrained = fake_from_pretrained
    try:
        final = dist.train(
            student, pb, sb, vocab, "fake-parent", out,
            epochs=epochs, batch_size=2, accum=2, log_every=1,
            seed=seed, ckpt_every=ckpt_every, resume=resume,
            device=torch.device("cpu"),
        )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = real_pre
    losses = ([float(m) for m in LOSS_RE.findall(capsys.readouterr().out)]
              if capsys is not None else [])
    ckpt = torch.load(final, map_location="cpu", weights_only=False)
    return final, ckpt, losses


def test_reprise_mid_epoch_meme_trajectoire(tmp_path, capsys):
    # 32 fenêtres, save 4, bs 2, accum 2 -> 16 micros, 8 steps/epoch.
    # Les deux runs ont le MÊME total (1 epoch = 8 steps) : même scheduler,
    # même ordre (seed), mêmes données. La reprise doit être bit-identique.
    pb, sb, vocab = _data()
    final1, ckpt1, l1 = _run(str(tmp_path / "r1_e.pt"), pb, sb, vocab,
                             epochs=1, seed=123, ckpt_every=0, capsys=capsys)
    assert ckpt1["step"] == 8 and len(l1) == 8

    _, ckpt_a, l_a = _run(str(tmp_path / "r2a_e.pt"), pb, sb, vocab,
                          epochs=1, seed=123, ckpt_every=2, capsys=capsys)
    assert len(l_a) == 8
    # Seulement les 2 derniers checkpoints périodiques sont gardés
    # (saves aux steps 2,4,6 ; pas de périodique au step 8 = fin du run,
    # couverte par la sauvegarde finale).
    ckpts = sorted(glob.glob(str(tmp_path / "r2a_e_ckpt*.pt")))
    assert len(ckpts) == 2, ckpts
    last = ckpts[-1]
    c = torch.load(last, map_location="cpu", weights_only=False)
    # Vraie reprise mid-epoch (step 6/8).
    assert c["step"] == 6
    for k in ("epoch", "micro", "order", "rng", "opt", "sched", "model"):
        assert k in c, k
    assert c["epoch"] == 0 and c["micro"] == 12 and len(c["order"]) == len(sb)

    final_b, ckpt_b, l_b = _run(str(tmp_path / "r2b_e.pt"), pb, sb, vocab,
                                epochs=1, seed=123, ckpt_every=0,
                                resume=last, capsys=capsys)
    assert ckpt_b["step"] == 8 and len(l_b) == 2  # steps 7-8

    # Trajectoire bit-identique : mêmes loss imprimées...
    assert l_a[:6] + l_b == l1
    # ...et mêmes poids finaux (comparaison exacte, pas d'arrondi).
    m1 = torch.load(final1, map_location="cpu", weights_only=False)["model"]
    mb = torch.load(final_b, map_location="cpu", weights_only=False)["model"]
    assert m1.keys() == mb.keys()
    for k in m1:
        assert torch.equal(m1[k], mb[k]), k


def test_ordre_deterministe_par_seed(tmp_path, capsys):
    """Même seed -> même ordre de batchs (mêmes loss) ; seed != -> ordre !=."""
    pb, sb, vocab = _data()
    _, _, l1 = _run(str(tmp_path / "s1_e.pt"), pb, sb, vocab,
                    epochs=1, seed=5, ckpt_every=0, capsys=capsys)
    _, _, l2 = _run(str(tmp_path / "s2_e.pt"), pb, sb, vocab,
                    epochs=1, seed=5, ckpt_every=0, capsys=capsys)
    _, _, l3 = _run(str(tmp_path / "s3_e.pt"), pb, sb, vocab,
                    epochs=1, seed=6, ckpt_every=0, capsys=capsys)
    assert l1 == l2  # bit-identique : même seed, mêmes ops
    assert l1 != l3  # un seed différent mélange différemment
