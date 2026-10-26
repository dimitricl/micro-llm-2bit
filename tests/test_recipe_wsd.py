"""Tests de la recette F : WSD, flips ternaires, WD programmable.

Ces tests ne remplacent pas les 171 tests existants ; ils verrouillent les
ajouts de l'étape 4 (scheduler WSD, comptage de flips, weight decay -> 0).
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import distill as dist
from data import make_batches
from model import TINY, TinyTransformer, TernaryLinear


def _opt(lr=1.0):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([p], lr=lr)


def test_wsd_plateau_puis_decroissance():
    opt = _opt(lr=1.0)
    sched = dist.wsd_with_warmup(opt, warmup_steps=5, total_steps=100,
                                 decay_frac=0.2, min_ratio=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert abs(lrs[0] - 0.2) < 1e-6          # (0+1)/5
    assert abs(lrs[4] - 1.0) < 1e-6          # fin de warmup
    assert abs(lrs[50] - 1.0) < 1e-6         # plateau
    assert lrs[79] > 0.99                     # encore en plateau à 79
    assert abs(lrs[80] - 1.0) < 1e-6          # début de décroissance
    assert lrs[81] < 1.0                      # décroissance entamée
    assert 0.1 - 1e-6 <= lrs[-1] < 0.12       # proche du LR final
    # Monotone après le warmup.
    assert all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(4, 99))


def test_wsd_warmup_zero_ne_depasse_pas_1():
    opt = _opt(lr=1.0)
    sched = dist.wsd_with_warmup(opt, warmup_steps=0, total_steps=10,
                                 decay_frac=0.5, min_ratio=0.0)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert max(lrs) <= 1.0 + 1e-9
    assert abs(lrs[0] - 1.0) < 1e-6


def test_ternary_snapshot_et_flips():
    cfg = dataclasses.replace(
        TINY, vocab_size=64, d_model=64, n_layers=2, n_heads_q=4,
        n_heads_kv=2, ffn_dim=128, seq_max=32, group_size=32)
    model = TinyTransformer(cfg)
    snap = dist.ternary_snapshot(model)
    total = sum(int(q.numel()) for q in snap.values())
    assert total > 0
    # Aucun flip sans modification.
    assert dist.ternary_flips(model, snap) == (0, total)
    # Force un poids latent très positif : au moins un flip ternaire.
    lin = next(m for m in model.modules() if isinstance(m, TernaryLinear))
    with torch.no_grad():
        lin.weight.view(-1)[0] = 100.0
    flipped, n = dist.ternary_flips(model, snap)
    assert n == total and flipped >= 1


def test_train_wsd_wd_flips_smoke(tmp_path):
    """Bout-en-bout alpha=0 : WSD + WD 0,1 -> 0 + flips, sans parent."""
    torch.manual_seed(0)
    v = 64
    vocab = {"parent": "fake", "vocab_size": v, "kept_ids": list(range(v)),
             "unk_new": 0, "eos_new": 1}
    stream = torch.randint(0, v, (512,)).tolist()
    sb = make_batches(stream, 16, 4, shuffle=True, seed=0)
    vstream = torch.randint(0, v, (256,)).tolist()
    vsb = make_batches(vstream, 16, 4, shuffle=False, seed=0)
    cfg = dataclasses.replace(
        TINY, vocab_size=v, d_model=64, n_layers=2, n_heads_q=4,
        n_heads_kv=2, ffn_dim=128, seq_max=32, group_size=32)
    student = TinyTransformer(cfg)
    out = str(tmp_path / "enfant.pt")
    final = dist.train(
        student, sb, sb, vocab, "fake-parent", out,
        epochs=1, batch_size=2, accum=2, alpha=0.0, lr=1e-3,
        sched_kind="wsd", wsd_warmup_frac=0.2, wsd_decay_frac=0.2,
        weight_decay=0.1, wd_zero_frac=0.5, track_flips=True,
        eval_every=1, val_parent_batches=sb, val_student_batches=vsb,
        log_every=1000, device=torch.device("cpu"))
    assert os.path.exists(final)
    ckpt = torch.load(final, map_location="cpu", weights_only=False)
    assert ckpt["step"] > 0
    # Le WD final est bien 0 dans l'état optimiseur sauvegardé.
    assert ckpt["opt"]["param_groups"][0]["weight_decay"] == 0.0
