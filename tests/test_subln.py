"""Tests V4 (subln) : normes supp désactivées par défaut, export refusé."""

import dataclasses
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import TINY, TinyTransformer


def _cfg(**kw):
    return dataclasses.replace(
        TINY, vocab_size=64, d_model=64, n_layers=2, n_heads_q=4,
        n_heads_kv=2, ffn_dim=128, seq_max=32, group_size=32, **kw)


def test_subln_desactive_par_defaut():
    """Sans subln : aucune norme supp, anciens checkpoints compatibles."""
    m = TinyTransformer(_cfg())
    for layer in m.layers:
        assert layer.attn.o_norm is None
        assert layer.mlp.down_norm is None
    # Ancien cfg picklé (sans l'attribut) : construction + forward OK.
    cfg_old = _cfg()
    del cfg_old.subln
    m_old = TinyTransformer(cfg_old)
    out = m_old(torch.randint(0, 64, (2, 8)))
    assert out.shape == (2, 8, 64)


def test_subln_actif_normes_et_gradients():
    m = TinyTransformer(_cfg(subln=True))
    for layer in m.layers:
        assert layer.attn.o_norm is not None
        assert layer.mlp.down_norm is not None
    x = torch.randint(0, 64, (2, 8))
    loss = m(x).float().sum()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    # Sortie différente du modèle sans subln (même seed d'init).
    torch.manual_seed(0)
    a = TinyTransformer(_cfg(subln=True))(x).detach()
    torch.manual_seed(0)
    b = TinyTransformer(_cfg())(x).detach()
    assert not torch.allclose(a, b)


def test_export_refuse_subln(tmp_path):
    """L'export .bin refuse un modèle subln (format inchangé)."""
    import main as M

    ckpt = {"cfg": _cfg(subln=True), "vocab": {"kept_ids": [0] * 64},
            "model": {}}
    path = str(tmp_path / "sub.pt")
    torch.save(ckpt, path)
    with pytest.raises(SystemExit, match="subln"):
        M.cmd_export(__import__("argparse").Namespace(ckpt=path, out="x.bin"))
