"""Tests pack/unpack 2 bits : aller-retour exact."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from quant import pack_ternary, unpack_ternary


def test_roundtrip_exact():
    torch.manual_seed(0)
    for n in [1, 3, 4, 5, 63, 64, 65, 1000]:
        q = torch.randint(-1, 2, (n,), dtype=torch.int8)
        packed, m = pack_ternary(q)
        assert m == n
        assert packed.shape == ((n + 3) // 4,)
        back = unpack_ternary(packed, m)
        assert torch.equal(back, q), f"échec aller-retour pour n={n}"


def test_codage_documente():
    # 0 -> 00, +1 -> 01, -1 -> 11 ; premier poids en bits faibles.
    q = torch.tensor([0, 1, -1, 0], dtype=torch.int8)
    packed, _ = pack_ternary(q)
    assert packed.numel() == 1
    assert int(packed[0]) == (0b00 | (0b01 << 2) | (0b11 << 4) | (0b00 << 6))


def test_code_reserve_decode_zero():
    # 0b10 réservé -> décodé comme 0 (robustesse).
    import torch as _t

    packed = _t.tensor([0b10], dtype=_t.uint8)
    assert unpack_ternary(packed, 1).tolist() == [0]


def test_valeurs_invalides_rejetees():
    with pytest.raises(ValueError):
        pack_ternary(torch.tensor([0, 2, -1], dtype=torch.int8))


def test_2d_et_bourrage():
    q = torch.ones(7, 8, dtype=torch.int8)
    packed, n = pack_ternary(q)
    assert n == 56
    assert packed.numel() == 14
    assert torch.equal(unpack_ternary(packed, n), torch.ones(56, dtype=torch.int8))
