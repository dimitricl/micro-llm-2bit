"""Tests B5 : pick_device() (cuda > mps > cpu) + fp16 parent sur cuda.

Aucun changement de comportement sur mps/cpu. Les tests d'ordre cuda par
mock ne touchent pas au matériel ; le test d'intégration cuda réel est
ignoré si cuda est absent.
"""

import os
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import distill as dist


def test_pick_device_ordre_cuda_dabord():
    """cuda dispo -> cuda, même si mps est dispo."""
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.backends.mps, "is_available", return_value=True):
        assert dist.pick_device() == torch.device("cuda")


def test_pick_device_mps_sans_cuda():
    with patch.object(torch.cuda, "is_available", return_value=False), \
         patch.object(torch.backends.mps, "is_available", return_value=True):
        assert dist.pick_device() == torch.device("mps")


def test_pick_device_cpu_sans_accelerateur():
    with patch.object(torch.cuda, "is_available", return_value=False), \
         patch.object(torch.backends.mps, "is_available", return_value=False):
        assert dist.pick_device() == torch.device("cpu")


def test_pick_device_reel_sans_cuda_sur_mac():
    """Sur cette machine (pas de cuda) : mps ou cpu, jamais d'exception."""
    d = dist.pick_device()
    assert d.type in ("mps", "cpu")


def test_parent_dtype():
    """fp16 sur mps ET cuda, fp32 sur cpu (objet device seul, sans matériel)."""
    assert dist.parent_dtype(torch.device("cpu")) == torch.float32
    assert dist.parent_dtype(torch.device("mps")) == torch.float16
    assert dist.parent_dtype(torch.device("cuda")) == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda absent")
def test_pick_device_reel_cuda():
    assert dist.pick_device().type == "cuda"
