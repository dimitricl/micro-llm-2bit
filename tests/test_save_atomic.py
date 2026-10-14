"""F0b : staging checkpoints — écriture interrompue => pas de .pt tronqué."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_save_atomic_ok(tmp_path):
    from distill import save_atomic
    p = str(tmp_path / "m.pt")
    save_atomic({"a": 1}, p)
    import torch
    assert torch.load(p, weights_only=False) == {"a": 1}


def test_save_atomic_eintr_puis_succes(tmp_path, monkeypatch):
    """EINTR 2× puis succès : contenu final intact, pas de résidu."""
    import errno
    import torch
    from distill import save_atomic
    calls = {"n": 0}
    real_save = torch.save

    def flaky(obj, f, *a, **k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError(errno.EINTR, "Interrupted system call")
        return real_save(obj, f, *a, **k)

    monkeypatch.setattr(torch, "save", flaky)
    p = str(tmp_path / "m.pt")
    save_atomic({"a": 2}, p)
    assert torch.load(p, weights_only=False) == {"a": 2}
    assert calls["n"] == 3
    assert [f for f in os.listdir(str(tmp_path)) if f.startswith(".tmp")] == []


def test_save_atomic_echec_total_preserve_cible(tmp_path, monkeypatch):
    """Échec persistant : l'ancienne cible reste intacte, rien de tronqué."""
    import errno
    import torch
    from distill import save_atomic
    import pytest
    p = str(tmp_path / "m.pt")
    save_atomic({"a": 1}, p)

    def boom(obj, f, *a, **k):
        raise OSError(errno.EINTR, "Interrupted system call")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        save_atomic({"a": 2}, p, retries=2)
    assert torch.load(p, weights_only=False) == {"a": 1}
