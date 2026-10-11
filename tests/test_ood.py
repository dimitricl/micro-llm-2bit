"""Tests H4 : jeu OOD (cohérence manifeste + disjonction + taille)."""

import hashlib
import json
import os
import re

import pytest

sys_path = os.path.join(os.path.dirname(__file__), "..")
import sys

sys.path.insert(0, sys_path)

ROOT = os.path.join(sys_path, "runs", "abl_data")


def _norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def _h(t):
    return hashlib.sha1(_norm(t).encode("utf-8")).hexdigest()


def _docs(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line).get("text", "")
                if t.strip():
                    out.append(t)
    return out


def test_ood_taille_et_manifeste():
    man = json.load(open(os.path.join(ROOT, "ood_manifest.json"),
                         encoding="utf-8"))
    docs = _docs(os.path.join(ROOT, "ood.jsonl"))
    assert len(docs) == man["n_docs"] >= 3
    assert man["chars"] == sum(len(t) for t in docs) >= 200_000
    assert man["docs"] == [_h(t) for t in docs]
    assert man["seed"] == 0
    for s in man["sources"]:
        assert {"titre", "auteur", "url", "page", "sha1_brut",
                "sha1_nettoye"} <= set(s)
        assert s["url"].startswith("https://www.gutenberg.org/")


def test_ood_disjoint_des_tranches_et_valbig():
    man = json.load(open(os.path.join(ROOT, "ood_manifest.json"),
                         encoding="utf-8"))
    inter = set()
    for i in range(10):
        for t in _docs(f"/Volumes/Lexar/clean-corpus/slices/slice_0{i}.jsonl"):
            inter.add(_h(t))
    for t in _docs("/Volumes/Lexar/clean-corpus/slice_100M.jsonl"):
        inter.add(_h(t))
    for t in _docs(os.path.join(ROOT, "val_big.jsonl")):
        inter.add(_h(t))
    assert not (set(man["docs"]) & inter)
