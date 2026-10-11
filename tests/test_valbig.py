"""Tests G2 : disjonction val_big (hash) + bootstrap (déterministe)."""

import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.eval_valbig import bootstrap, variant_sums

ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "abl_data")


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


def test_valbig_disjointe_des_slices():
    """Les 300 docs val_big : hashes absents des slices 00-09."""
    manifest = json.load(open(os.path.join(ROOT, "val_big_manifest.json"),
                             encoding="utf-8"))
    docs = _docs(os.path.join(ROOT, "val_big.jsonl"))
    assert len(docs) == manifest["n_docs"] >= 300
    # Cohérence manifeste <-> fichiers.
    assert manifest["doc_hashes"] == [_h(t) for t in docs]
    assert manifest["chars"] == sum(len(t) for t in docs)
    assert manifest["source"].startswith("reste disjoint de")
    forbidden = set(json.load(open(os.path.join(ROOT, "forbidden_sha1.json"),
                                   encoding="utf-8")))
    assert manifest["forbidden_count"] == len(forbidden) == 99965
    assert not (set(manifest["doc_hashes"]) & forbidden)


def test_variant_sums_coherent():
    nll = np.array([1.0, 5.0])
    unk = np.array([False, True])
    nb = np.array([3.0, 2.0])
    nc = np.array([3.0, 2.0])
    s = variant_sums(nll, unk, nb, nc)
    assert s["nll_a"] == 6.0 and s["n_unk"] == 1 and s["n"] == 2
    assert s["nll_b"] >= s["nll_a"]  # plancher byte-uniforme
    assert s["nll_c"] == 1.0 and s["chars_c"] == 3.0


def test_bootstrap_deterministe_et_ordonne():
    rng = np.random.default_rng(0)
    per_doc = [{"nll_a": float(x), "nll_b": float(x) + 1.0,
                "nll_c": float(x), "chars_c": 100.0, "n": 50, "n_unk": 0}
               for x in rng.uniform(1.0, 5.0, size=40)]
    r1 = bootstrap(per_doc, chars_total=4000, n_draws=200, seed=7)
    r2 = bootstrap(per_doc, chars_total=4000, n_draws=200, seed=7)
    assert r1 == r2  # seed fixée -> identique
    for v in ("a", "b", "c"):
        lo, hi = r1[v]["bpc_ci"]
        assert lo <= r1[v]["bpc"] <= hi
        plo, phi = r1[v]["ppl_ci"]
        assert plo <= r1[v]["ppl"] <= phi
    r3 = bootstrap(per_doc, chars_total=4000, n_draws=200, seed=8)
    assert r3 != r1  # autre seed -> autre tirage (quasi sûrement)
