"""Tests subset R1/R2 : manifeste vs fichier (rapide, sans retokenizer)."""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "abl_data")


def test_resume_subset_manifeste():
    man = json.load(open(os.path.join(ROOT, "resume_5M_manifest.json"),
                         encoding="utf-8"))
    docs = []
    with open(os.path.join(ROOT, "resume_5M.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line).get("text", "")
                if t.strip():
                    docs.append(t)
    assert len(docs) == man["K"] == 905
    assert man["n_train"] + man["n_val"] == 905
    assert man["train_tokens"] >= 5_000_000
    assert man["plan"]["steps"] == 306
    assert man["epochs_cumul"] == 15 and man["remaining"] == 342
    assert hashlib.sha1("".join(docs).encode()).hexdigest() == man["sha1"]
