#!/bin/bash
# Vérification de reproductibilité v1/v2 (hash des JSONL régénérés). Jetable.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
$V tools/make_qa_corpus.py --version v1 \
  --out-jsonl data/qa/_repro_v1.jsonl \
  --out-manifest data/qa/_repro_v1.manifest.json > logs/repro_v1.log 2>&1
echo "v1_exit=$?" >> logs/repro_v1.log
$V tools/make_qa_corpus.py --version v2 \
  --out-jsonl data/qa/_repro_v2.jsonl \
  --out-manifest data/qa/_repro_v2.manifest.json > logs/repro_v2.log 2>&1
echo "v2_exit=$?" >> logs/repro_v2.log
$V - <<'PY' > logs/repro_hash.txt 2>&1
import hashlib, json
for v in ("v1", "v2"):
    p = f"data/qa/_repro_{v}.jsonl"
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    ref = json.load(open(f"data/qa/qa_fr_pilot_{v}.manifest.json"))["sha256"]
    print(v, h, "OK" if h == ref else f"DIFF != {ref}")
PY
date > logs/repro.done
