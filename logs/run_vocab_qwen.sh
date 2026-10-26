#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nice -n 10 .venv/bin/python - > logs/vocab_qwen.log 2>&1 <<'PYEOF'
import sys, json, hashlib, time
sys.path.insert(0, '.')
import data as D
from main import load_tokenizer
t0 = time.time()
tok = load_tokenizer('Qwen/Qwen3-0.6B-Base')
texts = D.load_texts('runs/abl_data/resume_5M.jsonl')
print(f"texts={len(texts)}", flush=True)
vocab = D.build_reduced_vocab(tok, texts, 12000)
with open('runs/abl_data/vocab_qwen.json', 'w') as f:
    json.dump(vocab, f)
kept = vocab['kept_ids']
# coverage check on train split
tr, _ = D.train_val_split(texts, 0.05, 0)
keptset = set(k for k in kept if k >= 0)
tot, hit = 0, 0
for t in tr:
    ids = tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
    tot += len(ids); hit += sum(1 for i in ids if i in keptset)
man = {
  "parent": vocab.get("parent"),
  "vocab_size": len(kept),
  "unk_new": vocab.get("unk_new"),
  "eos_new": vocab.get("eos_new"),
  "unk_distinct": vocab.get("unk_distinct"),
  "vocab_version": vocab.get("vocab_version"),
  "n_texts": len(texts),
  "train_ids": tot,
  "train_covered": hit,
  "train_coverage": hit / max(tot, 1),
  "sha1_texts": hashlib.sha1("".join(texts).encode()).hexdigest(),
  "build_s": round(time.time() - t0, 1),
}
with open('docs/results/vocab_qwen_manifest.json', 'w') as f:
    json.dump(man, f, indent=1)
print("MANIFEST: " + json.dumps(man), flush=True)
print("DONE", flush=True)
PYEOF
echo $? > logs/vocab_qwen.exit
date > logs/vocab_qwen.done
