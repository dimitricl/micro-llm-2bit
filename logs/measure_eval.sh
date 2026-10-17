#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
.venv/bin/python - > logs/measure_eval.log 2>&1 <<'PYEOF'
import sys, time
sys.path.insert(0, '.')
import torch
from model import TinyTransformer
from distill import eval_loss
import data as D
from main import load_tokenizer
import json
t0 = time.time()
ckpt = torch.load('runs/E-t01/model.pt', map_location='cpu', weights_only=False)
model = TinyTransformer(ckpt['cfg'])
model.load_state_dict(ckpt['model'])
dev = torch.device('mps')
model.to(dev).eval()
vocab = json.load(open('runs/2026-10-08-tranche00/tranche00.pt.vocab.json'))
tok = load_tokenizer(vocab['parent'])
texts = D.load_texts('runs/abl_data/val_E_frozen.jsonl')
print(f"val docs: {len(texts)}", flush=True)
stream = []
for t in texts:
    stream += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
new_of = {o: n for n, o in enumerate(vocab['kept_ids'])}
ss = [new_of.get(i, vocab['unk_new']) for i in stream]
B = [torch.tensor(ss[i:i+8*129], dtype=torch.long).view(8,129) for i in range(0, len(ss)-128, 8*129)]
B = [b for b in B if b.shape == (8,129)]
print(f"val batches: {len(B)}", flush=True)
t1 = time.time()
ce = eval_loss(model, B, dev)
t2 = time.time()
print(f"eval_loss: ce={ce:.4f} en {t2-t1:.0f}s (prep {t1-t0:.0f}s)", flush=True)
print("DONE", flush=True)
PYEOF
echo $? > logs/measure_eval.exit
date > logs/measure_eval.done
