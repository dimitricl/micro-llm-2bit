#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nice -n 10 .venv/bin/python - > logs/plan_t01.log 2>&1 <<'PYEOF'
import sys, math, json
sys.path.insert(0, '.')
import data as D
from main import load_tokenizer
from distill import plan_epoch
import torch
tok = load_tokenizer('HuggingFaceTB/SmolLM-135M')
texts = D.load_texts('/Volumes/Lexar/clean-corpus/dedup/tranche_01.jsonl')
print('docs:', len(texts), flush=True)
stream = []
for t in texts:
    stream += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
W = [stream[i:i + 129] for i in range(0, len(stream) - 1, 128)]
W = [w for w in W if len(w) == 129]
B = [torch.tensor(W[i:i + 8], dtype=torch.long) for i in range(0, len(W), 8)]
plan = plan_epoch(B, batch_size=8, accum=16)
print('plan:', plan, flush=True)
PREV = 4248
E = (PREV + plan['steps'] + plan['steps'] - 1) // plan['steps']
print(f'E_CUMUL={E} total={E * plan["steps"]} restants={E * plan["steps"] - PREV}', flush=True)
print(f'ETA_restants_s~={ (E * plan["steps"] - PREV) * 16384 / 2500:.0f} (a 2500 tok/s)', flush=True)
print('PLAN-TERMINE', flush=True)
PYEOF
echo $? > logs/plan_t01.exit
date > logs/plan_t01.done
