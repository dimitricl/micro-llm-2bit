#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nice -n 10 .venv/bin/python - > logs/parents_samp.log 2>&1 <<'PYEOF'
import sys, json, statistics
sys.path.insert(0, '.')
import torch
from tools.eval_fixed import PROMPTS, metriques
from transformers import AutoModelForCausalLM, AutoTokenizer
for name, tag in [("HuggingFaceTB/SmolLM-135M", "smol135"), ("HuggingFaceTB/SmolLM-360M", "smol360")]:
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cpu().eval()
    rng = torch.Generator().manual_seed(1234)
    reps, bous = [], []
    with torch.no_grad():
        for cat, prompt in PROMPTS:
            ids = tok.encode(prompt, add_special_tokens=False)
            x = torch.tensor([ids])
            out = model.generate(x, max_new_tokens=60, do_sample=True, temperature=0.7,
                                 top_k=40, top_p=0.9, eos_token_id=tok.eos_token_id)
            gen = out[0][len(ids):].tolist()
            txt = tok.decode(gen, skip_special_tokens=True)
            m = metriques(gen, txt)
            reps.append(m["taux_repetition_4g"]); bous.append(m["plus_longue_boucle"])
    print(f"{tag} sampling: repet={statistics.mean(reps):.4f} max={max(reps):.4f} boucle={statistics.mean(bous):.1f}", flush=True)
    del model
print("DONE", flush=True)
PYEOF
echo $? > logs/parents_samp.exit
date > logs/parents_samp.done
