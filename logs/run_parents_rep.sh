#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nice -n 10 .venv/bin/python - > logs/parents_rep.log 2>&1 <<'PYEOF'
import sys, json, statistics
sys.path.insert(0, '.')
import torch
from tools.eval_fixed import PROMPTS, metriques
from transformers import AutoModelForCausalLM, AutoTokenizer
for name, tag in [("HuggingFaceTB/SmolLM-135M", "smol135"), ("HuggingFaceTB/SmolLM-360M", "smol360")]:
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cpu().eval()
    reps, bous, diss = [], [], []
    with torch.no_grad():
        for cat, prompt in PROMPTS:
            ids = tok.encode(prompt, add_special_tokens=False)
            x = torch.tensor([ids])
            out = model.generate(x, max_new_tokens=60, do_sample=False,
                                 eos_token_id=tok.eos_token_id)
            gen = out[0][len(ids):].tolist()
            txt = tok.decode(gen, skip_special_tokens=True)
            m = metriques(gen, txt)
            reps.append(m["taux_repetition_4g"]); bous.append(m["plus_longue_boucle"]); diss.append(m["mots_distincts"])
    print(f"{tag}: repet4g moy={statistics.mean(reps):.4f} max={max(reps):.4f} boucle moy={statistics.mean(bous):.1f} max={max(bous)} distincts moy={statistics.mean(diss):.1f}", flush=True)
    json.dump({"rep": reps, "bou": bous, "dis": diss}, open(f"docs/results/parents_rep_{tag}.json", "w"))
    del model
print("DONE", flush=True)
PYEOF
echo $? > logs/parents_rep.exit
date > logs/parents_rep.done
