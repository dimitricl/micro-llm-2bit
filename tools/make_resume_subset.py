"""Construit resume_5M.jsonl (R1/R2) : premiers docs de slice_01 avec
TRAIN (split seed 0 / val_ratio 0.05 répliqué, cf. data.train_val_split)
>= 5,0M tokens. Sorties (ignorées par git, régénérables) :
  runs/abl_data/resume_5M.jsonl (K premiers docs, ordre d'origine)
  runs/abl_data/resume_5M_manifest.json (K, SHA1, tokens train, plan)
Le plan d'epoch est celui du TRAIN seul (le trainer réserve 5 % en val).
"""

import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

import data as D
from distill import plan_epoch
from main import load_tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "runs", "abl_data")
SLICE = "/Volumes/Lexar/clean-corpus/slices/slice_01.jsonl"
PREV_STEP = 4248


def main() -> None:
    tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")
    cands = [t for t in D.load_texts(SLICE) if t.strip()]
    K = None
    for k in range(835, 1200, 10):
        sub = cands[:k]
        tr, _ = D.train_val_split(sub, 0.05, 0)
        s = sum(len(tok.encode(t, add_special_tokens=False)) + 1 for t in tr)
        if s >= 5_000_000:
            K = k
            break
    assert K is not None, "subset introuvable"
    sub = cands[:K]
    tr, va = D.train_val_split(sub, 0.05, 0)
    stream = []
    for t in tr:
        stream += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
    wins = [stream[i:i + 129] for i in range(0, len(stream) - 1, 128)]
    wins = [w for w in wins if len(w) == 129]
    batches = [torch.tensor(wins[i:i + 8], dtype=torch.long)
               for i in range(0, len(wins), 8)]
    plan = plan_epoch(batches, batch_size=8, accum=16)
    E = math.ceil((PREV_STEP + plan["steps"]) / plan["steps"])
    with open(os.path.join(ROOT, "resume_5M.jsonl"), "w",
              encoding="utf-8") as f:
        for t in sub:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    man = {"K": K, "n_train": len(tr), "n_val": len(va),
           "train_tokens": plan["tokens"], "plan": plan,
           "epochs_cumul": E,
           "remaining": E * plan["steps"] - PREV_STEP,
           "sha1": hashlib.sha1(
               "".join(sub).encode("utf-8")).hexdigest()}
    json.dump(man, open(os.path.join(ROOT, "resume_5M_manifest.json"), "w"),
              indent=1)
    print(f"[subset] K={K} train={len(tr)} val={len(va)} "
          f"tokens={plan['tokens']} E={E} reste={man['remaining']}",
          flush=True)
    print("[subset] TERMINE", flush=True)


if __name__ == "__main__":
    main()
