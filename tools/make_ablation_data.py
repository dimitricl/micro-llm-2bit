"""Construit le subset 5M tokens + vocab partagé pour les ablations D.

- Subset : premiers docs du split train slice_00 (seed 0) jusqu'à ~5,2M
  tokens (fenêtres 128 + eos/doc). Écrit runs/abl_data/train_5M.jsonl.
- Val commune : docs val du même split -> runs/abl_data/val.jsonl.
- Vocab partagé (unk distinct, nouveau format) construit sur le subset,
  taille 12000 -> runs/abl_data/vocab.json. Réutilisé via --vocab-from.
- Jeux communs 128/256 : fenêtres + mapping, pour l'éval post-hoc.
"""
import json, sys
sys.path.insert(0, "/Volumes/Lexar/micro-llm-2bit")
import os
import data as D
from transformers import AutoTokenizer

os.makedirs("/Volumes/Lexar/micro-llm-2bit/runs/abl_data", exist_ok=True)
ROOT = "/Volumes/Lexar/micro-llm-2bit/runs/abl_data"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M",
                                    trust_remote_code=True)
texts = D.load_texts("/Volumes/Lexar/clean-corpus/slices/slice_00.jsonl")
train_docs, val_docs = D.train_val_split(texts, 0.05, 0)
print(f"[abl] train={len(train_docs)} val={len(val_docs)}", flush=True)

# Subset ~5,2M tokens (fenêtres 128).
SEQ = 128
need = 5_200_000
got, sub = 0, []
for t in train_docs:
    n = len(tok.encode(t, add_special_tokens=False)) + 1
    sub.append(t)
    got += n
    if got >= need:
        break
print(f"[abl] subset: {len(sub)} docs, ~{got} tokens", flush=True)
with open(f"{ROOT}/train_5M.jsonl", "w") as f:
    for t in sub:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
with open(f"{ROOT}/val.jsonl", "w") as f:
    for t in val_docs:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

vocab = D.build_reduced_vocab(tok, sub, 12000)
assert vocab.get("unk_distinct")
with open(f"{ROOT}/vocab.json", "w") as f:
    json.dump(vocab, f)
print(f"[abl] vocab: {len(vocab['kept_ids'])} tokens, "
      f"unk={vocab['unk_new']} eos={vocab['eos_new']}", flush=True)

# Jeux communs : windows 128 et 256 mappées + chars pour BPC.
new_of = {o: n for n, o in enumerate(vocab["kept_ids"]) if o >= 0}
val_parent = []
for t in val_docs:
    val_parent += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
val_student = [new_of.get(t, vocab["unk_new"]) for t in val_parent]
unk_frac = sum(1 for t in val_student if t == vocab["unk_new"]) / len(val_student)
print(f"[abl] val: {len(val_parent)} tokens, part unk={unk_frac*100:.2f}%", flush=True)
import torch
chars_val = sum(len(t) for t in val_docs)
for seq in (128, 256):
    wins = [val_student[i:i + seq + 1]
            for i in range(0, len(val_student) - seq, seq)]
    wins = [w for w in wins if len(w) == seq + 1]
    torch.save(torch.tensor(wins, dtype=torch.long), f"{ROOT}/val_{seq}.pt")
    print(f"[abl] jeu val_{seq}: {len(wins)} fenetres", flush=True)
json.dump({"chars_val": chars_val, "ntok_val_parent": len(val_parent)},
          open(f"{ROOT}/meta.json", "w"))
print("[abl] TERMINE", flush=True)
