#!/usr/bin/env python3
"""Prépare les données QA G1 (181 979 exemples) pour le fine-tuning MLX 46M.

Chaque exemple : prompt + " " + réponse + "\\n\\n", packé dans des fenêtres
de seq_len+1 tokens ; le masque vaut 1 uniquement sur les tokens de réponse
et le séparateur. Le tokenizer 16k n'a pas d'EOS : "\\n\\n" sert de borne.

Sortie : .npz avec `ids` (uint16, N×(seq_len+1)) et `mask` (uint8, même
forme). Usage :
  .venv/bin/python tools/prepare_qa_mlx.py --field prompt \
      --out runs/G5-qa-data/qa_prompt_packed.npz
  .venv/bin/python tools/prepare_qa_mlx.py --field control_prompt \
      --out runs/G5-qa-data/qa_ctrl_packed.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from tokenizers import Tokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="runs/G1-data/qa_train.jsonl")
    ap.add_argument("--tokenizer", default="models/tokenizer_fr_16k")
    ap.add_argument("--field", default="prompt", choices=["prompt", "control_prompt"])
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
    tok.no_truncation()
    sep = tok.encode("\n\n").ids
    print(f"[prep] sep={sep} ({len(sep)} tokens), field={args.field}", flush=True)

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    T = args.seq_len + 1
    ids_out: list[list[int]] = []
    mask_out: list[list[int]] = []
    cur: list[int] = []
    curm: list[int] = []
    n_answer = 0
    skipped = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal cur, curm
        if not cur:
            return
        pad = T - len(cur)
        ids_out.append(cur + [sep[0]] * pad)
        mask_out.append(curm + [0] * pad)
        cur, curm = [], []

    for i, ex in enumerate(rows):
        prompt = ex.get(args.field, "")
        answer = ex.get("answer", "")
        if not prompt or not answer:
            skipped += 1
            continue
        p = tok.encode(prompt).ids
        a = tok.encode(" " + answer).ids
        seq = p + a + sep
        if len(seq) > T:
            skipped += 1
            continue
        if cur and len(cur) + len(seq) > T:
            flush()
        cur.extend(seq)
        curm.extend([0] * len(p) + [1] * (len(a) + len(sep)))
        n_answer += len(a) + len(sep)
        if (i + 1) % 25000 == 0:
            print(f"[prep] {i+1}/{len(rows)} exemples, {len(ids_out)} fenêtres, "
                  f"{time.time()-t0:.0f}s", flush=True)
    flush()

    ids = np.asarray(ids_out, dtype=np.uint16)
    mask = np.asarray(mask_out, dtype=np.uint8)
    assert ids.shape == mask.shape and ids.shape[1] == T
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, ids=ids, mask=mask)
    print(f"[prep] {args.out}: {ids.shape[0]} fenêtres × {T} tokens, "
          f"{int(mask.sum()):,} tokens de réponse utiles, {skipped} exemples sautés, "
          f"{os.path.getsize(args.out)/1e6:.0f} Mo, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
