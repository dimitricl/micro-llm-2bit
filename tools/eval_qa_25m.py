#!/usr/bin/env python3
"""Banc QA-passage 0-shot pour le modèle 25M from-scratch (MLX).

Métriques identiques à eval_qa_passage_G.py : EM strict normalisé,
EM « contient », F1 token, copie (sortie ⊆ passage), refus.
Génération greedy, pénalité 1.15, no-repeat 3 (règles G).

Attention : ce modèle n'a jamais vu le format QA (pré-entraînement brut) ;
c'est donc une mesure 0-shot, sans EOS (le tokenizer n'en a pas).

Usage :
  .venv/bin/python tools/eval_qa_25m.py \
      --items data/qa/g_passage_test_v1.jsonl \
      --out docs/results/g5_25m/qa_0shot.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
from tokenizers import Tokenizer

from train_25m import MicroLLM
from generate_25m import banned


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = re.sub(r"^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$", "", s, flags=re.U)
    return re.sub(r"\s+", " ", s)


def tokens(s: str) -> list[str]:
    return re.findall(r"[0-9a-zàâäéèêëîïôöùûüç]+", norm(s))


def f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    cp, cr = Counter(p), Counter(r)
    common = sum((cp & cr).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def contains_answer(pred: str, answer: str) -> bool:
    return f" {norm(answer)} " in f" {norm(pred)} "


def is_refusal(pred: str) -> bool:
    p = norm(pred)
    return (not p) or any(k in p for k in ("je ne sais pas", "impossible",
                                           "je ne peux pas", "aucune idée"))


def bootstrap_ci(vals: list[int], n: int = 2000, seed: int = 0) -> list[float]:
    import numpy as np
    if not vals:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=np.float64)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    means = a[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def generate(model: MicroLLM, tok: Tokenizer, prompt: str, max_new: int,
             rep_pen: float, no_rep: int) -> tuple[str, int]:
    ids = tok.encode(prompt).ids
    out = list(ids)
    new: list[int] = []
    for _ in range(max_new):
        x = mx.array([out[-1024:]])
        logits = model(x)[0, -1, :].astype(mx.float32)
        if rep_pen != 1.0:
            for t in set(out[-64:]):
                logits[t] = logits[t] / rep_pen if logits[t] > 0 else logits[t] * rep_pen
        ban = banned(out, no_rep) if no_rep >= 2 else set()
        if ban:
            logits[mx.array(sorted(ban))] = -1e9
        nxt = int(mx.argmax(logits).item())
        out.append(nxt)
        new.append(nxt)
    return tok.decode(new).strip(), len(new)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/qa/g_passage_test_v1.jsonl")
    ap.add_argument("--ckpt", default="models/llm_25m_fr/model.safetensors")
    ap.add_argument("--tokenizer", default="models/tokenizer_fr_16k")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--no-stop-newline", action="store_true",
                    help="ne pas couper au premier saut de ligne (le modèle "
                         "QA apprend '\\n\\n' comme borne, faute d'EOS)")
    ap.add_argument("--quant-bits", type=int, default=0,
                    help="0 = fp32 ; 4/8 = export quantifié")
    ap.add_argument("--quant-group", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
    tok.no_truncation()
    if args.quant_bits:
        from export_25m import QuantLLM
        import mlx.nn as nn
        model = QuantLLM(vocab_size=tok.get_vocab_size())
        nn.quantize(model, group_size=args.quant_group, bits=args.quant_bits)
    else:
        model = MicroLLM(vocab_size=tok.get_vocab_size())
    model.load_weights(args.ckpt)
    model.eval()
    mx.eval(model.parameters())

    rows = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        pred, n_new = generate(model, tok, it["prompt"], args.max_new,
                               args.rep_penalty, args.no_repeat_ngram)
        pred_raw = pred
        stopped = False
        if not args.no_stop_newline and "\n" in pred:
            pred = pred.split("\n")[0].strip()
            stopped = True
        rows.append({
            "id": it["id"], "source": it["source"], "type": it["type"],
            "answer": it["answer"], "pred": pred, "pred_raw": pred_raw,
            "stopped": stopped, "n_new": n_new,
            "em": norm(pred) == norm(it["answer"]),
            "em_contains": contains_answer(pred, it["answer"]),
            "f1": f1(pred, it["answer"]),
            "copy": bool(norm(pred) and norm(pred) in norm(it["passage"])),
            "refusal": is_refusal(pred),
        })
        if i % 25 == 0 or i == len(items):
            em = sum(r["em"] for r in rows) / len(rows)
            print(f"[qa] {i}/{len(items)} EM={em:.1%} ({time.time()-t0:.0f}s)",
                  flush=True)

    n = len(rows)
    em = [int(r["em"]) for r in rows]
    summary = {
        "model": "llm_25m_fr",
        "n": n,
        "em": sum(em) / n,
        "em_ci95": bootstrap_ci(em),
        "em_contains": sum(r["em_contains"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "copy": sum(r["copy"] for r in rows) / n,
        "refusal": sum(r["refusal"] for r in rows) / n,
        "stopped_nl": sum(r["stopped"] for r in rows) / n,
        "elapsed_s": round(time.time() - t0, 1),
        "by_source": {
            s: {"n": sum(1 for r in rows if r["source"] == s),
                "em": sum(r["em"] for r in rows if r["source"] == s)
                      / max(1, sum(1 for r in rows if r["source"] == s))}
            for s in sorted({r["source"] for r in rows})
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1, ensure_ascii=False)
    print("[qa] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
