"""Évalue un ou plusieurs exports .bin sur le banc QA-passage gelé.

Métriques : exact-match normalisé (strict), exact-match « contient la
réponse », F1 token, taux de copie (la sortie est dans le passage), refus.
Génération greedy, `--repetition-penalty 1.15 --no-repeat-ngram 3` par
défaut (règle de déploiement G).

Usage :
  .venv/bin/python tools/eval_qa_passage_G.py \
    --items data/qa/g_passage_test_v1.jsonl \
    --model E-t08=exports/E-t08.bin \
    --model F-B=exports/F-B.bin \
    --out docs/results/qa_passage_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import data as D
import engine as E

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
TOKENIZER = "HuggingFaceTB/SmolLM-135M"


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
    # F1 multiset simple.
    from collections import Counter
    cp, cr = Counter(p), Counter(r)
    common = sum((cp & cr).values())
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(r)
    return 2 * prec * rec / (prec + rec)


def contains_answer(pred: str, answer: str) -> bool:
    p = f" {norm(pred)} "
    a = f" {norm(answer)} "
    return a in p


def is_refusal(pred: str) -> bool:
    p = norm(pred)
    return (not p) or any(k in p for k in ("je ne sais pas", "impossible",
                                           "je ne peux pas", "aucune idée"))


def bootstrap_ci(vals: list[int], n: int = 2000, seed: int = 0) -> list[float]:
    if not vals:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=np.float64)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    means = a[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def load_items(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def eval_model(name: str, path: str, items: list[dict], tok, max_new: int,
               rep_pen: float, no_rep: int) -> dict:
    vocab_path = path + ".vocab.json"
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    eng = E.Engine(path)
    rows = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        ids = D.encode_mapped(tok, vocab, it["prompt"], add_eos=False)
        ids = ids[-eng.meta["seq_max"]:]
        out = eng.generate(
            ids, max_new=max_new, temperature=0.0, top_k=0, top_p=1.0,
            stop={vocab["eos_new"]},
            mask_id=vocab["unk_new"] if vocab.get("unk_distinct") else -1,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=no_rep,
        )
        pred = ""
        # Décodage identique à cmd_infer (sans importer main pour éviter
        # d'initialiser argparse) ; unk distinct -> marqueur.
        kept = vocab["kept_ids"]
        if vocab.get("unk_distinct"):
            parts, cur = [], []
            for t in out:
                if t == vocab["unk_new"] or not (0 <= t < len(kept)):
                    if cur:
                        parts.append(tok.decode([kept[x] for x in cur],
                                                 skip_special_tokens=True))
                        cur = []
                else:
                    cur.append(t)
            if cur:
                parts.append(tok.decode([kept[x] for x in cur],
                                         skip_special_tokens=True))
            pred = "".join(parts)
        else:
            pred = tok.decode(
                [kept[t] if 0 <= t < len(kept) else kept[vocab["unk_new"]]
                 for t in out], skip_special_tokens=True)
        em = norm(pred) == norm(it["answer"])
        row = {
            "id": it["id"], "source": it["source"], "type": it["type"],
            "answer": it["answer"], "prompt": it["prompt"], "pred": pred,
            "raw_ids": out, "em": bool(em),
            "em_contains": bool(contains_answer(pred, it["answer"])),
            "f1": f1(pred, it["answer"]),
            "copy": bool(norm(pred) and norm(pred) in norm(it["passage"])),
            "refusal": bool(is_refusal(pred)),
        }
        rows.append(row)
        if i % 25 == 0 or i == len(items):
            em_rate = sum(r["em"] for r in rows) / len(rows)
            print(f"[qa-eval] {name} {i}/{len(items)} EM={em_rate:.1%} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return {"model": name, "path": path, "rows": rows,
            "elapsed_s": time.time() - t0}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    em = [int(r["em"]) for r in rows]
    return {
        "n": n,
        "em": sum(em) / n,
        "em_ci95": bootstrap_ci(em),
        "em_contains": sum(r["em_contains"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "copy": sum(r["copy"] for r in rows) / n,
        "refusal": sum(r["refusal"] for r in rows) / n,
        "by_source": {
            s: {
                "n": len([r for r in rows if r["source"] == s]),
                "em": sum(r["em"] for r in rows if r["source"] == s)
                / max(1, len([r for r in rows if r["source"] == s])),
            } for s in sorted({r["source"] for r in rows})
        },
        "by_type": {
            t: {
                "n": len([r for r in rows if r["type"] == t]),
                "em": sum(r["em"] for r in rows if r["type"] == t)
                / max(1, len([r for r in rows if r["type"] == t])),
            } for t in sorted({r["type"] for r in rows})
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/qa/g_passage_test_v1.jsonl")
    ap.add_argument("--model", action="append", default=[],
                    help="nom=chemin.bin (répétable)")
    ap.add_argument("--out", default="docs/results/qa_passage_baseline.json")
    ap.add_argument("--peritem", default=None,
                    help="défaut : <out sans .json>_peritem.json (ne pas "
                         "écraser un fichier partagé entre exécutions)")
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not a.peritem:
        stem = a.out[:-5] if a.out.endswith(".json") else a.out
        a.peritem = f"{stem}_peritem.json"
    items = load_items(a.items)
    if a.limit:
        items = items[:a.limit]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    models = []
    for spec in a.model:
        if "=" not in spec:
            raise SystemExit(f"--model doit être nom=chemin : {spec}")
        name, path = spec.split("=", 1)
        models.append((name, path))
    if not models:
        raise SystemExit("Aucun --model fourni.")
    out = {
        "items": a.items, "n_items": len(items),
        "generation": {"temperature": 0.0,
                       "repetition_penalty": a.repetition_penalty,
                       "no_repeat_ngram": a.no_repeat_ngram,
                       "max_new": a.max_new},
        "models": {},
    }
    peritem = {}
    for name, path in models:
        print(f"[qa-eval] === {name} ({path}) ===", flush=True)
        r = eval_model(name, path, items, tok, a.max_new,
                       a.repetition_penalty, a.no_repeat_ngram)
        out["models"][name] = {**{k: v for k, v in r.items() if k != "rows"},
                               "summary": summarize(r["rows"])}
        peritem[name] = r["rows"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(a.peritem, "w", encoding="utf-8") as f:
        json.dump(peritem, f, ensure_ascii=False, indent=1)
    print(f"[qa-eval] -> {a.out}, {a.peritem}")
    for name, r in out["models"].items():
        s = r["summary"]
        print(f"[qa-eval] {name}: EM={s['em']:.1%} "
              f"[{s['em_ci95'][0]:.1%},{s['em_ci95'][1]:.1%}] "
              f"contains={s['em_contains']:.1%} F1={s['f1']:.3f} "
              f"copy={s['copy']:.1%} refus={s['refusal']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
