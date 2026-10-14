"""E-PREP : val GELÉE unique pour les 9 tranches (CPU, déterministe).

500 docs tirés seed 0 dans la tranche dédupliquée 00 (jamais entraînée
en E : E utilise t01-t09). Disjonction SHA1 exigée contre : tranches
t01-t09 (train E), val_big, OOD, slice_00-09 d'origine. Échec bruyant
sinon. Sorties : runs/abl_data/val_E_frozen.jsonl (ignoré git) +
docs/results/val_E_frozen_manifest.json (hash committé).
Usage : .venv/bin/python tools/make_frozen_val.py
"""
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

N_VAL = 500
SEED = 0


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def h(t):
    return hashlib.sha1(norm(t).encode()).hexdigest()


def docs_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                if t.strip():
                    out.append(t)
    return out


def main():
    rng = random.Random(SEED)
    pool = docs_jsonl("/Volumes/Lexar/clean-corpus/dedup/tranche_00.jsonl")
    print(f"[valfroide] pool t00 : {len(pool)} docs", flush=True)
    assert len(pool) >= N_VAL, "pool trop petit"
    val = rng.sample(pool, N_VAL)
    vh = {h(t) for t in val}
    # Disjonction.
    others = {}
    for ti in range(1, 10):
        others[f"dedup_t{ti:02d}"] = docs_jsonl(
            f"/Volumes/Lexar/clean-corpus/dedup/tranche_{ti:02d}.jsonl")
    others["val_big"] = docs_jsonl(
        "/Volumes/Lexar/micro-llm-2bit/runs/abl_data/val_big.jsonl")
    others["ood"] = docs_jsonl(
        "/Volumes/Lexar/micro-llm-2bit/runs/abl_data/ood.jsonl")
    bad = {}
    for name, docs in others.items():
        inter = vh & {h(t) for t in docs}
        if inter:
            bad[name] = len(inter)
    if bad:
        raise SystemExit(f"[valfroide] ÉCHEC disjonction : {bad}")
    print("[valfroide] disjonction OK (t01-t09, val_big, ood)", flush=True)
    out = "/Volumes/Lexar/micro-llm-2bit/runs/abl_data/val_E_frozen.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for t in val:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    man = {"seed": SEED, "n": len(val),
           "chars": sum(len(t) for t in val),
           "source": "/Volumes/Lexar/clean-corpus/dedup/tranche_00.jsonl",
           "sha1_texts": hashlib.sha1(
               "".join(val).encode()).hexdigest(),
           "doc_hashes": sorted(vh)}
    json.dump(man, open("/Volumes/Lexar/micro-llm-2bit/docs/results/"
                        "val_E_frozen_manifest.json", "w"), indent=1)
    print(f"[valfroide] {len(val)} docs, {man['chars']} chars -> {out}",
          flush=True)
    print("[valfroide] TERMINE", flush=True)


if __name__ == "__main__":
    main()
