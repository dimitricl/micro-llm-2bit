"""G1 — génère le corpus QA-passage d'entraînement (t01–t09).

Chaque exemple : passage + question (templates TRAIN, jamais ceux du banc)
→ réponse extraite. Le champ `control_prompt` retire la question et garde
le même passage et la même réponse (bras contrôle apparié).

Sortie : runs/G1-data/qa_train.jsonl + manifeste.
Aucune ligne de la val gelée / OOD / t00 n'est utilisée.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import data as D
import tools.build_qa_passage_G as B

CORPUS = "/Volumes/Lexar/corpus"
OUTDIR = os.path.join(ROOT, "runs", "G1-data")
OUT = os.path.join(OUTDIR, "qa_train.jsonl")
MANIFEST = os.path.join(OUTDIR, "qa_train.manifest.json")
VOCAB = os.path.join(ROOT, "runs", "E-t08", "model.pt.vocab.json")
TOKENIZER = "HuggingFaceTB/SmolLM-135M"
TARGET = 30_300_000
MAX_PER_DOC = 8


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=TARGET)
    ap.add_argument("--max-per-doc", type=int, default=MAX_PER_DOC)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    with open(VOCAB, encoding="utf-8") as f:
        vocab = json.load(f)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # Corpus réel : /Volumes/Lexar/clean-corpus/dedup/tranche_XX.jsonl
    corpus = "/Volumes/Lexar/clean-corpus/dedup"
    files = [os.path.join(corpus, f"tranche_{i:02d}.jsonl")
             for i in range(1, 10)]
    if not all(os.path.exists(p) for p in files):
        raise SystemExit(f"Corpus introuvable : {files[0]}")
    total = 0
    n = 0
    counts = {"annee": 0, "nombre": 0, "entite": 0}
    docs_seen = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for fp in files:
            tag = os.path.basename(fp).replace(".jsonl", "")
            with open(fp, encoding="utf-8") as f:
                for li, line in enumerate(f):
                    if a.max_docs and docs_seen >= a.max_docs:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    text = json.loads(line).get("text", "")
                    if len(text) < 200:
                        continue
                    docs_seen += 1
                    sents = B.sentences(text)
                    n_doc = 0
                    for si, s in enumerate(sents):
                        for cand in B.candidates_for_sentence(s):
                            doc = {"doc_id": f"{tag}-{li}"}
                            it = B.make_item(doc, tag, sents, si, cand,
                                             B.TRAIN_TEMPLATES, tok, vocab, 230)
                            if not it:
                                continue
                            it["control_prompt"] = (
                                f"Passage :\n{it['passage']}\n\nRéponse :")
                            it["source"] = tag
                            out.write(json.dumps(it, ensure_ascii=False) + "\n")
                            total += it["n_prompt_tokens"] + it["n_answer_tokens"] + 1
                            n += 1
                            counts[cand["type"]] += 1
                            n_doc += 1
                            if (n_doc >= a.max_per_doc
                                    or total >= a.target_tokens):
                                break
                        if n_doc >= a.max_per_doc or total >= a.target_tokens:
                            break
                    if n % 5000 == 0 and n:
                        print(f"[qa-train] {n} ex, {total/1e6:.1f}M tokens, "
                              f"{docs_seen} docs", flush=True)
                    if total >= a.target_tokens:
                        break
            print(f"[qa-train] {tag}: total {n} ex, {total/1e6:.1f}M tokens",
                  flush=True)
            if total >= a.target_tokens:
                break
    manifest = {
        "target_tokens": a.target_tokens,
        "tokens": total,
        "n_examples": n,
        "docs_seen": docs_seen,
        "counts": counts,
        "templates": B.TRAIN_TEMPLATES,
        "source": files,
        "tokenizer": TOKENIZER,
        "vocab": VOCAB,
        "sha256": sha256_file(a.out),
        "note": "prompt + answer ; control_prompt = passage + Réponse :",
    }
    with open(a.out + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[qa-train] -> {a.out} ({total/1e6:.2f}M tokens, {n} ex)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
