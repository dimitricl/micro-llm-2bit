"""Phase 0 G — tokenizer : fragmentation FR et comparaison à un BPE 16k.

Mesures (aucun modèle de langage entraîné) :
- SmolLM-135M natif (GPT-2 byte-level BPE, ~49k tokens) ;
- le même ramené au vocabulaire réduit E-t08 (12 000 tokens, unk) ;
- un BPE byte-level **entraîné sur le corpus** (16 000 tokens, et 12 000
  pour comparer à taille égale), entraîné sur 40 Mo de tranche_01 et évalué
  sur val_E_frozen (jamais vue).

Indicateurs : tokens par mot, part de mots fragmentés (types pondérés par
fréquence), part de tokens unk, et découpage des 4 mots de la confusion +
des 50 mots FR les plus fréquents.

Sorties : docs/results/tokenizer_fr_G.json, docs/TOKENIZER_FR_G.md.
Usage : .venv/bin/python tools/tokenizer_fr_g.py
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)

TRAIN_SRC = "/Volumes/Lexar/clean-corpus/dedup/tranche_01.jsonl"
EVAL_SRC = "runs/abl_data/val_E_frozen.jsonl"
VOCAB = "exports/E-t08.bin.vocab.json"
PARENT = "HuggingFaceTB/SmolLM-135M"
OUT_JSON = "docs/results/tokenizer_fr_G.json"
OUT_MD = "docs/TOKENIZER_FR_G.md"
TARGETS = ["capitale", "capitaine", "capitole", "capitalle"]
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ]+")
TRAIN_CHARS = 40_000_000


def _fix(piece: str) -> str:
    try:
        return piece.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return piece


def load_docs(path: str, max_chars: int = 0) -> list[str]:
    out, n = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line).get("text", "")
            if not t.strip():
                continue
            out.append(t)
            n += len(t)
            if max_chars and n >= max_chars:
                break
    return out


def train_bpe(texts: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    t0 = time.time()
    tok.train_from_iterator(texts, trainer=trainer)
    print(f"[tok-fr] BPE {vocab_size} entraîné en {time.time()-t0:.0f}s "
          f"({len(texts)} docs)", flush=True)
    return tok


def word_freq(texts: list[str]) -> collections.Counter:
    c: collections.Counter[str] = collections.Counter()
    for t in texts:
        c.update(w.lower() for w in WORD_RE.findall(t) if len(w) >= 2)
    return c


def enc(tok, text: str) -> list[int]:
    """Encode avec l'API HF (transformers) ou tokenizers.Tokenizer."""
    if isinstance(tok, Tokenizer):
        return tok.encode(text).ids
    return tok.encode(text, add_special_tokens=False)


def piece_of(tok, i: int) -> str:
    if isinstance(tok, Tokenizer):
        return tok.id_to_token(i)
    return tok.convert_ids_to_tokens(i)


def pieces_of(tok, word: str) -> list[str]:
    return [_fix(piece_of(tok, i)) for i in enc(tok, " " + word)]


def evaluate(name: str, tok, texts: list[str], freqs: collections.Counter,
             reduced: dict | None = None, vocab_size: int | None = None) -> dict:
    """tokens/mot, fragmentation pondérée, unk (si vocabulaire réduit)."""
    total_words = total_tokens = 0
    for t in texts:
        total_words += len(WORD_RE.findall(t))
        total_tokens += len(enc(tok, t))
    # Fragmentation des types, pondérée par leur fréquence dans l'éval.
    frag_w = n_types = unk_tokens = unk_words = tot_tok_words = 0
    per_word: dict[str, int] = {}
    for w, f in freqs.items():
        p = pieces_of(tok, w)
        per_word[w] = len(p)
        n_types += 1
        if len(p) > 1:
            frag_w += f
        tot_tok_words += len(p) * f
    if reduced is not None:
        kept = reduced["kept_ids"]
        new_of = {old: new for new, old in enumerate(kept) if old >= 0}
        unk_new = reduced["unk_new"]
        for t in texts:
            ids = enc(tok, t)
            unk_tokens += sum(1 for i in ids if i not in new_of)
        for w, f in freqs.items():
            if any(i not in new_of for i in enc(tok, " " + w)):
                unk_words += f
    out = {
        "vocab_size": vocab_size or (tok.get_vocab_size()
                                     if isinstance(tok, Tokenizer)
                                     else tok.vocab_size),
        "tokens_par_mot": total_tokens / max(total_words, 1),
        "mots_evalues": total_words,
        "types_evalues": n_types,
        "taux_mots_fragmentes_pondere": frag_w / max(sum(freqs.values()), 1),
        "tokens_par_mot_types": tot_tok_words / max(sum(freqs.values()), 1),
        "cible": {w: {"n_tokens": per_word.get(w),
                      "pieces": pieces_of(tok, w)} for w in TARGETS},
        "exemples_50": [
            {"mot": w, "freq": f, "n_tokens": per_word[w],
             "pieces": pieces_of(tok, w)}
            for w, f in freqs.most_common(50)
        ],
    }
    if reduced is not None:
        out["taux_tokens_unk"] = unk_tokens / max(total_tokens, 1)
        out["taux_mots_unk_pondere"] = unk_words / max(sum(freqs.values()), 1)
    print(f"[tok-fr] {name}: {out['tokens_par_mot']:.3f} tok/mot, "
          f"{100*out['taux_mots_fragmentes_pondere']:.1f}% types fragmentés"
          + (f", {100*out['taux_tokens_unk']:.1f}% unk" if reduced else ""),
          flush=True)
    return out


def main() -> int:
    train_texts = load_docs(TRAIN_SRC, TRAIN_CHARS)
    eval_texts = load_docs(EVAL_SRC)
    freqs = word_freq(eval_texts)
    vocab = json.load(open(VOCAB, encoding="utf-8"))
    smol = AutoTokenizer.from_pretrained(PARENT)
    bpe16 = train_bpe(train_texts, 16000)
    bpe12 = train_bpe(train_texts, 12000)

    payload = {
        "parent": PARENT,
        "vocab_reduit": VOCAB,
        "train_tokenizer": f"{TRAIN_SRC} ({TRAIN_CHARS/1e6:.0f} Mo)",
        "eval": EVAL_SRC,
        "n_docs_eval": len(eval_texts),
        "n_mots_eval": sum(len(WORD_RE.findall(t)) for t in eval_texts),
        "smollm": evaluate("smollm", smol, eval_texts, freqs),
        "smollm_reduit": evaluate("smollm+reduit", smol, eval_texts, freqs,
                                  reduced=vocab,
                                  vocab_size=int(vocab.get("vocab_size", 12000))),
        "bpe16k": evaluate("bpe16k", bpe16, eval_texts, freqs),
        "bpe12k": evaluate("bpe12k", bpe12, eval_texts, freqs),
        "hypothese": (
            "Un BPE FR de 12-16k réduit le nombre de tokens par mot et la "
            "fragmentation ; l'écart chiffré indique le gain maximal "
            "attendu d'un tokenizer natif, à comparer au coût d'un "
            "changement de vocabulaire (embeddings + tête liée à réinit.)"),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(payload, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    lines = [
        "# Tokenizer FR — fragmentation et comparaison BPE 16k",
        "",
        f"SmolLM-135M natif vs vocabulaire réduit E-t08 vs BPE byte-level "
        f"entraîné sur {TRAIN_CHARS/1e6:.0f} Mo de `tranche_01`, évalué sur "
        f"`{EVAL_SRC}` ({payload['n_mots_eval']:,} mots).".replace(",", " "),
        "",
        "| tokenizer | vocab | tokens/mot | types fragmentés (pondéré) | "
        "tokens unk | mots unk (pondéré) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [("smollm", "SmolLM natif"),
                       ("smollm_reduit", "SmolLM -> réduit E-t08"),
                       ("bpe16k", "BPE FR 16k"),
                       ("bpe12k", "BPE FR 12k")]:
        d = payload[key]
        lines.append(
            f"| {label} | {d['vocab_size']:,} | {d['tokens_par_mot']:.3f} | "
            f"{100*d['taux_mots_fragmentes_pondere']:.1f} % | "
            f"{100*d.get('taux_tokens_unk', 0):.2f} % | "
            f"{100*d.get('taux_mots_unk_pondere', 0):.2f} % |"
            .replace(",", " "))
    lines += ["", "## Mots de la confusion", "",
              "| mot | SmolLM | réduit E-t08 | BPE 16k |", "|---|---|---|---|"]
    for w in TARGETS:
        def fmt(d):
            return " + ".join(f"`{p}`" for p in d["cible"][w]["pieces"])
        lines.append(f"| **{w}** | {fmt(payload['smollm'])} | "
                     f"{fmt(payload['smollm_reduit'])} | "
                     f"{fmt(payload['bpe16k'])} |")
    lines += ["", "## 50 mots FR les plus fréquents (extrait)", "",
              "| mot | freq | SmolLM (n) | BPE 16k (n) |", "|---|---:|---|---|"]
    for row in payload["smollm"]["exemples_50"][:50]:
        w = row["mot"]
        b = next(x for x in payload["bpe16k"]["exemples_50"] if x["mot"] == w)
        lines.append(
            f"| {w} | {row['freq']} | "
            + " ".join(f"`{p}`" for p in row["pieces"])
            + f" ({row['n_tokens']}) | "
            + " ".join(f"`{p}`" for p in b["pieces"])
            + f" ({b['n_tokens']}) |")
    lines += ["", "## Lecture", "", payload["hypothese"], "",
              "## Limites", "",
              "- Le BPE est entraîné sur 40 Mo seulement (sous-échantillon) : "
              "un entraînement sur les 1,06 Go ferait légèrement mieux.",
              "- Les tokens/mot incluent ponctuation et espaces ; la colonne "
              "« types fragmentés » isole la fragmentation des mots.",
              "- Aucun modèle de langage n'est entraîné ici : c'est un "
              "estimateur de coût de tokenisation, pas une preuve de gain "
              "de qualité.", ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"[tok-fr] -> {OUT_JSON}\n[tok-fr] -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
