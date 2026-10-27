#!/usr/bin/env python3
"""Prépare le SFT MLX masqué à partir des conversations et du code locaux."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def repeated_word_pairs(text: str) -> int:
    return len(re.findall(r"\b([\wÀ-ÿ]+)[ \t]+\1\b", text, flags=re.I))


def chat_pairs(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            history = []
            for message in row.get("messages", []):
                role = message.get("role")
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                if role == "assistant" and history:
                    prompt = "\n".join(
                        f"{'Utilisateur' if r == 'user' else 'Assistant'} : {c}"
                        for r, c in history
                    )
                    yield prompt, content, "chat"
                history.append((role, content))


def synthetic_pairs(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("instruction", "")).strip()
            answer = str(row.get("response", "")).strip()
            if prompt and answer:
                yield prompt, answer, str(row.get("mode", "chat"))


def code_pairs(path: Path, tokenizer: Tokenizer, max_samples: int):
    """Crée des paires de complétion sur des frontières de lignes."""
    made = 0
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip() or made >= max_samples:
                continue
            row = json.loads(line)
            source = str(row.get("text", "")).replace("\r\n", "\n").strip()
            lines = [s for s in source.splitlines() if s.strip()]
            if len(lines) < 4:
                continue
            # Quelques points de coupure par fichier, sans découpage au milieu d'une ligne.
            stride = max(1, len(lines) // 4)
            cuts = sorted(set(range(max(2, len(lines) // 3), len(lines), stride)))[:4]
            for cut in cuts:
                prefix_lines = lines[max(0, cut - 48):cut]
                suffix_lines = lines[cut:min(len(lines), cut + 24)]
                if not prefix_lines or not suffix_lines:
                    continue
                header = ("Complète le code JavaScript à partir du préfixe suivant. "
                          "Continue directement le code, sans explication.\n"
                          "```javascript\n")
                while prefix_lines:
                    prompt = header + "\n".join(prefix_lines) + "\n"
                    if len(tokenizer.encode(prompt).ids) <= 300:
                        break
                    prefix_lines = prefix_lines[1:]
                while suffix_lines:
                    answer = "\n".join(suffix_lines) + "\n```"
                    if len(tokenizer.encode(answer).ids) <= 180:
                        break
                    suffix_lines = suffix_lines[:-1]
                if (not prefix_lines or not suffix_lines or
                        len(tokenizer.encode(prompt).ids) < 8 or
                        len(tokenizer.encode(answer).ids) < 8):
                    continue
                yield prompt, answer, "code"
                made += 1
                if made >= max_samples:
                    break


def encode_pair(tokenizer: Tokenizer, prompt: str, answer: str,
                seq_len: int) -> tuple[list[int], list[int]] | None:
    prompt = prompt.strip()
    answer = answer.strip()
    if not prompt or not answer or len(answer) < 8:
        return None
    if repeated_word_pairs(answer) >= 2:
        return None
    prefix = tokenizer.encode(f"Utilisateur : {prompt}\nAssistant :\n").ids
    target = tokenizer.encode(answer + "\n\n").ids
    limit = seq_len + 1
    if len(prefix) + len(target) > limit or not target:
        return None
    ids = prefix + target
    mask = [0] * len(prefix) + [1] * len(target)
    pad = limit - len(ids)
    return ids + [0] * pad, mask + [0] * pad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("models/tokenizer_fr_16k"))
    ap.add_argument("--chat", type=Path, default=Path("data/chat/chat_pilot_v1.jsonl"))
    ap.add_argument("--synthetic", type=Path, default=Path("data/hybrid/synthetic.jsonl"))
    ap.add_argument("--code", type=Path, default=Path("data/hybrid/javascript_code.jsonl"))
    ap.add_argument("--max-code-samples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("runs/hybrid-sft/sft_packed.npz"))
    args = ap.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    tokenizer.no_truncation()
    pairs = []
    pairs.extend(chat_pairs(args.chat) or [])
    pairs.extend(synthetic_pairs(args.synthetic) or [])
    pairs.extend(code_pairs(args.code, tokenizer, args.max_code_samples) or [])

    seen = set()
    ids_rows, mask_rows = [], []
    counts = {"chat": 0, "code": 0, "rejected": 0, "duplicates": 0}
    for prompt, answer, kind in pairs:
        signature = (key(prompt), key(answer))
        if signature in seen:
            counts["duplicates"] += 1
            continue
        seen.add(signature)
        encoded = encode_pair(tokenizer, prompt, answer, args.seq_len)
        if encoded is None:
            counts["rejected"] += 1
            continue
        ids, mask = encoded
        ids_rows.append(ids)
        mask_rows.append(mask)
        counts["code" if kind == "code" else "chat"] += 1

    if not ids_rows:
        raise SystemExit("Aucun exemple SFT valide après filtrage.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        ids=np.asarray(ids_rows, dtype=np.uint16),
        mask=np.asarray(mask_rows, dtype=np.uint8),
    )
    meta = {
        "tokenizer": str(args.tokenizer),
        "seq_len": args.seq_len,
        "examples": len(ids_rows),
        "supervised_tokens": int(np.asarray(mask_rows, dtype=np.uint8).sum()),
        "counts": counts,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
