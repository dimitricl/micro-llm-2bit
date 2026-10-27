"""Préparation des données : tokenisation, packing, masque de perte."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

# Tokens spéciaux à ajouter au tokenizer (ids attribués à l'entraînement).
SPECIAL_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|tool_call|>",
    "<|tool_result|>",
]

CHAT_TEMPLATE = (
    "<|system|>{system}<|end|>"
    "<|user|>{user}<|end|>"
    "<|assistant|>{assistant}<|end|>"
)


def iter_jsonl_texts(paths: Iterable[str | Path], key: str = "text") -> Iterator[str]:
    """Lit des fichiers JSONL et yield le champ texte (ou `content`)."""
    for path in paths:
        path = Path(path)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    yield line
                    continue
                text = row.get(key) or row.get("content") or row.get("text")
                if text:
                    yield str(text)


def encode_corpus(
    tokenizer,
    texts: Iterable[str],
    eos_id: int,
    add_eos: bool = True,
    max_tokens: int | None = None,
) -> np.ndarray:
    """Tokenise un flux de textes en un tableau uint16 plat, avec EOS."""
    chunks: list[np.ndarray] = []
    total = 0
    for text in texts:
        ids = tokenizer.encode(text).ids
        if add_eos:
            ids = ids + [eos_id]
        arr = np.asarray(ids, dtype=np.uint16)
        chunks.append(arr)
        total += arr.size
        if max_tokens is not None and total >= max_tokens:
            break
    if not chunks:
        return np.empty(0, dtype=np.uint16)
    out = np.concatenate(chunks)
    return out[:max_tokens] if max_tokens is not None else out


def pack_sequences(tokens: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Découpe un flux en (entrées, cibles) de longueur seq_len.

    La cible est décalée d'un token. Les tokens restants sont ignorés.
    """
    n = (len(tokens) - 1) // seq_len
    if n <= 0:
        return (
            np.empty((0, seq_len), dtype=np.int32),
            np.empty((0, seq_len), dtype=np.int32),
        )
    x = np.stack([tokens[i * seq_len : (i + 1) * seq_len] for i in range(n)])
    y = np.stack([tokens[i * seq_len + 1 : (i + 1) * seq_len + 1] for i in range(n)])
    return x.astype(np.int32), y.astype(np.int32)


def make_loss_mask(prompt_len: int, seq_len: int, pad_len: int = 0) -> np.ndarray:
    """Masque 1 sur la réponse, 0 sur le prompt et le padding."""
    mask = np.zeros(seq_len, dtype=np.float32)
    end = max(prompt_len, seq_len - pad_len)
    mask[prompt_len:end] = 1.0
    return mask


def build_sft_example(
    tokenizer,
    system: str,
    user: str,
    assistant: str,
    seq_len: int,
    pad_id: int,
    eos_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit un exemple SFT (ids, cibles, masque) de longueur fixe."""
    prompt = f"<|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>"
    prompt_ids = tokenizer.encode(prompt).ids
    answer_ids = tokenizer.encode(assistant).ids + [eos_id]
    ids = (prompt_ids + answer_ids)[:seq_len]
    prompt_len = min(len(prompt_ids), seq_len)
    pad_len = seq_len - len(ids)
    ids = ids + [pad_id] * pad_len
    targets = ids[1:] + [pad_id]
    mask = make_loss_mask(prompt_len, seq_len, pad_len)
    return (
        np.asarray(ids, dtype=np.int32),
        np.asarray(targets, dtype=np.int32),
        mask,
    )


class TokenBatches:
    """Itérateur de batchs aléatoires sur un corpus tokenisé."""

    def __init__(self, tokens: np.ndarray, seq_len: int, batch_size: int, seed: int = 0):
        self.x, self.y = pack_sequences(tokens, seq_len)
        if len(self.x) == 0:
            raise ValueError("corpus trop court pour seq_len")
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x) // self.batch_size

    def batch(self):
        idx = self.rng.integers(0, len(self.x), size=self.batch_size)
        return self.x[idx], self.y[idx]
