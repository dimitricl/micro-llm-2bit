"""Données : chargement texte/jsonl, vocabulaire réduit, batchs.

Le parent (ex. Qwen2.5-0.5B : 151k tokens) a un vocabulaire trop gros pour
100 Mo. On le réduit à `vocab_size` (défaut 8000) :
- on garde les tokens spéciaux (pad, eos, bos, unk) ;
- on garde les 256 octets de base si présents (sécurité bytes) ;
- on complète avec les tokens les plus fréquents du corpus `data/`.
Les tokens hors vocab sont rabattus sur `unk`. Le choix est sauvegardé
(`kept_ids`) pour la distillation (gather des logits parent) et l'export.
"""

from __future__ import annotations

import collections
import glob
import html
import json
import os
import re

import torch

TEXT_FIELD = "text"

_TAG_RE = re.compile(r"<[^>]+>")


def train_val_split(
    texts: list[str], val_ratio: float = 0.05, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Split train/val reproductible au niveau documents, sans chevauchement.

    Mélange déterministe des indices (seed fixe), les `val_ratio` derniers
    vont en val. Le vocabulaire doit être construit sur le train SEULEMENT
    (pas de fuite val -> vocab).
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio doit être dans ]0, 1[.")
    n = len(texts)
    n_val = max(1, int(n * val_ratio)) if n > 1 else 0
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    val_idx = set(perm[-n_val:]) if n_val else set()
    train = [t for i, t in enumerate(texts) if i not in val_idx]
    val = [t for i, t in enumerate(texts) if i in val_idx]
    return train, val


def _read_xml(fp: str) -> str:
    """Extrait le texte d'un XML/RSS : supprime les balises, dé-échappe."""
    with open(fp, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = _TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def load_texts(path: str | list[str], field: str = TEXT_FIELD) -> list[str]:
    """Charge .txt, .jsonl et .xml depuis un fichier, un dossier (récursif)
    ou une liste de chemins (plusieurs --data)."""
    paths = [path] if isinstance(path, str) else list(path)
    files: list[str] = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        else:
            for ext in ("*.txt", "*.jsonl", "*.xml"):
                files += sorted(glob.glob(os.path.join(p, "**", ext), recursive=True))
    if not files:
        raise FileNotFoundError(f"Aucun .txt/.jsonl/.xml trouvé sous {path}")
    texts: list[str] = []
    for fp in files:
        if fp.endswith(".jsonl"):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    texts.append(json.loads(line).get(field, ""))
        elif fp.endswith(".xml"):
            texts.append(_read_xml(fp))
        else:
            with open(fp, encoding="utf-8", errors="replace") as f:
                texts.append(f.read())
    texts = [t for t in texts if t.strip()]
    if not texts:
        raise ValueError(f"Corpus vide sous {path}")
    return texts


def build_reduced_vocab(tokenizer, texts: list[str], vocab_size: int = 8000) -> dict:
    """Construit le vocab réduit. Retourne le descripteur sérialisable.

    Descripteur : {"parent": name, "kept_ids": [...], "new_of_old": {old: new},
    "special": {...}}. `kept_ids[new] = old`.
    """
    # 1) Spéciaux toujours gardés.
    specials = set()
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            specials.add(int(tid))
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        raise ValueError("Le tokenizer parent n'a pas de token unk.")
    # 2) Fréquences sur le corpus.
    counter: collections.Counter[int] = collections.Counter()
    for t in texts:
        counter.update(tokenizer.encode(t, add_special_tokens=False))
    # 3) Sélection : spéciaux d'abord, puis plus fréquents.
    kept: list[int] = sorted(specials)
    for old, _ in counter.most_common():
        if len(kept) >= vocab_size:
            break
        if old not in specials:
            kept.append(int(old))
    if len(kept) < vocab_size:
        # Complète avec les premiers ids manquants (déterministe).
        cand = 0
        have = set(kept)
        while len(kept) < vocab_size:
            if cand not in have:
                kept.append(cand)
            cand += 1
    new_of_old = {old: new for new, old in enumerate(kept)}
    return {
        "parent": getattr(tokenizer, "name_or_path", "parent"),
        "vocab_size": len(kept),
        "kept_ids": kept,
        "unk_new": new_of_old[int(unk_id)],
        "eos_new": new_of_old.get(int(tokenizer.eos_token_id), new_of_old[int(unk_id)]),
    }


def check_vocab_compatible(vocab: dict, ckpt_vocab: dict) -> None:
    """Refuse l'enchaînement si le vocab diffère de celui du checkpoint.

    Indispensable pour l'entraînement par tranches : le mapping
    token->id doit rester identique d'une tranche à l'autre, sinon les
    poids (embeddings, tête liée) ne correspondent plus à rien.
    """
    a, b = vocab.get("kept_ids"), (ckpt_vocab or {}).get("kept_ids")
    if a is None or b is None:
        raise ValueError("Vocabulaire incomplet : kept_ids manquant.")
    if list(a) != list(b):
        raise ValueError(
            "Vocabulaire incompatible avec le checkpoint "
            f"({len(a)} vs {len(b)} tokens ou ordre différent). "
            "Pour enchaîner les tranches, réutilisez le même vocab "
            "(--vocab-from tranche précédente) au lieu de le recalculer."
        )


def encode_mapped(tokenizer, vocab: dict, text: str, add_eos: bool = True) -> list[int]:
    """Encode puis rabat sur le vocab réduit (hors-vocab -> unk)."""
    unk = vocab["unk_new"]
    new_of = {old: new for new, old in enumerate(vocab["kept_ids"])}
    ids = [new_of.get(t, unk) for t in tokenizer.encode(text, add_special_tokens=False)]
    if add_eos:
        ids.append(vocab["eos_new"])
    return ids


def make_batches(
    all_ids: list[int],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Découpe en fenêtres (seq_len+1) pour LM causale : x=[:-1], y=[1:]."""
    wins = [
        all_ids[i : i + seq_len + 1] for i in range(0, len(all_ids) - seq_len, seq_len)
    ]
    wins = [w for w in wins if len(w) == seq_len + 1]
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(wins), generator=g).tolist()
        wins = [wins[i] for i in perm]
    return [
        torch.tensor(wins[i : i + batch_size], dtype=torch.long)
        for i in range(0, len(wins), batch_size)
    ]
