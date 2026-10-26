#!/usr/bin/env python3
"""
Train a Byte-Level BPE tokenizer for French corpus.

Usage:
    python tools/1_train_tokenizer.py \
        --corpus-dir data/corpus \
        --output-dir models/tokenizer_fr_16k \
        --vocab-size 16000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel


def train_tokenizer(corpus_dir: Path, output_dir: Path, vocab_size: int) -> None:
    """
    Entraîne un ByteLevel BPE tokenizer sur le corpus français.

    Args:
        corpus_dir: Dossier contenant les fichiers .txt ou .jsonl
        output_dir: Dossier de sortie pour tokenizer.json et tokenizer_config.json
        vocab_size: Taille du vocabulaire (ex: 16000)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Récupère tous les fichiers texte
    files = list(corpus_dir.rglob("*.txt")) + list(corpus_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .txt ou .jsonl trouvé dans {corpus_dir}")

    print(f"[tokenizer] {len(files)} fichiers trouvés dans {corpus_dir}")

    # Initialise le tokenizer Byte-Level BPE
    tokenizer = ByteLevelBPETokenizer()

    # Entraîne sur les fichiers
    tokenizer.train(
        files=[str(f) for f in files],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<|endoftext|>",
            "<|pad|>",
        ],
        show_progress=True,
    )

    # Configure le post-processor ByteLevel (nécessaire pour AutoTokenizer)
    tokenizer._tokenizer.post_processor = ByteLevel(trim_offsets=False)

    # Active le padding
    pad_id = tokenizer.token_to_id("<|pad|>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<|pad|>")

    # NE PAS activer la truncation pour l'entraînement du tokenizer
    # (on la gérera côté data loader / modèle)

    # Sauvegarde tokenizer.json
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)
    print(f"[tokenizer] Sauvegardé : {tokenizer_path}")

    # Génère tokenizer_config.json compatible AutoTokenizer
    config = {
        "tokenizer_class": "ByteLevelBPETokenizer",
        "vocab_size": vocab_size,
        "model_max_length": 1024,
        "padding_side": "right",
        "truncation_side": "right",
        "special_tokens": {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|endoftext|>",
        },
        "clean_up_tokenization_spaces": False,
    }
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[tokenizer] Config sauvegardée : {config_path}")

    # Test rapide
    test_text = "La capitale de la France est Paris."
    encoded = tokenizer.encode(test_text)
    print(f"[tokenizer] Test : '{test_text}' -> {encoded.tokens} (ids: {encoded.ids})")
    decoded = tokenizer.decode(encoded.ids)
    print(f"[tokenizer] Décodé : '{decoded}'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ByteLevel BPE tokenizer for French")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"),
                        help="Dossier contenant les fichiers .txt/.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("models/tokenizer_fr_16k"),
                        help="Dossier de sortie pour tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=16000,
                        help="Taille du vocabulaire")
    args = parser.parse_args()

    try:
        train_tokenizer(args.corpus_dir, args.output_dir, args.vocab_size)
        print("[tokenizer] ✅ Entraînement terminé avec succès")
        return 0
    except Exception as e:
        print(f"[tokenizer] ❌ Erreur : {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())