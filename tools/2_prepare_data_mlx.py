#!/usr/bin/env python3
"""
Tokenise le corpus complet et sauvegarde en numpy memmap (uint16) pour MLX.

Usage:
    python tools/2_prepare_data_mlx.py \
        --tokenizer-dir models/tokenizer_fr_16k \
        --corpus-dir data/corpus \
        --output data/corpus.bin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def prepare_data(tokenizer_dir: Path, corpus_dir: Path, output_path: Path) -> None:
    """
    Tokenise tout le corpus et sauvegarde en memmap uint16.

    Args:
        tokenizer_dir: Dossier contenant tokenizer.json
        corpus_dir: Dossier source (.txt/.jsonl)
        output_path: Fichier de sortie .bin (memmap uint16)
    """
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer non trouvé : {tokenizer_path}")

    # Charge le tokenizer (sans truncation pour la préparation des données)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Désactive la truncation pour encoder tout le texte
    tokenizer.no_truncation()
    print(f"[prepare] Tokenizer chargé : {tokenizer_path} (vocab={tokenizer.get_vocab_size()})")

    # Récupère les fichiers
    files = list(corpus_dir.rglob("*.txt")) + list(corpus_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .txt ou .jsonl dans {corpus_dir}")

    print(f"[prepare] {len(files)} fichiers à traiter")

    # Compte total de tokens (première passe)
    total_tokens = 0
    file_tokens = []

    print("[prepare] Comptage des tokens...")
    for fpath in files:
        try:
            file_tokens_count = 0
            if fpath.suffix == ".jsonl":
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            text = json.loads(line).get("text", "")
                        except json.JSONDecodeError:
                            text = line
                        if not text.strip():
                            continue
                        encoded = tokenizer.encode(text)
                        n = len(encoded.ids)
                        if n > 0:
                            file_tokens_count += n
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                if not text.strip():
                    continue
                encoded = tokenizer.encode(text)
                file_tokens_count = len(encoded.ids)

            if file_tokens_count > 0:
                total_tokens += file_tokens_count
                file_tokens.append((fpath, file_tokens_count))

        except Exception as e:
            print(f"[prepare] ⚠ Erreur lecture {fpath}: {e}", file=sys.stderr)
            continue

    if total_tokens == 0:
        raise ValueError("Aucun token généré - corpus vide ?")

    print(f"[prepare] Total tokens : {total_tokens:,} (~{total_tokens / 1e6:.2f}M)")

    # Crée le memmap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))

    # Deuxième passe : écriture
    print("[prepare] Écriture du memmap...")
    idx = 0
    for fpath, expected_n in file_tokens:
        try:
            if fpath.suffix == ".jsonl":
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            text = json.loads(line).get("text", "")
                        except json.JSONDecodeError:
                            text = line
                        if not text.strip():
                            continue
                        encoded = tokenizer.encode(text)
                        ids = encoded.ids
                        if ids:
                            memmap[idx:idx + len(ids)] = np.array(ids, dtype=np.uint16)
                            idx += len(ids)
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                if not text.strip():
                    continue
                encoded = tokenizer.encode(text)
                ids = encoded.ids
                if ids:
                    memmap[idx:idx + len(ids)] = np.array(ids, dtype=np.uint16)
                    idx += len(ids)

        except Exception as e:
            print(f"[prepare] ⚠ Erreur écriture {fpath}: {e}", file=sys.stderr)
            continue

    # Flush pour s'assurer que tout est écrit
    memmap.flush()
    print(f"[prepare] ✅ Memmap sauvegardé : {output_path} ({idx:,} tokens)")
    print(f"[prepare] Taille fichier : {output_path.stat().st_size / 1e9:.2f} GB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare corpus data as uint16 memmap for MLX")
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("models/tokenizer_fr_16k"),
                        help="Dossier du tokenizer (tokenizer.json)")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"),
                        help="Dossier corpus source")
    parser.add_argument("--output", type=Path, default=Path("data/corpus.bin"),
                        help="Fichier de sortie .bin")
    args = parser.parse_args()

    try:
        prepare_data(args.tokenizer_dir, args.corpus_dir, args.output)
        return 0
    except Exception as e:
        print(f"[prepare] ❌ Erreur : {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())