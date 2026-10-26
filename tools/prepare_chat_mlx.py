#!/usr/bin/env python3
"""Prépare des conversations JSONL pour le fine-tuning MLX existant."""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/chat/chat_pilot_v1.jsonl")
    ap.add_argument("--tokenizer", default="models/tokenizer_fr_16k")
    ap.add_argument("--output", default="runs/G6-chat-data/chat_packed.npz")
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    tok = Tokenizer.from_file(
        str(Path(args.tokenizer) / "tokenizer.json")
    )
    tok.no_truncation()
    width = args.seq_len + 1

    all_ids, all_masks = [], []
    skipped = 0

    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]

            # Sérialisation explicite du dialogue.
            # Seuls les tokens de réponse assistant sont supervisés.
            ids, mask = [], []
            valid = True

            for msg in messages:
                role = msg["role"]
                content = msg["content"].strip()

                if role == "user":
                    part = tok.encode(
                        f"Utilisateur : {content}\n"
                    ).ids
                    ids.extend(part)
                    mask.extend([0] * len(part))
                elif role == "assistant":
                    prefix = tok.encode("Assistant : ").ids
                    answer = tok.encode(content).ids
                    ending = tok.encode("\n\n").ids

                    ids.extend(prefix)
                    mask.extend([0] * len(prefix))
                    ids.extend(answer)
                    mask.extend([1] * len(answer))
                    ids.extend(ending)
                    mask.extend([1] * len(ending))
                else:
                    raise ValueError(
                        f"Ligne {line_no}: rôle inconnu {role!r}"
                    )

            if not ids or len(ids) > width or not any(mask):
                skipped += 1
                continue

            # Padding neutre : il ne contribue jamais à la loss.
            pad = width - len(ids)
            ids.extend([0] * pad)
            mask.extend([0] * pad)

            all_ids.append(ids)
            all_masks.append(mask)

    if not all_ids:
        raise SystemExit("Aucun exemple valide. Vérifie les données.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=np.asarray(all_ids, dtype=np.uint16),
        mask=np.asarray(all_masks, dtype=np.uint8),
    )

    print(f"Exemples préparés : {len(all_ids)}")
    print(f"Exemples ignorés   : {skipped}")
    print(f"Dimensions IDs     : {len(all_ids)} × {width}")
    print(f"Tokens supervisés  : {int(np.asarray(all_masks).sum()):,}")
    print(f"Fichier créé       : {output}")
    print("Vérification du masque :")
    for i in range(min(3, len(all_ids))):
        print(
            f"  Exemple {i + 1}: "
            f"{int(np.asarray(all_masks)[i].sum())} tokens supervisés"
        )


if __name__ == "__main__":
    main()
