"""Convertit un checkpoint .pt (ancien vocab, unk rabattu sur eos) vers le
vocab à unk distinct (B2), pour repartir des poids existants.

- kept_ids inchangés SAUF le dernier (le moins fréquent), remplacé par la
  sentinelle -1 (ligne unk dédiée). Taille du vocab inchangée.
- La ligne d'embedding correspondante (tête liée : rien d'autre à toucher)
  est réinitialisée à la moyenne des lignes : le token évincé perd son sens,
  la ligne devient la ligne unk à entraîner.
- check_vocab_compatible refuse d'ailleurs de mélanger les deux formats :
  utilisez ce script (ou réentraînez le vocab) avant toute reprise.

Usage :
  uv run python tools/convert_vocab.py --ckpt wiki_slice00_bestval.pt \
      --out wiki_slice00_unk.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def convert_vocab_in_ckpt(ckpt: dict) -> tuple[dict, dict]:
    """Convertit le vocab du checkpoint en place (modèle + vocab retournés)."""
    vocab = ckpt.get("vocab")
    if vocab is None or vocab.get("kept_ids") is None:
        raise ValueError("Checkpoint sans vocabulaire (kept_ids manquant).")
    if vocab.get("unk_distinct"):
        raise ValueError("Checkpoint déjà en unk distinct, rien à convertir.")
    kept = list(vocab["kept_ids"])
    victim = len(kept) - 1
    if victim == vocab.get("eos_new"):
        raise ValueError("Vocabulaire trop petit pour un unk distinct.")
    new_kept = kept[:victim] + [-1] + kept[victim + 1:]
    new_vocab = dict(vocab)
    new_vocab.update({
        "kept_ids": new_kept,
        "unk_new": victim,
        "unk_distinct": True,
        "vocab_version": 2,
    })
    model = ckpt.get("model")
    if model is None or "embed_latent" not in model:
        raise ValueError("Checkpoint sans poids ('model.embed_latent').")
    emb = model["embed_latent"]
    if emb.shape[0] != len(kept):
        raise ValueError(
            f"Embedding ({emb.shape[0]} lignes) incohérent avec le vocab "
            f"({len(kept)} tokens).")
    with torch.no_grad():
        mean = emb.float().mean(dim=0).to(emb.dtype)
        emb[victim] = mean
    ckpt["vocab"] = new_vocab
    return ckpt, new_vocab


def main() -> None:
    p = argparse.ArgumentParser(description="Convertit un .pt vers unk distinct.")
    p.add_argument("--ckpt", required=True, help="Checkpoint source (ancien vocab).")
    p.add_argument("--out", required=True, help="Checkpoint converti (.pt).")
    p.add_argument("--vocab-out", default="",
                   help="Vocab JSON de sortie (défaut : <out>.vocab.json).")
    a = p.parse_args()
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    evicted = ckpt["vocab"]["kept_ids"][-1]
    ckpt, new_vocab = convert_vocab_in_ckpt(ckpt)
    torch.save(ckpt, a.out)
    vpath = a.vocab_out or (a.out + ".vocab.json")
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(new_vocab, f)
    print(f"[convert] {a.ckpt} -> {a.out} (+ {vpath}) : "
          f"token {evicted} évincé, ligne {new_vocab['unk_new']} = unk "
          f"(moyenne des embeddings), eos ligne {new_vocab['eos_new']}.")


if __name__ == "__main__":
    main()
