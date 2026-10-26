#!/usr/bin/env python3
"""Pipeline d'entraînement hybride pour modèle conversation + code.

Trois étapes :
1. Pré-entraînement langage mixte (50% code + 50% conversation)
2. Fine-tuning instruction-following (format chat)
3. Fine-tuning QA-passage (pour factualité)

Usage:
    # Étape 1 - Pré-entraînement
    python tools/train_hybrid.py \
        --stage pretrain \
        --data-dir data/hybrid \
        --output runs/hybrid-pretrain \
        --config hybrid

    # Étape 2 - Fine-tuning instruction
    python tools/train_hybrid.py \
        --stage instruct \
        --checkpoint runs/hybrid-pretrain/model.pt \
        --data-dir data/hybrid \
        --output runs/hybrid-instruct \
        --config hybrid

    # Étape 3 - Fine-tuning QA
    python tools/train_hybrid.py \
        --stage qa \
        --checkpoint runs/hybrid-instruct/model.pt \
        --qa-data data/qa/g_passage_v1.jsonl \
        --output runs/hybrid-final \
        --config hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def prepare_pretrain_data(data_dir: Path, output_dir: Path) -> Path:
    """Prépare les données de pré-entraînement mixte."""
    print("[pretrain] Préparation données mixtes code + conversation")

    # Charger tous les datasets
    datasets = []
    for file in data_dir.glob("*.jsonl"):
        if file.name.startswith("synthetic"):
            continue  # Skip synthétique pour pré-entraînement
        print(f"[pretrain]   Chargement {file.name}")
        with open(file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    datasets.append(json.loads(line))

    print(f"[pretrain]   Total documents: {len(datasets)}")

    # Mélanger et sauvegarder
    import random
    random.shuffle(datasets)

    output_file = output_dir / "pretrain_mixed.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in datasets:
            text = item.get("text", "")
            if text:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"[pretrain]   Sauvegardé: {output_file}")
    return output_file


def prepare_instruct_data(data_dir: Path, output_dir: Path) -> Path:
    """Prépare les données d'instruction-following avec format chat."""
    print("[instruct] Préparation données instruction-following")

    output_file = output_dir / "instruct_formatted.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        # Charger données synthétiques (déjà formatées)
        synthetic_file = data_dir / "synthetic.jsonl"
        if synthetic_file.exists():
            with open(synthetic_file, encoding="utf-8") as sf:
                for line in sf:
                    item = json.loads(line)
                    # Format: <user>instruction</user><assistant>response</assistant>
                    formatted = (
                        f"<user>{item['instruction']}</user>"
                        f"<assistant>{item['response']}</assistant>"
                    )
                    f.write(json.dumps({"text": formatted}, ensure_ascii=False) + "\n")

    print(f"[instruct]   Sauvegardé: {output_file}")
    return output_file


def run_pretrain(data_file: Path, output: Path, config: str) -> None:
    """Lance le pré-entraînement avec main.py train."""
    print(f"[pretrain] Lancement entraînement sur {data_file}")

    cmd = [
        "python",
        "main.py",
        "train",
        "--data", str(data_file),
        "--from-scratch",
        "--config", config,
        "--out", str(output / "model.pt"),
        "--vocab-size", "16000",
        "--seq-len", "512",
        "--save-batch", "8",
        "--batch-size", "8",
        "--accum", "16",
        "--epochs", "2",
        "--lr", "1e-3",
        "--alpha", "0",  # CE-only pour from-scratch
        "--temperature", "1.0",
        "--sched", "cosine",
        "--weight-decay", "0.1",
        "--wsd-warmup-frac", "0.05",
        "--wsd-decay-frac", "0.2",
        "--wsd-min-ratio", "0.1",
        "--wd-zero-frac", "0.5",
        "--ckpt-every", "500",
    ]

    print(f"[pretrain] Commande: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_instruct(data_file: Path, checkpoint: Path, output: Path, config: str) -> None:
    """Lance le fine-tuning instruction-following."""
    print(f"[instruct] Lancement fine-tuning sur {data_file}")

    cmd = [
        "python",
        "main.py",
        "train",
        "--data", str(data_file),
        "--parent", "dummy",  # Pas utilisé pour CE-only
        "--from-scratch",
        "--resume", str(checkpoint),
        "--config", config,
        "--vocab-from", str(checkpoint) + ".vocab.json",
        "--out", str(output / "model.pt"),
        "--seq-len", "512",
        "--save-batch", "8",
        "--batch-size", "8",
        "--accum", "16",
        "--epochs", "1",
        "--lr", "3e-4",
        "--alpha", "0",
        "--temperature", "1.0",
        "--sched", "cosine",
        "--weight-decay", "0.01",
        "--lr-resume", "3e-4",
        "--rewarmup-ratio", "0.05",
        "--reset-optimizer",
        "--ckpt-every", "200",
    ]

    print(f"[instruct] Commande: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_qa(qa_data: Path, checkpoint: Path, output: Path, config: str) -> None:
    """Lance le fine-tuning QA-passage (technique G1 validée)."""
    print(f"[qa] Lancement fine-tuning QA sur {qa_data}")

    cmd = [
        "python",
        "main.py",
        "train",
        "--qa-data", str(qa_data),
        "--parent", "HuggingFaceTB/SmolLM-135M",
        "--resume", str(checkpoint),
        "--config", config,
        "--vocab-from", str(checkpoint) + ".vocab.json",
        "--out", str(output / "model.pt"),
        "--seq-len", "256",
        "--save-batch", "8",
        "--batch-size", "8",
        "--accum", "16",
        "--qa-steps", "924",
        "--lr", "3e-4",
        "--alpha", "0",
        "--temperature", "2.0",
        "--sched", "cosine",
        "--weight-decay", "0.01",
        "--lr-resume", "3e-4",
        "--rewarmup-ratio", "0.05",
        "--reset-optimizer",
        "--mask-prompt", "answer",
        "--ckpt-every", "250",
    ]

    print(f"[qa] Commande: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline d'entraînement hybride chat + code"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["pretrain", "instruct", "qa"],
        help="Étape d'entraînement"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/hybrid"),
        help="Dossier des données"
    )
    parser.add_argument(
        "--qa-data",
        type=Path,
        help="Fichier QA (requis pour stage qa)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint précédent (requis pour instruct/qa)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Dossier de sortie"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hybrid",
        help="Configuration du modèle"
    )

    args = parser.parse_args()

    if args.stage == "pretrain":
        data_file = prepare_pretrain_data(args.data_dir, args.output)
        run_pretrain(data_file, args.output, args.config)

    elif args.stage == "instruct":
        if not args.checkpoint:
            parser.error("--checkpoint requis pour stage instruct")
        data_file = prepare_instruct_data(args.data_dir, args.output)
        run_instruct(data_file, args.checkpoint, args.output, args.config)

    elif args.stage == "qa":
        if not args.checkpoint:
            parser.error("--checkpoint requis pour stage qa")
        if not args.qa_data:
            parser.error("--qa-data requis pour stage qa")
        run_qa(args.qa_data, args.checkpoint, args.output, args.config)


if __name__ == "__main__":
    main()
