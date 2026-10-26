#!/usr/bin/env python3
"""Télécharge et prépare les datasets pour l'entraînement hybride chat + code.

Datasets cibles :
- Conversation FR : OpenAssistant/oasst1, HuggingFaceH4/no_robots
- Code : bigcode/the-stack (Python, JS, TS), HuggingFaceH4/codeparrot
- Instruction-following : Open-Orca/OpenOrca

Usage:
    python tools/download_hybrid_data.py --output-dir data/hybrid --max-samples 100000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "datasets non installé. Installez avec: pip install datasets"
    )


def download_oasst1(output_dir: Path, max_samples: int = 50000) -> Path:
    """Télécharge OpenAssistant OASST1 (dialogues FR)."""
    print("[download] OASST1 - dialogue FR")
    output_path = output_dir / "oasst1_fr.jsonl"

    if output_path.exists():
        print(f"[download] {output_path} existe déjà, skip")
        return output_path

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    # Filtrer les dialogues en français
    fr_dialogues = [
        item for item in ds
        if item.get("lang") == "fr" and item.get("text")
    ][:max_samples]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in fr_dialogues:
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")

    print(f"[download] OASST1 FR: {len(fr_dialogues)} exemples -> {output_path}")
    return output_path


def download_no_robots(output_dir: Path, max_samples: int = 30000) -> Path:
    """Télécharge no_robots (instruction-following FR)."""
    print("[download] no_robots - instruction-following FR")
    output_path = output_dir / "no_robots_fr.jsonl"

    if output_path.exists():
        print(f"[download] {output_path} existe déjà, skip")
        return output_path

    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    # Filtrer les instructions en français
    fr_instructions = [
        item for item in ds
        if item.get("language") == "french" and item.get("instruction")
    ][:max_samples]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in fr_instructions:
            text = f"Instruction: {item['instruction']}\nRéponse: {item['response']}"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"[download] no_robots FR: {len(fr_instructions)} exemples -> {output_path}")
    return output_path


def download_the_stack(output_dir: Path, max_samples: int = 50000) -> Path:
    """Télécharge The Stack (code Python, JS, TS)."""
    print("[download] The Stack - code (Python, JS, TS)")
    output_path = output_dir / "the_stack_code.jsonl"

    if output_path.exists():
        print(f"[download] {output_path} existe déjà, skip")
        return output_path

    # Télécharger un subset pour Python, JS, TS
    languages = ["Python", "JavaScript", "TypeScript"]
    all_code = []

    for lang in languages:
        print(f"[download]   {lang}...")
        try:
            ds = load_dataset(
                "bigcode/the-stack",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True
            )
            lang_code = []
            for item in ds:
                if item.get("content"):
                    lang_code.append({"text": item["content"]})
                    if len(lang_code) >= max_samples // len(languages):
                        break
            all_code.extend(lang_code)
            print(f"[download]   {lang}: {len(lang_code)} exemples")
        except Exception as e:
            print(f"[download]   Erreur {lang}: {e}")
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_code:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[download] The Stack total: {len(all_code)} exemples -> {output_path}")
    return output_path


def download_codeparrot(output_dir: Path, max_samples: int = 30000) -> Path:
    """Télécharge CodeParrot (code Python)."""
    print("[download] CodeParrot - Python")
    output_path = output_dir / "codeparrot_python.jsonl"

    if output_path.exists():
        print(f"[download] {output_path} existe déjà, skip")
        return output_path

    ds = load_dataset("HuggingFaceH4/codeparrot", split="train", streaming=True)
    python_code = []

    for item in ds:
        if item.get("content"):
            python_code.append({"text": item["content"]})
            if len(python_code) >= max_samples:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        for item in python_code:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[download] CodeParrot: {len(python_code)} exemples -> {output_path}")
    return output_path


def download_openorca(output_dir: Path, max_samples: int = 40000) -> Path:
    """Télécharge OpenOrca (instruction-following multilingue)."""
    print("[download] OpenOrca - instruction-following")
    output_path = output_dir / "openorca_instructions.jsonl"

    if output_path.exists():
        print(f"[download] {output_path} existe déjà, skip")
        return output_path

    ds = load_dataset("Open-Orca/OpenOrca", split="train")
    instructions = []

    for item in ds:
        question = item.get("question", "")
        response = item.get("response", "")
        if question and response:
            text = f"Question: {question}\nRéponse: {response}"
            instructions.append({"text": text})
            if len(instructions) >= max_samples:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        for item in instructions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[download] OpenOrca: {len(instructions)} exemples -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Télécharge datasets pour entraînement hybride chat + code"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hybrid"),
        help="Dossier de sortie pour les datasets"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Nombre max d'échantillons par dataset"
    )
    parser.add_argument(
        "--skip-code",
        action="store_true",
        help="Skip les datasets de code"
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip les datasets de conversation"
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] Output dir: {output_dir}")
    print(f"[download] Max samples per dataset: {args.max_samples}")

    datasets = []

    # Datasets conversation
    if not args.skip_chat:
        datasets.append(download_oasst1(output_dir, args.max_samples))
        datasets.append(download_no_robots(output_dir, args.max_samples // 2))
        datasets.append(download_openorca(output_dir, args.max_samples))

    # Datasets code
    if not args.skip_code:
        datasets.append(download_the_stack(output_dir, args.max_samples))
        datasets.append(download_codeparrot(output_dir, args.max_samples // 2))

    print(f"\n[download] Téléchargement terminé: {len(datasets)} datasets")
    for ds in datasets:
        size_mb = ds.stat().st_size / 1024 / 1024
        print(f"  - {ds.name}: {size_mb:.2f} Mo")


if __name__ == "__main__":
    main()
