#!/usr/bin/env python3
"""Génère des données d'entraînement synthétiques hybrides avec qwen2.5-coder:1.5b.

Utilise Ollama pour générer des paires instruction/réponse dans deux modes:
- code: complétion de code et explications de code
- chat: conversation en français

Usage:
    python tools/generate_hybrid_synthetic.py \
        --output data/hybrid/synthetic.jsonl \
        --num-samples 10000 \
        --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Literal


def call_ollama(
    prompt: str,
    model: str = "qwen2.5-coder:1.5b",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Appelle Ollama pour générer une réponse."""
    cmd = [
        "ollama",
        "run",
        model,
        prompt,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[ollama] Erreur: {result.stderr}")
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("[ollama] Timeout")
        return ""
    except Exception as e:
        print(f"[ollama] Exception: {e}")
        return ""


def generate_code_sample(model: str) -> dict | None:
    """Génère un échantillon de code."""
    templates = [
        "Écris une fonction Python qui {task}.",
        "Comment implémenter {task} en JavaScript ?",
        "Explique ce code: {code}",
        "Corrige ce bug: {code}",
    ]

    tasks = [
        "trie une liste",
        "inverse une chaîne",
        "calcule la factorielle",
        "vérifie si un nombre est premier",
        "convertit Celsius en Fahrenheit",
        "fait une requête HTTP",
        "parse un JSON",
        "gère les erreurs",
    ]

    template = random.choice(templates)
    task = random.choice(tasks)

    if "{code}" in template:
        # Pour l'explication/correction de code
        code_samples = [
            "def foo(x): return x + 1",
            "function add(a, b) { return a + b }",
            "x = [1, 2, 3]; x.sort()",
        ]
        code = random.choice(code_samples)
        prompt = template.format(code=code)
    else:
        prompt = template.format(task=task)

    response = call_ollama(prompt, model, temperature=0.3)

    if not response:
        return None

    return {
        "mode": "code",
        "instruction": prompt,
        "response": response,
    }


def generate_chat_sample(model: str) -> dict | None:
    """Génère un échantillon de conversation FR."""
    templates = [
        "Explique {concept} en termes simples.",
        "Quelle est la différence entre {a} et {b} ?",
        "Donne un exemple de {concept}.",
        "Comment fonctionne {concept} ?",
        "Pourquoi {concept} est important ?",
    ]

    concepts = [
        "l'intelligence artificielle",
        "la blockchain",
        "le machine learning",
        "les réseaux de neurones",
        "le cloud computing",
        "la cybersécurité",
        "DevOps",
        "le microservices",
    ]

    pairs = [
        ("Python", "JavaScript"),
        ("SQL", "NoSQL"),
        ("frontend", "backend"),
        ("git", "svn"),
        ("monolithique", "microservices"),
    ]

    if random.random() < 0.7:
        template = random.choice(templates)
        if "{a}" in template and "{b}" in template:
            a, b = random.choice(pairs)
            prompt = template.format(a=a, b=b)
        else:
            concept = random.choice(concepts)
            prompt = template.format(concept=concept)
    else:
        # Questions générales
        questions = [
            "Qu'est-ce que tu penses de l'avenir de l'IA ?",
            "Quel est ton framework préféré et pourquoi ?",
            "Explique ton processus de debugging.",
            "Quel conseil donnerais-tu à un développeur junior ?",
        ]
        prompt = random.choice(questions)

    response = call_ollama(prompt, model, temperature=0.7)

    if not response:
        return None

    return {
        "mode": "chat",
        "instruction": prompt,
        "response": response,
    }


def generate_hybrid_sample(model: str, code_ratio: float = 0.5) -> dict | None:
    """Génère un échantillon hybride (code ou chat)."""
    if random.random() < code_ratio:
        return generate_code_sample(model)
    else:
        return generate_chat_sample(model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère données synthétiques hybrides avec Ollama"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hybrid/synthetic.jsonl"),
        help="Fichier de sortie"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Nombre d'échantillons à générer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder:1.5b",
        help="Modèle Ollama à utiliser"
    )
    parser.add_argument(
        "--code-ratio",
        type=float,
        default=0.5,
        help="Ratio code vs chat (0.0 = 100% chat, 1.0 = 100% code)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Taille des batches pour écriture"
    )

    args = parser.parse_args()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[generate] Output: {output_path}")
    print(f"[generate] Model: {args.model}")
    print(f"[generate] Num samples: {args.num_samples}")
    print(f"[generate] Code ratio: {args.code_ratio}")

    samples = []
    failed = 0

    for i in range(args.num_samples):
        if i % 100 == 0:
            print(f"[generate] Progress: {i}/{args.num_samples} ({failed} failed)")

        sample = generate_hybrid_sample(args.model, args.code_ratio)

        if sample:
            samples.append(sample)
        else:
            failed += 1

        # Écrire par batch
        if len(samples) >= args.batch_size:
            with open(output_path, "a", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            samples.clear()

        # Petite pause pour éviter de surcharger Ollama
        time.sleep(0.1)

    # Écrire les derniers
    if samples:
        with open(output_path, "a", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n[generate] Terminé: {args.num_samples - failed}/{args.num_samples} réussis")
    print(f"[generate] Output: {output_path}")


if __name__ == "__main__":
    main()
