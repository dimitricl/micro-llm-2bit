#!/usr/bin/env python3
"""Interface d'inférence hybride avec détection automatique du mode.

Détecte si l'input ressemble à du code ou du chat, et applique les
paramètres de génération appropriés.

Usage:
    python tools/infer_hybrid.py \
        --model runs/hybrid-final/model.bin \
        --prompt "comment trier une liste en Python ?" \
        --mode auto
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal


def detect_mode(prompt: str) -> Literal["code", "chat"]:
    """Détecte si le prompt ressemble à du code ou du chat."""
    # Indicateurs de code
    code_keywords = [
        "function", "def ", "class ", "import ", "from ",
        "return", "if ", "for ", "while ", "try:",
        "const ", "let ", "var ", "=>", "() =>",
        "#", "//", "/*", "*/", "<script",
        "python", "javascript", "typescript", "java", "c++",
        "function", "method", "class", "algorithm",
        "code", "program", "script", "syntax",
        "bug", "error", "debug", "fix",
    ]

    # Indicateurs de chat
    chat_keywords = [
        "bonjour", "salut", "merci", "svp", "s'il vous plaît",
        "explique", "qu'est-ce que", "comment", "pourquoi",
        "dis-moi", "parle-moi", "raconte",
        "penses-tu", "opinion", "avis",
        "histoire", "exemple", "définition",
    ]

    prompt_lower = prompt.lower()

    # Compter les occurrences
    code_score = sum(1 for kw in code_keywords if kw in prompt_lower)
    chat_score = sum(1 for kw in chat_keywords if kw in prompt_lower)

    # Détecter patterns de code
    code_patterns = [
        r"def\s+\w+\s*\(",  # Python function
        r"function\s+\w+\s*\(",  # JS function
        r"class\s+\w+",  # Class definition
        r"import\s+\w+",  # Import
        r"=\s*function",  # Arrow function
        r"#.*$",  # Python comment
        r"//.*$",  # JS comment
    ]

    for pattern in code_patterns:
        if re.search(pattern, prompt, re.MULTILINE):
            code_score += 2

    # Décision
    if code_score > chat_score:
        return "code"
    else:
        return "chat"


def get_generation_params(mode: Literal["code", "chat"]) -> dict:
    """Retourne les paramètres de génération selon le mode."""
    if mode == "code":
        return {
            "temperature": 0.2,
            "top_k": 10,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "no_repeat_ngram": 0,
        }
    else:  # chat
        return {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "no_repeat_ngram": 3,
        }


def format_chat_prompt(prompt: str) -> str:
    """Formate le prompt pour le mode chat."""
    return f"<user>{prompt}</user><assistant>"


def format_code_prompt(prompt: str) -> str:
    """Formate le prompt pour le mode code."""
    # Pour le code, on garde tel quel ou on ajoute un préfixe si nécessaire
    if not prompt.startswith("def") and not prompt.startswith("function"):
        # Ajouter un préfixe subtil si ça ressemble à une question
        if "?" in prompt or "comment" in prompt.lower():
            return f"# {prompt}\n"
    return prompt


def infer_hybrid(
    model: Path,
    prompt: str,
    mode: Literal["auto", "code", "chat"] = "auto",
    max_new: int = 256,
) -> tuple[str, str]:
    """Effectue l'inférence hybride."""
    # Détection du mode
    if mode == "auto":
        detected_mode = detect_mode(prompt)
        print(f"[infer] Mode détecté: {detected_mode}")
    else:
        detected_mode = mode
        print(f"[infer] Mode forcé: {detected_mode}")

    # Paramètres de génération
    params = get_generation_params(detected_mode)
    print(f"[infer] Params: {params}")

    # Formatage du prompt
    if detected_mode == "chat":
        formatted_prompt = format_chat_prompt(prompt)
    else:
        formatted_prompt = format_code_prompt(prompt)

    # Construction de la commande main.py infer
    cmd = [
        "python",
        "main.py",
        "infer",
        "--model", str(model),
        "--prompt", formatted_prompt,
        "--max-new", str(max_new),
        "--temperature", str(params["temperature"]),
        "--top-k", str(params["top_k"]),
        "--top-p", str(params["top_p"]),
        "--repetition-penalty", str(params["repetition_penalty"]),
        "--no-repeat-ngram", str(params["no_repeat_ngram"]),
    ]

    # Exécution
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        response = result.stdout.strip()
        return response, detected_mode
    except subprocess.CalledProcessError as e:
        print(f"[infer] Erreur: {e.stderr}")
        return "", detected_mode


def interactive_mode(model: Path, mode: Literal["auto", "code", "chat"] = "auto") -> None:
    """Mode interactif."""
    print(f"[interactive] Mode: {mode}")
    print("[interactive] Tapez 'quit' pour sortir")

    while True:
        try:
            prompt = input("\n> ")
            if prompt.lower() in ["quit", "exit", "q"]:
                break

            response, detected_mode = infer_hybrid(model, prompt, mode)
            print(f"\n[{detected_mode}] {response}")

        except KeyboardInterrupt:
            print("\n[interactive] Interruption")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inférence hybride avec détection automatique"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Chemin vers le modèle .bin"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt unique (sinon mode interactif)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "code", "chat"],
        help="Mode de génération"
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=256,
        help="Nombre max de tokens à générer"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Mode interactif"
    )

    args = parser.parse_args()

    if args.interactive or not args.prompt:
        interactive_mode(args.model, args.mode)
    else:
        response, detected_mode = infer_hybrid(
            args.model,
            args.prompt,
            args.mode,
            args.max_new,
        )
        print(f"[{detected_mode}] {response}")


if __name__ == "__main__":
    main()
