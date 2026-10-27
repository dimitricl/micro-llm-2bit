#!/usr/bin/env python3
"""Build a filtered French chat + verified JavaScript SFT source set.

Sources:
  * angeluriot/french_instruct (MIT), human-authored and human-style subset
  * REXX-NEW/github-top-code (MIT), files explicitly labelled JavaScript

Requires `datasets` and Node.js. Emits JSONL files consumed by
`prepare_hybrid_sft_mlx.py`; it does not start training.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_chat(max_examples: int, seed: int) -> tuple[list[dict], int]:
    from datasets import load_dataset

    rng = random.Random(seed)
    reservoir: list[dict] = []
    eligible = 0
    ds = load_dataset("angeluriot/french_instruct", split="train")
    for row in ds:
        if row.get("author") != "human" or row.get("style") != "human":
            continue
        if row.get("code"):
            continue
        conversation = row.get("conversation") or []
        messages = []
        for message in conversation:
            role = message.get("role")
            content = str(message.get("text", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        if (len(messages) < 2 or messages[0]["role"] != "user" or
                not any(m["role"] == "assistant" for m in messages)):
            continue
        eligible += 1
        item = {"messages": messages, "source": row.get("source", "")}
        if len(reservoir) < max_examples:
            reservoir.append(item)
        else:
            slot = rng.randrange(eligible)
            if slot < max_examples:
                reservoir[slot] = item
    rng.shuffle(reservoir)
    return reservoir, eligible


PYTHON_SIGNATURES = re.compile(
    r"(?m)^\s*(?:async\s+)?def\s+\w+\s*\(|"
    r"^\s*from\s+[\w.]+\s+import\s+|"
    r"^\s*import\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*$"
)


def valid_javascript(path: str, content: str) -> bool:
    suffix = Path(path).suffix.lower()
    if suffix not in {".js", ".mjs", ".cjs"}:
        return False
    if len(content) < 120 or len(content) > 50_000 or len(content.splitlines()) < 5:
        return False
    if PYTHON_SIGNATURES.search(content):
        return False
    # Parse as an ES module to verify syntax instead of guessing from keywords.
    result = subprocess.run(
        ["node", "--check", "--input-type=module"], input=content,
        text=True, capture_output=True, timeout=10,
    )
    return result.returncode == 0


def collect_javascript(max_files: int) -> tuple[list[dict], int, int]:
    from datasets import load_dataset

    rows: list[dict] = []
    seen = 0
    rejected = 0
    ds = load_dataset("REXX-NEW/github-top-code", split="train", streaming=True)
    for row in ds:
        if str(row.get("file_language", "")).lower() != "javascript":
            continue
        seen += 1
        path = str(row.get("file_path", ""))
        content = str(row.get("content", "")).strip()
        if not valid_javascript(path, content):
            rejected += 1
            continue
        rows.append({"language": "javascript", "file_path": path,
                     "text": content, "repo": row.get("repo_name", "")})
        if len(rows) >= max_files:
            break
    return rows, seen, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/hybrid-v2"))
    parser.add_argument("--max-chat", type=int, default=5000)
    parser.add_argument("--max-js-files", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chat, eligible_chat = collect_chat(args.max_chat, args.seed)
    print(f"[build] Chat humain/humain conservé: {len(chat)} / {eligible_chat}", flush=True)
    js, seen_js, rejected_js = collect_javascript(args.max_js_files)
    print(f"[build] JavaScript analysé: {seen_js}, rejeté: {rejected_js}, "
          f"conservé: {len(js)}", flush=True)
    if not chat or not js:
        raise SystemExit("Jeu incomplet: aucun chat ou JavaScript validé; aucun fichier écrit.")

    write_jsonl(args.out_dir / "chat_train.jsonl", chat)
    write_jsonl(args.out_dir / "instruction_train.jsonl", [])
    write_jsonl(args.out_dir / "javascript_train.jsonl", js)
    manifest = {
        "chat_source": "angeluriot/french_instruct (author=human, style=human, code=false)",
        "chat_license": "MIT",
        "chat_eligible": eligible_chat,
        "chat_written": len(chat),
        "javascript_source": "REXX-NEW/github-top-code (file_language=JavaScript)",
        "javascript_license": "MIT (dataset card; permissive source licenses)",
        "javascript_inspected": seen_js,
        "javascript_rejected": rejected_js,
        "javascript_written": len(js),
        "seed": args.seed,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
