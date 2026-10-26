#!/usr/bin/env python3
"""Petit client Telegram sans dépendance (urllib uniquement).

Lit le token et le destinataire depuis l'environnement, sinon depuis ~/.env
(le fichier de configuration du bot OpenCode Telegram).

Usage :
    python tools/telegram_notify.py "message"
    echo "message" | python tools/telegram_notify.py -
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_ENV_FILE = Path.home() / ".env"
API_BASE = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 3900  # marge sous la limite Telegram de 4096


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse un fichier .env simple (KEY=VALUE, commentaires #)."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_telegram_config(env_file: Path | None = None) -> tuple[str, str]:
    """Retourne (token, chat_id) depuis l'env puis ~/.env.

    Lève RuntimeError si la configuration est incomplète.
    """
    file_values = _parse_env_file(env_file or DEFAULT_ENV_FILE)

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or file_values.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or file_values.get("TELEGRAM_CHAT_ID", "")

    if not chat_id:
        allowed = os.environ.get("ALLOWED_USERS") or file_values.get("ALLOWED_USERS", "")
        chat_id = next((part.strip() for part in allowed.split(",") if part.strip()), "")

    token = token.strip()
    chat_id = chat_id.strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN absent (env ou ~/.env)")
    if not chat_id:
        raise RuntimeError("Aucun destinataire : TELEGRAM_CHAT_ID / ALLOWED_USERS vide")
    return token, chat_id


def _split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Découpe un message trop long sur les sauts de ligne."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                chunks.append("".join(current).rstrip())
                current, size = [], 0
            chunks.append(line[:limit])
            line = line[limit:]
        if size + len(line) > limit:
            chunks.append("".join(current).rstrip())
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current).rstrip())
    return [c for c in chunks if c]


def send_message(
    text: str,
    *,
    token: str | None = None,
    chat_id: str | None = None,
    silent: bool = False,
    retries: int = 3,
) -> bool:
    """Envoie un message texte. Retourne True si tous les morceaux sont partis."""
    if token is None or chat_id is None:
        token, chat_id = load_telegram_config()

    ok = True
    for chunk in _split_message(text):
        payload = urllib.parse.urlencode(
            {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": "true",
                "disable_notification": "true" if silent else "false",
            }
        ).encode("utf-8")

        sent = False
        last_error = ""
        for attempt in range(1, retries + 1):
            request = urllib.request.Request(
                f"{API_BASE}/bot{token}/sendMessage",
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if body.get("ok"):
                    sent = True
                    break
                last_error = str(body.get("description", body))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
            if attempt < retries:
                time.sleep(2 * attempt)

        if not sent:
            print(f"[telegram] échec d'envoi : {last_error}", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Envoie un message Telegram")
    parser.add_argument("message", help="Texte à envoyer, ou '-' pour lire stdin")
    parser.add_argument("--silent", action="store_true", help="Notification sans son")
    args = parser.parse_args()

    text = sys.stdin.read() if args.message == "-" else args.message
    text = text.strip()
    if not text:
        print("[telegram] message vide", file=sys.stderr)
        return 2

    try:
        ok = send_message(text, silent=args.silent)
    except RuntimeError as exc:
        print(f"[telegram] configuration invalide : {exc}", file=sys.stderr)
        return 2
    print("[telegram] ✅ envoyé" if ok else "[telegram] ❌ échec", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
