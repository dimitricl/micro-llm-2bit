#!/usr/bin/env python3
"""Surveille le pipeline MLX hybride complet et envoie les événements sur Telegram."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from tools.telegram_notify import load_telegram_config, send_message
except ImportError:
    from telegram_notify import load_telegram_config, send_message


STEP_RE = re.compile(
    r"\[train\] step\s+(\d+)/(\d+) \| loss:?\s*([0-9.]+)"
    r"(?: \| ppl [0-9.]+)? \| lr:?\s*([0-9.eE+-]+)"
    r" \| ([\d,]+) tok/s(?: \| elapsed:?\s*(\d+)s)?"
)
CHECKPOINT_RE = re.compile(r"Checkpoint sauvegardé:\s*(.+?)\s*$")
STAGE_RE = re.compile(r"^\[hybrid\] (.+)$")
ERROR_MARKS = ("Traceback (most recent call last)", "OutOfMemory", "RuntimeError")


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    return f"{days}j {hours:02}h" if days else f"{hours}h {minutes:02}m"


class Watcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.offset = 0
        self.stage = "initialisation"
        self.step = self.total = self.tok_s = 0
        self.loss = 0.0
        self.log_changed = time.time()
        self.last_status = time.time()
        self.last_step_notification = 0
        self.last_checkpoint = ""
        self.finished = False
        self.failed = False

    def send(self, text: str) -> None:
        print(f"[watch-hybrid] {datetime.now():%H:%M:%S} {text.splitlines()[0]}", flush=True)
        try:
            if not send_message(text):
                print("[watch-hybrid] envoi Telegram échoué", file=sys.stderr, flush=True)
        except Exception as exc:  # Network/API failures must not kill monitoring.
            print(f"[watch-hybrid] Telegram indisponible: {exc}", file=sys.stderr, flush=True)

    def eta(self) -> str:
        if self.step <= 0 or self.total <= 0 or self.tok_s <= 0:
            return "en attente des premières mesures"
        remaining_tokens = (self.total - self.step) * self.args.tokens_per_step
        return duration(remaining_tokens / self.tok_s)

    def progress_message(self) -> str:
        pct = 100 * self.step / self.total if self.total else 0
        return (f"📊 MLX hybride — {self.stage}\n"
                f"Étape {self.step:,}/{self.total:,} ({pct:.1f} %) · loss {self.loss:.4f}\n"
                f"{self.tok_s:,} tokens/s · ETA de l’étape {self.eta()}")

    def consume(self, line: str, notify: bool = True) -> None:
        line = line.rstrip()
        stage_match = STAGE_RE.match(line)
        if stage_match:
            self.stage = stage_match.group(1)
            if "Pipeline terminé" in self.stage:
                self.finished = True
                if notify:
                    self.send("✅ Pipeline MLX hybride terminé : pré-entraînement, SFT, export et benchmark.")
            elif notify:
                self.send(f"🔄 Pipeline MLX hybride — {self.stage}")

        match = STEP_RE.search(line)
        if match:
            self.step, self.total = int(match.group(1)), int(match.group(2))
            self.loss, self.tok_s = float(match.group(3)), int(match.group(5).replace(",", ""))
            if notify and self.step and self.step % self.args.notify_every_steps == 0:
                if self.step != self.last_step_notification:
                    self.last_step_notification = self.step
                    self.send(self.progress_message())

        checkpoint = CHECKPOINT_RE.search(line)
        if checkpoint and notify:
            path = Path(checkpoint.group(1).strip())
            if str(path) != self.last_checkpoint:
                self.last_checkpoint = str(path)
                try:
                    size = f"{path.stat().st_size / 1e6:.1f} Mo"
                except OSError:
                    size = "taille indisponible"
                self.send(f"📦 Checkpoint MLX sauvegardé — step {self.step:,}/{self.total:,}\n{path.name} ({size})")

        if any(mark in line for mark in ERROR_MARKS) and not self.failed:
            self.failed = True
            if notify:
                self.send(f"⚠️ Erreur dans le pipeline MLX hybride\n{line[:600]}")

    def read_new(self, notify: bool = True) -> bool:
        try:
            size = self.args.log.stat().st_size
        except OSError:
            return False
        if size < self.offset:
            self.offset = 0
        if size == self.offset:
            return False
        with self.args.log.open("r", encoding="utf-8", errors="replace") as log:
            log.seek(self.offset)
            for line in log:
                self.consume(line, notify=notify)
            self.offset = log.tell()
        self.log_changed = time.time()
        return True

    @staticmethod
    def alive(pid: int) -> bool:
        if not pid:
            return True
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def run(self) -> int:
        if not self.args.log.exists():
            print(f"[watch-hybrid] journal absent: {self.args.log}", file=sys.stderr)
            return 2
        # Reconstitute current state without replaying old alerts.
        self.read_new(notify=False)
        self.send("👀 Surveillance Telegram activée pour le pipeline MLX hybride.\n" +
                  self.progress_message())
        self.last_status = time.time()

        while not self.finished:
            self.read_new()
            now = time.time()
            if self.args.status_every_min and now - self.last_status >= self.args.status_every_min * 60:
                self.send(self.progress_message())
                self.last_status = now
            if self.args.max_stall_min and now - self.log_changed >= self.args.max_stall_min * 60:
                self.send(f"🟠 Le journal MLX n’a pas changé depuis {duration(now - self.log_changed)}.\n" +
                          self.progress_message())
                self.log_changed = now  # One reminder per stall interval.
            if not self.alive(self.args.pid):
                if not self.finished and not self.failed:
                    self.send("🔴 Le processus du pipeline s’est arrêté avant le message de fin.\n" +
                              self.progress_message())
                return 1 if self.failed or not self.finished else 0
            time.sleep(self.args.poll)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Surveille le pipeline MLX hybride sur Telegram")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--poll", type=int, default=15)
    parser.add_argument("--status-every-min", type=int, default=60)
    parser.add_argument("--max-stall-min", type=int, default=20)
    parser.add_argument("--notify-every-steps", type=int, default=500)
    parser.add_argument("--tokens-per-step", type=int, default=8192)
    args = parser.parse_args()
    try:
        load_telegram_config()
    except RuntimeError as exc:
        print(f"[watch-hybrid] configuration Telegram invalide: {exc}", file=sys.stderr)
        return 2
    return Watcher(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
