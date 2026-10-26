#!/usr/bin/env python3
"""Surveille l'entraînement MLX et pousse l'état sur Telegram.

Surveille `logs/train_25m.log` (ou --log) et envoie :
  - un message au démarrage avec l'état courant ;
  - une notification à chaque checkpoint sauvegardé ;
  - un point d'étape périodique (--status-every-min) ;
  - une alerte si le processus meurt ou si le log stagne ;
  - un message final quand l'entraînement se termine.

Le token Telegram est lu depuis l'environnement ou ~/.env (voir
tools/telegram_notify.py). Aucun secret n'est écrit dans les logs.

Usage :
    python tools/watch_training.py --pid 67588
    python tools/watch_training.py --status-now     # état immédiat, puis sortie
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:  # exécution depuis la racine du dépôt
    from tools.telegram_notify import load_telegram_config, send_message
except ImportError:  # exécution directe : python tools/watch_training.py
    from telegram_notify import load_telegram_config, send_message

STEP_RE = re.compile(
    r"\[train\] step\s+(\d+)/(\d+) \| loss: ([0-9.]+) \| lr: ([0-9.eE+-]+) "
    r"\| ([\d,]+) tok/s \| elapsed: (\d+)s"
)
CKPT_RE = re.compile(r"Checkpoint sauvé\s*:\s*(.+?)\s*$")
DONE_MARK = "Entraînement terminé"
ERROR_MARKS = ("Traceback (most recent call last)", "❌", "RuntimeError", "OutOfMemory")


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rest = divmod(seconds, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes = rest // 60
    if days:
        return f"{days}j {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def fmt_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    return f"{size / 1e6:.1f} Mo"


class TrainingWatcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log_path: Path = args.log
        self.state_path: Path = args.state
        self.state = self._load_state()

        self.step = 0
        self.total_steps = 0
        self.loss = 0.0
        self.lr = 0.0
        self.tok_s = 0
        self.elapsed = 0
        self.done = bool(self.state.get("done"))
        self.last_change = time.time()
        self.last_size = 0
        self.stall_alerted = False
        self.error_sent = bool(self.state.get("error_sent"))
        self.last_notified_step = int(self.state.get("last_notified_step", 0))

    # ── état persistant ──────────────────────────────────────────────────────
    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self) -> None:
        self.state.update(
            {
                "last_checkpoint": self.state.get("last_checkpoint"),
                "last_status_ts": self.state.get("last_status_ts", 0),
                "done": self.done,
                "error_sent": self.error_sent,
                "last_notified_step": self.last_notified_step,
            }
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    # ── parsing ──────────────────────────────────────────────────────────────
    def _consume_line(self, line: str) -> None:
        line = line.rstrip("\n")

        match = STEP_RE.search(line)
        if match:
            self.step = int(match.group(1))
            self.total_steps = int(match.group(2))
            self.loss = float(match.group(3))
            self.lr = float(match.group(4))
            self.tok_s = int(match.group(5).replace(",", ""))
            self.elapsed = int(match.group(6))

            # Notify every 500 steps
            if self.step > 0 and self.step % 500 == 0 and self.step != self.last_notified_step:
                self.last_notified_step = self.step
                self._save_state()
                self._notify_progress()
            return

        match = CKPT_RE.search(line)
        if match:
            path = match.group(1).strip()
            if path != self.state.get("last_checkpoint"):
                self.state["last_checkpoint"] = path
                self._save_state()
                self._notify_checkpoint(Path(path))
            return

        if DONE_MARK in line:
            self.done = True
            self._save_state()
            self._notify_done()
            return

        if any(mark in line for mark in ERROR_MARKS) and not self.error_sent:
            self.error_sent = True
            self._save_state()
            self._notify_error(line)

    # ── messages ─────────────────────────────────────────────────────────────
    def _eta(self) -> str:
        if self.step <= 0 or self.total_steps <= 0:
            return "?"
        remaining = (self.total_steps - self.step) * (self.elapsed / self.step)
        return fmt_duration(remaining)

    def _progress(self) -> str:
        if not self.total_steps:
            return "?"
        return f"{100 * self.step / self.total_steps:.1f} %"

    def _last_lines(self, count: int = 6) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "(log illisible)"
        return "\n".join(lines[-count:])

    def _notify_checkpoint(self, path: Path) -> None:
        text = (
            f"📦 Checkpoint {fmt_int(self.step)} / {fmt_int(self.total_steps)} "
            f"({self._progress()})\n"
            f"loss {self.loss:.4f} · lr {self.lr:.2e}\n"
            f"{fmt_int(self.tok_s)} tok/s · écoulé {fmt_duration(self.elapsed)}\n"
            f"ETA {self._eta()}\n"
            f"Fichier : {path.name} ({fmt_size(path)})"
        )
        self._send(text)

    def _notify_progress(self) -> None:
        text = (
            f"📊 Progression {fmt_int(self.step)} / {fmt_int(self.total_steps)} "
            f"({self._progress()})\n"
            f"loss {self.loss:.4f} · lr {self.lr:.2e}\n"
            f"{fmt_int(self.tok_s)} tok/s · écoulé {fmt_duration(self.elapsed)}\n"
            f"ETA {self._eta()}"
        )
        self._send(text)

    def _notify_status(self) -> None:
        if self.done:
            return
        text = (
            f"📊 Training 25M — {fmt_int(self.step)} / {fmt_int(self.total_steps)} "
            f"({self._progress()})\n"
            f"loss {self.loss:.4f} · lr {self.lr:.2e} · {fmt_int(self.tok_s)} tok/s\n"
            f"écoulé {fmt_duration(self.elapsed)} · ETA {self._eta()}"
        )
        self._send(text)
        self.state["last_status_ts"] = time.time()
        self._save_state()

    def _notify_done(self) -> None:
        text = (
            f"✅ Entraînement terminé — {fmt_int(self.step)} / {fmt_int(self.total_steps)}\n"
            f"loss finale {self.loss:.4f} · lr {self.lr:.2e}\n"
            f"durée {fmt_duration(self.elapsed)}"
        )
        self._send(text)

    def _notify_error(self, line: str) -> None:
        text = (
            "⚠️ Erreur détectée dans le log d'entraînement\n\n"
            f"step {fmt_int(self.step)} / {fmt_int(self.total_steps)}\n"
            f"{line[:500]}"
        )
        self._send(text)

    def _notify_start(self) -> None:
        text = (
            f"👀 Surveillance active — {self.log_path.name}\n"
            f"step {fmt_int(self.step)} / {fmt_int(self.total_steps)} ({self._progress()})\n"
            f"loss {self.loss:.4f} · {fmt_int(self.tok_s)} tok/s · ETA {self._eta()}\n"
            f"Checkpoints : toutes les {fmt_int(max(0, self.args.save_every))} steps"
        )
        self._send(text)

    def _notify_stall(self) -> None:
        if self.args.pid and not self._pid_alive(self.args.pid):
            text = (
                f"🔴 Le processus d'entraînement ({self.args.pid}) n'existe plus.\n"
                f"Dernier step {fmt_int(self.step)} / {fmt_int(self.total_steps)} · "
                f"loss {self.loss:.4f}"
            )
        else:
            text = (
                f"🟠 Aucune progression du log depuis "
                f"{fmt_duration(time.time() - self.last_change)}.\n"
                f"Dernier step {fmt_int(self.step)} / {fmt_int(self.total_steps)} · "
                f"loss {self.loss:.4f}"
            )
        self._send(text)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _send(self, text: str) -> None:
        print(f"[watch] {datetime.now():%H:%M:%S} → Telegram ({text.splitlines()[0]})", flush=True)
        try:
            send_message(text)
        except RuntimeError as exc:
            print(f"[watch] Telegram indisponible : {exc}", flush=True)

    # ── boucle ───────────────────────────────────────────────────────────────
    def send_status_now(self) -> None:
        self._read_new(reset=True)
        self._notify_status()

    def run(self) -> int:
        if not self.log_path.exists():
            print(f"[watch] log introuvable : {self.log_path}", file=sys.stderr)
            return 2

        self.last_size = self.log_path.stat().st_size
        self.last_change = time.time()
        self._read_new(reset=True)
        self._notify_start()
        # Le message de démarrage contient déjà l'état : évite un doublon immédiat.
        self.state["last_status_ts"] = time.time()
        self._save_state()

        while not self.done:
            grew = self._read_new()
            now = time.time()
            if grew:
                self.last_change = now
                self.stall_alerted = False

            last_status = float(self.state.get("last_status_ts", 0))
            if self.args.status_every_min > 0 and now - last_status >= self.args.status_every_min * 60:
                self._notify_status()

            if (
                self.args.max_stall_min > 0
                and not self.stall_alerted
                and now - self.last_change >= self.args.max_stall_min * 60
            ):
                self.stall_alerted = True
                self._notify_stall()

            time.sleep(self.args.poll)

        return 0

    def _read_new(self, reset: bool = False) -> bool:
        """Lit les nouvelles lignes. reset=True relit tout le fichier."""
        try:
            size = self.log_path.stat().st_size
        except OSError:
            return False

        if reset:
            self.last_size = 0

        grew = size != self.last_size
        if not grew:
            return False

        with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.last_size)
            for line in handle:
                self._consume_line(line)
            self.last_size = handle.tell()
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Notifie l'entraînement sur Telegram")
    parser.add_argument("--log", type=Path, default=Path("logs/train_25m.log"))
    parser.add_argument("--state", type=Path, default=Path("logs/train_25m_notifier.json"))
    parser.add_argument("--pid", type=int, default=0, help="PID du processus d'entraînement")
    parser.add_argument("--poll", type=int, default=10, help="Intervalle de lecture (s)")
    parser.add_argument("--status-every-min", type=int, default=60)
    parser.add_argument("--max-stall-min", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--status-now", action="store_true", help="Envoie l'état puis sort")
    args = parser.parse_args()

    try:
        load_telegram_config()
    except RuntimeError as exc:
        print(f"[watch] configuration Telegram invalide : {exc}", file=sys.stderr)
        return 2

    watcher = TrainingWatcher(args)
    if args.status_now:
        watcher.send_status_now()
        return 0
    try:
        return watcher.run()
    except KeyboardInterrupt:
        print("[watch] arrêt demandé", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
