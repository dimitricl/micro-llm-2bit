"""Courbe d'un run : parse train/val PPL + tok/s, écrit docs/img/<run>.png.

Usage : nice -n 10 .venv/bin/python tools/plot_run.py
            --log runs/2026-10-08-tranche00/train.log --run tranche00
Sans GPU (matplotlib Agg). Fonctions parse_* testées (tests/test_plots.py).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib

matplotlib.use("Agg")

TRAIN_RE = re.compile(
    r"\[train\] step (\d+)/(\d+) loss=([0-9.]+) kl=([0-9.]+) ce=([0-9.]+) "
    r"ppl~([0-9.]+) ([0-9.]+) tok/s")
VAL_RE = re.compile(
    r"\[val\] step (\d+)/(\d+) val_ce=([0-9.]+) val_ppl~([0-9.]+)")
EVAL_RE = re.compile(
    r"\[eval\] train-PPL ~([0-9.]+) .* vs val-PPL ~([0-9.]+)")


def parse_log(path: str) -> dict:
    """Extrait steps, PPL train/val et tok/s d'un log de train."""
    steps, ppl_tr, toks = [], [], []
    vsteps, ppl_va = [], []
    eval_fin = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                ppl_tr.append(float(m.group(6)))
                toks.append(float(m.group(7)))
                continue
            m = VAL_RE.search(line)
            if m:
                vsteps.append(int(m.group(1)))
                ppl_va.append(float(m.group(4)))
                continue
            m = EVAL_RE.search(line)
            if m:
                eval_fin = (float(m.group(1)), float(m.group(2)))
    return {"steps": steps, "ppl_tr": ppl_tr, "toks": toks,
            "vsteps": vsteps, "ppl_va": ppl_va, "eval_fin": eval_fin}


def trace(d: dict, run: str, out: str) -> None:
    """Écrit le PNG : PPL (log) + tok/s."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    if d["steps"]:
        ax1.plot(d["steps"], d["ppl_tr"], label="train-PPL", linewidth=1)
    if d["vsteps"]:
        ax1.plot(d["vsteps"], d["ppl_va"], "o-", label="val-PPL", markersize=3)
    ax1.set_yscale("log")
    ax1.set_xlabel("step")
    ax1.set_ylabel("PPL (log)")
    if d["toks"]:
        ax2 = ax1.twinx()
        ax2.plot(d["steps"], d["toks"], color="green", alpha=0.4,
                 label="tok/s", linewidth=1)
        ax2.set_ylabel("tok/s")
    fig.suptitle(f"run {run}")
    fig.legend(loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print(f"[plot] -> {out} ({os.path.getsize(out) // 1024} Ko)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Courbe PNG d'un run.")
    ap.add_argument("--log", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = parse_log(a.log)
    print(f"[plot] {len(d['steps'])} points train, "
          f"{len(d['vsteps'])} évals val", flush=True)
    trace(d, a.run, a.out or f"docs/img/{a.run}.png")


if __name__ == "__main__":
    main()
