"""Tableau des runs : lit runs/*/config.json + logs -> docs/RUNS.md.

Usage : .venv/bin/python tools/runs_table.py [--runs runs] [--out docs/RUNS.md]
Lecture seule (logs), sans GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.plot_run import parse_log


def ligne_run(dossier: str) -> str:
    cfg_path = os.path.join(dossier, "config.json")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return f"| {os.path.basename(dossier)} | (pas de config.json) |"
    nom = os.path.basename(dossier)
    # Log : le premier train*.log trouvé.
    logs = sorted(glob.glob(os.path.join(dossier, "train*.log")))
    if not logs:
        return (f"| {nom} | {cfg.get('config', '?')} | "
                f"{os.path.basename(cfg.get('tranche', '?'))} | - | - | - | - | - |")
    d = parse_log(logs[0])
    val_fin = f"{d['ppl_va'][-1]:.1f}" if d["ppl_va"] else "-"
    ratio = "-"
    if d["eval_fin"]:
        tr, va = d["eval_fin"]
        ratio = f"{va / tr:.2f}" if tr > 0 else "-"
    toks = f"{sum(d['toks']) / len(d['toks']):.0f}" if d["toks"] else "-"
    steps = str(d["steps"][-1]) if d["steps"] else "-"
    duree = "-"
    return (f"| {nom} | {cfg.get('config', '?')} | "
            f"{os.path.basename(cfg.get('tranche', '?'))} | {steps} | "
            f"{val_fin} | {ratio} | {toks} | {duree} |")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tableau markdown des runs.")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs/RUNS.md")
    a = ap.parse_args()
    lignes = ["# Runs", "",
              "| run | config | tranche | steps | val-PPL finale | "
              "ratio val/train | tok/s moyen | durée |",
              "|---|---|---|---|---|---|---|---|"]
    for dossier in sorted(glob.glob(os.path.join(a.runs, "*"))):
        if os.path.isdir(dossier):
            lignes.append(ligne_run(dossier))
    lignes += [""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    print(f"[runs] -> {a.out} ({len(lignes) - 4} runs)")


if __name__ == "__main__":
    main()
