#!/bin/bash
# R2b : copie exacte R2 (CE, lr 3e-4), seed 1
set -euo pipefail
cd /Volumes/Lexar/micro-llm-2bit

V=.venv/bin/python
DATA=runs/abl_data/resume_5M.jsonl
PREV_PT=runs/2026-10-08-tranche00/tranche00.pt
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
OUTDIR=runs/R2b-ce-lr3e4-seed1
OUT=$OUTDIR/model.pt

mkdir -p "$OUTDIR"

# Précheck
pgrep -f "main.py train" >/dev/null 2>&1 && { echo "[R2b] AUTRE TRAIN ACTIF"; exit 1; }
curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"' && { echo "[R2b] OLLAMA CHARGE"; exit 1; }

caffeinate -dims $V main.py train --data "$DATA" --parent HuggingFaceTB/SmolLM-135M \
  --out "$OUT" --config tiny --vocab-size 12000 --vocab-from "$VOCAB" \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs 15 --seed 1 --alpha 0.0 --lr 3e-4 --temperature 2.0 \
  --val-ratio 0.05 --eval-every 50 --ckpt-every 500 \
  --resume runs/2026-10-08-tranche00/tranche00.pt --rewarmup-ratio 0.05 --lr-resume 3e-4 \
  2>&1 | tee "$OUTDIR/train.log"

echo $? > logs/r2b.exit
date > logs/r2b.done