#!/bin/bash
# R3-ORIGINAL : KD SmolLM-135M resume tranche00 — le test D3 jamais lancé.
# Commande = EXACTEMENT celle de R2 (tools/run_resume.sh, logs/r1r2.log),
# SEULES différences : --parent SmolLM-135M (explicite) --alpha 0.7
# --seed 0 --out runs/R3-smol135-kd/. Même tokenizer des deux côtés :
# le piège R3-Qwen est structurellement impossible (garde-fou F0a
# silencieux attendu — s'il claque, STOP).
# Schedule B4 cumulative : --epochs 15 → total 4590, reprise step 4248
# → « 342 steps restants » attendu dans le log, sinon kill avant 50 steps.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
OUTDIR=runs/R3-smol135-kd
OUT=$OUTDIR/model.pt
mkdir -p "$OUTDIR"

if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[R3] AUTRE TRAIN ACTIF -> abandon"; exit 1
fi
if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
  echo "[R3] MODÈLE OLLAMA CHARGÉ -> abandon"; exit 1
fi

caffeinate -dims $V main.py train --data runs/abl_data/resume_5M.jsonl \
  --parent HuggingFaceTB/SmolLM-135M \
  --out "$OUT" --config tiny --vocab-size 12000 \
  --vocab-from runs/2026-10-08-tranche00/tranche00.pt.vocab.json \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs 15 --seed 0 --alpha 0.7 --lr 3e-4 --temperature 2.0 \
  --val-ratio 0.05 --eval-every 50 --ckpt-every 500 \
  --resume runs/2026-10-08-tranche00/tranche00.pt \
  --rewarmup-ratio 0.05 --lr-resume 3e-4 \
  2>&1 | tee "$OUTDIR/train.log"
# Veilleur log-only sur le VRAI log (leçon logs/R4.log : pas de chemin faux).
$V tools/watch_run.py --log "$OUTDIR/train.log" --log-only
