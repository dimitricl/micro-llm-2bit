#!/usr/bin/env bash
set -euo pipefail

cd /Volumes/Lexar/micro-llm-2bit
VENV_PYTHON=.venv/bin/python
DATA_ARGS=(/Volumes/Lexar/clean-corpus/dedup data/hybrid)

if [[ -e runs/hybrid-mlx/final.safetensors || -e runs/hybrid-sft/final.safetensors ]]; then
  echo "[hybrid] Refus d'écraser un entraînement existant. Choisis un nouveau nom de run."
  exit 2
fi

mkdir -p logs runs/hybrid-mlx runs/hybrid-sft exports

echo "[hybrid] Préparation du jeu SFT chat + complétion code"
"$VENV_PYTHON" tools/prepare_hybrid_sft_mlx.py \
  --tokenizer models/tokenizer_fr_16k \
  --chat data/chat/chat_pilot_v1.jsonl \
  --synthetic data/hybrid/synthetic.jsonl \
  --code data/hybrid/javascript_code.jsonl \
  --seq-len 512 --out runs/hybrid-sft/sft_packed.npz

echo "[hybrid] Pré-entraînement MLX sur les tranches Wikipédia + data/hybrid"
"$VENV_PYTHON" -u tools/train_hybrid_mlx.py \
  --stage pretrain --data "${DATA_ARGS[@]}" \
  --tokenizer models/tokenizer_fr_16k \
  --output runs/hybrid-mlx \
  --batch-size 16 --seq-len 512 --epochs 1 --lr 3e-4 \
  --weight-decay 0.1 --ckpt-every 500 --log-every 50

echo "[hybrid] Fine-tuning masqué chat/code avec 10% de replay langue"
"$VENV_PYTHON" -u tools/train_hybrid_mlx.py \
  --stage sft --data "${DATA_ARGS[@]}" \
  --sft-data runs/hybrid-sft/sft_packed.npz \
  --token-cache runs/hybrid-mlx/tokens.uint16.bin \
  --tokenizer models/tokenizer_fr_16k \
  --output runs/hybrid-sft \
  --resume runs/hybrid-mlx/final.safetensors \
  --batch-size 16 --seq-len 512 --epochs 3 --lr 5e-5 \
  --weight-decay 0.01 --replay-ratio 0.1 \
  --ckpt-every 100 --log-every 20

echo "[hybrid] Export int4 MLX"
"$VENV_PYTHON" tools/export_hybrid_mlx.py export \
  --checkpoint runs/hybrid-sft/final.safetensors \
  --bits 4 --out exports/hybrid_mlx_int4.safetensors

echo "[hybrid] Benchmark export"
"$VENV_PYTHON" tools/export_hybrid_mlx.py bench \
  --model exports/hybrid_mlx_int4.safetensors --bits 4

echo "[hybrid] Pipeline terminé"
