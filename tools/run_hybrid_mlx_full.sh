#!/usr/bin/env bash
set -euo pipefail

cd /Volumes/Lexar/micro-llm-2bit
VENV_PYTHON=.venv/bin/python
DATA_ARGS=(/Volumes/Lexar/clean-corpus/dedup)

if [[ -e runs/hybrid-mlx/final.safetensors || -e runs/hybrid-sft/final.safetensors ]]; then
  echo "[hybrid] Refus d'écraser un entraînement existant. Choisis un nouveau nom de run."
  exit 2
fi

mkdir -p logs runs/hybrid-mlx runs/hybrid-sft exports

if [[ -s runs/hybrid-sft/sft_packed.npz && -s runs/hybrid-sft/sft_packed.json ]]; then
  echo "[hybrid] Réutilisation du jeu SFT déjà préparé"
else
  echo "[hybrid] Préparation du jeu SFT chat + complétion code"
  "$VENV_PYTHON" tools/prepare_hybrid_sft_mlx.py \
    --tokenizer models/tokenizer_fr_16k \
    --chat data/chat/chat_pilot_v1.jsonl \
    --synthetic data/hybrid/synthetic.jsonl \
    --code data/hybrid/javascript_code.jsonl \
    --seq-len 512 --out runs/hybrid-sft/sft_packed.npz
fi

resume_args=()
resume_step=0
shopt -s nullglob
checkpoints=(runs/hybrid-mlx/checkpoint_*.safetensors)
for checkpoint in "${checkpoints[@]}"; do
  step_text="${checkpoint##*_}"
  step_text="${step_text%.safetensors}"
  step=$((10#$step_text))
  if (( step > resume_step )); then
    resume_step=$step
    resume_checkpoint=$checkpoint
  fi
done
if (( resume_step > 0 )); then
  echo "[hybrid] Reprise du pré-entraînement au step $resume_step depuis $resume_checkpoint"
  resume_args=(--resume "$resume_checkpoint" --resume-step "$resume_step")
else
  echo "[hybrid] Aucun checkpoint trouvé : pré-entraînement depuis zéro"
fi

echo "[hybrid] Reprise du pré-entraînement MLX depuis le cache complet du corpus"
"$VENV_PYTHON" -u tools/train_hybrid_mlx.py \
  --stage pretrain --data "${DATA_ARGS[@]}" \
  --token-cache runs/hybrid-mlx/tokens.uint16.bin \
  --tokenizer models/tokenizer_fr_16k \
  --output runs/hybrid-mlx \
  --batch-size 16 --seq-len 512 --epochs 1 --lr 3e-4 \
  --weight-decay 0.1 --ckpt-every 500 --log-every 50 \
  "${resume_args[@]}"

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
