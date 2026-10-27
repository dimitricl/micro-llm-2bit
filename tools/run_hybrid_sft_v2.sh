#!/usr/bin/env bash
set -euo pipefail

cd /Volumes/Lexar/micro-llm-2bit
VENV_PYTHON=.venv/bin/python
BASE=runs/hybrid-mlx/final.safetensors
SFT_DATA=runs/hybrid-sft-v2/sft_packed.npz
TOKEN_CACHE=runs/hybrid-mlx/tokens.uint16.bin
OUT=runs/hybrid-sft-v2
EXPORT=exports/hybrid_mlx_int4_v2.safetensors
DATA_ARGS=(/Volumes/Lexar/clean-corpus/dedup)

for path in "$BASE" "$SFT_DATA" "$TOKEN_CACHE"; do
  if [[ ! -s "$path" ]]; then
    echo "[hybrid] Fichier requis absent ou vide : $path" >&2
    exit 2
  fi
done
if [[ -e "$OUT/final.safetensors" || -e "$EXPORT" ]]; then
  echo "[hybrid] Refus d'écraser une sortie SFT v2 existante." >&2
  exit 2
fi

mkdir -p "$OUT" exports logs
echo "[hybrid] SFT v2 vérifié depuis le checkpoint de pré-entraînement"
"$VENV_PYTHON" -u tools/train_hybrid_mlx.py \
  --stage sft --data "${DATA_ARGS[@]}" \
  --sft-data "$SFT_DATA" --token-cache "$TOKEN_CACHE" \
  --tokenizer models/tokenizer_fr_16k --output "$OUT" \
  --resume "$BASE" --batch-size 16 --seq-len 512 --epochs 3 \
  --lr 5e-5 --weight-decay 0.01 --replay-ratio 0.1 \
  --ckpt-every 100 --log-every 20

echo "[hybrid] Export int4 SFT v2"
"$VENV_PYTHON" tools/export_hybrid_mlx.py export \
  --checkpoint "$OUT/final.safetensors" --bits 4 --out "$EXPORT"

echo "[hybrid] Benchmark export SFT v2"
"$VENV_PYTHON" tools/export_hybrid_mlx.py bench \
  --model "$EXPORT" --bits 4 --max-new 256

echo "[hybrid] Pipeline terminé"
