#!/bin/bash
# E-PILOTE t01 seule (~4,1h) — GO E-PILOTE requis. t02-t09 INTERDITES.
# Recette spec §E (docs/ABLATIONS.md) : KD alpha 0.7, SmolLM-135M,
# lr 3e-4, bs 8/accum 16, val GELÉE (jamais re-tirée), vocab tranche00,
# resume tranche00.pt step 4248, seed 0. Namespace runs/E-*.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
TRANCHE=/Volumes/Lexar/clean-corpus/dedup/tranche_01.jsonl
VALDATA=runs/abl_data/val_E_frozen.jsonl
PREV_PT=runs/2026-10-08-tranche00/tranche00.pt
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
OUTDIR=runs/E-t01
OUT=$OUTDIR/model.pt
mkdir -p "$OUTDIR"

for f in "$TRANCHE" "$VALDATA" "$PREV_PT" "$VOCAB"; do
  [ -f "$f" ] || { echo "[E-t01] MANQUANT : $f"; exit 2; }
done
if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[E-t01] AUTRE TRAIN ACTIF -> abandon"; exit 1
fi
if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
  echo "[E-t01] MODÈLE OLLAMA CHARGÉ -> abandon"; exit 1
fi

# Epochs cumulatives B4 : total couvre step 4248 + ~1 époque tranche.
CHARS=$(wc -c < "$TRANCHE")
SPE=$((CHARS * 100 / 282 / 128 / 8 / 16))
[ "$SPE" -lt 1 ] && SPE=1
PREV_STEPS=$($V -c "import torch;print(torch.load('$PREV_PT',map_location='cpu',weights_only=False).get('step',0))")
E_CUMUL=$(( (PREV_STEPS + SPE + SPE - 1) / SPE ))
echo "[E-t01] SPE~$SPE steps/epoch, reprise step $PREV_STEPS -> epochs=$E_CUMUL"

if [ "${1:-}" != "--go" ]; then
  echo "[E-t01] PRÉPARÉ (rien lancé). Relancer avec --go."
  exit 0
fi

caffeinate -dims $V main.py train --data "$TRANCHE" --parent HuggingFaceTB/SmolLM-135M \
  --out "$OUT" --config tiny --vocab-size 12000 --vocab-from "$VOCAB" \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs "$E_CUMUL" --seed 0 --alpha 0.7 --lr 3e-4 --temperature 2.0 \
  --val-data "$VALDATA" --eval-every 50 --ckpt-every 500 \
  --resume "$PREV_PT" --rewarmup-ratio 0.05 --lr-resume 3e-4 \
  2>&1 | tee "$OUTDIR/train.log"
$V tools/watch_run.py --log "$OUTDIR/train.log" --log-only
