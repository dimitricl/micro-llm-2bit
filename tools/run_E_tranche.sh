#!/bin/bash
# E-SUITE t02-t09 — un appel par tranche : bash tools/run_E_tranche.sh NN PREV
# ex. bash tools/run_E_tranche.sh 02 runs/E-t01
# Recette spec §E + amendements Q0 : KD 0.7, lr 3e-4, bs 8/accum 16,
# val GELÉE, eval-every 100 (Q0a), resume FINAL du run précédent (Q0b),
# seed 0. Garde schedule-guard : tuer < 50 steps si total inattendu.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
NN="${1:?usage: run_E_tranche.sh NN PREVDIR}"; PREVDIR="$2"
TRANCHE=/Volumes/Lexar/clean-corpus/dedup/tranche_${NN}.jsonl
VALDATA=runs/abl_data/val_E_frozen.jsonl
PREV_PT=$PREVDIR/model.pt
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
OUTDIR=runs/E-t${NN}
OUT=$OUTDIR/model.pt
mkdir -p "$OUTDIR"

for f in "$TRANCHE" "$VALDATA" "$PREV_PT" "$VOCAB"; do
  [ -f "$f" ] || { echo "[E-t$NN] MANQUANT : $f"; exit 2; }
done
if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[E-t$NN] AUTRE TRAIN ACTIF -> abandon"; exit 1
fi
if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
  echo "[E-t$NN] MODÈLE OLLAMA CHARGÉ -> abandon"; exit 1
fi

# Epochs cumulatives B4 depuis le step du run précédent.
MANTOK=$($V -c "import json;print(json.load(open('docs/results/dedup_report.json'))['tranches'][$((10#$NN))]['tokens'])")
SPE=$(( (MANTOK + 16384 - 1) / 16384 ))
[ "$SPE" -lt 1 ] && SPE=1
PREV_STEPS=$($V -c "import torch;print(torch.load('$PREV_PT',map_location='cpu',weights_only=False).get('step',0))")
# Epochs : on vise ~1 époque de données nouvelles. ceil() surestime
# dès que PREV n'est pas un multiple de SPE (t02 : ceil -> 5 epochs =
# 2 époques). round() avec garde : remaining >= SPE/2 sinon +1.
SPE2=$((SPE / 2))
E_CUMUL=$(( (PREV_STEPS + SPE + SPE2) / SPE ))
[ $(( E_CUMUL * SPE - PREV_STEPS )) -lt "$SPE2" ] && E_CUMUL=$((E_CUMUL + 1))
echo "[E-t$NN] tokens=$MANTOK SPE~$SPE steps/tranche, reprise step $PREV_STEPS -> epochs=$E_CUMUL (restants ~$((E_CUMUL * SPE - PREV_STEPS)))"

if [ "${3:-}" != "--go" ]; then
  echo "[E-t$NN] PRÉPARÉ (rien lancé). Relancer avec --go."
  exit 0
fi

caffeinate -dims $V main.py train --data "$TRANCHE" --parent HuggingFaceTB/SmolLM-135M \
  --out "$OUT" --config tiny --vocab-size 12000 --vocab-from "$VOCAB" \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs "$E_CUMUL" --seed 0 --alpha 0.7 --lr 3e-4 --temperature 2.0 \
  --val-data "$VALDATA" --eval-every 100 --ckpt-every 500 \
  --resume "$PREV_PT" --rewarmup-ratio 0.05 --lr-resume 3e-4 \
  2>&1 | tee "$OUTDIR/train.log"
$V tools/watch_run.py --log "$OUTDIR/train.log" --log-only
