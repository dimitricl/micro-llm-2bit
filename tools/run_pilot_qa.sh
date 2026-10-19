#!/bin/bash
# PILOTE QA — un seul run GPU, recette QA étroite (capitales + unites).
# Corpus gelé : data/qa/qa_fr_pilot_v1.jsonl (manifeste + SHA256).
# Reprise : runs/E-t08/model.pt (step 22740), 1 époque neuve exactement.
# Recette identique à la chaîne E (KD 0.7, lr 3e-4, bs 8/accum 16, val
# GELÉE, seed 0) ; seule la DONNÉE change (QA répété) + epochs cumulatives.
# Usage : bash tools/run_pilot_qa.sh [--go]
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
DATA=data/qa/qa_fr_pilot_v1.jsonl
MANIFEST=data/qa/qa_fr_pilot_v1.manifest.json
VALDATA=runs/abl_data/val_E_frozen.jsonl
PREV_PT=runs/E-t08/model.pt
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
OUTDIR=runs/E-pilot-qa
OUT=$OUTDIR/model.pt
LOCK=logs/gpu.lock
mkdir -p "$OUTDIR"

for f in "$DATA" "$MANIFEST" "$VALDATA" "$PREV_PT" "$VOCAB"; do
  [ -f "$f" ] || { echo "[pilot-qa] MANQUANT : $f"; exit 2; }
done
if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[pilot-qa] AUTRE TRAIN ACTIF -> abandon"; exit 1
fi
if [ -f "$LOCK" ]; then
  echo "[pilot-qa] LOCK GPU PRÉSENT ($LOCK) -> abandon"; exit 1
fi
if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
  echo "[pilot-qa] MODÈLE OLLAMA CHARGÉ -> abandon (ollama stop)"; exit 1
fi

# Intégrité du corpus gelé.
WANT=$($V -c "import json;print(json.load(open('$MANIFEST'))['sha256'])")
GOT=$($V -c "import hashlib;print(hashlib.sha256(open('$DATA','rb').read()).hexdigest())")
if [ "$WANT" != "$GOT" ]; then
  echo "[pilot-qa] SHA256 CORPUS INVALIDE ($GOT != $WANT) -> abandon"; exit 1
fi

# Epochs cumulatives B4 : SPE = steps/époque du corpus QA (manifeste).
SPE=$($V -c "import json;print(json.load(open('$MANIFEST'))['steps_par_epoque'])")
PREV_STEPS=$($V -c "import torch;print(torch.load('$PREV_PT',map_location='cpu',weights_only=False).get('step',0))")
E_CUMUL=$(( PREV_STEPS / SPE + 1 ))
RESTANT=$(( E_CUMUL * SPE - PREV_STEPS ))
echo "[pilot-qa] SPE=$SPE steps/époque ; reprise step $PREV_STEPS -> E_CUMUL=$E_CUMUL (restants=$RESTANT, cible~$SPE)"
if [ "$RESTANT" -lt $(( SPE / 2 )) ]; then
  echo "[pilot-qa] GARDE schedule-guard : restants $RESTANT < SPE/2 -> abandon"; exit 1
fi
if [ "$RESTANT" -gt $(( SPE + SPE / 2 )) ]; then
  echo "[pilot-qa] GARDE schedule-guard : restants $RESTANT > 1,5*SPE -> abandon"; exit 1
fi

if [ "${1:-}" != "--go" ]; then
  echo "[pilot-qa] PRÉPARÉ (rien lancé). Relancer avec --go."
  exit 0
fi

echo $$ > "$LOCK"
echo "[pilot-qa] lock pris ($LOCK, pid $$)"
trap 'rm -f "$LOCK"; echo "[pilot-qa] lock libéré"' EXIT

caffeinate -dims $V main.py train --data "$DATA" --parent HuggingFaceTB/SmolLM-135M \
  --out "$OUT" --config tiny --vocab-size 12000 --vocab-from "$VOCAB" \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs "$E_CUMUL" --seed 0 --alpha 0.7 --lr 3e-4 --temperature 2.0 \
  --val-data "$VALDATA" --eval-every 100 --ckpt-every 500 \
  --resume "$PREV_PT" --rewarmup-ratio 0.05 --lr-resume 3e-4 \
  > "$OUTDIR/train.log" 2>&1
$V tools/watch_run.py --log "$OUTDIR/train.log" --log-only
