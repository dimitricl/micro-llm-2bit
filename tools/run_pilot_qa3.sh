#!/bin/bash
# PILOTE QA #3 — un seul run GPU (budget x3 + corpus naturel, 4 catégories).
# Corpus gelé : data/qa/qa_fr_pilot_v3.jsonl (manifeste + SHA256).
# Reprise : runs/E-t08/model.pt (step 22740), E_CUMUL=16 -> exactement
# 1516 steps restants = 1 époque neuve de 24,85 M tokens.
# Recette identique à la chaîne E / pilotes #1-#2 (KD 0.7, lr 3e-4,
# bs 8/accum 16, val GELÉE, seed 0) ; seule la DONNÉE change.
# Usage : bash tools/run_pilot_qa3.sh [--go]
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
DATA=data/qa/qa_fr_pilot_v3.jsonl
MANIFEST=data/qa/qa_fr_pilot_v3.manifest.json
VALDATA=runs/abl_data/val_E_frozen.jsonl
PREV_PT=runs/E-t08/model.pt
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
OUTDIR=runs/E-pilot-qa3
OUT=$OUTDIR/model.pt
LOCK=logs/gpu.lock
mkdir -p "$OUTDIR"

for f in "$DATA" "$MANIFEST" "$VALDATA" "$PREV_PT" "$VOCAB"; do
  [ -f "$f" ] || { echo "[pilot-qa3] MANQUANT : $f"; exit 2; }
done
if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[pilot-qa3] AUTRE TRAIN ACTIF -> abandon"; exit 1
fi
if [ -f "$LOCK" ]; then
  echo "[pilot-qa3] LOCK GPU PRÉSENT ($LOCK) -> abandon"; exit 1
fi
if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
  echo "[pilot-qa3] MODÈLE OLLAMA CHARGÉ -> abandon (ollama stop)"; exit 1
fi

# Intégrité du corpus gelé.
WANT=$($V -c "import json;print(json.load(open('$MANIFEST'))['sha256'])")
GOT=$($V -c "import hashlib;print(hashlib.sha256(open('$DATA','rb').read()).hexdigest())")
if [ "$WANT" != "$GOT" ]; then
  echo "[pilot-qa3] SHA256 CORPUS INVALIDE ($GOT != $WANT) -> abandon"; exit 1
fi

# Epochs cumulatives B4/K3 : SPE = steps/époque du corpus QA (manifeste).
SPE=$($V -c "import json;print(json.load(open('$MANIFEST'))['steps_par_epoque'])")
PREV_STEPS=$($V -c "import torch;print(torch.load('$PREV_PT',map_location='cpu',weights_only=False).get('step',0))")
E_CUMUL=$(( PREV_STEPS / SPE + 1 ))
RESTANT=$(( E_CUMUL * SPE - PREV_STEPS ))
echo "[pilot-qa3] SPE=$SPE steps/époque ; reprise step $PREV_STEPS -> E_CUMUL=$E_CUMUL (restants=$RESTANT, cible~$SPE)"
LIMIT_LOW=$(( SPE * 8 / 10 ))
LIMIT_HIGH=$(( SPE * 3 / 2 ))
if [ "$RESTANT" -lt "$LIMIT_LOW" ]; then
  echo "[pilot-qa3] GARDE schedule-guard : restants $RESTANT < 0,8*SPE -> abandon"; exit 1
fi
if [ "$RESTANT" -gt "$LIMIT_HIGH" ]; then
  echo "[pilot-qa3] GARDE schedule-guard : restants $RESTANT > 1,5*SPE -> abandon"; exit 1
fi
if [ "$RESTANT" -lt 50 ]; then
  echo "[pilot-qa3] GARDE schedule-guard : restants $RESTANT < 50 steps -> tuer"; exit 1
fi

if [ "${1:-}" != "--go" ]; then
  echo "[pilot-qa3] PRÉPARÉ (rien lancé). Relancer avec --go."
  exit 0
fi

echo $$ > "$LOCK"
echo "[pilot-qa3] lock pris ($LOCK, pid $$)"
trap 'rm -f "$LOCK"; echo "[pilot-qa3] lock libéré"' EXIT

caffeinate -dims $V main.py train --data "$DATA" --parent HuggingFaceTB/SmolLM-135M \
  --out "$OUT" --config tiny --vocab-size 12000 --vocab-from "$VOCAB" \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs "$E_CUMUL" --seed 0 --alpha 0.7 --lr 3e-4 --temperature 2.0 \
  --val-data "$VALDATA" --eval-every 100 --ckpt-every 500 \
  --resume "$PREV_PT" --rewarmup-ratio 0.05 --lr-resume 3e-4 \
  > "$OUTDIR/train.log" 2>&1
RC=$?
echo "[pilot-qa3] train exit=$RC" >> "$OUTDIR/train.log"
$V tools/watch_run.py --log "$OUTDIR/train.log" --log-only
exit $RC
