#!/bin/bash
# Tranche01 — PRÉPARÉ, NE PAS LANCER sans "go tranche01" explicite.
# Recette V6 (CE seule alpha=0, lr 1e-3, seq 128) + reprise tranche00.
# Lancement : nohup bash tools/run_tranche01.sh > runs/tranche01.log 2>&1 &
# (caffeinate -dims est DANS la commande train : anti-veille macOS).
# Revalidation ensuite : voir tools/revalidate_tranche01.py et la section
# "Mesures de revalidation" ci-dessous.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
PARENT=HuggingFaceTB/SmolLM-135M  # ignoré en CE seule (B6), requis par le CLI
SLICE=/Volumes/Lexar/clean-corpus/slices/slice_01.jsonl
PREV_PT=runs/2026-10-08-tranche00/tranche00.pt
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
OUTDIR=runs/2026-10-09-tranche01
OUT=$OUTDIR/tranche01.pt
mkdir -p "$OUTDIR"

# Échec rapide si la base n'existe pas.
[ -f "$PREV_PT" ] || { echo "[t01] MANQUANT : $PREV_PT"; exit 2; }
[ -f "$VOCAB" ] || { echo "[t01] MANQUANT : $VOCAB"; exit 2; }
[ -f "$SLICE" ] || { echo "[t01] MANQUANT : $SLICE"; exit 2; }

# Précheck GPU : aucun autre train, aucun modèle Ollama chargé.
if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[t01] AUTRE TRAIN ACTIF -> abandon"; exit 1
fi
if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
  echo "[t01] MODÈLE OLLAMA CHARGÉ -> abandon"; exit 1
fi

# Steps/epoch estimés (décompte B1 : fenêtres/bs/accum ; 2,82 chars/token
# mesurés sur slice_00) + epochs cumulées (scheduler absolu, cf run_slices).
CHARS=$(wc -c < "$SLICE")
SPE=$((CHARS * 100 / 282 / 128 / 8 / 16))
[ "$SPE" -lt 1 ] && SPE=1
PREV_STEPS=$(python3 -c "import torch;print(torch.load('$PREV_PT',map_location='cpu',weights_only=False).get('step',0))")
E_CUMUL=$(( (PREV_STEPS + SPE + SPE - 1) / SPE ))
echo "[t01] SPE~$SPE steps/epoch, reprise step $PREV_STEPS -> epochs=$E_CUMUL"

if [ "${1:-}" != "--go" ]; then
  echo "[t01] PRÉPARÉ (rien lancé). Relancer avec --go après validation."
  echo "[t01] epochs=$E_CUMUL out=$OUT"
  exit 0
fi

caffeinate -dims $V main.py train --data "$SLICE" --parent "$PARENT" \
  --out "$OUT" --config tiny --vocab-size 12000 --vocab-from "$VOCAB" \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs "$E_CUMUL" --seed 0 --alpha 0.0 --lr 1e-3 --temperature 2.0 \
  --val-ratio 0.05 --eval-every 50 --ckpt-every 500 \
  --resume "$PREV_PT" --rewarmup-ratio 0.05 --lr-resume 1e-3 \
  2>&1 | tee "$OUTDIR/train.log"

# Mesures de revalidation (commandes, à lancer après le run) :
# 1) BPC val élargie (même protocole G2) :
#  uv run python tools/eval_valbig.py --ckpt $OUT \
#    --docs runs/abl_data/val_big.jsonl \
#    --vocab $VOCAB --out $OUTDIR/eval_valbig.json
# 2) Boucles greedy (20 prompts fixes) :
#  uv run python main.py export --ckpt $OUT --out $OUTDIR/tranche01.bin
#  nice -n 10 uv run python tools/eval_fixed.py \
#    --model $OUTDIR/tranche01.bin --out $OUTDIR/samples.md
#  (taux = moyenne des plus_longue_boucle greedy ; réf tranche00 : à mesurer
#   une fois avec la même commande sur exports/tranche00.bin)
# 3) Verdict :
#  uv run python tools/revalidate_tranche01.py --ref $OUTDIR/../ref_tranche00.json \
#    --new $OUTDIR/new_tranche01.json --train-log $OUTDIR/train.log
#  (new_tranche01.json = {bpc_a, bpc_a_ci, loop, has_nan} assemblé depuis
#   eval_valbig.json + samples.md + grep du log ; ref_tranche00.json écrit
#   après les mesures G2.)
