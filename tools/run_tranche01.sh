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

# Veilleur H5.2 (bloquant : STOP si divergence, val-PPL > 2× initiale,
# ou pas de retour sous l'initiale à 25 % de l'epoch).
$V tools/watch_run.py --log "$OUTDIR/train.log" || exit 3

# Mesures de revalidation — GATE APPARIÉ S5 (tools/gate_paire.py, ε=0,03,
# veto sur b/c, MDE documenté ; voir docs/DIAGNOSTICS.md C2-ter).
# L'ancien tools/revalidate_tranche01.py (IC non appariés) est SUPERSEDÉ :
# conservé pour l'historique, ne plus l'utiliser pour tranche01.
# 1) BPC val élargie (même protocole S5.1, per-doc pour le gate) :
#  caffeinate -dims uv run python tools/eval_valbig.py --ckpt $OUT \
#    --docs runs/abl_data/val_big.jsonl \
#    --vocab $VOCAB --out $OUTDIR/eval_valbig.summary.json \
#    --dump-per-doc $OUTDIR/eval_valbig.perdoc.json --draws 1000 --seed 0
# 2) Boucles greedy (32 prompts v2, metrics JSON pour le gate) :
#  uv run python main.py export --ckpt $OUT --out $OUTDIR/tranche01.bin
#  nice -n 10 uv run python tools/eval_fixed.py \
#    --model $OUTDIR/tranche01.bin --out $OUTDIR/samples.md \
#    --metrics-out $OUTDIR/metrics.json
#  (ref boucles tranche00 : eval/tranche00/metrics_v2.json, 32 prompts)
# 3) Verdict (ref per-doc : docs/results/valbig_tranche00.json) :
#  uv run python tools/gate_paire.py \
#    --ref docs/results/valbig_tranche00.json \
#    --new $OUTDIR/eval_valbig.perdoc.json \
#    --ref-loops eval/tranche00/metrics_v2.json \
#    --new-loops $OUTDIR/metrics.json
