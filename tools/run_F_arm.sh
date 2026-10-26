#!/bin/bash
# Étape 4 — un bras F, une tranche (tranche_09), reprise depuis E-t08.
#
#   bash tools/run_F_arm.sh A|B|C [--go]
#
# Les trois bras voient exactement les mêmes données et le même nombre de
# tokens que t09 : reprise step 22740, epochs=16 (16 x 1479 = 23664),
# soit 924 steps restants (0,625 époque de tranche_09, 15,1 M tokens).
# L'ordre des batchs est seed=0, epoch 0 : identique à t09.
#
# Différence volontaire avec t09 : l'optimiseur est réinitialisé pour les
# trois bras (moments AdamW neufs), afin de comparer des recettes depuis un
# même point de départ. t09 reste la référence historique "reprise complète".
#
# Recettes :
#   A contrôle : KD alpha 0.7, lr 3e-4, cosine + rewarmup 0.05, WD 0.01
#   B CE seule : alpha 0, lr 1e-3, WSD 5%/20%/0.1, WD 0.1 -> 0 à 50 %
#   C KD       : alpha 0.7, lr 1e-3, WSD 5%/20%/0.1, WD 0.1 -> 0 à 50 %
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
DATA=/Volumes/Lexar/clean-corpus/dedup/tranche_09.jsonl
VALDATA=runs/abl_data/val_E_frozen.jsonl
PREV_PT=runs/E-t08/model.pt
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
LOCK=logs/gpu.lock
HEART=logs/F-heartbeat.md

ARM="${1:?usage: run_F_arm.sh A|B|C [--go]}"
case "$ARM" in
  A) LABEL="A contrôle (KD 0.7, lr 3e-4, cosine)"
     ALPHA=0.7; LR=3e-4; SCHED=cosine; WD=0.01; WDFRAC=0.0 ;;
  B) LABEL="B CE seule (lr 1e-3, WSD, WD 0.1->0)"
     ALPHA=0.0; LR=1e-3; SCHED=wsd; WD=0.1; WDFRAC=0.5 ;;
  C) LABEL="C KD 0.7 (lr 1e-3, WSD, WD 0.1->0)"
     ALPHA=0.7; LR=1e-3; SCHED=wsd; WD=0.1; WDFRAC=0.5 ;;
  *) echo "[F-$ARM] bras inconnu (A|B|C)"; exit 2 ;;
esac
OUTDIR=runs/F-$ARM
OUT=$OUTDIR/model.pt
mkdir -p "$OUTDIR"

for f in "$DATA" "$VALDATA" "$PREV_PT" "$VOCAB"; do
  [ -f "$f" ] || { echo "[F-$ARM] MANQUANT : $f"; exit 2; }
done
if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[F-$ARM] AUTRE TRAIN ACTIF -> abandon"; exit 1
fi
if [ -f "$LOCK" ]; then
  echo "[F-$ARM] LOCK GPU PRÉSENT ($LOCK) -> abandon"; exit 1
fi
# Ollama : tolère un petit modèle résident (gemma3:1b, < 2 Go) mais refuse
# un gros modèle qui prendrait la VRAM du train (MPS, mémoire unifiée).
OLLAMA_VRAM=$(curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null \
  | python3 -c 'import json,sys;print(sum(m.get("size_vram",0) for m in json.load(sys.stdin).get("models",[])))' \
  2>/dev/null || echo 0)
if [ "${OLLAMA_VRAM:-0}" -gt 2000000000 ]; then
  echo "[F-$ARM] OLLAMA > 2 Go en VRAM (${OLLAMA_VRAM} octets) -> abandon"
  exit 1
fi
if [ -f "$OUTDIR/.done" ]; then
  echo "[F-$ARM] déjà terminé ($OUTDIR/.done)"; exit 0
fi
if [ -f "$OUT" ]; then
  echo "[F-$ARM] $OUT existe déjà (run terminé ?) -> marquer .done et sortir"
  touch "$OUTDIR/.done"; exit 0
fi

# Reprise après crash : dernier checkpoint périodique (optimiseur + scheduler
# conservés). Premier lancement : poids de t08 + optimiseur neuf.
RESUME="$PREV_PT"
FRESH_FLAGS=(--reset-optimizer --lr-resume "$LR" --rewarmup-ratio 0.05)
LATEST=$(ls -t "$OUTDIR"/model_ckpt*.pt 2>/dev/null | head -1 || true)
if [ -n "${LATEST:-}" ]; then
  RESUME="$LATEST"
  FRESH_FLAGS=()
  echo "[F-$ARM] reprise après crash : $LATEST (optimiseur/scheduler conservés)"
fi

COMMON=(--data "$DATA" --parent HuggingFaceTB/SmolLM-135M --out "$OUT"
  --config tiny --vocab-size 12000 --vocab-from "$VOCAB"
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16
  --epochs 16 --seed 0 --alpha "$ALPHA" --lr "$LR" --temperature 2.0
  --val-data "$VALDATA" --eval-every 100 --ckpt-every 250
  --resume "$RESUME" --track-flips)
if [ "$SCHED" = "wsd" ]; then
  COMMON+=(--sched wsd --wsd-warmup-frac 0.05 --wsd-decay-frac 0.2
           --wsd-min-ratio 0.1 --weight-decay "$WD" --wd-zero-frac "$WDFRAC")
else
  COMMON+=(--sched cosine --weight-decay "$WD")
fi

echo "[F-$ARM] $LABEL"
echo "[F-$ARM] data=$DATA (même tranche que t09), resume=$RESUME"
echo "[F-$ARM] steps restants=924 (epochs=16 x 1479), alpha=$ALPHA lr=$LR"
echo "[F-$ARM] sched=$SCHED wd=$WD wd_zero_frac=$WDFRAC seed=0"
if [ "${2:-}" != "--go" ]; then
  echo "[F-$ARM] PRÉPARÉ (rien lancé). Relancer avec --go."
  echo "[F-$ARM] commande :"
  echo "  caffeinate -dims $V main.py train ${COMMON[*]} ${FRESH_FLAGS[*]}"
  exit 0
fi

echo $$ > "$LOCK"
echo "[F-$ARM] lock pris ($LOCK, pid $$)"
{
  echo ""
  echo "## $(date '+%Y-%m-%d %H:%M') — étape 4, bras $ARM : départ"
  echo ""
  echo "- $LABEL"
  echo "- données : \`$DATA\` (tranche_09, mêmes tokens que t09)"
  echo "- reprise : \`$RESUME\` ; 924 steps ; seed 0"
  echo "- logs : \`$OUTDIR/train.log\`"
} >> "$HEART"
trap 'rm -f "$LOCK"; echo "[F-$ARM] lock libéré"' EXIT

t0=$(date +%s)
caffeinate -dims "$V" main.py train "${COMMON[@]}" "${FRESH_FLAGS[@]}" \
  > "$OUTDIR/train.log" 2>&1
RC=$?
t1=$(date +%s)
echo "[F-$ARM] train exit=$RC durée=$((t1-t0))s" >> "$OUTDIR/train.log"
{
  echo "- fin bras $ARM : exit=$RC, durée=$((t1-t0))s"
} >> "$HEART"
if [ "$RC" -eq 0 ] && [ -f "$OUT" ]; then
  touch "$OUTDIR/.done"
  echo "[F-$ARM] OK -> $OUT"
else
  echo "[F-$ARM] ÉCHEC exit=$RC (voir $OUTDIR/train.log)"
fi
"$V" tools/watch_run.py --log "$OUTDIR/train.log" --log-only 2>/dev/null || true
exit "$RC"
