#!/bin/bash
# Boucle d'entraînement par tranches : train -> purge -> export -> bench.
# Affiche la val-PPL de chaque tranche. Par défaut : ESTIMATION SEULEMENT.
# Lancer avec --go pour vraiment entraîner (long !).
#
# Usage : bash data_fetch/run_slices.sh <dir_tranches> <prefix_out> [epochs] [--go]
# Ex.   : bash data_fetch/run_slices.sh /Volumes/Lexar/clean-corpus/slices \
#             checkpoints/tranche --epochs 1
# Ex. go: ... --epochs 1 --go
#
# Convention : tranche_00 -> vocab construit ; tranches suivantes réutilisent
# le vocab (--vocab-from) et reprennent le checkpoint précédent (vocab
# vérifié, refus sinon). Seuls les 1-2 derniers checkpoints sont gardés.

set -u
DIR=${1:?dossier de tranches}
PREFIX=${2:?préfixe out (ex. checkpoints/tranche)}
EPOCHS=1
GO=0
for a in "$@"; do
  [ "$a" = "--go" ] && GO=1
  case "$a" in --epochs) : ;; esac
done
[[ " $* " == *" --epochs "* ]] && EPOCHS=$(echo "$*" | sed -E 's/.*--epochs ([0-9]+).*/\1/')

V=.venv/bin/python
TOK_S=750  # vitesse mesurée (tok/s) pour l'estimation
CHARS_PER_TOK=4
SEQ=256; BATCH=2; ACCUM=8
SAVE=$BATCH  # B1 : --save-batch == --batch-size (micro-batch réel)

echo "=== Estimation (aucun entraînement sans --go) ==="
TOTAL_S=0
for SL in "$DIR"/slice_*.jsonl; do
  CHARS=$(wc -c < "$SL")
  # Décompte B1 : fenêtres = TOK/SEQ, 1 micro-batch par batch stocké
  # (save == bs), 1 step par ACCUM micros.
  TOK=$((CHARS / CHARS_PER_TOK))
  STEPS=$((TOK / SEQ / BATCH / ACCUM * EPOCHS))
  SEC=$((TOK * EPOCHS / TOK_S))
  TOTAL_S=$((TOTAL_S + SEC))
  printf "%s : ~%d tokens, ~%d steps, ~%dm\n" "$(basename "$SL")" $((TOK * EPOCHS)) "$STEPS" $((SEC / 60))
done
echo "TOTAL estimé : ~$((TOTAL_S / 3600))h$((TOTAL_S % 3600 / 60))m pour $EPOCHS epoch(s)/tranche"
if [ "$GO" -eq 0 ]; then
  echo "Relancez avec --go pour entraîner."
  exit 0
fi

PREV=""; VOCAB=""; PREV_STEPS=0
for SL in "$DIR"/slice_*.jsonl; do
  NAME=$(basename "$SL" .jsonl)
  OUT="${PREFIX}_${NAME}.pt"
  # Steps/epoch estimés pour cette tranche (décompte B1 : save == bs,
  # steps = fenêtres / bs / accum).
  CHARS=$(wc -c < "$SL")
  SPE=$((CHARS / CHARS_PER_TOK / SEQ / BATCH / ACCUM))
  [ "$SPE" -lt 1 ] && SPE=1
  ARGS=(--data "$SL" --parent HuggingFaceTB/SmolLM-360M --out "$OUT"
        --config base --vocab-size 12000 --seq-len 256 --hidden-weight 0.0
        --save-batch "$SAVE" --batch-size "$BATCH" --accum "$ACCUM"
        --val-ratio 0.05 --eval-every 200)
  if [ -z "$PREV" ]; then
    ARGS+=(--epochs "$EPOCHS")
  else
    # Les steps du scheduler sont ABSOLUS : il faut des epochs cumulées
    # pour que (SPE * epochs) dépasse le step de reprise + steps voulus.
    NEED=$((PREV_STEPS + SPE * EPOCHS))
    E_CUMUL=$(( (NEED + SPE - 1) / SPE ))
    ARGS+=(--epochs "$E_CUMUL" --resume "$PREV" --vocab-from "$VOCAB"
           --rewarmup-ratio 0.05 --lr-resume 3e-4)
  fi
  echo "=== train $NAME ==="
  $V main.py train "${ARGS[@]}" 2>&1 | tee "checkpoints/train_${NAME}.log"
  # Purge : ne garde que le final de cette tranche (+ bestval).
  ls "${OUT%.pt}"_e*.pt 2>/dev/null | sort | head -n -1 | xargs -r rm -f
  VOCAB="${OUT}.vocab.json"
  echo "=== export+bench $NAME ==="
  $V main.py export --ckpt "$OUT" --out "${OUT%.pt}.bin"
  $V main.py bench --model "${OUT%.pt}.bin" --bench-tokens 32
  grep -E "^\[eval\]" "checkpoints/train_${NAME}.log" | tail -2
  # Steps totaux atteints (pour les epochs cumulées de la tranche suivante).
  PREV_STEPS=$(grep -oE "step [0-9]+/[0-9]+" "checkpoints/train_${NAME}.log" | tail -1 | sed -E 's|step [0-9]+/||')
  [ -z "$PREV_STEPS" ] && PREV_STEPS=0
  PREV="$OUT"
done
echo "=== Terminé. Val-PPL par tranche ci-dessus ([eval]). ==="
