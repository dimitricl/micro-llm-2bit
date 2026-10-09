#!/bin/bash
# Chaine d'ablations D : un run a la fois, precheck GPU avant chacun.
# Arret sur anomalie (crash, loss non finie, GPU occupe). Une ligne
# docs/ABLATIONS.md par run termine. Protocole : voir docs/ABLATIONS.md.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
DATE=2026-10-09
PARENT=HuggingFaceTB/SmolLM-135M
DATA=runs/abl_data/train_5M.jsonl
VOCAB=runs/abl_data/vocab.json
ABL=docs/ABLATIONS.md
SPE_SUBSET=305  # steps/epoch du subset (save8/bs8/accum16/seq128, seed 0 ; 303 en seed 1)

precheck() {
  # 1) Pas d'autre train (hors descendants de ce shell).
  for pid in $(pgrep -f "main.py train" 2>/dev/null); do
    p=$pid
    descendant=0
    while [ "$p" -ne 1 ] && [ "$p" -ne 0 ]; do
      [ "$p" -eq "$$" ] && descendant=1 && break
      p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " " || echo 0)
      [ -z "$p" ] && break
    done
    if [ "$descendant" -eq 0 ]; then
      echo "[precheck] AUTRE TRAIN ACTIF (pid $pid) -> abandon"
      return 1
    fi
  done
  # 2) Aucun modele Ollama charge en local.
  if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
    echo "[precheck] MODELE OLLAMA CHARGE -> abandon"
    return 1
  fi
  echo "[precheck] GPU libre OK"
  return 0
}

ligne_md() {  # $1=log $2=dir $3=nom $4=label $5=seed $6=steps $7=tokens $8=note $9=facteur BPC
  LOG=$1; D=$2; K=${9:-0.53288}  # BPC = val_ce * K (K = tok/char du val in-run / ln2 ; 0.51815 en seed 1)
  V100=$(grep -E "^\[val\] step 100/" "$LOG" | tail -1 | sed -E 's/.*val_ppl~([0-9.]+).*/\1/')
  C100=$(grep -E "^\[val\] step 100/" "$LOG" | tail -1 | sed -E 's/.*val_ce=([0-9.]+).*/\1/')
  V200=$(grep -E "^\[val\] step 200/" "$LOG" | tail -1 | sed -E 's/.*val_ppl~([0-9.]+).*/\1/')
  C200=$(grep -E "^\[val\] step 200/" "$LOG" | tail -1 | sed -E 's/.*val_ce=([0-9.]+).*/\1/')
  V300=$(grep -E "^\[val\] step 300/" "$LOG" | tail -1 | sed -E 's/.*val_ppl~([0-9.]+).*/\1/')
  C300=$(grep -E "^\[val\] step 300/" "$LOG" | tail -1 | sed -E 's/.*val_ce=([0-9.]+).*/\1/')
  B100=$(awk "BEGIN{printf \"%.4f\", ${C100:-0} * $K}")
  B200=$(awk "BEGIN{printf \"%.4f\", ${C200:-0} * $K}")
  B300=$(awk "BEGIN{printf \"%.4f\", ${C300:-0} * $K}")
  TOKS=$(grep -E "^\[train\] step " "$LOG" | tail -1 | sed -E 's/.* ([0-9]+) tok\/s.*/\1/')
  WALL=$(grep -E "terminé en" "$LOG" | tail -1 | sed -E 's/.*terminé en ([0-9]+)s.*/\1/')
  EV=$($V -c "
import json
try:
    e = json.load(open('$D/eval.json'))
    g = lambda k, f: e.get(k, {}).get(f, '?')
    print(f\"{g('128','ppl')}/{g('128','bpc')}|{g('256','ppl')}/{g('256','bpc')}\")
except Exception:
    print('?/?|?/?')
")
  echo "| $3 | $4 | $5 | $6 | $7 | ${V100:--} | ${V200:--} | ${V300:--} | $B100 | $B200 | $B300 | ${EV%%|*} | ${EV##*|} | ${TOKS:--} | ${WALL:--} | $8 |"
}

lancer() {  # $1=nom $2=label $3=seed $4=alpha $5=lr $6=seq $7=bs $8=accum $9=epochs $10=extra
  NOM=$1; LABEL=$2; SEED=$3; ALPHA=$4; LR=$5; SEQ=$6; BS=$7; ACC=$8; EP=$9; EXTRA=${10}
  D="runs/$DATE-abl-$NOM"
  mkdir -p "$D"
  $V -c "import json;json.dump({'run':'$NOM','variante':'$LABEL','date':'$DATE','config':'tiny','parent':'$PARENT','data':'$DATA','vocab_from':'$VOCAB','seed':$SEED,'alpha':$ALPHA,'lr':$LR,'seq_len':$SEQ,'save_batch':$BS,'batch_size':$BS,'accum':$ACC,'epochs':$EP,'temperature':2.0,'eval_every':50,'ckpt_every':100,'extra':'$EXTRA'},open('$D/config.json','w'),indent=1)"
  echo "=== [abl] $NOM [$LABEL] seed=$SEED alpha=$ALPHA lr=$LR seq=$SEQ bs=$BS acc=$ACC ep=$EP $EXTRA ==="
  precheck || { echo "[abl] ANOMALIE precheck $NOM -> arret chaine"; return 1; }
  # shellcheck disable=SC2086
  $V main.py train --data "$DATA" --parent "$PARENT" --out "$D/model.pt" \
    --config tiny --vocab-size 12000 --vocab-from "$VOCAB" \
    --seq-len "$SEQ" --save-batch "$BS" --batch-size "$BS" --accum "$ACC" \
    --epochs "$EP" --seed "$SEED" --alpha "$ALPHA" --lr "$LR" --temperature 2.0 \
    --val-ratio 0.05 --eval-every 50 --ckpt-every 100 $EXTRA \
    2>&1 | tee "$D/train.log"
  if grep -q "non finie" "$D/train.log" || ls "$D"/model_nonfinite.pt >/dev/null 2>&1; then
    echo "[abl] ANOMALIE loss non finie ($NOM) -> arret chaine"
    return 1
  fi
  grep -q "terminé en" "$D/train.log" || { echo "[abl] ANOMALIE run incomplet ($NOM) -> arret"; return 1; }
  $V tools/eval_common.py "$D/model.pt" "$D/eval.json" 2>&1 | tee -a "$D/train.log"
  # Steps totaux = second nombre de "step X/Y" (le dernier [train] affiche X < Y).
  ST=$(grep -oE "step [0-9]+/[0-9]+" "$D/train.log" | grep -v "\[val\]" | tail -1 | sed -E 's|step [0-9]*/([0-9]+)|\1|')
  TV=$((ST * BS * ACC * SEQ))
  K=0.53288; [ "$SEED" -eq 1 ] && K=0.51815  # facteur BPC du split val in-run
  [ "$NOM" = "V5-seq256" ] && K=0.53258  # seq 256 : ratio tok/char du val in-run légèrement différent
  NOTE=""; [ "$NOM" = "V1b-temps-egal" ] && NOTE="@100/200/300 = mi-parcours (610 steps), non comparables"
  ligne_md "$D/train.log" "$D" "$NOM" "$LABEL" "$SEED" "$ST" "$TV" "$NOTE" "$K" >> "$ABL"
  echo "[abl] $NOM termine : steps=$ST tokens=$TV"
  rm -f "$D"/model_ckpt*.pt "$D"/model_e*.pt
  return 0
}

lancer V0-seed0 "baseline KD" 0 0.7 3e-4 128 8 16 1 "" || exit 1
lancer V0-seed1 "baseline KD, seed bruit" 1 0.7 3e-4 128 8 16 1 "" || exit 1
lancer V1a-alpha0 "CE seule, memes tokens" 0 0.0 3e-4 128 8 16 1 "" || exit 1
# V1b : budget temps = mur de V0-seed0, debit = V1a (meme config CE seule).
W0=$(grep -E "terminé en" runs/$DATE-abl-V0-seed0/train.log | tail -1 | sed -E 's/.*terminé en ([0-9]+)s.*/\1/')
S1=$(grep -oE "step [0-9]+/[0-9]+" runs/$DATE-abl-V1a-alpha0/train.log | grep -v "\[val\]" | tail -1 | sed -E 's|step [0-9]*/([0-9]+)|\1|')
W1=$(grep -E "terminé en" runs/$DATE-abl-V1a-alpha0/train.log | tail -1 | sed -E 's/.*terminé en ([0-9]+)s.*/\1/')
EP1B=$(( (W0 * S1 + W1 * SPE_SUBSET - 1) / (W1 * SPE_SUBSET) ))
[ "$EP1B" -lt 1 ] && EP1B=1
echo "[abl] V1b : mur V0=${W0}s, ${S1} steps V1a en ${W1}s -> epochs=$EP1B"
lancer V1b-temps-egal "CE seule, temps egal V0" 0 0.0 3e-4 128 8 16 "$EP1B" "" || exit 1
lancer V2-alpha03 "alpha 0.3" 0 0.3 3e-4 128 8 16 1 "" || exit 1
lancer V3-lr1e3 "lr 1e-3" 0 0.7 1e-3 128 8 16 1 "" || exit 1
lancer V4-subln "subln (normes o/down)" 0 0.7 3e-4 128 8 16 1 "--subln" || exit 1
lancer V5-seq256 "seq 256, 16384 tok/step" 0 0.7 3e-4 256 4 16 1 "" || exit 1
echo "[abl] CHAINE TERMINEE"
