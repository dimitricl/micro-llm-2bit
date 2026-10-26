#!/bin/bash
# Étape 4 — campagne complète, STRICTEMENT séquentielle :
#   bras A -> mesures -> bras B -> mesures -> bras C -> mesures -> tableau
#
#   bash tools/run_F_step4.sh            # affiche le plan, ne lance rien
#   bash tools/run_F_step4.sh --go       # lance (≈ 5,5–6,5 h)
#
# Un seul job GPU à la fois : chaque train prend logs/gpu.lock, chaque
# campagne de mesures le reprend après le train. Arrêt au premier échec.
# Reprise après crash : relancer la même commande ; run_F_arm.sh repart du
# dernier model_ckpt*.pt et measure_F_arm.sh réutilise les sorties déjà là.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
LOCK=logs/gpu.lock
HEART=logs/F-heartbeat.md
ARMS=(A B C)

echo "[step4] campagne F — 3 bras, tranche_09, reprise E-t08 (poids seuls)"
echo "[step4] A : KD 0.7, lr 3e-4, cosine + rewarmup 5%, WD 0.01"
echo "[step4] B : CE seule (alpha 0), lr 1e-3, WSD 5/20/0.1, WD 0.1 -> 0 à 50%"
echo "[step4] C : KD 0.7, lr 1e-3, WSD 5/20/0.1, WD 0.1 -> 0 à 50%"
echo "[step4] 924 steps par bras, 15,1 M tokens, seed 0, val gelée"
echo "[step4] ordre : ${ARMS[*]} (mesures après chaque bras)"

if [ "${1:-}" != "--go" ]; then
  echo "[step4] PRÉPARÉ (rien lancé). Relancer avec --go."
  exit 0
fi

if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[step4] TRAIN DÉJÀ ACTIF -> abandon"; exit 1
fi
if [ -f "$LOCK" ]; then
  echo "[step4] LOCK GPU PRÉSENT ($LOCK) -> abandon"; exit 1
fi
OLLAMA_VRAM=$(curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null \
  | python3 -c 'import json,sys;print(sum(m.get("size_vram",0) for m in json.load(sys.stdin).get("models",[])))' \
  2>/dev/null || echo 0)
if [ "${OLLAMA_VRAM:-0}" -gt 2000000000 ]; then
  echo "[step4] OLLAMA > 2 Go en VRAM (${OLLAMA_VRAM} octets) -> abandon"
  exit 1
fi

# Garde-fou espace disque : ~0,7 Go de checkpoints par bras + exports.
FREE_KB=$(df -k . | awk 'NR==2{print $4}')
if [ "$FREE_KB" -lt 5000000 ]; then
  echo "[step4] ESPACE DISQUE FAIBLE (${FREE_KB} Ko libres < 5 Go) -> abandon"
  exit 1
fi

t0=$(date +%s)
{
  echo ""
  echo "## $(date '+%Y-%m-%d %H:%M') — étape 4 : campagne A/B/C"
  echo ""
  echo "- ordre ${ARMS[*]} ; 924 steps/bras ; reprise E-t08 poids seuls"
  echo "- log global : \`logs/F-step4.log\`"
} >> "$HEART"

# Chien de garde : les bras B/C tournent à lr 1e-3 avec 46 steps de warmup.
# Une divergence franche (val-PPL > 120, soit 6x la référence 19,8, ou loss
# NaN) doit arrêter le bras au lieu de brûler 2 h de GPU. Seuil volontairement
# très large : aucune val observée sur E ne dépasse ~31 en cours de run.
watchdog() {
  local arm="$1" log="runs/F-$1/train.log"
  while kill -0 "$2" 2>/dev/null; do
    if [ -f "$log" ]; then
      if grep -q "loss=nan\|val_ce=nan" "$log" 2>/dev/null; then
        echo "[watchdog-$arm] loss NaN -> arrêt du train" | tee -a "$log"
        touch "runs/F-$arm/.diverged"
        pkill -f "main.py train" 2>/dev/null
        return
      fi
      local ppl
      ppl=$(grep -o 'val_ppl~[0-9.]*' "$log" 2>/dev/null | tail -1 | cut -d'~' -f2)
      if [ -n "${ppl:-}" ] && awk "BEGIN{exit !($ppl > 120)}"; then
        echo "[watchdog-$arm] val_ppl=$ppl > 120 -> arrêt du train" | tee -a "$log"
        touch "runs/F-$arm/.diverged"
        pkill -f "main.py train" 2>/dev/null
        return
      fi
    fi
    sleep 60
  done
}

for ARM in "${ARMS[@]}"; do
  echo ""
  echo "[step4] ===== bras $ARM : train ====="
  bash tools/run_F_arm.sh "$ARM" --go &
  TRAIN_PID=$!
  watchdog "$ARM" "$TRAIN_PID" &
  WD_PID=$!
  wait "$TRAIN_PID"
  RC=$?
  kill "$WD_PID" 2>/dev/null
  wait "$WD_PID" 2>/dev/null
  if [ -f "runs/F-$ARM/.diverged" ]; then
    echo "[step4] bras $ARM : DIVERGENCE détectée -> arrêt (voir runs/F-$ARM/train.log)"
    exit 4
  fi
  if [ "$RC" -ne 0 ]; then
    echo "[step4] bras $ARM : train ÉCHEC exit=$RC -> arrêt"
    exit "$RC"
  fi
  echo "[step4] ===== bras $ARM : mesures ====="
  bash tools/measure_F_arm.sh "$ARM"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "[step4] bras $ARM : mesures ÉCHEC exit=$RC -> arrêt"
    exit "$RC"
  fi
done

echo "[step4] ===== tableau comparatif ====="
"$V" tools/summarize_F.py || echo "[step4] summarize_F.py a échoué (non bloquant)"
t1=$(date +%s)
echo "[step4] TERMINÉ en $((t1-t0))s"
{
  echo "- fin campagne : durée totale $((t1-t0))s"
} >> "$HEART"
