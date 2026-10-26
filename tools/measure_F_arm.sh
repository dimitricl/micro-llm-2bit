#!/bin/bash
# Étape 4 — mesures d'un bras F, APRÈS son entraînement.
#
#   bash tools/measure_F_arm.sh A|B|C
#
# Séquentiel, GPU exclusif (lock logs/gpu.lock), aucun fichier de référence
# écrasé. Sorties propres au bras :
#   exports/F-$ARM.bin (+ .vocab.json)
#   docs/results/valbig_F$ARM.json           (résumé, IC bootstrap)
#   docs/results/valbig_F${ARM}_perdoc.json  (per-doc, pour gate_paire)
#   docs/results/ood_F$ARM.json / ood_F${ARM}_perdoc.json
#   logs/F/gate_valbig_F$ARM.txt / gate_ood_F$ARM.txt
#   eval/F-$ARM-fixed/{samples.md,metrics.json}
#   eval/F-$ARM-probe/{samples.md,results.json}
#   runs/F-$ARM/measure.log
# Régénérés (cumulatifs, jamais des références) :
#   docs/results/surface_F.json, docs/SURFACE.md
#
# Le val-PPL « gelé » de référence reste celui du journal d'entraînement
# (même val_E_frozen, même protocole, cf. t08 19,8 / t09 19,8).
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
ARM="${1:?usage: measure_F_arm.sh A|B|C}"
case "$ARM" in A|B|C) ;; *) echo "[mes-$ARM] bras inconnu (A|B|C)"; exit 2 ;; esac

OUTDIR=runs/F-$ARM
CKPT=$OUTDIR/model.pt
BIN=exports/F-$ARM.bin
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
VALBIG=runs/abl_data/val_big.jsonl
OOD=runs/abl_data/ood.jsonl
LOCK=logs/gpu.lock
HEART=logs/F-heartbeat.md
LOG=$OUTDIR/measure.log
mkdir -p "$OUTDIR" eval/F-$ARM-fixed eval/F-$ARM-probe logs/F

[ -f "$CKPT" ] || { echo "[mes-$ARM] MANQUANT : $CKPT (train non terminé ?)"; exit 2; }
for f in "$VALBIG" "$OOD" "$VOCAB"; do
  [ -f "$f" ] || { echo "[mes-$ARM] MANQUANT : $f"; exit 2; }
done
if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[mes-$ARM] TRAIN ACTIF -> abandon"; exit 1
fi
if [ -f "$LOCK" ]; then
  echo "[mes-$ARM] LOCK GPU PRÉSENT ($LOCK) -> abandon"; exit 1
fi
OLLAMA_VRAM=$(curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null \
  | python3 -c 'import json,sys;print(sum(m.get("size_vram",0) for m in json.load(sys.stdin).get("models",[])))' \
  2>/dev/null || echo 0)
if [ "${OLLAMA_VRAM:-0}" -gt 2000000000 ]; then
  echo "[mes-$ARM] OLLAMA > 2 Go en VRAM (${OLLAMA_VRAM} octets) -> abandon"
  exit 1
fi

echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
t0=$(date +%s)
exec > >(tee -a "$LOG") 2>&1
echo "[mes-$ARM] $(date '+%Y-%m-%d %H:%M') — début des mesures"

# 1) Export .bin (déploiement CPU/NEON, mêmes conditions que t08/pilot-qa3).
if [ -f "$BIN" ]; then
  echo "[mes-$ARM] $BIN existe déjà -> réutilisé (non écrasé)"
else
  echo "[mes-$ARM] export -> $BIN"
  "$V" main.py export --ckpt "$CKPT" --out "$BIN" || {
    echo "[mes-$ARM] ÉCHEC export"; exit 3; }
fi

# 2) val_big appariée (mêmes docs, S=128, seed 0, 1000 tirages).
echo "[mes-$ARM] === val_big ==="
caffeinate -dims "$V" tools/eval_valbig.py --ckpt "$CKPT" \
  --docs "$VALBIG" --vocab "$VOCAB" \
  --out "docs/results/valbig_F${ARM}.json" \
  --dump-per-doc "docs/results/valbig_F${ARM}_perdoc.json" \
  --draws 1000 --seed 0 || { echo "[mes-$ARM] ÉCHEC val_big"; exit 3; }

# 3) OOD apparié.
echo "[mes-$ARM] === OOD ==="
caffeinate -dims "$V" tools/eval_valbig.py --ckpt "$CKPT" \
  --docs "$OOD" --vocab "$VOCAB" \
  --out "docs/results/ood_F${ARM}.json" \
  --dump-per-doc "docs/results/ood_F${ARM}_perdoc.json" \
  --draws 1000 --seed 0 || { echo "[mes-$ARM] ÉCHEC OOD"; exit 3; }

# 4) Gate apparié vs tranche00 (mêmes réfs que t08).
echo "[mes-$ARM] === gate val_big ==="
"$V" tools/gate_paire.py --ref docs/results/valbig_tranche00.json \
  --new "docs/results/valbig_F${ARM}_perdoc.json" \
  | tee "logs/F/gate_valbig_F${ARM}.txt"
echo "[mes-$ARM] === gate OOD ==="
"$V" tools/gate_paire.py --ref docs/results/ood_tranche00.json \
  --new "docs/results/ood_F${ARM}_perdoc.json" \
  | tee "logs/F/gate_ood_F${ARM}.txt"

# 5) Boucles fixes — décodage de DÉPLOIEMENT (pénalité 1,15 + no-repeat 3g),
# identique à t08/pilot-qa3 (P3). Sans ces flags, le greedy « nu » n'est pas
# comparable d'un modèle à l'autre (t08 nu = 0,5057, déploiement = 0,0000).
echo "[mes-$ARM] === eval_fixed (déploiement) ==="
"$V" tools/eval_fixed.py --model "$BIN" \
  --out "eval/F-$ARM-fixed/samples.md" \
  --metrics-out "eval/F-$ARM-fixed/metrics.json" \
  --repetition-penalty 1.15 --no-repeat-ngram 3 \
  || { echo "[mes-$ARM] ÉCHEC eval_fixed"; exit 3; }

# 6) Sonde FR gelée (exact-match, greedy de déploiement).
echo "[mes-$ARM] === sonde gelée ==="
"$V" tools/eval_probe.py --model "$BIN" \
  --out "eval/F-$ARM-probe/samples.md" \
  --results-out "eval/F-$ARM-probe/results.json" \
  || { echo "[mes-$ARM] ÉCHEC sonde"; exit 3; }

# 7) Surface + classement log-proba, cumulatif (t08, qa3, bras déjà mesurés).
SPECS=(
  't08|eval/F-Et08-fixed/samples.md|eval/Et08-probe/results.json|exports/E-t08.bin'
  'pilot-qa3|eval/F-Epilotqa3-fixed/samples.md|eval/Epilotqa3-probe/results.json|runs/E-pilot-qa3/model.bin'
)
for X in A B C; do
  if [ -f "eval/F-$X-fixed/samples.md" ] && [ -f "eval/F-$X-probe/results.json" ] \
     && [ -f "exports/F-$X.bin" ]; then
    SPECS+=("F-$X|eval/F-$X-fixed/samples.md|eval/F-$X-probe/results.json|exports/F-$X.bin")
  fi
done
echo "[mes-$ARM] === surface (${#SPECS[@]} modèles) ==="
SURF_ARGS=()
for s in "${SPECS[@]}"; do SURF_ARGS+=(--spec "$s"); done
"$V" tools/eval_surface.py "${SURF_ARGS[@]}" \
  --lexicon-cache runs/F-diag/lexique_fr.tsv.gz \
  --min-freq 5 20 \
  --out-json docs/results/surface_F.json \
  --out-md docs/SURFACE.md \
  || { echo "[mes-$ARM] ÉCHEC surface"; exit 3; }

# 8) Traces val-PPL (journal d'entraînement) pour le tableau final.
{
  echo "=== $ARM : val gelée (journal d'entraînement) ==="
  grep -h '\[val\]' "$OUTDIR/train.log" | tail -3
  grep -h '\[eval\]' "$OUTDIR/train.log" | tail -1
} | tee "$OUTDIR/val_ppl.txt"

t1=$(date +%s)
echo "[mes-$ARM] terminé en $((t1-t0))s"
{
  echo ""
  echo "## $(date '+%Y-%m-%d %H:%M') — étape 4, bras $ARM : mesures"
  echo ""
  echo "- durée mesures : $((t1-t0))s ; log \`$LOG\`"
  echo "- val_big : \`docs/results/valbig_F${ARM}.json\` ; gate \`logs/F/gate_valbig_F${ARM}.txt\`"
  echo "- OOD : \`docs/results/ood_F${ARM}.json\` ; gate \`logs/F/gate_ood_F${ARM}.txt\`"
  echo "- surface : \`docs/results/surface_F.json\` (cumulatif)"
  echo "- val gelée : \`$OUTDIR/val_ppl.txt\`"
} >> "$HEART"
