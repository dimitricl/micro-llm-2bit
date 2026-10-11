#!/bin/bash
# Runs R1/R2 de RESUME (H5.1) — PRÉPARÉS, NE PAS LANCER sans "go resume".
# Base : runs/2026-10-08-tranche00/tranche00.pt (step 4248) + son vocab
# (ancien format unk==eos==0, supporté). Sous-ensemble COMMUN aux deux runs :
# runs/abl_data/resume_5M.jsonl (835 premiers docs de slice_01, ordre
# d'origine, seed 0 ; 5 005 775 tokens ; régénérable : voir le bloc
# « subset » ci-dessous). CE seule (alpha 0, SANS parent, ~4 400 tok/s),
# seq 128. R1 : lr 1e-3 (+lr-resume) ; R2 : lr d'origine tranche00 =
# 3e-4 (config.json : "lr": 0.0003).
# Steps (plan_epoch réel sur le TRAIN du subset, split seed 0 / val 0.05
# répliqué : 860 docs train / 5 012 944 tokens → 306 steps/epoch) :
# E_CUMUL=15 → total 4 590 steps, reprise à 4 248 ⇒ 342 steps/run
# ≈ 5,6M tokens ≈ 21 min/run à 4 400 tok/s (débit réel mesuré au run).
# Schedule : rewarmup 5 % (17 steps), cosine vers 0,1× (P2 vérifié).
# Lancement : bash tools/run_resume.sh --go (R1 puis R2, séquentiels,
# nohup + caffeinate + log + veilleur conseillés, GPU exclusif).
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
PARENT=HuggingFaceTB/SmolLM-135M  # requis par le CLI, ignoré en CE seule
DATA=runs/abl_data/resume_5M.jsonl
PREV_PT=runs/2026-10-08-tranche00/tranche00.pt
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
EPOCHS=15

#bloc subset : tools/make_resume_subset.py (exécuté : 905 premiers docs
# slice_01, split train/val seed 0 / 0.05 répliqué → 860 docs train,
# 5 012 944 tokens ; E_CUMUL=15, 342 steps restants).
# Manifeste : runs/abl_data/resume_5M_manifest.json (test test_subset.py).

[ -f "$PREV_PT" ] || { echo "[resume] MANQUANT : $PREV_PT"; exit 2; }
[ -f "$VOCAB" ] || { echo "[resume] MANQUANT : $VOCAB"; exit 2; }
[ -f "$DATA" ] || { echo "[resume] MANQUANT : $DATA (voir bloc subset)"; exit 2; }

if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[resume] AUTRE TRAIN ACTIF -> abandon"; exit 1
fi
if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
  echo "[resume] MODÈLE OLLAMA CHARGÉ -> abandon"; exit 1
fi

run_one() {  # $1 tag, $2 lr
  local tag="$1"
  local lr="$2"
  local out="runs/${tag}/model.pt"
  mkdir -p "runs/${tag}"
  echo "[resume] === $tag (lr=$lr) ==="
  caffeinate -dims $V main.py train --data "$DATA" --parent "$PARENT" \
    --out "$out" --config tiny --vocab-size 12000 --vocab-from "$VOCAB" \
    --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
    --epochs "$EPOCHS" --seed 0 --alpha 0.0 --lr "$lr" --temperature 2.0 \
    --val-ratio 0.05 --eval-every 50 --ckpt-every 500 \
    --resume "$PREV_PT" --rewarmup-ratio 0.05 --lr-resume "$lr" \
    2>&1 | tee "runs/${tag}/train.log"
  # P3 : log-only (abort sur loss non finie seule ; seuils 2×/25 %
  # informatifs — la récupération CE après reprise peut être non monotone).
  # Val in-run toutes les 50 steps conservée (courbe de récupération).
  $V tools/watch_run.py --log "runs/${tag}/train.log" --log-only
}

if [ "${1:-}" != "--go" ]; then
  echo "[resume] PRÉPARÉ (rien lancé). Relancer avec --go."
  echo "[resume] R1 : lr 1e-3 ; R2 : lr 3e-4 (origine tranche00) ; ~20 min/run."
  exit 0
fi
run_one R1-ce-lr1e3 1e-3
run_one R2-ce-lr3e4 3e-4
echo "[resume] TERMINE"
