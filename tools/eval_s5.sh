#!/bin/bash
# Campagne S5.1 (exécutée le 11/10/2026, log ignoré runs/eval_s5.log).
# GPU exclusif MPS : précheck, évals séquentielles sous caffeinate.
# Résultats : docs/results/valbig_{tranche00,smol135,smol360}.json
# (+ tranche00_win32 : contrôle ctx32 ; e0_init : INVALIDE — voir note).
# Note 360M : OOM MPS en lots de 16 fenêtres → relance --pb 4
# (runs/eval_360_retry.log). Note e0 : tranche00_e0.pt est un doublon
# bit-identique du final (même hash) — pas un init, contrôle abandonné
# (substitué par win32 et smol135-vs-smol360, cf. DIAGNOSTICS C2-ter).
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
DOCS=runs/abl_data/val_big.jsonl
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json

if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[s5] TRAIN ACTIF -> abandon"; exit 1
fi
if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
  echo "[s5] OLLAMA CHARGE -> abandon"; exit 1
fi

run_eval() {  # $1 = modele-flag, $2 = valeur, $3 = tag, $4+ = options
  local flag="$1" val="$2" tag="$3"; shift 3
  echo "[s5] === $tag ==="
  caffeinate -dims $V tools/eval_valbig.py "$flag" "$val" \
    --docs "$DOCS" --vocab "$VOCAB" \
    --out "docs/results/valbig_${tag}.summary.json" \
    --dump-per-doc "docs/results/valbig_${tag}.json" \
    --draws 1000 --seed 0 "$@"
}

run_eval --ckpt runs/2026-10-08-tranche00/tranche00.pt tranche00
run_eval --parent HuggingFaceTB/SmolLM-135M smol135
run_eval --parent HuggingFaceTB/SmolLM-360M smol360 --pb 4
run_eval --ckpt runs/2026-10-08-tranche00/tranche00.pt tranche00_win32 --win 32
echo "[s5] TERMINE"
