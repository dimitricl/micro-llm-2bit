#!/bin/bash
# Évals OOD (H4) — à lancer quand le GPU est libre (après S5.1).
# Protocole apparié identique (S=128, seed 0, 1000 tirages).
# Résultats : docs/results/ood_{tranche00,smol135,smol360}.json
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
DOCS=runs/abl_data/ood.jsonl
VOCAB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json

if pgrep -f "main.py train" >/dev/null 2>&1; then
  echo "[ood] TRAIN ACTIF -> abandon"; exit 1
fi
if curl -s --max-time 5 http://localhost:11434/api/ps 2>/dev/null | grep -q '"name"'; then
  echo "[ood] OLLAMA CHARGE -> abandon"; exit 1
fi

run_eval() {
  local flag="$1" val="$2" tag="$3"; shift 3
  echo "[ood] === $tag ==="
  caffeinate -dims $V tools/eval_valbig.py "$flag" "$val" \
    --docs "$DOCS" --vocab "$VOCAB" \
    --out "docs/results/ood_${tag}.summary.json" \
    --dump-per-doc "docs/results/ood_${tag}.json" \
    --draws 1000 --seed 0 "$@"
}

run_eval --ckpt runs/2026-10-08-tranche00/tranche00.pt tranche00
run_eval --parent HuggingFaceTB/SmolLM-135M smol135
run_eval --parent HuggingFaceTB/SmolLM-360M smol360 --pb 4
echo "[ood] TERMINE"
