#!/bin/bash
# Éval downstream (H5.3) — PRÉPARÉ, smoke testé, PAS d'éval complète lancée.
# Tâche FR : belebele_fra_Latn (compréhension, multiple_choice).
# Cibles : parent SmolLM-135M (référence) ; tranche00/tranche01 = archi
# custom, NON compatibles HF : adaptateur à écrire le jour du run
# (wrapper CausalLM ou export + scoring externe) — NE PAS lancer avant.
# Smoke : bash tools/eval_downstream.sh --smoke (CPU, ~5 instances).
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
TASK=belebele_fra_Latn
if [ "${1:-}" = "--smoke" ]; then
  .venv/bin/python -m lm_eval run --model hf \
    --model_args pretrained=HuggingFaceTB/SmolLM-135M --device cpu \
    --tasks "$TASK" --limit 5 --batch_size 1 2>&1 | tail -15
  exit 0
fi
echo "PRÉPARÉ (rien lancé)."
echo "Complet parent : .venv/bin/python -m lm_eval run --model hf \\"
echo "  --model_args pretrained=HuggingFaceTB/SmolLM-135M --tasks $TASK"
echo "tranche00/tranche01 : adaptateur requis (archi custom tiny)."
