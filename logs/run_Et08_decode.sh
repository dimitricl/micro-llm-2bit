#!/bin/bash
# Balayage des leviers de decodage sur t08 (CPU pur, noyau NEON).
# Aucun GPU/MPS, aucun train, aucun lock GPU, aucun Ollama.
cd /Volumes/Lexar/micro-llm-2bit || exit 1
M=runs/E-t08/model.bin
OUT=eval/Et08-decode
run() {  # $1 = config, $2.. = options
  cfg="$1"; shift
  mkdir -p "$OUT/$cfg"
  nice -n 10 .venv/bin/python tools/eval_fixed.py \
      --model "$M" \
      --out "$OUT/$cfg/samples.md" \
      --metrics-out "$OUT/$cfg/metrics.json" "$@" \
      > "logs/eval_${cfg}.log" 2>&1
  echo "$cfg exit=$?" >> logs/Et08-decode.exit
}

run C0
run C1 --repetition-penalty 1.15
run C2 --repetition-penalty 1.30
run C3 --no-repeat-ngram 3
run C4 --repetition-penalty 1.15 --no-repeat-ngram 3
run C5
date > logs/Et08-decode.done
