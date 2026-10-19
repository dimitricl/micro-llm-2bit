#!/bin/bash
# Baseline sonde FR v1 sur t08 (CPU/NEON). Aucun GPU, aucun train, aucun lock.
cd /Volumes/Lexar/micro-llm-2bit || exit 1
mkdir -p eval/Et08-probe
nice -n 10 .venv/bin/python tools/eval_probe.py \
    --model runs/E-t08/model.bin \
    --out eval/Et08-probe/samples.md \
    --results-out eval/Et08-probe/results.json \
    > logs/eval_probe_Et08.log 2>&1
echo $? > logs/Et08probe.exit
date > logs/Et08probe.done
