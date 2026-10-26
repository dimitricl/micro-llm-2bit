#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
.venv/bin/python tools/eval_valbig.py --ckpt runs/R3-smol135-kd/model.pt --docs runs/abl_data/val_big.jsonl --vocab runs/2026-10-08-tranche00/tranche00.pt.vocab.json --out docs/results/valbig_R3.json --dump-per-doc docs/results/valbig_R3_perdoc.json --draws 1000 --seed 0 > logs/eval_R3_valbig.log 2>&1
.venv/bin/python tools/eval_valbig.py --ckpt runs/R3-smol135-kd/model.pt --docs runs/abl_data/ood.jsonl --vocab runs/2026-10-08-tranche00/tranche00.pt.vocab.json --out docs/results/ood_R3.json --dump-per-doc docs/results/ood_R3_perdoc.json --draws 1000 --seed 0 > logs/eval_R3_ood.log 2>&1
echo $? > logs/R3eval.exit
date > logs/R3eval.done
