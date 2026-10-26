#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
.venv/bin/python tools/eval_valbig.py --ckpt runs/R4-qwen-kd/model.pt --docs runs/abl_data/val_big.jsonl --vocab runs/abl_data/vocab_qwen.json --out docs/results/valbig_R4.json --dump-per-doc docs/results/valbig_R4_perdoc.json --draws 1000 --seed 0 > logs/eval_R4_valbig.log 2>&1
.venv/bin/python tools/eval_valbig.py --ckpt runs/R4-qwen-kd/model.pt --docs runs/abl_data/ood.jsonl --vocab runs/abl_data/vocab_qwen.json --out docs/results/ood_R4.json --dump-per-doc docs/results/ood_R4_perdoc.json --draws 1000 --seed 0 > logs/eval_R4_ood.log 2>&1
echo $? > logs/R4eval.exit
date > logs/R4eval.done
