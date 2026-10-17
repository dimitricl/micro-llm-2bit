#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
V=runs/E-t04/model.pt
VB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
.venv/bin/python tools/eval_valbig.py --ckpt $V --docs runs/abl_data/val_big.jsonl --vocab $VB --out docs/results/valbig_Et04.json --dump-per-doc docs/results/valbig_Et04_perdoc.json --draws 1000 --seed 0 > logs/eval_Et04_valbig.log 2>&1
.venv/bin/python tools/eval_valbig.py --ckpt $V --docs runs/abl_data/ood.jsonl --vocab $VB --out docs/results/ood_Et04.json --dump-per-doc docs/results/ood_Et04_perdoc.json --draws 1000 --seed 0 > logs/eval_Et04_ood.log 2>&1
echo $? > logs/Et04eval.exit
date > logs/Et04eval.done
