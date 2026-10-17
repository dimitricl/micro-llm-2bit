#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
V=runs/E-t03/model.pt
VB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
.venv/bin/python tools/eval_valbig.py --ckpt $V --docs runs/abl_data/val_big.jsonl --vocab $VB --out docs/results/valbig_Et03.json --dump-per-doc docs/results/valbig_Et03_perdoc.json --draws 1000 --seed 0 > logs/eval_Et03_valbig.log 2>&1
.venv/bin/python tools/eval_valbig.py --ckpt $V --docs runs/abl_data/ood.jsonl --vocab $VB --out docs/results/ood_Et03.json --dump-per-doc docs/results/ood_Et03_perdoc.json --draws 1000 --seed 0 > logs/eval_Et03_ood.log 2>&1
echo $? > logs/Et03eval.exit
date > logs/Et03eval.done
