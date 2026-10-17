#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
V=runs/E-t02/model.pt
VB=runs/2026-10-08-tranche00/tranche00.pt.vocab.json
.venv/bin/python tools/eval_valbig.py --ckpt $V --docs runs/abl_data/val_big.jsonl --vocab $VB --out docs/results/valbig_Et02.json --dump-per-doc docs/results/valbig_Et02_perdoc.json --draws 1000 --seed 0 > logs/eval_Et02_valbig.log 2>&1
.venv/bin/python tools/eval_valbig.py --ckpt $V --docs runs/abl_data/ood.jsonl --vocab $VB --out docs/results/ood_Et02.json --dump-per-doc docs/results/ood_Et02_perdoc.json --draws 1000 --seed 0 > logs/eval_Et02_ood.log 2>&1
echo $? > logs/Et02eval.exit
date > logs/Et02eval.done
