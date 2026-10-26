#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
mkdir -p runs/R4-qwen-kd
nice -n 10 caffeinate -dims .venv/bin/python main.py train \
  --data runs/abl_data/resume_5M.jsonl \
  --parent Qwen/Qwen3-0.6B-Base \
  --out runs/R4-qwen-kd/model.pt \
  --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab_qwen.json \
  --seq-len 128 --save-batch 4 --batch-size 4 --accum 16 \
  --epochs 1 --seed 0 --alpha 0.7 --lr 3e-4 --temperature 2.0 \
  --val-ratio 0.05 --eval-every 50 --ckpt-every 500 \
  > logs/R4.log 2>&1
echo $? > logs/R4.exit
date > logs/R4.done
.venv/bin/python tools/watch_run.py --log logs/R4.log --log-only >> logs/R4.log 2>&1 || true
