#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup .venv/bin/python main.py train --data /Volumes/Lexar/clean-corpus/dedup/tranche_04.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/E-t04/model.pt --config tiny --vocab-size 12000 --vocab-from runs/2026-10-08-tranche00/tranche00.pt.vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 6 --seed 0 --alpha 0.7 --lr 3e-4 --temperature 2.0 --val-data runs/abl_data/val_E_frozen.jsonl --eval-every 100 --ckpt-every 500 --resume runs/E-t04/model_ckpt0012000.pt --rewarmup-ratio 0.05 --lr-resume 3e-4 >> logs/Et04.log 2>&1 &
echo $! > logs/Et04_resume.pid
echo "resume pid $(cat logs/Et04_resume.pid)"
