#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_E_tranche.sh 07 runs/E-t06 --go > logs/Et07.log 2>&1 &
echo $! > logs/Et07.pid
echo "t07 launcher pid $(cat logs/Et07.pid)"
