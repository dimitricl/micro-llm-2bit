#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_E_tranche.sh 06 runs/E-t05 --go > logs/Et06.log 2>&1 &
echo $! > logs/Et06.pid
echo "t06 launcher pid $(cat logs/Et06.pid)"
