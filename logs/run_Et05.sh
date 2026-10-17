#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_E_tranche.sh 05 runs/E-t04 --go > logs/Et05.log 2>&1 &
echo $! > logs/Et05.pid
echo "t05 launcher pid $(cat logs/Et05.pid)"
