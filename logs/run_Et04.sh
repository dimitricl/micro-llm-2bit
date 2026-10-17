#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_E_tranche.sh 04 runs/E-t03 --go > logs/Et04.log 2>&1 &
echo $! > logs/Et04.pid
echo "t04 launcher pid $(cat logs/Et04.pid)"
