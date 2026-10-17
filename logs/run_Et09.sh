#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_E_tranche.sh 09 runs/E-t08 --go > logs/Et09.log 2>&1 &
echo $! > logs/Et09.pid
echo "t09 launcher pid $(cat logs/Et09.pid)"
