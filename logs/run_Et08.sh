#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_E_tranche.sh 08 runs/E-t07 --go > logs/Et08.log 2>&1 &
echo $! > logs/Et08.pid
echo "t08 launcher pid $(cat logs/Et08.pid)"
