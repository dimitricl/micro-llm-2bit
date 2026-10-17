#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_E_tranche.sh 02 runs/E-t01 --go > logs/Et02.log 2>&1 &
echo $! > logs/Et02.pid
echo "t02 launcher pid $(cat logs/Et02.pid)"
