#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_E_tranche.sh 03 runs/E-t02 --go > logs/Et03.log 2>&1 &
echo $! > logs/Et03.pid
echo "t03 launcher pid $(cat logs/Et03.pid)"
