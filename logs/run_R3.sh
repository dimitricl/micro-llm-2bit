#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_r3_original.sh > logs/R3.log 2>&1 &
echo $! > logs/R3.pid
echo "R3 pid $(cat logs/R3.pid)"
