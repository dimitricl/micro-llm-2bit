#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nohup bash tools/run_E_t01.sh --go > logs/Et01.log 2>&1 &
echo $! > logs/Et01.pid
echo "t01 launcher pid $(cat logs/Et01.pid)"
