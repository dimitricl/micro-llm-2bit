#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit || exit 1
rm -f logs/E-pilot-qa.exit logs/E-pilot-qa.done
nohup nice -n 10 bash tools/run_pilot_qa.sh --go > logs/E-pilot-qa.log 2>&1 &
echo $! > logs/E-pilot-qa.pid
wait $!
echo $? > logs/E-pilot-qa.exit
date > logs/E-pilot-qa.done
