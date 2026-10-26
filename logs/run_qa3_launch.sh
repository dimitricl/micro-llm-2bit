#!/bin/bash
# Lanceur nohup du pilote QA #3 (job unique). Jetable / non committé.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
bash tools/run_pilot_qa3.sh --go
RC=$?
echo "$RC" > logs/E-pilot-qa3.exit
date > logs/E-pilot-qa3.done
exit $RC
