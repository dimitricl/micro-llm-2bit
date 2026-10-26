#!/bin/bash
# Mesure du pilote QA #3 (CPU/NEON uniquement, après la fin du run). Jetable.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
V=.venv/bin/python
M=runs/E-pilot-qa3/model.bin
mkdir -p eval/Epilotqa3-probe eval/Epilotqa3-fixed
$V main.py export --ckpt runs/E-pilot-qa3/model.pt --out "$M" \
  > logs/E-pilot-qa3-export.log 2>&1
echo "export=$?" >> logs/E-pilot-qa3-export.log
$V tools/eval_probe.py --model "$M" \
  --out eval/Epilotqa3-probe/samples.md \
  --results-out eval/Epilotqa3-probe/results.json \
  > logs/E-pilot-qa3-probe.log 2>&1
echo "probe=$?" >> logs/E-pilot-qa3-probe.log
$V tools/eval_fixed.py --model "$M" \
  --out eval/Epilotqa3-fixed/samples_deploy.md \
  --metrics-out eval/Epilotqa3-fixed/metrics_deploy.json \
  --repetition-penalty 1.15 --no-repeat-ngram 3 \
  > logs/E-pilot-qa3-fixed-deploy.log 2>&1
echo "fixed_deploy=$?" >> logs/E-pilot-qa3-fixed-deploy.log
$V tools/eval_fixed.py --model "$M" \
  --out eval/Epilotqa3-fixed/samples_bare.md \
  --metrics-out eval/Epilotqa3-fixed/metrics_bare.json \
  > logs/E-pilot-qa3-fixed-bare.log 2>&1
echo "fixed_bare=$?" >> logs/E-pilot-qa3-fixed-bare.log
$V logs/assemble_qa3.py > logs/E-pilot-qa3-assemble.log 2>&1
echo "assemble=$?" >> logs/E-pilot-qa3-assemble.log
date > logs/E-pilot-qa3-measure.done
