#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
nice -n 10 .venv/bin/python tools/dedup_corpus.py > logs/dedup.log 2>&1
echo $? > logs/dedup.exit
date > logs/dedup.done
