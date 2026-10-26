#!/bin/bash
# Génération du corpus QA v3 gelé (CPU, déterministe). Jetable / non committé.
set -u
cd /Volumes/Lexar/micro-llm-2bit || exit 1
nice -n 10 caffeinate -dims .venv/bin/python tools/make_qa_corpus.py --version v3
echo "exit=$?" > logs/gen_v3.exit
date > logs/gen_v3.done
