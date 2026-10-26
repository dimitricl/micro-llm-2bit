#!/bin/bash
cd /Volumes/Lexar/micro-llm-2bit
.venv/bin/python -m pytest -q > logs/pytest.log 2>&1
echo $? > logs/pytest.exit
date > logs/pytest.done
