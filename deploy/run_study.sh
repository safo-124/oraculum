#!/usr/bin/env bash
# Calibration study on the server: build both benchmarks, run every question
# through the live pipeline (resumable), fit the calibrator, write the report.
# Started by systemd-run (see deploy/README.md) so it survives SSH logouts.
#
# On a 4 GB server the study loads its own copy of the encoders, so stop the
# API first (sudo systemctl stop oraculum) and start it again afterwards.
# The benchmark is sized for CPU: 8 articles x 30 questions (240) plus the
# 40 in-domain questions, about 1.5-2.5 hours with the 0.5B model.
set -euo pipefail
cd /srv/oraculum
PY=.venv/bin/python
[ -f evals/benchmarks/squad/questions.jsonl ] || $PY -m research build-squad --articles 8 --per-article 30
[ -f evals/benchmarks/acme/questions.jsonl ] || $PY -m research build-acme
$PY -m research run squad
$PY -m research run acme
$PY -m research study --transfer acme
echo "Study finished. Start the API again to load the calibrator: sudo systemctl start oraculum"
