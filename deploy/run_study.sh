#!/usr/bin/env bash
# Calibration study on the server: build both benchmarks, run every question
# through the live pipeline (resumable), fit the calibrator, write the report.
# Started by systemd-run (see deploy/README.md) so it survives SSH logouts.
set -euo pipefail
cd /srv/oraculum
PY=.venv/bin/python
[ -f evals/benchmarks/squad/questions.jsonl ] || $PY -m research build-squad
[ -f evals/benchmarks/acme/questions.jsonl ] || $PY -m research build-acme
$PY -m research run squad
$PY -m research run acme
$PY -m research study --transfer acme
echo "Study finished. Restart the API to load the calibrator: sudo systemctl restart oraculum"
