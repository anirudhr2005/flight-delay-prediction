#!/usr/bin/env bash
# Runs the whole pipeline end to end: inspect -> prepare -> EDA -> train -> evaluate.
# Usage: scripts/run_pipeline.sh [YEAR]   (default 2022; takes ~20 min, mostly SVM)
set -euo pipefail
YEAR="${1:-2022}"
cd "$(dirname "$0")/.."
export FLIGHT_DATA_DIR="${FLIGHT_DATA_DIR:-$HOME/Desktop/archive}"
PY=.venv/bin/python

$PY scripts/01_inspect_dataset.py --year "$YEAR"
$PY scripts/02_prepare_data.py --year "$YEAR"
$PY scripts/03_eda.py --year "$YEAR"
$PY scripts/04_train_models.py --year "$YEAR" | tee "reports/phase3_train_log.txt"
$PY scripts/05_evaluate.py --year "$YEAR" | tee "reports/phase4_eval_log.txt"
