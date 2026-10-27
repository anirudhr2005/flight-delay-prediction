#!/usr/bin/env bash
# Phase 1: download one year of the Kaggle Flight Delay Dataset (2018-2022).
# Usage: scripts/download_data.sh [YEAR]   (default 2022)
set -euo pipefail
YEAR="${1:-2022}"
cd "$(dirname "$0")/.."
.venv/bin/kaggle datasets download robikscube/flight-delay-dataset-20182022 \
  -f "Combined_Flights_${YEAR}.parquet" -p data/raw
