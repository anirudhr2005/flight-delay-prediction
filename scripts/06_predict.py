"""Predict arrival delays for new flights with a trained model.

Reads a CSV with one row per flight and the feature columns listed in
src/flight_delay/config.py (see examples/sample_flights.csv), and prints the
delay probability and the prediction at the threshold Phase 4 chose for the
given objective (see src/flight_delay/thresholds.py).

    python scripts/06_predict.py examples/sample_flights.csv
    python scripts/06_predict.py flights.csv --model "Logistic Regression" --out preds.csv
    python scripts/06_predict.py flights.csv --objective recall_70%
"""
import argparse

import joblib
import pandas as pd

from flight_delay.config import (
    DEFAULT_SPLIT, DEFAULT_YEAR, FEATURES, REPORTS_DIR, SPLITS, model_path, run_tag,
)
from flight_delay.models import MODEL_FILES
from flight_delay.thresholds import OBJECTIVES, PRIMARY

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("csv", help="flights to score")
parser.add_argument("--model", default="Random Forest", choices=list(MODEL_FILES))
parser.add_argument("--year", default=DEFAULT_YEAR, help="which trained run to use")
parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SPLITS, help="which trained run to use")
parser.add_argument("--objective", default=PRIMARY, choices=list(OBJECTIVES),
                    help="which Phase 4 threshold to apply")
parser.add_argument("--out", help="optional CSV to write predictions to")
args = parser.parse_args()

flights = pd.read_csv(args.csv)
missing = [c for c in FEATURES if c not in flights.columns]
if missing:
    parser.error(f"{args.csv} is missing columns: {', '.join(missing)}")

path = model_path(MODEL_FILES[args.model], args.year, args.split)
if not path.exists():
    parser.error(f"{path} not found - run scripts/04_train_models.py --split {args.split} first")
model = joblib.load(path)

threshold = 0.5
results_path = REPORTS_DIR / f"phase4_test_results_{run_tag(args.year, args.split)}.csv"
if results_path.exists():
    results = pd.read_csv(results_path)
    chosen = results[(results["model"] == args.model) & (results["objective"] == args.objective)]
    if chosen.empty or chosen["threshold"].isna().all():
        parser.error(f"no threshold for {args.objective} - its target was unreachable for {args.model}")
    threshold = float(chosen["threshold"].iloc[0])
else:
    print(f"{results_path.name} not found - using threshold 0.5")

proba = model.predict_proba(flights[FEATURES])[:, 1]
out = flights[["Airline", "Origin", "Dest", "DayofMonth", "dep_hour"]].assign(
    delay_probability=proba.round(3),
    predicted=["Delayed" if p >= threshold else "On time" for p in proba],
)
print(f"{args.model} ({args.split} split), {args.objective} threshold {threshold:.2f}\n")
print(out.to_string(index=False))
if args.out:
    out.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")
