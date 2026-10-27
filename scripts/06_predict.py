"""Predict arrival delays for new flights with a trained model.

Reads a CSV with one row per flight and the feature columns listed in
src/flight_delay/config.py (see examples/sample_flights.csv), and prints the
delay probability and the prediction at the model's tuned threshold from
Phase 4.

    python scripts/06_predict.py examples/sample_flights.csv
    python scripts/06_predict.py flights.csv --model "Logistic Regression" --out preds.csv
"""
import argparse

import joblib
import pandas as pd

from flight_delay.config import DEFAULT_YEAR, FEATURES, MODELS_DIR, REPORTS_DIR
from flight_delay.models import MODEL_FILES

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("csv", help="flights to score")
parser.add_argument("--model", default="Random Forest", choices=list(MODEL_FILES))
parser.add_argument("--year", default=DEFAULT_YEAR, help="which trained run to use")
parser.add_argument("--out", help="optional CSV to write predictions to")
args = parser.parse_args()

flights = pd.read_csv(args.csv)
missing = [c for c in FEATURES if c not in flights.columns]
if missing:
    parser.error(f"{args.csv} is missing columns: {', '.join(missing)}")

model_path = MODELS_DIR / f"{MODEL_FILES[args.model]}.joblib"
if not model_path.exists():
    parser.error(f"{model_path} not found - run scripts/04_train_models.py first")
model = joblib.load(model_path)

threshold = 0.5
results_path = REPORTS_DIR / f"phase4_test_results_{args.year}.csv"
if results_path.exists():
    results = pd.read_csv(results_path)
    tuned = results[(results["model"] == args.model) & (results["threshold_type"] == "tuned")]
    if not tuned.empty:
        threshold = float(tuned["threshold"].iloc[0])

proba = model.predict_proba(flights[FEATURES])[:, 1]
out = flights[["Airline", "Origin", "Dest", "Month", "DayofMonth", "dep_hour"]].assign(
    delay_probability=proba.round(3),
    predicted=["Delayed" if p >= threshold else "On time" for p in proba],
)
print(f"{args.model}, threshold {threshold:.2f}\n")
print(out.to_string(index=False))
if args.out:
    out.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")
