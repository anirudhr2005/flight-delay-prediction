"""Phase 3: train Logistic Regression, Decision Tree, Random Forest, SVM and
Naive Bayes on the stratified sample (model definitions in
src/flight_delay/models.py).

- Train/test split from --split (default chronological: train on earlier
  months, test on the last two); the test set is untouched until Phase 4.
- 5-fold stratified cross-validation on the training set gives mean +/- std
  of each metric, so the comparison isn't down to one lucky split.
- The out-of-fold probabilities from that cross-validation are saved for
  Phase 4 threshold tuning, so no model is cross-validated twice.
- Each model is then refitted on all its training rows and saved to models/.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from flight_delay.cli import year_parser
from flight_delay.config import (
    CV_FOLDS, FEATURES, PROCESSED_DIR, REPORTS_DIR, SEED, SVM_TRAIN_ROWS, model_path, oof_path,
    run_tag,
)
from flight_delay.data import describe_split, train_test, training_rows
from flight_delay.models import MODEL_FILES, build_models

args = year_parser(__doc__, split=True).parse_args()
YEAR, SPLIT, TAG = args.year, args.split, run_tag(args.year, args.split)
X_train, X_test, y_train, y_test = train_test(YEAR, SPLIT)
print(describe_split(YEAR, SPLIT))

SCORING = {
    "accuracy": "accuracy",
    "precision_delayed": "precision",
    "recall_delayed": "recall",
    "f1_delayed": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
model_path("x", YEAR, SPLIT).parent.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
rows, oof_frames = [], []

for name, pipe in build_models().items():
    Xt, yt = training_rows(name, X_train, y_train)
    print(f"\n{name}: cross-validating on {len(Xt):,} rows ...", flush=True)
    start = time.time()
    scores = cross_validate(pipe, Xt, yt, cv=cv, scoring=SCORING,
                            return_estimator=True, return_indices=True)
    cv_seconds = time.time() - start

    # Out-of-fold probability for every training row, from the fold model
    # that did not see it.
    oof = np.empty(len(Xt))
    for est, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof[val_idx] = est.predict_proba(Xt.iloc[val_idx])[:, 1]
    oof_frames.append(pd.DataFrame({"model": name, "row": Xt.index, "y_true": yt.to_numpy(),
                                    "proba": oof}))

    row = {"model": name, "train_rows": len(Xt)}
    for metric in SCORING:
        vals = scores[f"test_{metric}"]
        row[f"cv_{metric}_mean"], row[f"cv_{metric}_std"] = vals.mean(), vals.std()

    start = time.time()
    pipe.fit(Xt, yt)
    row["fit_seconds"], row["cv_seconds"] = time.time() - start, cv_seconds
    rows.append(row)
    joblib.dump(pipe, model_path(MODEL_FILES[name], YEAR, SPLIT), compress=3)
    print(f"  CV macro F1 {row['cv_f1_macro_mean']:.3f} ± {row['cv_f1_macro_std']:.3f}, "
          f"delayed F1 {row['cv_f1_delayed_mean']:.3f}, ROC-AUC {row['cv_roc_auc_mean']:.3f}  "
          f"({cv_seconds:.0f}s CV, {row['fit_seconds']:.0f}s final fit)", flush=True)

results = pd.DataFrame(rows)
results.to_csv(REPORTS_DIR / f"phase3_cv_results_{TAG}.csv", index=False)
pd.concat(oof_frames).to_parquet(oof_path(YEAR, SPLIT), index=False)
(REPORTS_DIR / f"phase3_split_{TAG}.json").write_text(json.dumps({
    "split": SPLIT, "train_rows": len(X_train), "test_rows": len(X_test),
    "train_delay_rate": float(y_train.mean()), "test_delay_rate": float(y_test.mean()),
    "svm_train_rows": SVM_TRAIN_ROWS,
    "cv_folds": CV_FOLDS, "features": FEATURES, "seed": SEED,
}, indent=2))

table = results[["model", "train_rows"] + [f"cv_{m}_mean" for m in SCORING]].copy()
table.columns = ["Model", "Train rows", "Accuracy", "Precision (delayed)", "Recall (delayed)",
                 "F1 (delayed)", "Macro F1", "ROC-AUC"]
print(f"\n{CV_FOLDS}-fold cross-validation ({SPLIT} split, training set, mean):")
print(table.sort_values("Macro F1", ascending=False).to_string(index=False, float_format="%.3f"))
