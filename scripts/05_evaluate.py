"""Phase 4: evaluate the five trained models on the held-out test set.

Threshold tuning is done without touching the test set: each model's
decision threshold is chosen to maximise macro F1 on the out-of-fold
probabilities saved by Phase 3 (scripts/04_train_models.py). The test set is
then scored once at the default 0.5 threshold and once at the tuned one.

Outputs:
  reports/phase4_test_results_<YEAR>.csv   metrics per model and threshold
  reports/phase4_summary_<YEAR>.txt        tables, reports, bootstrap CIs
  reports/figures/09-14_*.png              confusion matrices, ROC, PR, etc.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from flight_delay.cli import year_parser
from flight_delay.config import MODELS_DIR, REPORTS_DIR, SEED, oof_path
from flight_delay.data import train_test
from flight_delay.models import MODEL_FILES
from flight_delay.plotting import INK, INK_2, MODEL_COLORS as COLORS, SURFACE, apply_style, save

YEAR = year_parser(__doc__).parse_args().year
_, X_test, _, y_test = train_test(YEAR)
y_test = y_test.to_numpy()
if not oof_path(YEAR).exists():
    raise FileNotFoundError(f"{oof_path(YEAR)} not found - run scripts/04_train_models.py first")
oof_all = pd.read_parquet(oof_path(YEAR))
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
apply_style()


def macro_f1_curve(y_true, proba):
    return np.array([f1_score(y_true, proba >= t, average="macro") for t in THRESHOLDS])


def metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_delayed": precision_score(y_true, pred),
        "recall_delayed": recall_score(y_true, pred),
        "f1_delayed": f1_score(y_true, pred),
        "f1_ontime": f1_score(y_true, pred, pos_label=0),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def bootstrap_ci(y_true, pred, n=500):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    scores = [f1_score(y_true[i], pred[i], average="macro") for i in idx]
    return np.percentile(scores, [2.5, 97.5])


rows, probas, curves, summary = [], {}, {}, []
for name, fname in MODEL_FILES.items():
    model = joblib.load(MODELS_DIR / f"{fname}.joblib")
    oof = oof_all[oof_all["model"] == name]
    if oof.empty:
        raise ValueError(f"No out-of-fold predictions for {name} - rerun scripts/04_train_models.py")
    curve = macro_f1_curve(oof["y_true"].to_numpy(), oof["proba"].to_numpy())
    best_t = float(THRESHOLDS[curve.argmax()])
    curves[name] = curve

    proba = model.predict_proba(X_test)[:, 1]
    probas[name] = proba
    for label, t in [("default", 0.5), ("tuned", best_t)]:
        rows.append({"model": name, "threshold_type": label, "threshold": t,
                     **metrics(y_test, proba, t)})
    pred = (proba >= best_t).astype(int)
    lo, hi = bootstrap_ci(y_test, pred)
    rows[-1]["f1_macro_ci_low"], rows[-1]["f1_macro_ci_high"] = lo, hi
    summary += [f"\n=== {name} (tuned threshold {best_t:.2f}) ===",
                classification_report(y_test, pred, target_names=["On time", "Delayed"], digits=3),
                f"Macro F1 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]"]
    print(f"  tuned threshold {best_t:.2f}: test macro F1 {rows[-1]['f1_macro']:.3f}, "
          f"delayed F1 {rows[-1]['f1_delayed']:.3f}", flush=True)

results = pd.DataFrame(rows)
results.to_csv(REPORTS_DIR / f"phase4_test_results_{YEAR}.csv", index=False)
tuned = results[results["threshold_type"] == "tuned"].set_index("model")
default = results[results["threshold_type"] == "default"].set_index("model")
order = tuned.sort_values("f1_macro", ascending=False).index

cols = ["threshold", "accuracy", "balanced_accuracy", "precision_delayed", "recall_delayed",
        "f1_delayed", "f1_macro", "roc_auc", "pr_auc"]
header = [
    f"Test set: {len(y_test):,} flights, {y_test.mean():.2%} delayed.",
    "\nDefault threshold (0.50):", default.loc[order, cols].to_string(float_format="%.3f"),
    "\nTuned threshold (max macro F1 on out-of-fold training predictions):",
    tuned.loc[order, cols + ["f1_macro_ci_low", "f1_macro_ci_high"]].to_string(float_format="%.3f"),
]


# 09. Confusion matrices at the tuned threshold (row-normalised, with counts)
fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
for ax, name in zip(axes, order):
    t = tuned.loc[name, "threshold"]
    cm = confusion_matrix(y_test, probas[name] >= t)
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=LinearSegmentedColormap.from_list(
        "seq", ["#f0efec", "#2a78d6", "#184f95"]), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center",
                    color="white" if share[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["On time", "Delayed"])
    ax.set_yticks([0, 1], ["On time", "Delayed"])
    ax.set_xlabel("Predicted")
    if ax is axes[0]:
        ax.set_ylabel("Actual")
    ax.grid(False)
    ax.set_title(f"{name}\nthreshold {t:.2f}", fontsize=11)
fig.suptitle("Confusion matrices on the test set (tuned thresholds)", x=0.01, ha="left",
             fontweight="bold", fontsize=13)
save(fig, "09_confusion_matrices.png")

# 10. ROC curves
fig, ax = plt.subplots(figsize=(6.5, 6))
for name in order:
    fpr, tpr, _ = roc_curve(y_test, probas[name])
    ax.plot(fpr, tpr, color=COLORS[name], lw=2,
            label=f"{name} (AUC {tuned.loc[name, 'roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--", label="Random guess (AUC 0.500)")
ax.set_xlabel("False positive rate (on-time flights flagged)")
ax.set_ylabel("True positive rate (delays caught)")
ax.set_title("ROC curves on the test set")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
save(fig, "10_roc_curves.png")

# 11. Precision-recall curves (more informative than ROC under imbalance)
fig, ax = plt.subplots(figsize=(6.5, 6))
for name in order:
    prec, rec, _ = precision_recall_curve(y_test, probas[name])
    ax.plot(rec, prec, color=COLORS[name], lw=2,
            label=f"{name} (AP {tuned.loc[name, 'pr_auc']:.3f})")
ax.axhline(y_test.mean(), color=INK_2, lw=1, ls="--",
           label=f"Random guess (AP {y_test.mean():.3f})")
ax.set_xlabel("Recall (share of delays caught)")
ax.set_ylabel("Precision (share of alerts that were delays)")
ax.set_title("Precision–recall curves on the test set")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
save(fig, "11_precision_recall_curves.png")

# 12. Threshold tuning curves (out-of-fold, training data)
fig, ax = plt.subplots(figsize=(8, 4.5))
for name in order:
    ax.plot(THRESHOLDS, curves[name], color=COLORS[name], lw=2, label=name)
    best = curves[name].argmax()
    ax.plot(THRESHOLDS[best], curves[name][best], "o", ms=8, color=COLORS[name],
            mec=SURFACE, mew=2)
ax.axvline(0.5, color=INK_2, lw=1, ls="--", label="Default threshold 0.5")
ax.set_xlabel("Decision threshold (predict delayed if probability ≥ threshold)")
ax.set_ylabel("Macro F1 (out-of-fold)")
ax.set_title("Threshold tuning: macro F1 peaks at a different threshold per model")
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
save(fig, "12_threshold_tuning.png")

# 13. Metric comparison at tuned thresholds
metric_names = {"f1_macro": "Macro F1", "f1_delayed": "F1 (delayed)",
                "precision_delayed": "Precision (delayed)", "recall_delayed": "Recall (delayed)",
                "roc_auc": "ROC-AUC"}
fig, ax = plt.subplots(figsize=(11, 4.8))
width = 0.16
x = np.arange(len(metric_names))
for i, name in enumerate(order):
    vals = tuned.loc[name, list(metric_names)].to_numpy(dtype=float)
    ax.bar(x + (i - 2) * width, vals, width=width - 0.02, color=COLORS[name], label=name)
ax.set_xticks(x, metric_names.values())
ax.set_ylim(0, 1)
ax.set_ylabel("Score on test set")
ax.set_title("Model comparison on the test set (tuned thresholds)")
ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.grid(axis="x", visible=False)
save(fig, "13_model_comparison.png")

# 14. Permutation importance of the best model on the test set
best_name = order[0]
best_model = joblib.load(MODELS_DIR / f"{MODEL_FILES[best_name]}.joblib")
sub = np.random.default_rng(SEED).choice(len(X_test), size=10_000, replace=False)
print(f"Permutation importance for {best_name} ...", flush=True)
imp = permutation_importance(best_model, X_test.iloc[sub], y_test[sub], scoring="roc_auc",
                             n_repeats=5, random_state=SEED, n_jobs=-1)
imp = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.barh(imp.index, imp.values, color=COLORS[best_name], height=0.6)
ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
ax.set_title(f"What drives the {best_name} predictions")
ax.grid(axis="y", visible=False)
save(fig, "14_feature_importance.png")

text = "\n".join(header + ["\nPermutation importance (ROC-AUC drop), " + best_name + ":",
                           imp.sort_values(ascending=False).to_string(float_format="%.4f")]
                 + summary)
print(text)
(REPORTS_DIR / f"phase4_summary_{YEAR}.txt").write_text(text)
