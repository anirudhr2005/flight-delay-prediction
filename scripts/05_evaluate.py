"""Phase 4: evaluate the five trained models on the held-out test set.

Thresholds are chosen without touching the test set: for each model and
each objective in src/flight_delay/thresholds.py (balanced macro F1,
delayed-class F1, a recall target, a precision target), the threshold is
picked on the out-of-fold probabilities saved by Phase 3. The test set is
then scored once per model and objective.

For reference only, the threshold that would have maximised macro F1 on the
test set itself ("oracle") is also reported: the gap to the tuned threshold
shows how much the delay rate shifted between the training and test periods.

Outputs (TAG = <YEAR>_<SPLIT>):
  reports/phase4_test_results_<TAG>.csv    metrics per model and objective
  reports/phase4_summary_<TAG>.txt         tables, classification reports, CIs
  reports/figures/09-15_*.png              confusion matrices, ROC, PR, trade-offs
                                           (suffixed _random for the random split)
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
from flight_delay.config import DEFAULT_SPLIT, REPORTS_DIR, SEED, model_path, oof_path, run_tag
from flight_delay.data import describe_split, train_test
from flight_delay.models import MODEL_FILES
from flight_delay.plotting import INK, INK_2, MODEL_COLORS as COLORS, SURFACE, apply_style, save
from flight_delay.thresholds import (
    OBJECTIVES, PRIMARY, THRESHOLDS, choose_threshold, threshold_curves,
)

args = year_parser(__doc__, split=True).parse_args()
YEAR, SPLIT, TAG = args.year, args.split, run_tag(args.year, args.split)
_, X_test, _, y_test = train_test(YEAR, SPLIT)
y_test = y_test.to_numpy()
if not oof_path(YEAR, SPLIT).exists():
    raise FileNotFoundError(f"{oof_path(YEAR, SPLIT)} not found - run scripts/04_train_models.py "
                            f"--split {SPLIT} first")
oof_all = pd.read_parquet(oof_path(YEAR, SPLIT))
apply_style()
SPLIT_NOTE = f"{SPLIT} split"


def fig_name(base):
    """Default-split figures keep plain names; others get a suffix."""
    return base if SPLIT == DEFAULT_SPLIT else base.replace(".png", f"_{SPLIT}.png")


def metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_delayed": precision_score(y_true, pred, zero_division=0),
        "recall_delayed": recall_score(y_true, pred),
        "f1_delayed": f1_score(y_true, pred),
        "f1_ontime": f1_score(y_true, pred, pos_label=0),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "alert_rate": pred.mean(),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def bootstrap_ci(y_true, pred, n=500):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    scores = [f1_score(y_true[i], pred[i], average="macro") for i in idx]
    return np.percentile(scores, [2.5, 97.5])


rows, probas, oof_curves, thresholds, summary = [], {}, {}, {}, []
for name, slug in MODEL_FILES.items():
    model = joblib.load(model_path(slug, YEAR, SPLIT))
    oof = oof_all[oof_all["model"] == name]
    if oof.empty:
        raise ValueError(f"No out-of-fold predictions for {name} - rerun scripts/04_train_models.py")
    curves = threshold_curves(oof["y_true"].to_numpy(), oof["proba"].to_numpy())
    oof_curves[name] = curves
    probas[name] = proba = model.predict_proba(X_test)[:, 1]

    thresholds[name] = {}
    for objective in OBJECTIVES:
        t = choose_threshold(curves, objective)
        thresholds[name][objective] = t
        if t is None:
            rows.append({"model": name, "objective": objective, "threshold": np.nan})
            continue
        rows.append({"model": name, "objective": objective, "threshold": t,
                     **metrics(y_test, proba, t)})
        if objective == PRIMARY:
            pred = (proba >= t).astype(int)
            lo, hi = bootstrap_ci(y_test, pred)
            rows[-1]["f1_macro_ci_low"], rows[-1]["f1_macro_ci_high"] = lo, hi
            summary += [f"\n=== {name} ({PRIMARY}, threshold {t:.2f}) ===",
                        classification_report(y_test, pred, target_names=["On time", "Delayed"],
                                              digits=3),
                        f"Macro F1 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]"]

    test_curve = threshold_curves(y_test, proba)["f1_macro"]
    oracle_t = float(THRESHOLDS[test_curve.argmax()])
    rows.append({"model": name, "objective": "oracle_test_macro_f1", "threshold": oracle_t,
                 **metrics(y_test, proba, oracle_t)})
    primary = next(r for r in rows if r["model"] == name and r["objective"] == PRIMARY)
    print(f"{name}: {PRIMARY} threshold {primary['threshold']:.2f} -> test macro F1 "
          f"{primary['f1_macro']:.3f} (oracle {oracle_t:.2f} -> {rows[-1]['f1_macro']:.3f})",
          flush=True)

results = pd.DataFrame(rows)
results.to_csv(REPORTS_DIR / f"phase4_test_results_{TAG}.csv", index=False)
by_obj = {obj: results[results["objective"] == obj].set_index("model")
          for obj in results["objective"].unique()}
tuned = by_obj[PRIMARY]
order = tuned.sort_values("f1_macro", ascending=False).index

cols = ["threshold", "accuracy", "balanced_accuracy", "precision_delayed", "recall_delayed",
        "f1_delayed", "f1_macro", "roc_auc", "pr_auc"]
trade_cols = ["threshold", "precision_delayed", "recall_delayed", "f1_delayed", "f1_macro",
              "alert_rate"]
header = [describe_split(YEAR, SPLIT),
          f"\nPrimary objective ({OBJECTIVES[PRIMARY]}):",
          tuned.loc[order, cols + ["f1_macro_ci_low", "f1_macro_ci_high"]]
          .to_string(float_format="%.3f")]
for obj, desc in OBJECTIVES.items():
    if obj != PRIMARY:
        header += [f"\n{obj}: {desc}", by_obj[obj].loc[order, trade_cols].to_string(
            float_format="%.3f", na_rep="unreachable")]
header += ["\nReference only - threshold maximising macro F1 on the test set itself (oracle):",
           by_obj["oracle_test_macro_f1"].loc[order, trade_cols].to_string(float_format="%.3f")]

# 09. Confusion matrices at the primary threshold (row-normalised, with counts)
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
fig.suptitle(f"Confusion matrices on the test set (max macro F1 thresholds, {SPLIT_NOTE})",
             x=0.01, ha="left", fontweight="bold", fontsize=13)
save(fig, fig_name("09_confusion_matrices.png"))

# 10. ROC curves
fig, ax = plt.subplots(figsize=(6.5, 6))
for name in order:
    fpr, tpr, _ = roc_curve(y_test, probas[name])
    ax.plot(fpr, tpr, color=COLORS[name], lw=2,
            label=f"{name} (AUC {tuned.loc[name, 'roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--", label="Random guess (AUC 0.500)")
ax.set_xlabel("False positive rate (on-time flights flagged)")
ax.set_ylabel("True positive rate (delays caught)")
ax.set_title(f"ROC curves on the test set ({SPLIT_NOTE})")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
save(fig, fig_name("10_roc_curves.png"))

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
ax.set_title(f"Precision–recall curves on the test set ({SPLIT_NOTE})")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
save(fig, fig_name("11_precision_recall_curves.png"))

# 12. Threshold tuning curves (out-of-fold, training data)
fig, ax = plt.subplots(figsize=(8, 4.5))
for name in order:
    curve = oof_curves[name]["f1_macro"]
    ax.plot(THRESHOLDS, curve, color=COLORS[name], lw=2, label=name)
    best = curve.argmax()
    ax.plot(THRESHOLDS[best], curve[best], "o", ms=8, color=COLORS[name], mec=SURFACE, mew=2)
ax.axvline(0.5, color=INK_2, lw=1, ls="--", label="Default threshold 0.5")
ax.set_xlabel("Decision threshold (predict delayed if probability ≥ threshold)")
ax.set_ylabel("Macro F1 (out-of-fold)")
ax.set_title("Threshold tuning: macro F1 peaks at a different threshold per model")
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
save(fig, fig_name("12_threshold_tuning.png"))

# 13. Metric comparison at the primary threshold
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
ax.set_title(f"Model comparison on the test set (max macro F1 thresholds, {SPLIT_NOTE})")
ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.grid(axis="x", visible=False)
save(fig, fig_name("13_model_comparison.png"))

# 14. Permutation importance of the best model on the test set
best_name = order[0]
best_model = joblib.load(model_path(MODEL_FILES[best_name], YEAR, SPLIT))
sub = np.random.default_rng(SEED).choice(len(X_test), size=min(10_000, len(X_test)), replace=False)
print(f"Permutation importance for {best_name} ...", flush=True)
imp = permutation_importance(best_model, X_test.iloc[sub], y_test[sub], scoring="roc_auc",
                             n_repeats=5, random_state=SEED, n_jobs=-1)
imp = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.barh(imp.index, imp.values, color=COLORS[best_name], height=0.6)
ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
ax.set_title(f"What drives the {best_name} predictions ({SPLIT_NOTE})")
ax.grid(axis="y", visible=False)
save(fig, fig_name("14_feature_importance.png"))

# 15. Precision-recall trade-off for the best model, with each objective's
# operating point (threshold chosen on training data, scored on test).
OBJ_STYLE = {"default_0.5": ("s", INK_2), "max_macro_f1": ("o", "#1baf7a"),
             "max_delayed_f1": ("D", "#2a78d6")}
extra = [o for o in OBJECTIVES if o not in OBJ_STYLE]
OBJ_STYLE.update(zip(extra, [("^", "#eb6834"), ("v", "#6250d6")]))
test_curves = threshold_curves(y_test, probas[best_name])
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ax = axes[0]
for key, label, color, ls in [("precision", "Precision (delayed)", "#6250d6", "-"),
                              ("recall", "Recall (delayed)", "#eb6834", "-"),
                              ("f1_delayed", "F1 (delayed)", "#2a78d6", "-"),
                              ("f1_macro", "Macro F1", "#1baf7a", "--")]:
    ax.plot(THRESHOLDS, test_curves[key], color=color, lw=2, ls=ls, label=label)
for obj, (marker, color) in OBJ_STYLE.items():
    t = thresholds[best_name][obj]
    if t is not None:
        ax.axvline(t, color=color, lw=1, ls=":")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score on test set")
ax.set_ylim(0, 1)
ax.set_title(f"{best_name}: raising the threshold trades recall for precision")
ax.legend(frameon=False, fontsize=9, loc="upper right")

ax = axes[1]
prec, rec, _ = precision_recall_curve(y_test, probas[best_name])
ax.plot(rec, prec, color=COLORS[best_name], lw=2, label=f"{best_name} PR curve")
for obj, (marker, color) in OBJ_STYLE.items():
    if thresholds[best_name][obj] is None:
        continue
    r = by_obj[obj].loc[best_name]
    ax.plot(r["recall_delayed"], r["precision_delayed"], marker, ms=10, color=color,
            mec=SURFACE, mew=1.5,
            label=f"{obj} (t={r['threshold']:.2f}): P {r['precision_delayed']:.2f}, "
                  f"R {r['recall_delayed']:.2f}")
ax.axhline(y_test.mean(), color=INK_2, lw=1, ls="--", label=f"Random guess ({y_test.mean():.2f})")
ax.set_xlabel("Recall (share of delays caught)")
ax.set_ylabel("Precision (share of alerts that were delays)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Operating points for each objective (test set)")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
save(fig, fig_name("15_threshold_tradeoff.png"))

text = "\n".join(header + ["\nPermutation importance (ROC-AUC drop), " + best_name + ":",
                           imp.sort_values(ascending=False).to_string(float_format="%.4f")]
                 + summary)
print(text)
(REPORTS_DIR / f"phase4_summary_{TAG}.txt").write_text(text)
