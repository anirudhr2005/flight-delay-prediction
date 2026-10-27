"""Choosing a decision threshold for a clearly stated objective.

A model outputs a delay probability; the threshold turns it into a yes/no
alert. Which threshold is "best" depends on what the alert is for, so each
objective below is a different, explicit trade-off between catching delays
(recall) and not crying wolf (precision). Thresholds are always chosen on
out-of-fold training predictions, never on the test set.
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

THRESHOLDS = np.round(np.arange(0.01, 0.991, 0.01), 2)
RECALL_TARGET = 0.70
PRECISION_TARGET = 0.50

OBJECTIVES = {
    "default_0.5": "Fixed 0.5 cut-off (no tuning)",
    "max_macro_f1": "Balanced: maximise macro F1 (primary metric)",
    "max_delayed_f1": "Maximise F1 of the delayed class",
    f"recall_{RECALL_TARGET:.0%}": (f"Catch delays: highest precision with recall >= {RECALL_TARGET:.0%} "
                                    "(e.g. alert passengers / plan spare crew)"),
    f"precision_{PRECISION_TARGET:.0%}": (f"Few false alarms: highest recall with precision >= "
                                          f"{PRECISION_TARGET:.0%} (e.g. costly interventions)"),
}
PRIMARY = "max_macro_f1"


def threshold_curves(y_true, proba):
    """Precision, recall, delayed F1 and macro F1 at every threshold in THRESHOLDS."""
    y_true = np.asarray(y_true)
    curves = {"precision": [], "recall": [], "f1_delayed": [], "f1_macro": []}
    for t in THRESHOLDS:
        pred = proba >= t
        curves["precision"].append(precision_score(y_true, pred, zero_division=0))
        curves["recall"].append(recall_score(y_true, pred, zero_division=0))
        curves["f1_delayed"].append(f1_score(y_true, pred, zero_division=0))
        curves["f1_macro"].append(f1_score(y_true, pred, average="macro", zero_division=0))
    return {k: np.array(v) for k, v in curves.items()}


def choose_threshold(curves: dict, objective: str):
    """Threshold for `objective`, or None if its target can't be reached."""
    if objective == "default_0.5":
        return 0.5
    if objective == "max_macro_f1":
        return float(THRESHOLDS[curves["f1_macro"].argmax()])
    if objective == "max_delayed_f1":
        return float(THRESHOLDS[curves["f1_delayed"].argmax()])
    if objective.startswith("recall_"):
        ok = curves["recall"] >= RECALL_TARGET
        if not ok.any():
            return None
        best = np.where(ok, curves["precision"], -1).argmax()
        return float(THRESHOLDS[best])
    if objective.startswith("precision_"):
        ok = curves["precision"] >= PRECISION_TARGET
        if not ok.any():
            return None
        best = np.where(ok, curves["recall"], -1).argmax()
        return float(THRESHOLDS[best])
    raise ValueError(f"unknown objective {objective!r}")
