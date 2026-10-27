"""Compare test results of the chronological and random splits.

Run after scripts/05_evaluate.py has been run for both splits. Shows how
much optimism the random split hides when the task is predicting future
flights.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flight_delay.cli import year_parser
from flight_delay.config import REPORTS_DIR, SPLITS, run_tag
from flight_delay.plotting import INK_2, apply_style, save
from flight_delay.thresholds import PRIMARY

YEAR = year_parser(__doc__).parse_args().year
frames = []
for split in SPLITS:
    path = REPORTS_DIR / f"phase4_test_results_{run_tag(YEAR, split)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/05_evaluate.py --split {split}")
    frames.append(pd.read_csv(path).assign(split=split))
res = pd.concat(frames)
res = res[res["objective"] == PRIMARY]

metrics = ["f1_macro", "f1_delayed", "precision_delayed", "recall_delayed", "roc_auc", "pr_auc"]
table = res.pivot(index="model", columns="split", values=metrics)
for m in metrics:
    table[(m, "change")] = table[(m, "chronological")] - table[(m, "random")]
table = table[[(m, s) for m in metrics for s in [*SPLITS, "change"]]]
table = table.sort_values(("f1_macro", "chronological"), ascending=False)

text = (f"Test results at the {PRIMARY} threshold, chronological vs random split ({YEAR})\n\n"
        + table.to_string(float_format="%.3f"))
print(text)
(REPORTS_DIR / f"split_comparison_{YEAR}.txt").write_text(text)

apply_style()
order = table.index
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, m, label in [(axes[0], "f1_macro", "Macro F1"), (axes[1], "roc_auc", "ROC-AUC")]:
    x = np.arange(len(order))
    ax.bar(x - 0.18, table.loc[order, (m, "random")], width=0.34, color="#86b6ef",
           label="Random split")
    ax.bar(x + 0.18, table.loc[order, (m, "chronological")], width=0.34, color="#1c5cab",
           label="Chronological split")
    ax.set_xticks(x, [n.replace(" (RBF)", "").replace(" ", "\n") for n in order])
    lo = min(table[(m, s)].min() for s in SPLITS)
    ax.set_ylim(max(0, lo - 0.1), None)
    ax.set_ylabel(f"{label} on test set")
    ax.set_title(f"{label}: random vs chronological split")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
save(fig, "16_split_comparison.png")
