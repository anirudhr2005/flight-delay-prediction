"""Shared chart style so every figure in reports/figures/ matches."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .config import FIG_DIR

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# Fixed categorical order: one colour per model across every chart.
MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",
    "Decision Tree": "#eb6834",
    "Random Forest": "#1baf7a",
    "SVM (RBF)": "#eda100",
    "Naive Bayes": "#e87ba4",
}
PCT = PercentFormatter(1.0, decimals=0)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=160)
    plt.close(fig)
