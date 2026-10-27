"""Phase 2: exploratory charts of delay patterns in the stratified sample.

Saves PNG figures to reports/figures/ for the report and presentation.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from flight_delay.cli import year_parser
from flight_delay.config import REPORTS_DIR, TARGET
from flight_delay.data import load_sample
from flight_delay.plotting import BLUE, INK, INK_2, ORANGE, PCT, apply_style, save

YEAR = year_parser(__doc__).parse_args().year
df = load_sample(YEAR)
OVERALL = df[TARGET].mean()
apply_style()


def overall_line(ax, horizontal=False):
    (ax.axvline if horizontal else ax.axhline)(
        OVERALL, color=INK_2, lw=1, ls="--", label=f"Overall delay rate ({OVERALL:.1%})")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2,
              loc="lower right" if horizontal else "upper left")


# 1. Class balance
counts = df[TARGET].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar(["On time (<15 min)", "Delayed (15+ min)"], counts.values, color=[BLUE, ORANGE], width=0.55)
for i, n in enumerate(counts.values):
    ax.text(i, n, f"{n:,}\n({n / len(df):.1%})", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, counts.max() * 1.2)
ax.set_title("Class balance: about 1 in 5 flights arrives late")
ax.set_ylabel("Flights in sample")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
ax.grid(axis="x", visible=False)
save(fig, "01_class_balance.png")

# 2. Delay rate by scheduled departure hour
by_hour = df.groupby("dep_hour")[TARGET].agg(["mean", "size"])
by_hour = by_hour[by_hour["size"] >= 200]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(by_hour.index, by_hour["mean"], color=BLUE, width=0.7)
overall_line(ax)
ax.set_title("Delay risk builds through the day")
ax.set_xlabel("Scheduled departure hour")
ax.set_ylabel("Share of flights delayed")
ax.yaxis.set_major_formatter(PCT)
ax.set_xticks(range(0, 24, 2))
save(fig, "02_delay_by_hour.png")

# 3. Delay rate by month
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
by_month = df.groupby("Month")[TARGET].mean()
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar([months[m - 1] for m in by_month.index], by_month.values, color=BLUE, width=0.6)
overall_line(ax)
ax.set_title(f"Delay rate by month ({YEAR} data covers Jan–{months[by_month.index.max() - 1]})")
ax.set_ylabel("Share of flights delayed")
ax.yaxis.set_major_formatter(PCT)
ax.grid(axis="x", visible=False)
save(fig, "03_delay_by_month.png")

# 4. Delay rate by day of week
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_dow = df.groupby("DayOfWeek")[TARGET].mean()
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar([days[d - 1] for d in by_dow.index], by_dow.values, color=BLUE, width=0.6)
overall_line(ax)
ax.set_title("Delay rate by day of week")
ax.set_ylabel("Share of flights delayed")
ax.yaxis.set_major_formatter(PCT)
ax.grid(axis="x", visible=False)
save(fig, "04_delay_by_weekday.png")

# 5. Delay rate by airline
by_airline = df.groupby("Airline")[TARGET].agg(["mean", "size"])
by_airline = by_airline[by_airline["size"] >= 500].sort_values("mean")
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(by_airline.index, by_airline["mean"], color=BLUE, height=0.6)
overall_line(ax, horizontal=True)
ax.set_title("Delay rate varies widely by airline")
ax.set_xlabel("Share of flights delayed")
ax.xaxis.set_major_formatter(PCT)
ax.grid(axis="y", visible=False)
save(fig, "05_delay_by_airline.png")

# 6. Delay rate by the aircraft's leg of the day
leg = df["aircraft_leg_of_day"].clip(upper=7)
by_leg = df.groupby(leg)[TARGET].mean()
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar([str(i) if i < 7 else "7+" for i in by_leg.index], by_leg.values, color=BLUE, width=0.6)
overall_line(ax)
ax.set_title("Later legs of an aircraft's day are delayed more")
ax.set_xlabel("Aircraft's flight number of the day")
ax.set_ylabel("Share of flights delayed")
ax.yaxis.set_major_formatter(PCT)
ax.grid(axis="x", visible=False)
save(fig, "06_delay_by_aircraft_leg.png")

# 7. Busiest origin airports
top = df["Origin"].value_counts().head(15).index
by_origin = df[df["Origin"].isin(top)].groupby("Origin")[TARGET].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(by_origin.index, by_origin.values, color=BLUE, height=0.6)
overall_line(ax, horizontal=True)
ax.set_title("Delay rate at the 15 busiest origin airports")
ax.set_xlabel("Share of flights delayed")
ax.xaxis.set_major_formatter(PCT)
ax.grid(axis="y", visible=False)
save(fig, "07_delay_by_origin.png")

# 8. Correlation of numeric features with each other and the target
numeric = df.select_dtypes("number")
corr = numeric.corr()
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", ORANGE])
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iat[i, j]
        if abs(v) >= 0.3 or corr.columns[j] == TARGET:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson correlation")
ax.set_title("Correlation between numeric features")
save(fig, "08_correlation_heatmap.png")

# Text summary of the strongest single-feature signals
target_corr = corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
summary = ["Correlation with ArrDel15 (numeric features):"]
summary += [f"  {c:<25} {v:+.3f}" for c, v in target_corr.items()]
summary += ["", "Delay rate by departure hour:", by_hour["mean"].map("{:.1%}".format).to_string()]
summary += ["", "Delay rate by month:", by_month.map("{:.1%}".format).to_string()]
summary += ["", "Delay rate by airline:", by_airline["mean"].map("{:.1%}".format).to_string()]
summary += ["", "Delay rate by aircraft leg:", by_leg.map("{:.1%}".format).to_string()]
text = "\n".join(summary)
print(text)
(REPORTS_DIR / f"phase2_eda_{YEAR}.txt").write_text(text)
print("\nSaved figures 01-08 to reports/figures/")
