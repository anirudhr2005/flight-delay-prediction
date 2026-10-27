"""Phase 1: inspect the raw flight data before committing to it.

Reports size, columns, dtypes, missing values, duplicates and the
target-class distribution (ArrDel15 = arrival delayed by 15+ minutes).
"""
import pandas as pd

from flight_delay.cli import year_parser
from flight_delay.config import REPORTS_DIR, ROOT, TARGET, raw_path

args = year_parser(__doc__).parse_args()
path = raw_path(args.year)
df = pd.read_parquet(path)
lines = []
p = lines.append

p(f"File: {path.name} ({path.stat().st_size / 1e6:.1f} MB on disk)")
p(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
p(f"In-memory size: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
p(f"Date range: {df['FlightDate'].min():%Y-%m-%d} to {df['FlightDate'].max():%Y-%m-%d}\n")

p("Columns and dtypes:")
for col, dtype in df.dtypes.items():
    p(f"  {col:<45} {dtype}")

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
p("\nMissing values (columns with any):")
for col, n in missing.items():
    p(f"  {col:<45} {n:>10,}  ({n / len(df):.2%})")

p(f"\nDuplicate rows: {df.duplicated().sum():,}")

p(f"\nCancelled: {df['Cancelled'].sum():,}   Diverted: {df['Diverted'].sum():,}")
p(f"\nTarget {TARGET} distribution (all rows, incl. NaN for cancelled/diverted):")
for val, n in df[TARGET].value_counts(dropna=False).items():
    p(f"  {val!s:<6} {n:>10,}  ({n / len(df):.2%})")

flown = df[~df["Cancelled"] & ~df["Diverted"]]
rate = flown[TARGET].mean()
p(f"\nAmong completed flights: delayed rate = {rate:.2%}, "
  f"imbalance ratio on-time:delayed = {(1 - rate) / rate:.1f}:1")

p("\nTop airlines by flights:")
p(df["Airline"].value_counts().head(10).to_string())

report = "\n".join(lines)
print(report)
out = REPORTS_DIR / f"phase1_inspection_{args.year}.txt"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
out.write_text(report)
print(f"\nSaved to {out.relative_to(ROOT)}")
