"""Phase 2: clean the raw data, engineer pre-departure features, and draw a
stratified sample for modelling.

Only information known before departure is kept (the schedule, airline,
airports, calendar). Anything recorded after pushback (DepDelay, TaxiOut,
WheelsOff, AirTime, ArrTime, ...) would leak the answer and is never loaded.

Schedule-derived counts (airport traffic, aircraft leg of the day) are
computed on the full year before sampling, since the whole schedule is
published in advance.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from flight_delay.cli import year_parser
from flight_delay.config import (
    FEATURES, PROCESSED_DIR, REPORTS_DIR, ROOT, SAMPLE_SIZE, SEED, TARGET, raw_path, sample_path,
)

parser = year_parser(__doc__)
parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
args = parser.parse_args()

# Kept in the sample only so the EDA heatmap can show why they were dropped
# as model features (near-duplicates of Distance and Month).
EDA_ONLY = ["DistanceGroup", "Quarter"]

# Columns known before departure, plus what is needed to clean and build features.
COLUMNS = [
    "FlightDate", "Airline", "Origin", "Dest", "Cancelled", "Diverted",
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "DistanceGroup",
    "Quarter", "Month", "DayofMonth", "DayOfWeek", "Tail_Number", TARGET,
]


def clean(df: pd.DataFrame, log: list) -> pd.DataFrame:
    # Cancelled and diverted flights have no arrival outcome, so no label.
    df = df[~df["Cancelled"] & ~df["Diverted"]].drop(columns=["Cancelled", "Diverted"])
    log.append(f"After dropping cancelled/diverted: {len(df):,}")
    df = df.dropna(subset=[TARGET, "CRSElapsedTime"]).drop_duplicates()
    log.append(f"After dropping missing target/duplicates: {len(df):,}")
    return df


def add_features(df: pd.DataFrame, year: str) -> pd.DataFrame:
    df = df.copy()
    df["dep_hour"] = (df["CRSDepTime"] // 100).clip(0, 23)
    df["arr_hour"] = (df["CRSArrTime"] // 100).clip(0, 23)
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["overnight"] = (df["CRSArrTime"] < df["CRSDepTime"]).astype(int)

    # Congestion: how busy the airports are scheduled to be that day / hour.
    df["origin_daily_flights"] = df.groupby(["Origin", "FlightDate"])["Origin"].transform("size")
    df["dest_daily_flights"] = df.groupby(["Dest", "FlightDate"])["Dest"].transform("size")
    df["origin_hourly_flights"] = (
        df.groupby(["Origin", "FlightDate", "dep_hour"])["Origin"].transform("size"))

    # Which leg of the day this aircraft is flying; later legs inherit earlier
    # delays. Flights with no tail number are treated as a first leg.
    leg = df.groupby(["Tail_Number", "FlightDate"])["CRSDepTime"].rank(method="first")
    df["aircraft_leg_of_day"] = leg.fillna(1).astype(int)

    # Days to the nearest US federal holiday (travel peaks around them), capped at 30.
    holidays = USFederalHolidayCalendar().holidays(f"{int(year) - 1}-12-01", f"{int(year) + 1}-01-31")
    dates = pd.Series(df["FlightDate"].unique())
    gap = dates.map(lambda d: np.abs((holidays - d).days).min())
    df["days_to_holiday"] = df["FlightDate"].map(dict(zip(dates, gap))).clip(upper=30)

    df[TARGET] = df[TARGET].astype(int)
    return df[FEATURES + EDA_ONLY + [TARGET]]


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    frac = min(1.0, n / len(df))
    return (df.groupby(TARGET, group_keys=False)
              .sample(frac=frac, random_state=SEED)
              .sample(frac=1, random_state=SEED)  # shuffle
              .reset_index(drop=True))


log = []
raw = pd.read_parquet(raw_path(args.year), columns=COLUMNS)
log.append(f"Loaded {len(raw):,} rows from {raw_path(args.year).name}")
df = add_features(clean(raw, log), args.year)
sample = stratified_sample(df, args.sample_size)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
sample.to_parquet(sample_path(args.year), index=False)

log.append(f"Full clean delayed rate: {df[TARGET].mean():.2%}")
log.append(f"Sample: {len(sample):,} rows, delayed rate {sample[TARGET].mean():.2%}")
log.append(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}")
log.append(f"Missing values in sample: {int(sample.isna().sum().sum())}")
log.append(f"Saved to {sample_path(args.year).relative_to(ROOT)}")
report = "\n".join(log)
print(report)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / f"phase2_cleaning_{args.year}.txt").write_text(report)
