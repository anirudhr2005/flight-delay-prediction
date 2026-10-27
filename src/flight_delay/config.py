"""Project-wide paths, constants and the feature list.

Every script imports from here so the feature set, random seed and split
can't drift apart between training and evaluation.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("FLIGHT_DATA_DIR", ROOT / "data" / "raw")).expanduser()
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

DEFAULT_YEAR = "2022"
SEED = 42
SAMPLE_SIZE = 200_000
TEST_SIZE = 0.2
CV_FOLDS = 5
# Kernel SVM scales roughly quadratically with rows, so it trains on a
# stratified subset of the training set.
SVM_TRAIN_ROWS = 25_000

TARGET = "ArrDel15"

CATEGORICAL = ["Airline", "Origin", "Dest"]
NUMERIC = [
    "CRSElapsedTime", "Distance", "Month", "DayofMonth", "DayOfWeek",
    "dep_hour", "arr_hour", "is_weekend", "overnight",
    "origin_daily_flights", "dest_daily_flights", "origin_hourly_flights",
    "aircraft_leg_of_day", "days_to_holiday",
]
FEATURES = CATEGORICAL + NUMERIC
# Dropped as near-duplicates of kept features (see the Phase 2 correlation
# heatmap): DistanceGroup (~Distance), Quarter (~Month), DepTimeBlk
# (~dep_hour), OriginState/DestState (implied by the airport).

# Recorded at or after departure. Using any of them would leak the answer.
LEAKAGE_COLUMNS = [
    "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15", "DepartureDelayGroups",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn", "ArrTime", "ArrDelay",
    "ArrDelayMinutes", "ArrivalDelayGroups", "AirTime", "ActualElapsedTime",
    "Cancelled", "Diverted", "DivAirportLandings",
]


def raw_path(year: str) -> Path:
    return RAW_DIR / f"Combined_Flights_{year}.parquet"


def sample_path(year: str) -> Path:
    return PROCESSED_DIR / f"flights_{year}_sample.parquet"


def oof_path(year: str) -> Path:
    return PROCESSED_DIR / f"oof_predictions_{year}.parquet"
