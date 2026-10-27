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
# How the sample is split into train and test:
#   "chronological" (default): train on earlier dates, test on the last
#       TEST_MONTHS calendar months in the data - the realistic setting,
#       since the model is meant to predict future flights.
#   "random": stratified 80/20 split, kept for comparison.
SPLITS = ("chronological", "random")
DEFAULT_SPLIT = "chronological"
TEST_MONTHS = 2
TEST_SIZE = 0.2
CV_FOLDS = 5
# Kernel SVM scales roughly quadratically with rows, so it trains on a
# stratified subset of the training set.
SVM_TRAIN_ROWS = 25_000

TARGET = "ArrDel15"

CATEGORICAL = ["Airline", "Origin", "Dest"]
NUMERIC = [
    "CRSElapsedTime", "Distance", "DayofMonth", "DayOfWeek",
    "dep_hour", "arr_hour", "is_weekend", "overnight",
    "origin_daily_flights", "dest_daily_flights", "origin_hourly_flights",
    "aircraft_leg_of_day", "days_to_holiday",
]
FEATURES = CATEGORICAL + NUMERIC
# Not used as features:
# - Month / Quarter: under a chronological split the test months never
#   appear in training, so a month effect can't be learned for them.
# - DistanceGroup (~Distance), DepTimeBlk (~dep_hour), OriginState/DestState
#   (implied by the airport): near-duplicates of kept features (see the
#   Phase 2 correlation heatmap).

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


def run_tag(year: str, split: str) -> str:
    """Suffix that keeps outputs of different years/splits apart, e.g. 2022_chronological."""
    return f"{year}_{split}"


def oof_path(year: str, split: str) -> Path:
    return PROCESSED_DIR / f"oof_predictions_{run_tag(year, split)}.parquet"


def model_path(name_slug: str, year: str, split: str) -> Path:
    return MODELS_DIR / run_tag(year, split) / f"{name_slug}.joblib"
