"""Loading the processed sample and making the train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_SPLIT, FEATURES, SEED, SVM_TRAIN_ROWS, TARGET, TEST_MONTHS, TEST_SIZE, sample_path,
)


def load_sample(year: str) -> pd.DataFrame:
    path = sample_path(year)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/02_prepare_data.py first")
    return pd.read_parquet(path)


def chronological_cutoff(dates: pd.Series, test_months: int = TEST_MONTHS) -> pd.Timestamp:
    """First day of the last `test_months` calendar months present in `dates`."""
    last_month = dates.max().to_period("M")
    return (last_month - (test_months - 1)).to_timestamp()


def split_frame(df: pd.DataFrame, split: str = DEFAULT_SPLIT):
    """Return X_train, X_test, y_train, y_test for the chosen split."""
    if split == "chronological":
        cutoff = chronological_cutoff(df["FlightDate"])
        train, test = df[df["FlightDate"] < cutoff], df[df["FlightDate"] >= cutoff]
        return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET]
    if split == "random":
        return train_test_split(df[FEATURES], df[TARGET], test_size=TEST_SIZE,
                                stratify=df[TARGET], random_state=SEED)
    raise ValueError(f"unknown split {split!r}")


def train_test(year: str, split: str = DEFAULT_SPLIT):
    """The train/test split used by every script."""
    return split_frame(load_sample(year), split)


def describe_split(year: str, split: str) -> str:
    df = load_sample(year)
    X_train, X_test, y_train, y_test = split_frame(df, split)
    text = (f"{split} split: train {len(X_train):,} flights ({y_train.mean():.2%} delayed), "
            f"test {len(X_test):,} flights ({y_test.mean():.2%} delayed)")
    if split == "chronological":
        cutoff = chronological_cutoff(df["FlightDate"])
        text += (f"\n  train dates {df['FlightDate'].min():%Y-%m-%d} to "
                 f"{(cutoff - pd.Timedelta(days=1)):%Y-%m-%d}, test dates "
                 f"{cutoff:%Y-%m-%d} to {df['FlightDate'].max():%Y-%m-%d}")
    return text


def training_rows(name: str, X_train, y_train):
    """The rows a model trains on: everything, or a stratified subset for SVM."""
    if not name.startswith("SVM") or len(X_train) <= SVM_TRAIN_ROWS:
        return X_train, y_train
    X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=SVM_TRAIN_ROWS,
                                          stratify=y_train, random_state=SEED)
    return X_sub, y_sub
