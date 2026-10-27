"""Loading the processed sample and making the train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, SEED, SVM_TRAIN_ROWS, TARGET, TEST_SIZE, sample_path


def load_sample(year: str) -> pd.DataFrame:
    path = sample_path(year)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/02_prepare_data.py first")
    return pd.read_parquet(path)


def train_test(year: str):
    """Stratified 80/20 split, identical in every script."""
    df = load_sample(year)
    return train_test_split(df[FEATURES], df[TARGET], test_size=TEST_SIZE,
                            stratify=df[TARGET], random_state=SEED)


def training_rows(name: str, X_train, y_train):
    """The rows a model trains on: everything, or a stratified subset for SVM."""
    if not name.startswith("SVM") or len(X_train) <= SVM_TRAIN_ROWS:
        return X_train, y_train
    X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=SVM_TRAIN_ROWS,
                                          stratify=y_train, random_state=SEED)
    return X_sub, y_sub
