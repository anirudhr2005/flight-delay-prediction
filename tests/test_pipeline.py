"""Fast checks that guard the parts of the pipeline that are easy to break
silently: leakage, the train/test splits, threshold objectives, and that
every model trains and outputs valid probabilities."""
import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import f1_score, recall_score

from flight_delay.config import CATEGORICAL, FEATURES, LEAKAGE_COLUMNS, NUMERIC, TARGET
from flight_delay.data import chronological_cutoff, split_frame, training_rows
from flight_delay.models import MODEL_FILES, build_models
from flight_delay.thresholds import (
    PRECISION_TARGET, RECALL_TARGET, choose_threshold, threshold_curves,
)


def make_flights(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Airline": rng.choice(["AA", "DL", "UA", "WN"], n),
        "Origin": rng.choice(["ATL", "ORD", "DEN", "LAX", "JFK"], n),
        "Dest": rng.choice(["SEA", "MCO", "EWR", "PHX"], n),
        **{col: rng.integers(1, 24, n) for col in NUMERIC},
    })
    # Late departures are delayed more often, so models have something to learn.
    p = 0.1 + 0.03 * df["dep_hour"].clip(0, 20)
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df


def test_no_post_departure_columns_in_features():
    assert not set(FEATURES) & set(LEAKAGE_COLUMNS)
    assert TARGET not in FEATURES


def test_feature_lists_are_disjoint_and_complete():
    assert not set(CATEGORICAL) & set(NUMERIC)
    assert FEATURES == CATEGORICAL + NUMERIC


def test_every_model_has_a_file_name():
    assert set(build_models()) == set(MODEL_FILES)


@pytest.mark.parametrize("name", list(MODEL_FILES))
def test_model_trains_and_outputs_probabilities(name):
    df = make_flights()
    model = build_models()[name]
    model.fit(df[FEATURES], df[TARGET])
    proba = model.predict_proba(df[FEATURES])[:, 1]
    assert proba.shape == (len(df),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_model_handles_unseen_airport():
    df = make_flights()
    model = build_models()["Logistic Regression"].fit(df[FEATURES], df[TARGET])
    new = df[FEATURES].head(3).assign(Origin="ZZZ")
    assert model.predict_proba(new).shape == (3, 2)


def test_svm_subset_is_stratified(monkeypatch):
    monkeypatch.setattr("flight_delay.data.SVM_TRAIN_ROWS", 200)
    df = make_flights(1000)
    X, y = training_rows("SVM (RBF)", df[FEATURES], df[TARGET])
    assert len(X) == 200
    assert abs(y.mean() - df[TARGET].mean()) < 0.01
    X_all, _ = training_rows("Random Forest", df[FEATURES], df[TARGET])
    assert len(X_all) == 1000


def test_month_is_not_a_feature():
    # Test months never appear in training under the chronological split.
    assert "Month" not in FEATURES and "Quarter" not in FEATURES


def test_chronological_split_tests_on_later_dates():
    df = make_flights(700)
    df["FlightDate"] = pd.date_range("2022-01-01", periods=len(df), freq="7h")
    X_train, X_test, y_train, y_test = split_frame(df, "chronological")
    train_dates = df.loc[X_train.index, "FlightDate"]
    test_dates = df.loc[X_test.index, "FlightDate"]
    assert train_dates.max() < test_dates.min()
    assert len(X_train) + len(X_test) == len(df)
    # The test set is the last two calendar months present.
    cutoff = chronological_cutoff(df["FlightDate"])
    assert train_dates.max() < cutoff <= test_dates.min()
    assert test_dates.dt.to_period("M").nunique() == 2


def test_random_split_is_stratified():
    df = make_flights(1000)
    df["FlightDate"] = pd.Timestamp("2022-01-01")
    _, X_test, y_train, y_test = split_frame(df, "random")
    assert len(X_test) == 200
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_threshold_objectives():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.3 * y + rng.random(2000) * 0.7, 0, 1)
    curves = threshold_curves(y, proba)
    assert choose_threshold(curves, "default_0.5") == 0.5
    t_recall = choose_threshold(curves, f"recall_{RECALL_TARGET:.0%}")
    assert recall_score(y, proba >= t_recall) >= RECALL_TARGET
    t_macro = choose_threshold(curves, "max_macro_f1")
    assert f1_score(y, proba >= t_macro, average="macro") == pytest.approx(curves["f1_macro"].max())
    # An unreachable target returns None instead of a misleading threshold:
    # with ~20% positives and an uninformative model, precision stays ~0.2.
    y_rare = (rng.random(2000) < 0.2).astype(int)
    assert choose_threshold(threshold_curves(y_rare, np.full(2000, 0.5)),
                            f"precision_{PRECISION_TARGET:.0%}") is None
