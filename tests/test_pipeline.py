"""Fast checks that guard the parts of the pipeline that are easy to break
silently: leakage, the shared split, and that every model trains and
outputs valid probabilities."""
import numpy as np
import pandas as pd
import pytest

from flight_delay.config import CATEGORICAL, FEATURES, LEAKAGE_COLUMNS, NUMERIC, TARGET
from flight_delay.data import training_rows
from flight_delay.models import MODEL_FILES, build_models


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
