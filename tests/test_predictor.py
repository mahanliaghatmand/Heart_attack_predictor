"""Pre-processing tests. They use a stub model, so TensorFlow is NOT required."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heart_app.explain import what_if_contributions
from heart_app.predictor import HeartRiskPredictor, PredictionInputError
from heart_app.schema import (
    FEATURES,
    INPUT_COLUMNS,
    PRESETS,
    TARGET,
    risk_band,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data" / "Data.csv"

pytestmark = pytest.mark.skipif(
    not (DATA.exists() and (ARTIFACTS / "OneHotEncoder.pkl").exists()),
    reason="data/artifacts not available",
)


class StubModel:
    """Deterministic stand-in for the Keras model (depends on the first columns)."""

    def predict(self, matrix, verbose=0):
        score = (matrix[:, 0] - 50) / 20 + (matrix[:, 12] * 1.5) - 1
        return (1 / (1 + np.exp(-score))).reshape(-1, 1)


@pytest.fixture(scope="module")
def predictor():
    with open(ARTIFACTS / "OneHotEncoder.pkl", "rb") as fh:
        encoder = pickle.load(fh)
    with open(ARTIFACTS / "Config.pkl", "rb") as fh:
        config = pickle.load(fh)
    return HeartRiskPredictor(StubModel(), encoder, config)


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(DATA)


def test_schema_matches_csv_header(data):
    assert INPUT_COLUMNS + [TARGET] == list(data.columns)


def test_features_match_training_notebook_pipeline(predictor, data):
    """Re-implements the notebook's pipeline and compares it with the app's."""
    with open(ARTIFACTS / "OneHotEncoder.pkl", "rb") as fh:
        ohe = pickle.load(fh)
    cats = ["smoker_status", "chest_pain_type", "sex"]
    x = data.drop(columns=cats + [TARGET]).values.astype(np.float32)
    expected = np.hstack([x, ohe.transform(data[cats]).astype(np.float32)])

    actual = predictor.features(data)
    assert actual.shape == (len(data), predictor.config["INPUT_DIM"]) == (9000, 31)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected)


def test_defaults_inside_declared_ranges(data):
    for f in FEATURES:
        if f.kind in ("int", "float"):
            assert f.low <= f.default <= f.high, f.name
            assert data[f.name].min() >= f.low - 1e-9, f.name
            assert data[f.name].max() <= f.high + 1e-9, f.name


def test_presets_cover_every_feature():
    for name, values in PRESETS.items():
        assert set(values) == set(INPUT_COLUMNS), name


def test_predict_proba_single_and_batch(predictor):
    row = pd.DataFrame([PRESETS["typical"]])
    single = predictor.predict_proba(row)
    batch = predictor.predict_proba(pd.concat([row] * 5))
    assert single.shape == (1,) and batch.shape == (5,)
    assert 0 <= single[0] <= 1


def test_string_booleans_are_accepted(predictor):
    row = pd.DataFrame([PRESETS["typical"]]).astype({"family_history": object})
    row["family_history"] = "True"
    assert predictor.features(row).shape == (1, 31)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns=["age"]), "Missing required"),
        (lambda d: d.assign(sex="Other"), "unknown value"),
        (lambda d: d.assign(age=np.nan), "Empty values"),
        (lambda d: d.assign(age="abc"), "numeric"),
        (lambda d: d.assign(family_history="maybe"), "True/False"),
    ],
)
def test_invalid_input_raises(predictor, mutate, message):
    frame = mutate(pd.DataFrame([PRESETS["typical"]]))
    with pytest.raises(PredictionInputError, match=message):
        predictor.features(frame)


def test_what_if_contributions(predictor):
    result = what_if_contributions(predictor, PRESETS["at_risk"])
    assert set(result.columns) == {"feature", "delta"}
    assert len(result) == len(FEATURES)
    assert result["delta"].abs().is_monotonic_decreasing
    # A patient equal to the reference profile has (almost) no deviation.
    neutral = what_if_contributions(predictor, PRESETS["typical"])
    assert neutral["delta"].abs().max() < 1e-9


def test_risk_band():
    assert risk_band(0.1) == "low"
    assert risk_band(0.45) == "moderate"
    assert risk_band(0.9) == "high"
