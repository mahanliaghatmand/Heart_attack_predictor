"""Model loading, input validation and feature engineering.

Importing this module does NOT import TensorFlow/Keras: the Keras model is only
materialised (via ``pickle``) inside :meth:`HeartRiskPredictor.load`. That keeps
unit tests and CI light-weight.

SECURITY NOTE: ``pickle`` can execute arbitrary code. Only load artifact files
you created yourself or fully trust.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import BOOL_COLUMNS, INPUT_COLUMNS, NUMERIC_COLUMNS

_TRUE = {"true", "1", "1.0", "yes", "y", "t"}
_FALSE = {"false", "0", "0.0", "no", "n", "f"}


class PredictionInputError(ValueError):
    """Raised when the supplied data cannot be turned into model features."""


def _to_bool(series: pd.Series, column: str) -> pd.Series:
    def convert(value) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise PredictionInputError(
            f"Column '{column}' must be True/False (or 1/0); found {value!r}."
        )

    return series.map(convert)


class HeartRiskPredictor:
    """Wraps the Keras model + fitted OneHotEncoder used in training."""

    def __init__(self, model, encoder, config: dict):
        self._model = model
        self._encoder = encoder
        self.config = dict(config)
        self._cat_columns = [str(c) for c in encoder.feature_names_in_]
        self.feature_names = NUMERIC_COLUMNS + [
            str(n) for n in encoder.get_feature_names_out()
        ]
        expected = self.config.get("INPUT_DIM")
        if expected is not None and expected != len(self.feature_names):
            raise RuntimeError(
                f"Feature mismatch: the model expects {expected} inputs but the "
                f"app builds {len(self.feature_names)}. Check heart_app/schema.py."
            )

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, artifacts_dir: str | Path) -> "HeartRiskPredictor":
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        artifacts_dir = Path(artifacts_dir)

        def read(name: str):
            path = artifacts_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing artifact: {path}")
            with open(path, "rb") as handle:
                return pickle.load(handle)  # noqa: S301 - trusted, local artifacts

        config = read("Config.pkl")
        encoder = read("OneHotEncoder.pkl")
        model = read("Model.pkl")
        return cls(model, encoder, config)

    # ------------------------------------------------------------- preprocessing
    def features(self, df: pd.DataFrame) -> np.ndarray:
        """Return the ``(n_rows, INPUT_DIM)`` float32 matrix expected by the model."""
        missing = [c for c in INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise PredictionInputError("Missing required column(s): " + ", ".join(missing))

        data = df[INPUT_COLUMNS].copy().reset_index(drop=True)
        if data.empty:
            raise PredictionInputError("The file contains no rows.")
        null_cols = [c for c in INPUT_COLUMNS if data[c].isna().any()]
        if null_cols:
            raise PredictionInputError("Empty values found in: " + ", ".join(null_cols))

        for col in BOOL_COLUMNS:
            data[col] = _to_bool(data[col], col)
        for col in NUMERIC_COLUMNS:
            try:
                data[col] = pd.to_numeric(data[col])
            except (ValueError, TypeError) as exc:
                raise PredictionInputError(f"Column '{col}' must be numeric.") from exc

        for col, allowed in zip(self._cat_columns, self._encoder.categories_):
            data[col] = data[col].astype(str).str.strip()
            unknown = sorted(set(data[col]) - {str(a) for a in allowed})
            if unknown:
                raise PredictionInputError(
                    f"Column '{col}' has unknown value(s) {unknown}; "
                    f"allowed: {[str(a) for a in allowed]}."
                )

        numeric = data[NUMERIC_COLUMNS].astype("float32").to_numpy()
        categorical = np.asarray(
            self._encoder.transform(data[self._cat_columns]), dtype="float32"
        )
        return np.hstack([numeric, categorical]).astype("float32")

    # ---------------------------------------------------------------- prediction
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probability of heart disease for every row (1-D float array)."""
        matrix = self.features(df)
        output = self._model.predict(matrix, verbose=0)
        return np.asarray(output, dtype="float64").reshape(-1)
