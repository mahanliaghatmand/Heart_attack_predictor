"""
Loads the trained Keras model + fitted OneHotEncoder once at import time and
exposes a single `predict()` function that takes a plain dict of raw field
values (as produced by the Gradio form) and returns a result dict.

IMPORTANT: the feature order below must exactly match the order used during
training:
    1. The 22 numeric/boolean columns, in their original dataframe order.
    2. The one-hot encoded columns, built from
       df[["smoker_status", "chest_pain_type", "sex"]] via a scikit-learn
       OneHotEncoder(sparse_output=False), in that exact column order.
Changing this order will silently corrupt predictions because the model
only sees a flat float32 array, not named columns.
"""
from __future__ import annotations

import pickle
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from app import config

# Exact order the notebook produced after `df.drop([...])` on the raw CSV.
NUMERIC_FEATURE_ORDER = [
    "age",
    "resting_bp_systolic",
    "resting_bp_diastolic",
    "cholesterol_total",
    "hdl",
    "ldl",
    "triglycerides",
    "fasting_blood_sugar",
    "hba1c",
    "bmi",
    "resting_heart_rate",
    "max_heart_rate_achieved",
    "exercise_induced_angina",
    "st_depression",
    "family_history",
    "alcohol_units_per_week",
    "exercise_minutes_per_week",
    "sleep_hours",
    "stress_score",
    "wearable_owner",
    "daily_steps",
    "diet_quality_score",
]

# Column order fed into the OneHotEncoder during training.
CATEGORICAL_FEATURE_ORDER = ["smoker_status", "chest_pain_type", "sex"]

SEX_OPTIONS = ["Male", "Female"]
CHEST_PAIN_OPTIONS = [
    "Asymptomatic",
    "Non-Anginal Pain",
    "Atypical Angina",
    "Typical Angina",
]
SMOKER_OPTIONS = ["Never", "Current", "Former"]


class ModelService:
    """Thread-safe singleton wrapper around the model + encoder."""

    _lock = threading.Lock()

    def __init__(self) -> None:
        self.model = None
        self.encoder = None
        self._load()

    def _load(self) -> None:
        with self._lock:
            with open(config.ENCODER_PATH, "rb") as f:
                self.encoder = pickle.load(f)
            with open(config.MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.encoder is not None

    def _build_feature_vector(self, data: dict) -> np.ndarray:
        numeric_values = [float(data[name]) for name in NUMERIC_FEATURE_ORDER]
        numeric_array = np.array(numeric_values, dtype=np.float32).reshape(1, -1)

        categorical_row = [[data[name] for name in CATEGORICAL_FEATURE_ORDER]]
        categorical_array = self.encoder.transform(categorical_row).astype(np.float32)

        return np.hstack([numeric_array, categorical_array])

    def predict(self, data: dict) -> dict:
        if not self.is_ready:
            raise RuntimeError("Model/encoder not loaded")

        start = time.perf_counter()
        features = self._build_feature_vector(data)
        raw_output = self.model.predict(features, verbose=0)
        probability = float(np.asarray(raw_output).reshape(-1)[0])
        prediction = int(round(probability))
        latency_ms = (time.perf_counter() - start) * 1000

        if probability < config.RISK_LOW_MAX:
            risk_label = "low"
        elif probability < config.RISK_MEDIUM_MAX:
            risk_label = "medium"
        else:
            risk_label = "high"

        return {
            "prediction": prediction,
            "risk_probability": round(probability, 6),
            "risk_label": risk_label,
            "model_version": config.MODEL_VERSION,
            "latency_ms": round(latency_ms, 3),
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Singleton instance, imported by app.py
model_service = ModelService()
