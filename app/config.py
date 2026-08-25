"""
Central configuration, driven by environment variables (set as HF Space
secrets/variables) so the app runs the same locally and on Spaces.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Model artifacts -------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "artifacts" / "model.pkl"))
ENCODER_PATH = os.getenv("ENCODER_PATH", str(BASE_DIR / "artifacts" / "OneHotEncoder.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# --- Prediction log (used by the Evidently report) -------------------------
LOG_CSV_PATH = os.getenv("LOG_CSV_PATH", str(BASE_DIR / "data" / "predictions_log.csv"))
# Minimum number of logged rows before a drift report can be built
# (needs enough rows to split into a reference half and a current half).
MIN_ROWS_FOR_REPORT = int(os.getenv("MIN_ROWS_FOR_REPORT", "20"))

# --- ClearML (replaces MLflow) ---------------------------------------------
# Free community server at app.clear.ml. Set CLEARML_API_ACCESS_KEY /
# CLEARML_API_SECRET_KEY as HF Space secrets to enable. Left off by default
# so the app still boots cleanly before you've configured credentials.
ENABLE_CLEARML = os.getenv("ENABLE_CLEARML", "false").lower() == "true"
CLEARML_PROJECT_NAME = os.getenv("CLEARML_PROJECT_NAME", "heart-disease-risk")
CLEARML_TASK_NAME = os.getenv("CLEARML_TASK_NAME", "serving")

# --- Risk thresholds ---------------------------------------------------
# Probability cut points used only to turn a raw probability into a
# human-readable label. Adjust freely; they do not affect the underlying
# 0/1 prediction (which uses 0.5).
RISK_LOW_MAX = float(os.getenv("RISK_LOW_MAX", "0.3"))
RISK_MEDIUM_MAX = float(os.getenv("RISK_MEDIUM_MAX", "0.6"))
