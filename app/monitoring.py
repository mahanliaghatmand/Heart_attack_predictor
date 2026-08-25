"""
Free monitoring stack:
  - Every prediction's input features + output are appended to a CSV file
    (no database needed).
  - `build_drift_report()` uses Evidently to compare the first half of the
    logged rows (as a stand-in "reference" window) against the second half
    (the "current" window) and returns an HTML report string.
  - `log_to_clearml()` is a thin, optional wrapper: if ENABLE_CLEARML is
    false or the clearml package/credentials aren't configured, it's a
    silent no-op so the app still runs fine without it.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from app import config
from app.inference import CATEGORICAL_FEATURE_ORDER, NUMERIC_FEATURE_ORDER

LOG_FIELDS = NUMERIC_FEATURE_ORDER + CATEGORICAL_FEATURE_ORDER + [
    "risk_probability",
    "prediction",
    "risk_label",
    "latency_ms",
    "timestamp",
]


def log_prediction(input_data: dict, result: dict) -> None:
    """Append one row (inputs + outputs) to the CSV prediction log."""
    path = Path(config.LOG_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    row = {name: input_data[name] for name in NUMERIC_FEATURE_ORDER + CATEGORICAL_FEATURE_ORDER}
    row.update(
        {
            "risk_probability": result["risk_probability"],
            "prediction": result["prediction"],
            "risk_label": result["risk_label"],
            "latency_ms": result["latency_ms"],
            "timestamp": result["timestamp"],
        }
    )

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _load_log_df() -> Optional[pd.DataFrame]:
    path = Path(config.LOG_CSV_PATH)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def build_drift_report() -> str:
    """
    Build an Evidently data-drift HTML report from the logged predictions.
    Returns an HTML string (embeddable in gr.HTML), or a plain-text message
    if there isn't enough data yet.
    """
    df = _load_log_df()
    if df is None or len(df) < config.MIN_ROWS_FOR_REPORT:
        have = 0 if df is None else len(df)
        return (
            f"<p>هنوز داده کافی برای گزارش drift وجود ندارد "
            f"({have}/{config.MIN_ROWS_FOR_REPORT} رکورد). "
            f"چند پیش‌بینی دیگر انجام بده و دوباره امتحان کن.</p>"
        )

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        return "<p>پکیج evidently نصب نیست. آن را به requirements.txt اضافه و نصب کن.</p>"

    feature_cols = NUMERIC_FEATURE_ORDER  # numeric columns are enough for a useful drift view
    half = len(df) // 2
    reference = df.iloc[:half][feature_cols]
    current = df.iloc[half:][feature_cols]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    out_path = Path(config.BASE_DIR) / "data" / "drift_report.html"
    try:
        report.save_html(str(out_path))
        return out_path.read_text(encoding="utf-8")
    except Exception:
        # Fall back to whatever in-memory HTML Evidently can give us.
        try:
            return report.get_html()
        except Exception as exc:  # pragma: no cover
            return f"<p>خطا در ساخت گزارش Evidently: {exc}</p>"


_clearml_task = None


def _get_clearml_task():
    global _clearml_task
    if not config.ENABLE_CLEARML:
        return None
    if _clearml_task is not None:
        return _clearml_task
    try:
        from clearml import Task

        _clearml_task = Task.init(
            project_name=config.CLEARML_PROJECT_NAME,
            task_name=config.CLEARML_TASK_NAME,
            reuse_last_task_id=True,
        )
    except Exception:
        # Missing credentials, no network, package not installed, etc. —
        # never let monitoring break the actual prediction flow.
        _clearml_task = None
    return _clearml_task


def log_to_clearml(result: dict) -> None:
    """Best-effort: log one prediction's metrics to ClearML. No-op on any failure."""
    task = _get_clearml_task()
    if task is None:
        return
    try:
        logger = task.get_logger()
        logger.report_scalar(
            title="serving", series="risk_probability", value=result["risk_probability"], iteration=0
        )
        logger.report_scalar(
            title="serving", series="latency_ms", value=result["latency_ms"], iteration=0
        )
    except Exception:
        pass
