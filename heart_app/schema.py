"""Single source of truth for the model's input features.

The order of ``FEATURES`` mirrors the column order of ``data/Data.csv``.
The training notebook builds the model input as

    [all non-categorical columns, in CSV order]
    + OneHotEncoder(smoker_status, chest_pain_type, sex)

so ``NUMERIC_COLUMNS`` (CSV order, categoricals removed) must be preserved exactly.
``tests/test_predictor.py`` verifies this against the real dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Group identifiers (translated in i18n.py as ``grp_<id>``)
PROFILE, VITALS, LABS, CARDIAC, LIFESTYLE = "profile", "vitals", "labs", "cardiac", "lifestyle"
GROUPS = (PROFILE, VITALS, LABS, CARDIAC, LIFESTYLE)

TARGET = "has_heart_disease"

# Illustrative (non-clinical) probability bands used for the colour-coded result.
LOW_BAND_MAX = 0.30
MODERATE_BAND_MAX = 0.60


def risk_band(probability: float) -> str:
    """Map a probability to ``'low' | 'moderate' | 'high'``."""
    if probability < LOW_BAND_MAX:
        return "low"
    if probability < MODERATE_BAND_MAX:
        return "moderate"
    return "high"


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "int" | "float" | "bool" | "cat"
    group: str
    label_en: str
    label_fa: str
    default: object
    low: float | None = None
    high: float | None = None
    step: float | None = None
    options: tuple = field(default_factory=tuple)  # categorical values (raw, as in training)
    widget: str = "number"  # "number" | "slider"
    help_en: str = ""
    help_fa: str = ""

    def label(self, lang: str) -> str:
        return self.label_fa if lang == "fa" else self.label_en

    def help(self, lang: str) -> str:
        return (self.help_fa if lang == "fa" else self.help_en) or ""


# fmt: off
FEATURES: tuple[Feature, ...] = (
    Feature("age", "int", PROFILE, "Age (years)", "سن (سال)", 54, 18, 90, 1),
    Feature("sex", "cat", PROFILE, "Sex", "جنسیت", "Male", options=("Female", "Male")),
    Feature("resting_bp_systolic", "int", VITALS, "Systolic blood pressure (mmHg)", "فشار خون سیستولیک (mmHg)", 128, 85, 181, 1),
    Feature("resting_bp_diastolic", "int", VITALS, "Diastolic blood pressure (mmHg)", "فشار خون دیاستولیک (mmHg)", 81, 50, 126, 1),
    Feature("cholesterol_total", "int", LABS, "Total cholesterol (mg/dL)", "کلسترول تام (mg/dL)", 189, 90, 314, 1),
    Feature("hdl", "int", LABS, "HDL cholesterol (mg/dL)", "کلسترول HDL (mg/dL)", 55, 18, 110, 1),
    Feature("ldl", "int", LABS, "LDL cholesterol (mg/dL)", "کلسترول LDL (mg/dL)", 103, 35, 207, 1),
    Feature("triglycerides", "int", LABS, "Triglycerides (mg/dL)", "تری‌گلیسرید (mg/dL)", 151, 35, 390, 1),
    Feature("fasting_blood_sugar", "int", LABS, "Fasting blood sugar (mg/dL)", "قند خون ناشتا (mg/dL)", 119, 60, 204, 1),
    Feature("hba1c", "float", LABS, "HbA1c (%)", "هموگلوبین A1c (٪)", 5.8, 4.0, 8.6, 0.1),
    Feature("bmi", "float", PROFILE, "BMI (kg/m²)", "شاخص توده بدنی (kg/m²)", 25.3, 15.0, 43.3, 0.1),
    Feature("resting_heart_rate", "int", VITALS, "Resting heart rate (bpm)", "ضربان قلب در استراحت (bpm)", 81, 48, 111, 1),
    Feature("max_heart_rate_achieved", "int", VITALS, "Max heart rate achieved (bpm)", "حداکثر ضربان قلب حاصل‌شده (bpm)", 166, 93, 210, 1,
            help_en="Highest heart rate reached during an exercise test.",
            help_fa="بیشترین ضربان قلب ثبت‌شده در تست ورزش."),
    Feature("chest_pain_type", "cat", CARDIAC, "Chest pain type", "نوع درد قفسه سینه", "Asymptomatic",
            options=("Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina")),
    Feature("exercise_induced_angina", "bool", CARDIAC, "Exercise-induced angina", "آنژین ناشی از ورزش", False),
    Feature("st_depression", "float", CARDIAC, "ST depression (mm)", "افت قطعه ST (mm)", 0.7, 0.0, 6.5, 0.1,
            help_en="ST-segment depression induced by exercise relative to rest.",
            help_fa="میزان افت قطعه ST ناشی از ورزش نسبت به استراحت."),
    Feature("family_history", "bool", PROFILE, "Family history of heart disease", "سابقه خانوادگی بیماری قلبی", False),
    Feature("smoker_status", "cat", PROFILE, "Smoking status", "وضعیت مصرف سیگار", "Never", options=("Never", "Former", "Current")),
    Feature("alcohol_units_per_week", "float", LIFESTYLE, "Alcohol (units / week)", "الکل (واحد در هفته)", 4.2, 0.0, 45.9, 0.1),
    Feature("exercise_minutes_per_week", "int", LIFESTYLE, "Exercise (minutes / week)", "ورزش (دقیقه در هفته)", 139, 0, 366, 1),
    Feature("sleep_hours", "float", LIFESTYLE, "Sleep (hours / night)", "خواب (ساعت در شب)", 7.0, 3.1, 11.0, 0.1),
    Feature("stress_score", "float", LIFESTYLE, "Stress score (0–100)", "نمره استرس (۰ تا ۱۰۰)", 48.1, 0.0, 100.0, 0.1, widget="slider"),
    Feature("wearable_owner", "bool", LIFESTYLE, "Owns a wearable device", "دارای دستگاه پوشیدنی", False),
    Feature("daily_steps", "int", LIFESTYLE, "Daily steps", "تعداد گام روزانه", 6178, 500, 13950, 50),
    Feature("diet_quality_score", "float", LIFESTYLE, "Diet quality score (0–100)", "نمره کیفیت رژیم غذایی (۰ تا ۱۰۰)", 59.6, 4.8, 100.0, 0.1, widget="slider"),
)
# fmt: on

FEATURE_BY_NAME = {f.name: f for f in FEATURES}
INPUT_COLUMNS = [f.name for f in FEATURES]
NUMERIC_COLUMNS = [f.name for f in FEATURES if f.kind != "cat"]  # model input order
CATEGORICAL_COLUMNS = [f.name for f in FEATURES if f.kind == "cat"]
BOOL_COLUMNS = [f.name for f in FEATURES if f.kind == "bool"]

# Example profiles: per-class medians / modes computed from data/Data.csv.
PRESET_HEALTHY = {
    "age": 52, "sex": "Female", "resting_bp_systolic": 125, "resting_bp_diastolic": 79,
    "cholesterol_total": 186, "hdl": 57, "ldl": 98, "triglycerides": 148,
    "fasting_blood_sugar": 116, "hba1c": 5.7, "bmi": 24.6, "resting_heart_rate": 80,
    "max_heart_rate_achieved": 173, "chest_pain_type": "Asymptomatic",
    "exercise_induced_angina": False, "st_depression": 0.6, "family_history": False,
    "smoker_status": "Never", "alcohol_units_per_week": 4.0,
    "exercise_minutes_per_week": 149, "sleep_hours": 7.0, "stress_score": 47.2,
    "wearable_owner": False, "daily_steps": 6395, "diet_quality_score": 61.1,
}
PRESET_AT_RISK = {
    "age": 60, "sex": "Male", "resting_bp_systolic": 133, "resting_bp_diastolic": 84,
    "cholesterol_total": 197, "hdl": 51, "ldl": 114, "triglycerides": 159,
    "fasting_blood_sugar": 126, "hba1c": 6.0, "bmi": 26.7, "resting_heart_rate": 83,
    "max_heart_rate_achieved": 146, "chest_pain_type": "Asymptomatic",
    "exercise_induced_angina": True, "st_depression": 1.2, "family_history": False,
    "smoker_status": "Never", "alcohol_units_per_week": 4.5,
    "exercise_minutes_per_week": 114, "sleep_hours": 6.9, "stress_score": 50.3,
    "wearable_owner": False, "daily_steps": 5708, "diet_quality_score": 55.8,
}
PRESETS = {
    "typical": {f.name: f.default for f in FEATURES},
    "healthy": PRESET_HEALTHY,
    "at_risk": PRESET_AT_RISK,
}


def cast_value(feature: Feature, value):
    """Cast a raw value to the exact Python type the Streamlit widget expects."""
    if feature.kind == "int":
        return int(round(float(value)))
    if feature.kind == "float":
        return float(value)
    if feature.kind == "bool":
        return bool(value)
    return str(value)
