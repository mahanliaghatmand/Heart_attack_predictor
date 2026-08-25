"""
Hugging Face Spaces entry point (SDK: gradio, app_file: app.py).

Fully free stack:
  - Serving + UI:      Gradio (this file), hosted on HF Spaces free CPU tier
  - Experiment/serving tracking: ClearML (community server, see app/config.py)
  - Monitoring:         Evidently drift report (see app/monitoring.py) +
                         optional external Uptime Kuma instance (see DEPLOYMENT.md)
"""
from __future__ import annotations

import gradio as gr

from app.inference import (
    CHEST_PAIN_OPTIONS,
    NUMERIC_FEATURE_ORDER,
    SEX_OPTIONS,
    SMOKER_OPTIONS,
    model_service,
)
from app.config import MIN_ROWS_FOR_REPORT
from app.monitoring import build_drift_report, log_prediction, log_to_clearml

RISK_LABELS_FA = {"low": "پایین 🟢", "medium": "متوسط 🟡", "high": "بالا 🔴"}


def predict(
    age,
    sex,
    resting_bp_systolic,
    resting_bp_diastolic,
    cholesterol_total,
    hdl,
    ldl,
    triglycerides,
    fasting_blood_sugar,
    hba1c,
    bmi,
    resting_heart_rate,
    max_heart_rate_achieved,
    chest_pain_type,
    exercise_induced_angina,
    st_depression,
    family_history,
    smoker_status,
    alcohol_units_per_week,
    exercise_minutes_per_week,
    sleep_hours,
    stress_score,
    wearable_owner,
    daily_steps,
    diet_quality_score,
):
    data = {
        "age": age,
        "sex": sex,
        "resting_bp_systolic": resting_bp_systolic,
        "resting_bp_diastolic": resting_bp_diastolic,
        "cholesterol_total": cholesterol_total,
        "hdl": hdl,
        "ldl": ldl,
        "triglycerides": triglycerides,
        "fasting_blood_sugar": fasting_blood_sugar,
        "hba1c": hba1c,
        "bmi": bmi,
        "resting_heart_rate": resting_heart_rate,
        "max_heart_rate_achieved": max_heart_rate_achieved,
        "chest_pain_type": chest_pain_type,
        "exercise_induced_angina": bool(exercise_induced_angina),
        "st_depression": st_depression,
        "family_history": bool(family_history),
        "smoker_status": smoker_status,
        "alcohol_units_per_week": alcohol_units_per_week,
        "exercise_minutes_per_week": exercise_minutes_per_week,
        "sleep_hours": sleep_hours,
        "stress_score": stress_score,
        "wearable_owner": bool(wearable_owner),
        "daily_steps": daily_steps,
        "diet_quality_score": diet_quality_score,
    }

    result = model_service.predict(data)

    # Best-effort logging — never break the UI if either sink fails.
    try:
        log_prediction(data, result)
    except Exception as exc:
        print(f"[monitoring] CSV log failed: {exc}")
    try:
        log_to_clearml(result)
    except Exception as exc:
        print(f"[monitoring] ClearML log failed: {exc}")

    label = RISK_LABELS_FA.get(result["risk_label"], result["risk_label"])
    summary = (
        f"## نتیجه: ریسک {label}\n\n"
        f"- احتمال مدل: **{result['risk_probability']:.1%}**\n"
        f"- پیش‌بینی نهایی (0/1): **{result['prediction']}**\n"
        f"- نسخه مدل: `{result['model_version']}`\n"
        f"- زمان اجرا: {result['latency_ms']} ms"
    )
    return summary


def refresh_report():
    return build_drift_report()


with gr.Blocks(title="Heart Disease Risk Predictor") as demo:
    gr.Markdown(
        "# ❤️ پیش‌بینی ریسک بیماری قلبی\n"
        "این مدل بر اساس داده‌های بالینی و سبک زندگی، احتمال ابتلا به بیماری قلبی را تخمین می‌زند. "
        "این ابزار جایگزین تشخیص پزشکی نیست."
    )

    with gr.Tab("پیش‌بینی"):
        with gr.Row():
            with gr.Column():
                age = gr.Slider(0, 120, value=54, step=1, label="سن (age)")
                sex = gr.Radio(SEX_OPTIONS, value="Male", label="جنسیت (sex)")
                resting_bp_systolic = gr.Slider(60, 260, value=132, label="فشار خون سیستولیک")
                resting_bp_diastolic = gr.Slider(30, 180, value=84, label="فشار خون دیاستولیک")
                cholesterol_total = gr.Slider(50, 700, value=221, label="کلسترول کل")
                hdl = gr.Slider(5, 200, value=45, label="HDL")
                ldl = gr.Slider(5, 500, value=130, label="LDL")
                triglycerides = gr.Slider(10, 1000, value=150, label="تری‌گلیسیرید")
                fasting_blood_sugar = gr.Slider(40, 500, value=110, label="قند خون ناشتا")
                hba1c = gr.Slider(2.0, 18.0, value=5.8, label="HbA1c")
                bmi = gr.Slider(10.0, 70.0, value=27.4, label="BMI")
                resting_heart_rate = gr.Slider(30, 220, value=72, label="ضربان قلب در حالت استراحت")
            with gr.Column():
                max_heart_rate_achieved = gr.Slider(60, 250, value=158, label="حداکثر ضربان قلب")
                chest_pain_type = gr.Dropdown(CHEST_PAIN_OPTIONS, value="Atypical Angina", label="نوع درد قفسه سینه")
                exercise_induced_angina = gr.Checkbox(value=False, label="آنژین ناشی از ورزش")
                st_depression = gr.Slider(0.0, 10.0, value=1.2, label="ST depression")
                family_history = gr.Checkbox(value=True, label="سابقه خانوادگی")
                smoker_status = gr.Dropdown(SMOKER_OPTIONS, value="Former", label="وضعیت سیگار")
                alcohol_units_per_week = gr.Slider(0.0, 100.0, value=3.5, label="واحد الکل در هفته")
                exercise_minutes_per_week = gr.Slider(0, 2000, value=90, label="دقایق ورزش در هفته")
                sleep_hours = gr.Slider(0.0, 24.0, value=6.5, label="ساعات خواب")
                stress_score = gr.Slider(0.0, 10.0, value=6.2, label="نمره استرس")
                wearable_owner = gr.Checkbox(value=True, label="مالک دستگاه پوشیدنی")
                daily_steps = gr.Slider(0, 100000, value=6500, label="قدم روزانه")
                diet_quality_score = gr.Slider(0.0, 100.0, value=58.0, label="نمره کیفیت رژیم غذایی")

        predict_btn = gr.Button("پیش‌بینی ریسک", variant="primary")
        result_md = gr.Markdown()

        predict_btn.click(
            predict,
            inputs=[
                age, sex, resting_bp_systolic, resting_bp_diastolic, cholesterol_total,
                hdl, ldl, triglycerides, fasting_blood_sugar, hba1c, bmi,
                resting_heart_rate, max_heart_rate_achieved, chest_pain_type,
                exercise_induced_angina, st_depression, family_history, smoker_status,
                alcohol_units_per_week, exercise_minutes_per_week, sleep_hours,
                stress_score, wearable_owner, daily_steps, diet_quality_score,
            ],
            outputs=result_md,
        )

    with gr.Tab("مانیتورینگ (Evidently)"):
        gr.Markdown(
            "گزارش drift داده‌ها بر اساس پیش‌بینی‌های ثبت‌شده در همین سرویس. "
            f"حداقل {MIN_ROWS_FOR_REPORT} رکورد لازم است (در `app/config.py` قابل تنظیم است)."
        )
        report_btn = gr.Button("ساخت / بروزرسانی گزارش")
        report_html = gr.HTML()
        report_btn.click(refresh_report, inputs=None, outputs=report_html)

    with gr.Tab("درباره / لینک‌ها"):
        gr.Markdown(
            "- **ردیابی تجربه‌ها:** اگر `ENABLE_CLEARML=true` و کلیدهای API ست شده باشند، "
            "متریک هر پیش‌بینی در پروژه `heart-disease-risk` روی app.clear.ml ثبت می‌شود.\n"
            "- **آپتایم:** برای مانیتور کردن در‌دسترس‌بودن این Space از یک نمونه رایگان "
            "Uptime Kuma استفاده کن (به `DEPLOYMENT.md` مراجعه کن).\n"
            "- این ابزار صرفاً یک نمونه‌ی پرتفولیو است و جایگزین مشاوره پزشکی نیست."
        )


if __name__ == "__main__":
    demo.launch()
