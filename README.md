---
title: Heart Disease Risk Predictor
emoji: ❤️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Heart Disease Risk Prediction (free stack: Gradio + ClearML + Evidently)

A Gradio app that estimates a patient's probability of heart disease from
25 clinical/lifestyle features, deployed on Hugging Face Spaces — no paid
services anywhere in the stack.

| Component | Tool |
|---|---|
| Model | Keras `Sequential` (`31 → 16 → 8 → 1`), trained on a 9,000-row clinical dataset |
| Serving + UI | Gradio, hosted on HF Spaces (free CPU tier) |
| Experiment/serving tracking | ClearML (free community server) |
| Monitoring | Evidently data-drift report (in-app tab) + optional Uptime Kuma for uptime |

## Project structure

```
heart-disease-risk-predictor/
├── app.py                  # Gradio UI + entry point (HF Spaces app_file)
├── app/
│   ├── inference.py         # Model + encoder loading, feature vector, predict()
│   ├── monitoring.py        # CSV prediction log, Evidently report, ClearML logging
│   └── config.py            # Environment-driven configuration
├── artifacts/
│   ├── model.pkl             # Trained Keras model
│   └── OneHotEncoder.pkl     # Fitted encoder for categorical features
├── data/                     # predictions_log.csv is written here at runtime
├── requirements.txt
└── DEPLOYMENT.md             # Step-by-step free deployment guide
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Gradio will print a local URL (default `http://127.0.0.1:7860`).

## Monitoring

- **Evidently tab**: builds a data-drift report comparing the first half vs.
  second half of everything logged so far in `data/predictions_log.csv`.
  Needs at least `MIN_ROWS_FOR_REPORT` (default 20) logged predictions.
- **ClearML**: set `ENABLE_CLEARML=true` plus `CLEARML_API_ACCESS_KEY` /
  `CLEARML_API_SECRET_KEY` (as HF Space secrets) to log `risk_probability`
  and `latency_ms` per request to a free ClearML project.
- **Uptime Kuma**: not run inside this Space — see `DEPLOYMENT.md` for a
  free external instance you can point at this Space's URL.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full, step-by-step, completely
free deployment guide (Hugging Face Spaces + ClearML + Evidently/Uptime Kuma).

## Tech stack

Python · TensorFlow/Keras · scikit-learn · Gradio · Evidently · ClearML
