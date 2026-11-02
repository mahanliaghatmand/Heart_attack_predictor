# 🫀 Heart Attack Risk Predictor

A bilingual (English / فارسی) **Streamlit** app that serves a Keras neural network estimating the
probability of heart disease from 25 clinical, lab and lifestyle measurements.

> ⚠️ **Educational use only.** This is not a medical device and does not provide a diagnosis.

## Features

- **Assessment** – grouped input form, quick-fill example profiles, colour-coded risk gauge and a
  "what influenced this estimate?" chart (one-at-a-time what-if analysis).
- **Batch prediction** – upload a CSV, get risk per patient, summary metrics and a downloadable file
  (CSV template included, strict input validation with clear error messages).
- **Data explorer** – class balance, per-feature distributions by outcome, correlations.
- **About the model** – architecture, the 31-element input vector, training configuration, limitations.
- English / Persian UI with RTL support.

## Project structure

```
├── app.py                  # Streamlit entry point
├── heart_app/
│   ├── schema.py           # feature definitions, ranges, presets (single source of truth)
│   ├── predictor.py        # artifact loading, validation, feature engineering
│   ├── explain.py          # what-if contribution analysis
│   ├── charts.py           # Plotly figures
│   └── i18n.py             # EN / FA strings
├── artifacts/              # Model.pkl, Config.pkl, OneHotEncoder.pkl
├── data/Data.csv           # training data (9,000 rows)
├── notebooks/              # training notebook
├── tests/                  # pytest suite (no TensorFlow required)
├── .streamlit/config.toml
├── Dockerfile
└── .github/workflows/ci.yml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10 – 3.13 is supported (limited by TensorFlow).

## Deploy

**Streamlit Community Cloud** – push the repo to GitHub, then *New app* → select the repo, branch and
`app.py`. Choose Python 3.12 under *Advanced settings*.

**Docker**

```bash
docker build -t heart-risk-app .
docker run -p 8501:8501 heart-risk-app
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests re-implement the notebook's pre-processing and assert that the app builds an identical
`(n, 31)` feature matrix, so model and app cannot silently drift apart.

## Model notes

| Item | Value |
|---|---|
| Inputs | 22 numeric/boolean features + one-hot (`smoker_status`, `chest_pain_type`, `sex`) = **31** |
| Network | Dense 31 → 16 → 8 → Dropout 0.1 → 1 (sigmoid) |
| Training | Adam, binary cross-entropy, early stopping (patience 50) |
| Output | Probability of `has_heart_disease` |

Risk bands in the UI (low < 30 %, moderate 30–60 %, high ≥ 60 %) are illustrative, not clinical.

### Compatibility & security

- `Model.pkl` was saved with **Keras 3.15.1** and `OneHotEncoder.pkl` with **scikit-learn 1.9.0**.
  If loading fails after deployment, pin the versions you trained with
  (`pip freeze | grep -iE "keras|tensorflow|scikit-learn"`) in `requirements.txt`.
- The artifacts are Python **pickles**, which can execute code on load. Only use files you trust.
  For long-term use consider exporting the model as `.keras` and the encoder with `skops`.

## خلاصه فارسی

این پروژه یک اپ Streamlit دو‌زبانه (فارسی/انگلیسی) برای مدل پیش‌بینی خطر بیماری قلبی است.
اجرا: `pip install -r requirements.txt` و سپس `streamlit run app.py`.
این ابزار فقط جنبه آموزشی دارد و جایگزین تشخیص پزشک نیست.

## License

MIT – see [LICENSE](LICENSE). Replace `YOUR NAME` with your name.
