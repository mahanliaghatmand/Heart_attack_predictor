"""Heart Attack Risk Predictor - Streamlit application entry point.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heart_app import __version__
from heart_app.charts import COLORS, contribution_chart, gauge_chart
from heart_app.explain import what_if_contributions
from heart_app.i18n import LANGS, translator
from heart_app.predictor import HeartRiskPredictor, PredictionInputError
from heart_app.schema import (
    FEATURE_BY_NAME,
    FEATURES,
    GROUPS,
    INPUT_COLUMNS,
    NUMERIC_COLUMNS,
    PRESETS,
    TARGET,
    cast_value,
    risk_band,
)

ROOT = Path(__file__).parent
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_PATH = ROOT / "data" / "Data.csv"

st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

RTL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');
.stMarkdown, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"],
.stAlert p, .stTabs button p, h1, h2, h3 {
    font-family: 'Vazirmatn', sans-serif;
    direction: rtl;
    text-align: right;
}
</style>
"""


# --------------------------------------------------------------------------- state
def widget_key(name: str) -> str:
    return f"inp_{name}"


def init_state() -> None:
    st.session_state.setdefault("lang", "en")
    for feature in FEATURES:
        st.session_state.setdefault(widget_key(feature.name), cast_value(feature, feature.default))


def apply_preset(name: str) -> None:
    values = PRESETS[name]
    for feature in FEATURES:
        st.session_state[widget_key(feature.name)] = cast_value(feature, values[feature.name])


def collect_patient() -> dict:
    return {f.name: st.session_state[widget_key(f.name)] for f in FEATURES}


# ------------------------------------------------------------------------- caching
@st.cache_resource(show_spinner="Loading model…")
def get_predictor() -> HeartRiskPredictor:
    return HeartRiskPredictor.load(ARTIFACTS_DIR)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None


# ---------------------------------------------------------------------------- UI
def render_input(feature, lang: str, t) -> None:
    key = widget_key(feature.name)
    label = feature.label(lang)
    help_text = feature.help(lang) or None
    if feature.kind == "cat":
        st.selectbox(
            label,
            feature.options,
            key=key,
            format_func=lambda value: t(f"opt_{value}"),
            help=help_text,
        )
    elif feature.kind == "bool":
        st.toggle(label, key=key, help=help_text)
    elif feature.widget == "slider":
        st.slider(
            label,
            min_value=float(feature.low),
            max_value=float(feature.high),
            step=float(feature.step),
            key=key,
            help=help_text,
        )
    elif feature.kind == "int":
        st.number_input(
            label,
            min_value=int(feature.low),
            max_value=int(feature.high),
            step=int(feature.step),
            key=key,
            help=help_text,
        )
    else:
        st.number_input(
            label,
            min_value=float(feature.low),
            max_value=float(feature.high),
            step=float(feature.step),
            format="%.1f",
            key=key,
            help=help_text,
        )


def render_assessment(predictor: HeartRiskPredictor, lang: str, t) -> None:
    st.subheader(t("presets_title"), help=t("presets_help"))
    columns = st.columns(3)
    for column, preset in zip(columns, ("typical", "healthy", "at_risk")):
        column.button(
            t(f"preset_{preset}"),
            key=f"preset_btn_{preset}",
            on_click=apply_preset,
            args=(preset,),
        )

    with st.form("patient_form"):
        tabs = st.tabs([t(f"grp_{group}") for group in GROUPS])
        for tab, group in zip(tabs, GROUPS):
            with tab:
                members = [f for f in FEATURES if f.group == group]
                cols = st.columns(2)
                for index, feature in enumerate(members):
                    with cols[index % 2]:
                        render_input(feature, lang, t)
        submitted = st.form_submit_button(t("submit"), type="primary")

    if submitted:
        patient = collect_patient()
        probability = float(predictor.predict_proba(pd.DataFrame([patient]))[0])
        st.session_state["result"] = {
            "probability": probability,
            "patient": patient,
            "contributions": what_if_contributions(predictor, patient),
        }

    result = st.session_state.get("result")
    if result:
        render_result(result, lang, t)


def render_result(result: dict, lang: str, t) -> None:
    probability = result["probability"]
    band = risk_band(probability)

    st.divider()
    st.subheader(t("result_title"))
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge_chart(probability, band, t("gauge_title")))
    with right:
        message = f"**{t(f'band_{band}')}** — {probability:.1%}"
        {"low": st.success, "moderate": st.warning, "high": st.error}[band](message)
        st.caption(t("band_note"))
        export = pd.DataFrame([result["patient"]])
        export["risk_probability"] = round(probability, 4)
        export["risk_band"] = band
        st.download_button(
            t("download_result"),
            export.to_csv(index=False).encode("utf-8"),
            file_name="heart_risk_result.csv",
            mime="text/csv",
        )

    st.subheader(t("drivers_title"), help=t("drivers_help"))
    top = result["contributions"].head(8)
    labels = [FEATURE_BY_NAME[name].label(lang) for name in top["feature"]]
    deltas = (top["delta"] * 100).round(2).tolist()
    st.plotly_chart(contribution_chart(labels, deltas, t("drivers_axis")))


def render_batch(predictor: HeartRiskPredictor, lang: str, t) -> None:
    st.subheader(t("batch_title"))
    st.write(t("batch_help"))

    template = pd.DataFrame([{f.name: f.default for f in FEATURES}])
    st.download_button(
        t("batch_template"),
        template.to_csv(index=False).encode("utf-8"),
        file_name="patients_template.csv",
        mime="text/csv",
    )

    upload = st.file_uploader(t("batch_upload"), type=["csv"])
    if upload is None:
        return

    try:
        frame = pd.read_csv(upload)
    except Exception:  # noqa: BLE001 - any parser failure is a user-facing error
        st.error(t("batch_read_error"))
        return

    try:
        probabilities = predictor.predict_proba(frame)
    except PredictionInputError as exc:
        st.error(t("batch_error", msg=str(exc)))
        return

    output = frame.copy()
    output["risk_probability"] = probabilities.round(4)
    output["risk_band"] = [risk_band(p) for p in probabilities]

    m1, m2, m3 = st.columns(3)
    m1.metric(t("batch_rows"), f"{len(output):,}")
    m2.metric(t("batch_mean"), f"{probabilities.mean():.1%}")
    m3.metric(t("batch_high"), f"{(output['risk_band'] == 'high').sum():,}")

    st.markdown(f"**{t('batch_dist')}**")
    histogram = go.Figure(
        go.Histogram(x=probabilities * 100, nbinsx=20, marker_color=COLORS["high"])
    )
    histogram.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="%", yaxis_title=None
    )
    st.plotly_chart(histogram)

    st.dataframe(output, hide_index=True)
    st.download_button(
        t("batch_download"),
        output.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
        type="primary",
    )


def _value_label(feature, value, t) -> str:
    if feature.kind == "bool":
        return t("yes") if value else t("no")
    if feature.kind == "cat":
        return t(f"opt_{value}")
    return str(value)


def render_data(lang: str, t) -> None:
    st.subheader(t("data_title"))
    df = load_data()
    if df is None:
        st.info(t("data_missing_file"))
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("data_rows"), f"{len(df):,}")
    m2.metric(t("data_features"), len(INPUT_COLUMNS))
    m3.metric(t("data_prevalence"), f"{df[TARGET].mean():.1%}")
    m4.metric(t("data_missing"), int(df.isna().sum().sum()))

    name = st.selectbox(
        t("data_pick"),
        INPUT_COLUMNS,
        format_func=lambda n: FEATURE_BY_NAME[n].label(lang),
    )
    feature = FEATURE_BY_NAME[name]
    outcomes = ((0, t("neg"), COLORS["low"]), (1, t("pos"), COLORS["high"]))
    figure = go.Figure()
    for value, label, color in outcomes:
        subset = df.loc[df[TARGET] == value, name]
        if feature.kind in ("int", "float"):
            figure.add_trace(
                go.Histogram(x=subset, name=label, marker_color=color, opacity=0.65, nbinsx=40)
            )
        else:
            figure.add_trace(
                go.Histogram(
                    x=[_value_label(feature, v, t) for v in subset],
                    name=label,
                    marker_color=color,
                )
            )
    figure.update_layout(
        barmode="overlay" if feature.kind in ("int", "float") else "group",
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=feature.label(lang),
        legend_title_text=t("data_outcome"),
    )
    st.plotly_chart(figure)

    numeric = df[NUMERIC_COLUMNS + [TARGET]].astype(float)
    corr = numeric.corr()
    with_target = corr[TARGET].drop(TARGET).sort_values()
    st.markdown(f"**{t('corr_target_title')}**")
    bar = go.Figure(
        go.Bar(
            x=with_target.values,
            y=[FEATURE_BY_NAME[n].label(lang) for n in with_target.index],
            orientation="h",
            marker_color=[COLORS["high"] if v > 0 else COLORS["low"] for v in with_target.values],
        )
    )
    bar.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar)

    with st.expander(t("corr_heat")):
        axis = [FEATURE_BY_NAME[n].label(lang) if n in FEATURE_BY_NAME else n for n in corr.columns]
        heat = go.Figure(
            go.Heatmap(
                z=corr.values, x=axis, y=axis, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True
            )
        )
        heat.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(heat)

    with st.expander(t("data_preview")):
        st.dataframe(df.head(100), hide_index=True)


def render_model(predictor: HeartRiskPredictor, t) -> None:
    st.subheader(t("model_title"))
    st.markdown(f"**{t('model_arch_title')}**")
    st.markdown(t("model_arch"))

    st.markdown(f"**{t('model_features_title')}**")
    st.caption(t("model_features_help"))
    names = pd.DataFrame(
        {"#": range(1, len(predictor.feature_names) + 1), "feature": predictor.feature_names}
    )
    st.dataframe(names, hide_index=True)

    st.markdown(f"**{t('model_cfg_title')}**")
    st.json(predictor.config)

    st.markdown(f"**{t('model_limits_title')}**")
    st.markdown(t("model_limits"))


# ------------------------------------------------------------------------- main
def main() -> None:
    init_state()

    with st.sidebar:
        st.radio(
            "Language / زبان",
            options=list(LANGS),
            format_func=LANGS.get,
            key="lang",
            horizontal=True,
        )
    lang = st.session_state["lang"]
    t = translator(lang)
    if lang == "fa":
        st.markdown(RTL_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"### 🫀 {t('app_title')}")
        st.write(t("sidebar_about"))
        st.caption(t("footer", version=__version__))

    st.title(f"🫀 {t('app_title')}")
    st.write(t("app_subtitle"))
    st.info(t("disclaimer"))

    try:
        predictor = get_predictor()
    except Exception as exc:  # noqa: BLE001 - show a helpful message instead of a stack trace
        st.error(t("err_model"))
        with st.expander("Details"):
            st.exception(exc)
        st.stop()

    tab_assess, tab_batch, tab_data, tab_model = st.tabs(
        [t("tab_assess"), t("tab_batch"), t("tab_data"), t("tab_model")]
    )
    with tab_assess:
        render_assessment(predictor, lang, t)
    with tab_batch:
        render_batch(predictor, lang, t)
    with tab_data:
        render_data(lang, t)
    with tab_model:
        render_model(predictor, t)


main()
