"""Plotly figure builders."""

from __future__ import annotations

import plotly.graph_objects as go

from .schema import LOW_BAND_MAX, MODERATE_BAND_MAX

COLORS = {"low": "#2a9d8f", "moderate": "#f4a261", "high": "#d62839"}


def gauge_chart(probability: float, band: str, title: str) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": COLORS[band], "thickness": 0.32},
                "steps": [
                    {"range": [0, LOW_BAND_MAX * 100], "color": "rgba(42,157,143,0.22)"},
                    {
                        "range": [LOW_BAND_MAX * 100, MODERATE_BAND_MAX * 100],
                        "color": "rgba(244,162,97,0.22)",
                    },
                    {"range": [MODERATE_BAND_MAX * 100, 100], "color": "rgba(214,40,57,0.22)"},
                ],
            },
        )
    )
    figure.update_layout(height=300, margin=dict(l=30, r=30, t=70, b=10))
    return figure


def contribution_chart(labels: list[str], deltas_pp: list[float], x_title: str) -> go.Figure:
    """Horizontal bar chart; ``deltas_pp`` are percentage points."""
    # Reverse so the largest effect is drawn at the top.
    labels = list(reversed(labels))
    deltas_pp = list(reversed(deltas_pp))
    colors = [COLORS["high"] if d > 0 else COLORS["low"] for d in deltas_pp]
    figure = go.Figure(
        go.Bar(
            x=deltas_pp,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:+.1f} pp<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(280, 38 * len(labels) + 90),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=x_title,
        yaxis_title=None,
    )
    figure.add_vline(x=0, line_width=1, line_color="gray")
    return figure
