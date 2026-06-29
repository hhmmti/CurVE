"""CurVE recommendation-comparison figure (M3) — current vs recommended operating point.

A small, additive plotting module (does NOT modify the vendored chart files). Chart
generation only — no physics. The recommendation_comparison tool does the faithful
payload extraction; this renders the current-vs-recommended setpoint deltas as a
grouped horizontal bar strip (frequency, tubing pressure, liquid rate), each metric
on its own normalized row so the relative change reads at a glance regardless of unit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import plotly.graph_objects as go


_CURRENT_COLOR = "#57606a"
_RECOMMENDED_COLOR = "#1a7f37"


def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN → None


def build_recommendation_comparison_bars(
    compare_row: Dict[str, Any], title: str = "Current vs Recommended Operating Point"
) -> go.Figure:
    """Grouped bars of current vs recommended for the headline setpoint metrics."""
    metrics = [
        ("Motor frequency (Hz)", "cur_motor_frequency_hz", "rec_motor_frequency_hz"),
        ("Tubing pressure (psi)", "cur_tubing_pressure_psi", "rec_tubing_pressure_psi"),
        ("Liquid rate (bpd)", "cur_liquid_rate_bpd", "rec_liquid_rate_bpd"),
    ]

    labels = [m[0] for m in metrics]
    cur_vals = [_finite(compare_row.get(m[1])) for m in metrics]
    rec_vals = [_finite(compare_row.get(m[2])) for m in metrics]

    if not any(v is not None for v in cur_vals + rec_vals):
        fig = go.Figure()
        fig.add_annotation(
            text="Recommendation comparison unavailable",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color="#95A5A6", size=13),
        )
        fig.update_layout(
            height=160, template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        return fig

    fig = go.Figure()
    fig.add_bar(
        y=labels, x=cur_vals, name="Current", orientation="h",
        marker_color=_CURRENT_COLOR,
        text=[f"{v:,.1f}" if v is not None else "—" for v in cur_vals],
        textposition="auto",
    )
    fig.add_bar(
        y=labels, x=rec_vals, name="Recommended", orientation="h",
        marker_color=_RECOMMENDED_COLOR,
        text=[f"{v:,.1f}" if v is not None else "—" for v in rec_vals],
        textposition="auto",
    )
    fig.update_layout(
        title=title,
        barmode="group",
        height=320,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig
