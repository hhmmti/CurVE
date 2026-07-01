"""CurVE ``curve_position`` figures (M4 / 4b) — UI only.

Two real-vs-ideal figures for the *where am I on the curve* question, both reusing the
app's vendored ΔP overlays and adding ONE distinct marker for the representative
operating point (median flow, median ΔP_pump):

  * :func:`build_curve_position_overlay` — the *Single Frequency* overlay (observed
    cloud + the ideal well-scaled ΔP curve at the operating Hz + last-observation star).
  * :func:`build_curve_position_family` — the *Multi Frequency* affinity fan (the family
    of ΔP curves across the app's frequency set, the operating-Hz curve highlighted).

Plotting only — no physics (4-layer rule). The per-frequency curves are reconstructed
upstream (vendored ``build_multi_frequency_curves``); these builders only render. Both
figures go to the UI; the engine strips them before the model sees the result.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go

from plotting.preprocessed_charts import (
    build_ideal_multi_frequency_delta_p,
    build_ideal_single_frequency_delta_p,
)

# The operating-point marker — identical across both figures so the same point reads the
# same way on the single-frequency overlay and the affinity fan.
_OP_POINT_MARKER = dict(
    symbol="diamond",
    size=16,
    color="#DC2626",
    line=dict(color="#7F1D1D", width=1.5),
)


def _add_operating_point(fig: go.Figure, operating_point: Dict[str, Any]) -> go.Figure:
    """Mark the representative operating point (median flow, median ΔP_pump) on ``fig``."""
    q = operating_point.get("flow_bpd")
    dp = operating_point.get("delta_p_pump_psi")
    if q is not None and dp is not None:
        fig.add_trace(
            go.Scatter(
                x=[q],
                y=[dp],
                mode="markers",
                name="Operating Point (median)",
                marker=dict(_OP_POINT_MARKER),
            ),
            secondary_y=False,
        )
    return fig


def build_curve_position_overlay(
    observed_daily: pd.DataFrame,
    ideal_curve: pd.DataFrame,
    operating_point: Dict[str, Any],
    title: str = "Operating Point vs Ideal Curve",
) -> go.Figure:
    """Single-frequency observed-vs-ideal ΔP overlay with the operating point marked."""
    fig = build_ideal_single_frequency_delta_p(observed_daily, ideal_curve, title=title)
    return _add_operating_point(fig, operating_point)


def build_curve_position_family(
    observed_daily: pd.DataFrame,
    family_curves: Dict[float, pd.DataFrame],
    operating_point: Dict[str, Any],
    selected_frequency_hz: float,
    title: str = "Operating Point vs Ideal Curve — Frequency Family",
) -> go.Figure:
    """Affinity frequency-family ΔP fan (app's set) with the operating point marked.

    Ports the app's *Multi Frequency* ΔP plot (``build_ideal_multi_frequency_delta_p``):
    the fan of per-frequency ΔP curves with the operating-Hz curve highlighted and the
    observed cloud overlaid, then the same operating-point marker as the single-frequency
    overlay. The per-frequency curves are the vendored affinity reconstruction, passed in.
    """
    fig = build_ideal_multi_frequency_delta_p(
        observed_daily,
        family_curves,
        selected_frequency_hz=selected_frequency_hz,
        title=title,
    )
    return _add_operating_point(fig, operating_point)
