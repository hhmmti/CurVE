"""Plotting helpers for ML recommendation analysis charts."""

from typing import Dict, Iterable, Optional

import numpy as np
import plotly.graph_objects as go


def _is_finite_number(value) -> bool:
    try:
        return value is not None and np.isfinite(float(value))
    except Exception:
        return False


def build_recommendation_grid_view(
    grid_payload: Dict,
    title: str,
    show_out_of_bounds: bool = True,
    selected_scenario_ids: Optional[Iterable[str]] = None,
    current_anchor_id: Optional[str] = None,
    recommended_anchor_id: Optional[str] = None,
) -> go.Figure:
    """Build Panel 1 view for recommendation surface exploration.

    The chart uses scatter-style points to preserve all scenario points and keeps
    out-of-bounds points visually inactive.
    """
    fig = go.Figure()

    points_df = grid_payload.get("points_df")
    if points_df is None or points_df.empty:
        fig.update_layout(
            title=title,
            xaxis_title="Motor Frequency (Hz)",
            yaxis_title="Tubing Pressure (psi)",
            template="plotly_white",
            height=620,
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text="No scenario points found for latest run.",
            showarrow=False,
            font=dict(size=13, color="#666666"),
        )
        return fig

    df = points_df.copy()
    if not show_out_of_bounds:
        df = df[df["in_bounds"]].copy()

    df = df[df["motor_frequency_hz"].notna() & df["tubing_pressure_psi"].notna()].copy()

    in_bounds_df = df[df["in_bounds"]].copy()
    out_bounds_df = df[~df["in_bounds"]].copy()

    selected_ids = {str(x) for x in (selected_scenario_ids or [])}

    if not in_bounds_df.empty:
        customdata = np.column_stack(
            [
                in_bounds_df["scenario_id"].astype(str),
                in_bounds_df["violated_boundaries"].fillna("").astype(str),
                in_bounds_df["is_selectable"].astype(str),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=in_bounds_df["motor_frequency_hz"],
                y=in_bounds_df["tubing_pressure_psi"],
                mode="markers",
                name="In bounds",
                marker=dict(
                    size=11,
                    color=in_bounds_df["total_economics"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Max Oil objective score"),
                    line=dict(color="rgba(255,255,255,0.7)", width=0.6),
                    opacity=0.95,
                ),
                customdata=customdata,
                hovertemplate=(
                    "Freq=%{x:.2f} Hz"
                    "<br>Tubing=%{y:.2f} psi"
                    "<br>Max Oil objective score=%{marker.color:.3f}"
                    "<br>Scenario=%{customdata[0]}"
                    "<br>Selectable=%{customdata[2]}<extra></extra>"
                ),
            )
        )

    if show_out_of_bounds and not out_bounds_df.empty:
        customdata = np.column_stack(
            [
                out_bounds_df["scenario_id"].astype(str),
                out_bounds_df["violated_boundaries"].fillna("").astype(str),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=out_bounds_df["motor_frequency_hz"],
                y=out_bounds_df["tubing_pressure_psi"],
                mode="markers",
                name="Out of bounds (inactive)",
                marker=dict(
                    size=9,
                    color="rgba(140,140,140,0.55)",
                    symbol="x",
                    line=dict(color="rgba(90,90,90,0.7)", width=0.5),
                ),
                customdata=customdata,
                hovertemplate=(
                    "Freq=%{x:.2f} Hz"
                    "<br>Tubing=%{y:.2f} psi"
                    "<br>Scenario=%{customdata[0]}"
                    "<br>Inactive: out of bounds"
                    "<br>Violations=%{customdata[1]}<extra></extra>"
                ),
            )
        )

    if selected_ids:
        selected_df = df[df["scenario_id"].astype(str).isin(selected_ids) & df["in_bounds"]].copy()
        if not selected_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected_df["motor_frequency_hz"],
                    y=selected_df["tubing_pressure_psi"],
                    mode="markers",
                    name="Selected alternatives",
                    marker=dict(size=15, symbol="star", color="#111111", line=dict(color="#ffffff", width=1.2)),
                    hovertemplate=(
                        "Selected"
                        "<br>Freq=%{x:.2f} Hz"
                        "<br>Tubing=%{y:.2f} psi<extra></extra>"
                    ),
                )
            )

    for anchor_id, label, color, symbol in [
        (current_anchor_id, "Current", "#1f77b4", "circle"),
        (recommended_anchor_id, "Recommended", "#d62728", "diamond"),
    ]:
        if not anchor_id:
            continue
        anchor_df = df[df["scenario_id"].astype(str) == str(anchor_id)]
        if anchor_df.empty:
            continue
        x = anchor_df["motor_frequency_hz"].iloc[0]
        y = anchor_df["tubing_pressure_psi"].iloc[0]
        if not (_is_finite_number(x) and _is_finite_number(y)):
            continue
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                name=label,
                text=[label],
                textposition="top center",
                marker=dict(size=15, color=color, symbol=symbol, line=dict(color="white", width=1.2)),
                hovertemplate=(
                    f"{label}"
                    "<br>Freq=%{x:.2f} Hz"
                    "<br>Tubing=%{y:.2f} psi<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Motor Frequency (Hz)",
        yaxis_title="Tubing Pressure (psi)",
        template="plotly_white",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_current_vs_recommended_family(
    curve_payload: Dict,
    title: str,
    alternative_points: Optional[Iterable[Dict]] = None,
    show_current: bool = True,
    show_recommended: bool = True,
) -> go.Figure:
    """Build current vs recommended overlay on multi-frequency ideal family."""
    fig = go.Figure()

    family_curves = curve_payload["family_curves"]
    selected_curve = curve_payload["selected_curve"]
    selected_freq = curve_payload["selected_frequency_hz"]
    current_point = curve_payload["current_point"]
    recommended_point = curve_payload["recommended_point"]

    sorted_freqs = sorted(family_curves.keys())
    first_freq = sorted_freqs[0] if sorted_freqs else None

    for freq in sorted_freqs:
        curve = family_curves[freq]
        fig.add_trace(
            go.Scatter(
                x=curve["flow_bpd"],
                y=curve["delta_p_psi"],
                mode="lines",
                line=dict(color="rgba(120,120,120,0.35)", width=1),
                name=f"Ideal {int(freq)} Hz",
                showlegend=bool(freq == first_freq),
                hovertemplate=(
                    f"Freq={int(freq)} Hz"
                    "<br>Flow=%{x:.1f} bpd"
                    "<br>Ideal Pressure=%{y:.1f} psi<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=selected_curve["flow_bpd"],
            y=selected_curve["delta_p_psi"],
            mode="lines",
            line=dict(color="black", width=3),
            name=f"Ideal selected ({selected_freq:.1f} Hz)",
            hovertemplate=(
                "Selected ideal"
                "<br>Flow=%{x:.1f} bpd"
                "<br>Ideal Pressure=%{y:.1f} psi<extra></extra>"
            ),
        )
    )

    cur_x = current_point.get("flow_bpd")
    cur_y = current_point.get("delta_p_pump_psi")
    rec_x = recommended_point.get("flow_bpd")
    rec_y = recommended_point.get("delta_p_pump_psi")

    def _valid(v):
        return _is_finite_number(v)

    has_cur = _valid(cur_x) and _valid(cur_y)
    has_rec = _valid(rec_x) and _valid(rec_y)

    if has_cur and show_current:
        fig.add_trace(
            go.Scatter(
                x=[cur_x],
                y=[cur_y],
                mode="markers+text",
                marker=dict(size=13, color="#1f77b4", symbol="circle"),
                text=["Current"],
                textposition="top center",
                name="Current setpoint",
            )
        )

    if has_rec and show_recommended:
        fig.add_trace(
            go.Scatter(
                x=[rec_x],
                y=[rec_y],
                mode="markers+text",
                marker=dict(size=14, color="#d62728", symbol="diamond"),
                text=["Recommended (max_oil)"],
                textposition="bottom center",
                name="ML recommendation",
            )
        )

    if has_cur and has_rec and show_current and show_recommended:
        fig.add_trace(
            go.Scatter(
                x=[cur_x, rec_x],
                y=[cur_y, rec_y],
                mode="lines",
                line=dict(color="rgba(214,39,40,0.55)", width=2, dash="dash"),
                name="Current -> Recommended",
                hoverinfo="skip",
            )
        )

    alt_points = list(alternative_points or [])
    if alt_points:
        xs = []
        ys = []
        texts = []
        economics = []
        for i, point in enumerate(alt_points, start=1):
            x = point.get("flow_bpd")
            y = point.get("delta_p_pump_psi")
            if not (_valid(x) and _valid(y)):
                continue
            xs.append(float(x))
            ys.append(float(y))
            label = point.get("label") or f"Alt {i}"
            texts.append(str(label))
            econ = point.get("total_economics")
            economics.append(float(econ) if _valid(econ) else np.nan)

        if xs:
            econ_vals = np.array(economics, dtype=float)
            use_econ_scale = np.isfinite(econ_vals).any()
            marker = dict(size=12, symbol="star")
            if use_econ_scale:
                marker.update(
                    color=econ_vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Max Oil objective score"),
                    line=dict(color="white", width=0.8),
                )
            else:
                marker.update(color="#111111")

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    marker=marker,
                    text=texts,
                    textposition="bottom center",
                    name="Selected alternatives",
                    hovertemplate=(
                        "%{text}"
                        "<br>Flow=%{x:.1f} bpd"
                        "<br>Pump Delta P=%{y:.1f} psi"
                        "<br>Max Oil objective score=%{marker.color:.3f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Liquid Rate (oil + water), bpd",
        yaxis_title="Pump Delta P (psi)",
        template="plotly_white",
        height=680,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    if not (has_cur and has_rec):
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.98,
            showarrow=False,
            text="Current/recommended Delta P points unavailable (missing required intake/alloc summary values).",
            font=dict(size=11, color="#7A1F1F"),
            bgcolor="rgba(255,235,235,0.8)",
        )

    return fig


def build_gas_interference_trend_chart(
    screen_payload: Dict,
    title: str = "Normalized multi-signal trend screen",
) -> go.Figure:
    """Build normalized trend chart for Gas-Interference Risk Screening."""
    fig = go.Figure()

    rows = screen_payload.get("normalized_trend_series") if isinstance(screen_payload, dict) else None
    if not rows:
        fig.update_layout(
            title=title,
            xaxis_title="Timestamp",
            yaxis_title="Normalized value (0-1)",
            template="plotly_white",
            height=480,
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text="No normalized trend series available.",
            showarrow=False,
            font=dict(size=13, color="#666666"),
        )
        return fig

    data = {k: [row.get(k) for row in rows] for k in rows[0].keys()}
    ts = data.get("timestamp", [])

    # Map triggered_evidence.id values to their signal column keys
    _ID_TO_SIGNAL = {
        "pip_declining": "pump_intake_pressure_psi",
        "gor_increasing": "gor",
        "water_cut_increasing": "water_cut",
        "delta_p_volatility_increasing": "delta_p_pump_psi",
        "liquid_rate_declining_or_unstable": "liquid_rate_bbl_day",
        "motor_load_decreasing_with_instability": None,
    }
    triggered_evidence = screen_payload.get("triggered_evidence") or [] if isinstance(screen_payload, dict) else []
    triggered_signal_keys = {
        _ID_TO_SIGNAL[item["id"]]
        for item in triggered_evidence
        if item.get("id") in _ID_TO_SIGNAL and _ID_TO_SIGNAL[item["id"]] is not None
    }

    # Direction lookup from evidence_table: signal column key → direction string
    evidence_table = screen_payload.get("evidence_table") or [] if isinstance(screen_payload, dict) else []
    direction_by_signal = {
        row["signal"]: row.get("direction")
        for row in evidence_table
        if row.get("signal")
    }

    risk_label = screen_payload.get("risk_label") or "Insufficient data" if isinstance(screen_payload, dict) else "Insufficient data"
    n_triggered = len(triggered_evidence)
    annotations = []
    n_rendered = 0

    # Identity colors per signal — triggered signals keep their full color, passive signals are washed to gray.
    # This preserves signal identity while still communicating triggered vs passive through weight + opacity.
    _SIGNAL_COLOR = {
        "pump_intake_pressure_psi": "#1f77b4",   # blue
        "gor":                      "#d62728",   # red
        "water_cut":                "#17becf",   # cyan
        "liquid_rate_bbl_day":      "#2ca02c",   # green
        "delta_p_pump_psi":         "#9467bd",   # purple
    }
    _PASSIVE_COLOR = "rgba(180,180,180,0.40)"

    traces = [
        ("normalized_pump_intake_pressure_psi", "PIP",         "pump_intake_pressure_psi"),
        ("normalized_gor",                      "GOR",         "gor"),
        ("normalized_water_cut",                "Water Cut",   "water_cut"),
        ("normalized_liquid_rate_bbl_day",      "Liquid Rate", "liquid_rate_bbl_day"),
        ("normalized_delta_p_pump_psi",         "Pump Delta-P","delta_p_pump_psi"),
    ]
    for col_key, display_name, signal_key in traces:
        values = data.get(col_key)
        if values is None:
            continue
        if not any(_is_finite_number(v) for v in values):
            continue

        is_triggered = signal_key in triggered_signal_keys
        direction = direction_by_signal.get(signal_key)
        identity_color = _SIGNAL_COLOR.get(signal_key, "#888888")

        if is_triggered:
            color = identity_color
            width = 3.0
        else:
            color = _PASSIVE_COLOR
            width = 1.2

        fig.add_trace(
            go.Scatter(
                x=ts,
                y=values,
                mode="lines",
                name=display_name,
                line=dict(color=color, width=width),
                hovertemplate=f"{display_name}: %{{y:.3f}}<extra></extra>",
            )
        )
        n_rendered += 1

        # End-of-line direction arrow — only for triggered signals, larger size
        if is_triggered:
            arrow_map = {"increasing": "↗", "decreasing": "↘", "stable": "→"}
            arrow = arrow_map.get(direction)
            if arrow:
                last_x = last_y = None
                for i in range(len(values) - 1, -1, -1):
                    if _is_finite_number(values[i]):
                        last_x = ts[i] if i < len(ts) else None
                        last_y = float(values[i])
                        break
                if last_x is not None and last_y is not None:
                    annotations.append(
                        dict(
                            x=last_x,
                            y=last_y,
                            text=arrow,
                            showarrow=False,
                            font=dict(size=22, color=color),
                            xanchor="left",
                            xshift=8,
                        )
                    )

    if not fig.data:
        return build_gas_interference_trend_chart({}, title=title)

    # Risk badge — placed in chart margin above the plot area, legend moved below x-axis to avoid overlap
    _BADGE = {
        "Elevated": ("⚠ Elevated risk", "#C0392B"),
        "Watch":    ("⚠ Watch",         "#E67E22"),
        "Low":      ("✓ Low risk",       "#27AE60"),
    }
    badge_text, badge_color = _BADGE.get(risk_label, ("— Insufficient data", "#95A5A6"))
    badge_text += f"  |  {n_triggered}/{n_rendered} signals triggered  |  Screening prototype"

    annotations.append(
        dict(
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.0,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            text=badge_text,
            font=dict(size=12, color=badge_color),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=badge_color,
            borderwidth=1,
            borderpad=4,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Normalized value (0-1)",
        template="plotly_white",
        height=520,
        hovermode="x unified",
        margin=dict(t=100),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        annotations=annotations,
    )
    return fig
