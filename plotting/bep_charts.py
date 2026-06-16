"""BEP / Operating Range visualization charts."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# --- Color palette ---
_GREEN = "#2ECC71"
_GREEN_DARK = "#27AE60"
_AMBER = "#F39C12"
_RED = "#E74C3C"
_GRAY = "#95A5A6"
_SLATE = "#2C3E50"
_BLUE = "#3498DB"


def _is_finite(value) -> bool:
    try:
        return value is not None and np.isfinite(float(value))
    except Exception:
        return False


def _movement_color(movement_label: str) -> str:
    lbl = (movement_label or "").lower()
    if "closer" in lbl:
        return _GREEN
    if "farther" in lbl:
        return _RED
    return _GRAY


def build_bep_curve_chart(
    diagnostic: dict,
    rec_curve: Optional[pd.DataFrame],
    cur_curve: Optional[pd.DataFrame],
    compare_row: dict,
    summary_json: dict,
    family_curves: Dict[float, pd.DataFrame],
    tornado: bool = False,
) -> go.Figure:
    """Layer 1 — ΔP-Flow curve with operating points and zone bands."""
    fig = go.Figure()

    bep_bpd = diagnostic.get("bep_bpd")
    min_bpd = diagnostic.get("min_recommended_bpd")
    max_bpd = diagnostic.get("max_recommended_bpd")
    current = diagnostic.get("current") or {}
    recommended = diagnostic.get("recommended") or {}
    movement = diagnostic.get("movement") or {}
    movement_label = movement.get("movement_label") or ""
    movement_color = _movement_color(movement_label)
    trust_label = diagnostic.get("trust_label")

    cur_hz = compare_row.get("cur_motor_frequency_hz")
    rec_hz = compare_row.get("rec_motor_frequency_hz")
    cur_liquid_rate = compare_row.get("cur_liquid_rate_bpd")
    rec_liquid_rate = compare_row.get("rec_liquid_rate_bpd")
    cur_delta_p = compare_row.get("cur_delta_p_pump_psi")
    rec_delta_p = compare_row.get("rec_delta_p_pump_psi")

    # --- Zone bands ---
    if _is_finite(min_bpd) and _is_finite(max_bpd):
        fig.add_shape(
            type="rect",
            x0=min_bpd, x1=max_bpd, y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor="rgba(46,204,113,0.10)", line_width=0, layer="below",
        )
    if _is_finite(bep_bpd):
        fig.add_shape(
            type="rect",
            x0=bep_bpd * 0.90, x1=bep_bpd * 1.10, y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor="rgba(39,174,96,0.15)", line_width=0, layer="below",
        )

    # --- Tornado curves ---
    if tornado and family_curves:
        tornado_dfs = {
            hz: df for hz, df in family_curves.items()
            if hz not in (rec_hz, cur_hz) and df is not None and not df.empty
        }
        if tornado_dfs:
            gray_shades = ["#BBBBBB", "#AAAAAA", "#999999", "#888888", "#777777"]
            for i, (hz, df) in enumerate(sorted(tornado_dfs.items())):
                fig.add_trace(go.Scatter(
                    x=df["flow_bpd"], y=df["delta_p_psi"],
                    mode="lines",
                    line=dict(color=gray_shades[i % len(gray_shades)], width=1, dash="dash"),
                    name=f"f = {hz:.0f} Hz",
                    opacity=0.5,
                    showlegend=True,
                ))

            # Shaded envelope across tornado curves
            all_dfs = list(tornado_dfs.values())
            all_flows = np.sort(np.unique(np.concatenate([df["flow_bpd"].values for df in all_dfs])))
            interp_matrix = np.array([
                np.interp(all_flows, df["flow_bpd"], df["delta_p_psi"],
                          left=np.nan, right=np.nan)
                for df in all_dfs
            ])
            min_dp = np.nanmin(interp_matrix, axis=0)
            max_dp = np.nanmax(interp_matrix, axis=0)
            fig.add_trace(go.Scatter(
                x=all_flows, y=min_dp,
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=all_flows, y=max_dp,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(149,165,166,0.10)",
                showlegend=False, hoverinfo="skip",
            ))

    # --- Ghost curve (current Hz) ---
    if cur_curve is not None and not cur_curve.empty:
        hz_label = f"{cur_hz:.0f}" if _is_finite(cur_hz) else "?"
        fig.add_trace(go.Scatter(
            x=cur_curve["flow_bpd"], y=cur_curve["delta_p_psi"],
            mode="lines",
            line=dict(color=_GRAY, width=1.5, dash="dot"),
            name=f"f = {hz_label} Hz",
        ))

    # --- Main curve (recommended Hz) ---
    if rec_curve is not None and not rec_curve.empty:
        hz_label = f"{rec_hz:.0f}" if _is_finite(rec_hz) else "?"
        fig.add_trace(go.Scatter(
            x=rec_curve["flow_bpd"], y=rec_curve["delta_p_psi"],
            mode="lines",
            line=dict(color=_SLATE, width=2),
            name=f"f = {hz_label} Hz",
        ))

    # --- BEP marker ---
    if _is_finite(bep_bpd) and rec_curve is not None and not rec_curve.empty:
        delta_p_at_bep = float(np.interp(bep_bpd, rec_curve["flow_bpd"], rec_curve["delta_p_psi"]))
        fig.add_trace(go.Scatter(
            x=[bep_bpd], y=[delta_p_at_bep],
            mode="markers",
            marker=dict(symbol="star", size=14, color=_AMBER),
            name="BEP",
            hovertemplate=f"BEP: {bep_bpd:.0f} bbl/d<extra></extra>",
        ))

    # --- Current operating point ---
    if _is_finite(cur_liquid_rate) and _is_finite(cur_delta_p):
        oil_std = summary_json.get("alloc_oil_vol_1d_std")
        wat_std = summary_json.get("alloc_water_vol_1d_std")
        error_x = None
        try:
            total_std = float(oil_std or 0) + float(wat_std or 0)
            if total_std > 0:
                error_x = dict(type="data", array=[total_std], visible=True)
        except (TypeError, ValueError):
            pass

        fig.add_trace(go.Scatter(
            x=[cur_liquid_rate], y=[cur_delta_p],
            mode="markers",
            marker=dict(symbol="circle-open", size=12, color=_GRAY),
            error_x=error_x,
            name="Current",
            hovertemplate=(
                f"Current: {cur_liquid_rate:.0f} bbl/d | {cur_delta_p:.0f} psi<extra></extra>"
            ),
        ))

        if trust_label:
            trust_color = _AMBER if trust_label == "Estimated" else _BLUE
            fig.add_annotation(
                x=cur_liquid_rate, y=cur_delta_p,
                text=trust_label, showarrow=False,
                font=dict(size=9, color=trust_color),
                yshift=14,
            )

    # --- Recommended operating point ---
    if _is_finite(rec_liquid_rate) and _is_finite(rec_delta_p):
        rec_color = _GREEN if "closer" in movement_label.lower() else _RED
        fig.add_trace(go.Scatter(
            x=[rec_liquid_rate], y=[rec_delta_p],
            mode="markers",
            marker=dict(symbol="circle", size=12, color=rec_color),
            name="Recommended",
            hovertemplate=(
                f"Recommended: {rec_liquid_rate:.0f} bbl/d | {rec_delta_p:.0f} psi<extra></extra>"
            ),
        ))

    # --- Movement arrow ---
    if (
        _is_finite(cur_liquid_rate) and _is_finite(cur_delta_p)
        and _is_finite(rec_liquid_rate) and _is_finite(rec_delta_p)
    ):
        fig.add_annotation(
            x=rec_liquid_rate, y=rec_delta_p,
            ax=cur_liquid_rate, ay=cur_delta_p,
            axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor=movement_color,
            showarrow=True, text="",
        )

    # --- Data freshness annotation ---
    tele_h = summary_json.get("telemetry_hours_since_last_datapoint", "?")
    alloc_h = summary_json.get("telemetry_allocation_hours_difference", "?")
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=0,
        text=f"Telemetry: {tele_h}h ago  |  Allocation: {alloc_h}h ago",
        showarrow=False,
        font=dict(size=10, color=_GRAY),
        xanchor="left", yanchor="bottom",
    )

    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=1, y=1, xanchor="right", yanchor="top"),
        margin=dict(t=20, b=40, l=60, r=20),
        xaxis_title="Flow (bbl/d)",
        yaxis_title="ΔP Pump (psi)",
    )
    return fig


def build_bep_health_strip(diagnostic: dict) -> go.Figure:
    """Layer 2 — Compact %BEP health strip."""
    fig = go.Figure()

    current = diagnostic.get("current") or {}
    recommended = diagnostic.get("recommended") or {}
    movement = diagnostic.get("movement") or {}
    mov_label = movement.get("movement_label") or ""
    movement_color = _movement_color(mov_label)

    cur_dist = current.get("distance_from_bep_pct")
    rec_dist = recommended.get("distance_from_bep_pct")
    cur_bep_label = current.get("bep_position_label") or ""
    rec_bep_label = recommended.get("bep_position_label") or ""
    delta = movement.get("delta_abs_distance_pct") or 0.0

    # --- Zone background shapes ---
    zone_specs = [
        (-50, -25, "rgba(231,76,60,0.05)"),
        (-25, -10, "rgba(243,156,18,0.10)"),
        (-10,  10, "rgba(46,204,113,0.15)"),
        ( 10,  25, "rgba(243,156,18,0.10)"),
        ( 25,  50, "rgba(231,76,60,0.05)"),
    ]
    for x0, x1, color in zone_specs:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor=color, line_width=0, layer="below",
        )

    # --- BEP tick at x=0 ---
    fig.add_shape(
        type="line", x0=0, x1=0, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=_AMBER, dash="dash", width=1),
    )

    # --- Markers ---
    if _is_finite(cur_dist):
        fig.add_trace(go.Scatter(
            x=[cur_dist], y=[0],
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color=_GRAY),
            showlegend=False,
            hovertemplate=f"Current: {cur_dist:+.1f}%<extra></extra>",
        ))
        fig.add_annotation(
            x=cur_dist, y=-0.35, xref="x", yref="paper",
            text=f"{cur_dist:+.1f}%<br>{cur_bep_label}",
            showarrow=False,
            font=dict(size=9, color=_GRAY),
            yanchor="top",
        )

    if _is_finite(rec_dist):
        fig.add_trace(go.Scatter(
            x=[rec_dist], y=[0],
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color=movement_color),
            showlegend=False,
            hovertemplate=f"Recommended: {rec_dist:+.1f}%<extra></extra>",
        ))
        fig.add_annotation(
            x=rec_dist, y=-0.35, xref="x", yref="paper",
            text=f"{rec_dist:+.1f}%<br>{rec_bep_label}",
            showarrow=False,
            font=dict(size=9, color=movement_color),
            yanchor="top",
        )

    # --- Δ bracket ---
    if _is_finite(cur_dist) and _is_finite(rec_dist):
        fig.add_shape(
            type="line",
            x0=cur_dist, x1=rec_dist, y0=0.15, y1=0.15,
            xref="x", yref="paper",
            line=dict(color=movement_color, width=1.5),
        )
        for x in [cur_dist, rec_dist]:
            fig.add_shape(
                type="line", x0=x, x1=x, y0=0.10, y1=0.20,
                xref="x", yref="paper",
                line=dict(color=movement_color, width=1.5),
            )
        midpoint = (float(cur_dist) + float(rec_dist)) / 2
        fig.add_annotation(
            x=midpoint, y=0.25, xref="x", yref="paper",
            text=f"Δ = {delta:+.1f}%",
            showarrow=False,
            font=dict(size=9, color=movement_color),
        )

    # --- Right-side verdict ---
    if "closer" in mov_label.lower():
        verdict, verdict_color = "Closer ✓", _GREEN
    elif "farther" in mov_label.lower():
        verdict, verdict_color = "Farther ✗", _RED
    else:
        verdict, verdict_color = "Same →", _GRAY

    fig.add_annotation(
        xref="paper", yref="paper", x=1.0, y=0.5,
        text=f"<b>{verdict}</b>",
        showarrow=False,
        font=dict(size=14, color=verdict_color),
        xanchor="right", yanchor="middle",
    )

    fig.update_layout(
        height=110,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[-50, 50],
            title="% distance from BEP",
            tickvals=[-25, -10, 0, 10, 25],
            showgrid=True,
            gridcolor="rgba(150,150,150,0.2)",
        ),
        yaxis=dict(visible=False, range=[-1, 1]),
        showlegend=False,
        margin=dict(t=10, b=40, l=60, r=80),
    )
    return fig
