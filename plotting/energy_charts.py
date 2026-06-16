"""Energy / Efficiency power-balance card chart for tab7."""

import math
from typing import Optional

import plotly.graph_objects as go

_COLOR_HYDRAULIC_DIRECT = "rgba(44,62,80,0.85)"
_COLOR_HYDRAULIC_PROXY = "rgba(52,152,219,0.55)"
_COLOR_LOSSES = "rgba(149,165,166,0.25)"
_COLOR_IMPROVEMENT = "#2ECC71"
_COLOR_WORSE = "#E74C3C"
_COLOR_NEUTRAL = "#95A5A6"
_COLOR_PROXY_LABEL = "#3498DB"
_COLOR_DIRECT_LABEL = "#2C3E50"
_COLOR_UNAVAILABLE = "#95A5A6"
_COLOR_AMBER = "#F39C12"


def _is_finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _resolve_input_power(diag: dict) -> tuple[Optional[float], str]:
    """Return (input_power_kw, source_label) or (None, '')."""
    motor = diag.get("motor_power_kw")
    proxy = diag.get("proxy_power_kw")
    if _is_finite(motor):
        return float(motor), "direct"
    if _is_finite(proxy):
        return float(proxy), "proxy"
    return None, ""


def _trust_color(trust_label: str) -> str:
    tl = (trust_label or "").lower()
    if "direct" in tl:
        return _COLOR_DIRECT_LABEL
    if "proxy" in tl:
        return _COLOR_PROXY_LABEL
    return _COLOR_AMBER


def build_energy_power_cards(current_diag: dict, rec_diag: dict) -> go.Figure:
    """Two-column horizontal power bar figure: current (left) | recommended (right)."""

    cur_avail = bool(current_diag.get("available"))
    rec_avail = bool(rec_diag.get("available"))

    # Both unavailable — return minimal placeholder
    if not cur_avail and not rec_avail:
        fig = go.Figure()
        fig.add_annotation(
            text="Energy / Efficiency diagnostic unavailable",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=13, color=_COLOR_UNAVAILABLE),
        )
        fig.update_layout(
            height=200,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        return fig

    # Resolve input power for both states
    cur_input_kw, cur_source = _resolve_input_power(current_diag)
    rec_input_kw, rec_source = _resolve_input_power(rec_diag)

    cur_hydraulic_kw = current_diag.get("hydraulic_kw_estimate") if cur_avail else None
    rec_hydraulic_kw = rec_diag.get("hydraulic_kw_estimate") if rec_avail else None

    # Shared x-axis range
    candidates = [v for v in [cur_input_kw, rec_input_kw] if _is_finite(v)]
    x_max = max(candidates) * 1.15 if candidates else 1.0

    fig = go.Figure()

    # --- Left column (current) bars at y=1, right column (recommended) at y=0 ---
    # We use two separate y categories to stack the bars side by side in a single plot.
    # y-axis: "Current" and "Recommended" as category labels.

    def _add_state_bars(diag: dict, input_kw: Optional[float], hydraulic_kw, source: str, y_label: str):
        avail = bool(diag.get("available"))

        if not avail or input_kw is None:
            reason = diag.get("reason_unavailable") or "No power data"
            fig.add_trace(go.Bar(
                x=[x_max * 0.6],
                y=[y_label],
                orientation="h",
                marker_color=_COLOR_UNAVAILABLE,
                opacity=0.3,
                text=f"Unavailable — {reason}",
                textposition="inside",
                insidetextanchor="middle",
                showlegend=False,
                hoverinfo="skip",
            ))
            return

        hyd = float(hydraulic_kw) if _is_finite(hydraulic_kw) else 0.0
        losses = max(input_kw - hyd, 0.0)

        hyd_color = _COLOR_HYDRAULIC_DIRECT if source == "direct" else _COLOR_HYDRAULIC_PROXY

        # Hydraulic bar
        bar_kwargs = dict(marker_color=hyd_color, showlegend=False, hoverinfo="skip")
        if source == "proxy":
            bar_kwargs["marker"] = dict(
                color=_COLOR_HYDRAULIC_PROXY,
                pattern=dict(shape="/", fgcolor="rgba(52,152,219,0.8)", bgcolor=_COLOR_HYDRAULIC_PROXY),
            )

        fig.add_trace(go.Bar(
            x=[hyd],
            y=[y_label],
            orientation="h",
            name="Hydraulic",
            **bar_kwargs,
        ))

        # Losses bar
        fig.add_trace(go.Bar(
            x=[losses],
            y=[y_label],
            orientation="h",
            marker_color=_COLOR_LOSSES,
            name="Losses",
            showlegend=False,
            hoverinfo="skip",
        ))

        # Efficiency annotation
        if source == "direct":
            eff = diag.get("direct_power_efficiency_pct")
        else:
            eff = diag.get("proxy_power_efficiency_pct")

        eff_text = f"{eff:.0f}%" if _is_finite(eff) else "—"
        proxy_tag = " [proxy]" if source == "proxy" else ""
        ann_text = f"<b>{eff_text}</b><span style='color:{_COLOR_PROXY_LABEL};font-size:11px'>{proxy_tag}</span>"

        fig.add_annotation(
            x=input_kw / 2,
            y=y_label,
            text=ann_text,
            showarrow=False,
            font=dict(size=16, color="white" if source == "direct" else _COLOR_DIRECT_LABEL),
            xanchor="center",
            yanchor="middle",
        )

    _add_state_bars(current_diag, cur_input_kw, cur_hydraulic_kw, cur_source, "Current")
    _add_state_bars(rec_diag, rec_input_kw, rec_hydraulic_kw, rec_source, "Recommended")

    # --- Specific power delta annotation ---
    cur_sp = current_diag.get("specific_power_kwh_per_liquid_bbl") if cur_avail else None
    rec_sp = rec_diag.get("specific_power_kwh_per_liquid_bbl") if rec_avail else None

    if _is_finite(cur_sp) and _is_finite(rec_sp):
        cur_sp, rec_sp = float(cur_sp), float(rec_sp)
        delta = rec_sp - cur_sp
        delta_pct = (delta / cur_sp * 100) if cur_sp != 0 else 0.0
        if delta_pct < -5:
            delta_color = _COLOR_IMPROVEMENT
        elif delta_pct > 5:
            delta_color = _COLOR_WORSE
        else:
            delta_color = _COLOR_NEUTRAL

        sp_text = (
            f"Specific power: {cur_sp:.2f} → {rec_sp:.2f} kWh/bbl liquid  "
            f"<span style='color:{delta_color}'>{delta:+.2f} kWh/bbl ({delta_pct:+.0f}%)</span>"
        )
    elif _is_finite(cur_sp):
        sp_text = f"Specific power: {float(cur_sp):.2f} kWh/bbl (current) → N/A"
    elif _is_finite(rec_sp):
        sp_text = f"Specific power: N/A → {float(rec_sp):.2f} kWh/bbl (recommended)"
    else:
        sp_text = "<span style='color:#95A5A6'>Specific power: N/A</span>"

    # Column header annotations
    cur_trust = current_diag.get("trust_label") or ""
    rec_trust = rec_diag.get("trust_label") or ""
    cur_hz = current_diag.get("motor_frequency_hz") or ""
    rec_hz = rec_diag.get("motor_frequency_hz") or ""

    def _hz_suffix(hz):
        return f"  f={hz:.0f} Hz" if _is_finite(hz) else ""

    fig.update_layout(
        barmode="stack",
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=60),
        xaxis=dict(
            title="kW",
            range=[0, x_max],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=["Recommended", "Current"],
        ),
        annotations=[
            # Column-style header labels via title annotations at top
            dict(
                text=f"<b>Current</b>{_hz_suffix(cur_hz)}"
                     + (f"  <span style='color:{_trust_color(cur_trust)}'>[{cur_trust}]</span>" if cur_trust else ""),
                x=0.0, y=1.22, xref="paper", yref="paper",
                showarrow=False, xanchor="left",
                font=dict(size=12),
            ),
            dict(
                text=f"<b>Recommended</b>{_hz_suffix(rec_hz)}"
                     + (f"  <span style='color:{_trust_color(rec_trust)}'>[{rec_trust}]</span>" if rec_trust else ""),
                x=0.5, y=1.22, xref="paper", yref="paper",
                showarrow=False, xanchor="left",
                font=dict(size=12),
            ),
            # Specific power delta row at bottom
            dict(
                text=sp_text,
                x=0.5, y=-0.28, xref="paper", yref="paper",
                showarrow=False, xanchor="center",
                font=dict(size=11),
            ),
        ],
    )

    return fig
