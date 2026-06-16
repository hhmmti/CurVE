"""Horizontal strip charts for bubble-point and NPSH pressure/head comparisons."""

import plotly.graph_objects as go

_COLOR_SAFE = "#2ECC71"
_COLOR_WATCH = "#F39C12"
_COLOR_RISK = "#E74C3C"
_COLOR_OPERATING = "#2C3E50"
_COLOR_TRUST = "#95A5A6"

_FILL_GREEN = "rgba(46,204,113,0.12)"
_FILL_AMBER = "rgba(243,156,18,0.15)"
_FILL_AMBER_LIGHT = "rgba(243,156,18,0.12)"
_FILL_RED = "rgba(231,76,60,0.10)"
_FILL_RED_STRONG = "rgba(231,76,60,0.14)"

_LAYOUT_BASE = dict(
    height=160,
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    margin=dict(l=10, r=10, t=40, b=40),
    yaxis=dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[-0.5, 0.5],
    ),
)


def _unavailable_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=13, color=_COLOR_TRUST),
    )
    fig.update_layout(**_LAYOUT_BASE)
    return fig


def _rect(x0, x1, fill_color: str) -> dict:
    return dict(
        type="rect",
        xref="x", yref="paper",
        x0=x0, x1=x1,
        y0=0, y1=1,
        fillcolor=fill_color,
        line_width=0,
        layer="below",
    )


def build_bubble_point_strip(bpp_diag: dict) -> go.Figure:
    """Horizontal pressure strip: PIP vs bubble-point P_b."""
    if not bpp_diag:
        return _unavailable_fig("Bubble-point comparison unavailable")

    pb = bpp_diag.get("selected_bubble_point_psi")
    pip = bpp_diag.get("pressure_reference_psi")
    margin_psi = bpp_diag.get("margin_psi")
    margin_pct = bpp_diag.get("margin_pct")
    status = bpp_diag.get("status_label", "")
    trust = bpp_diag.get("trust_label", "Research prototype")

    if bpp_diag.get("available") is False or any(v is None for v in (pb, pip, margin_psi, margin_pct)):
        return _unavailable_fig("Bubble-point comparison unavailable")

    x_min = min(pb, pip) * 0.85
    x_max = max(pb, pip) * 1.20

    # Zone shapes
    shapes = []
    if status == "Above bubble point":
        shapes.append(_rect(x_min, pb, _FILL_RED))
        shapes.append(_rect(pb, x_max, _FILL_GREEN))
    elif status == "Near bubble point":
        shapes.append(_rect(x_min, x_max, _FILL_AMBER))
    else:  # Below bubble point
        shapes.append(_rect(x_min, pb, _FILL_RED_STRONG))
        shapes.append(_rect(pb, pb + pb * 0.05, _FILL_AMBER_LIGHT))

    # Threshold / operating point colors
    if status == "Above bubble point":
        pb_color = _COLOR_SAFE
        arrow_color = _COLOR_SAFE
    elif status == "Near bubble point":
        pb_color = _COLOR_WATCH
        arrow_color = _COLOR_WATCH
    else:
        pb_color = _COLOR_RISK
        arrow_color = _COLOR_RISK

    fig = go.Figure()

    # P_b threshold marker
    fig.add_trace(go.Scatter(
        x=[pb], y=[0],
        mode="markers+text",
        marker=dict(symbol="triangle-up", size=16, color=pb_color),
        text=[f"P_b<br>{pb:.0f} psi"],
        textposition="top center",
        textfont=dict(size=11),
        showlegend=False,
    ))

    # PIP operating point marker
    fig.add_trace(go.Scatter(
        x=[pip], y=[0],
        mode="markers+text",
        marker=dict(symbol="circle", size=13, color=_COLOR_OPERATING),
        text=[f"PIP<br>{pip:.0f} psi"],
        textposition="bottom center",
        textfont=dict(size=11),
        showlegend=False,
    ))

    # Arrow bracket: P_b → PIP
    mid_x = (pb + pip) / 2
    annotations = [
        dict(
            ax=pb, ay=0,
            x=pip, y=0,
            axref="x", ayref="y",
            xref="x", yref="y",
            arrowhead=2,
            arrowcolor=arrow_color,
            arrowwidth=2,
        ),
        dict(
            x=mid_x, y=0.22,
            xref="x", yref="paper",
            text=f"{margin_psi:+.1f} psi ({margin_pct:+.1f}%)",
            showarrow=False,
            font=dict(size=11, color=arrow_color),
        ),
        dict(
            x=0.99, y=1.10,
            xref="paper", yref="paper",
            text=trust,
            showarrow=False,
            font=dict(size=10, color=_COLOR_TRUST),
            xanchor="right",
        ),
    ]

    fig.update_layout(
        **_LAYOUT_BASE,
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(title="Pressure (psi)", range=[x_min, x_max]),
    )
    return fig


def build_npsh_strip(npsh_diag: dict) -> go.Figure:
    """Horizontal head strip: NPSHa vs NPSHr."""
    if not npsh_diag:
        return _unavailable_fig("NPSH comparison unavailable")

    npsha = npsh_diag.get("npsha_ft")
    npshr = npsh_diag.get("npshr_ft")
    margin = npsh_diag.get("margin_ft")
    status = npsh_diag.get("status_label", "")
    trust = npsh_diag.get("trust_label", "Proxy / Research prototype")

    if npsh_diag.get("available") is False or any(v is None for v in (npsha, npshr, margin)):
        return _unavailable_fig("NPSH comparison unavailable")

    x_max = max(npsha, npshr) * 1.35

    # Zone shapes
    shapes = []
    if status == "Safe":
        shapes.append(_rect(0, npshr, _FILL_RED))
        shapes.append(_rect(npshr, x_max, _FILL_GREEN))
    elif status == "Watch":
        shapes.append(_rect(0, npshr, _FILL_RED))
        shapes.append(_rect(npshr, npshr + 5, _FILL_AMBER))
        shapes.append(_rect(npshr + 5, x_max, _FILL_GREEN))
    else:  # Risk
        shapes.append(_rect(0, npshr, _FILL_RED_STRONG))
        shapes.append(_rect(npshr, x_max, _FILL_GREEN))

    # NPSHr marker color
    if status == "Safe":
        npshr_color = _COLOR_SAFE
        arrow_color = _COLOR_SAFE
    else:
        npshr_color = _COLOR_RISK
        arrow_color = _COLOR_WATCH if status == "Watch" else _COLOR_RISK

    # NPSHr label — append placeholder note if proxy/placeholder
    npshr_label = f"NPSHr<br>{npshr:.1f} ft"
    if "proxy" in trust.lower() or "placeholder" in trust.lower():
        npshr_label += " [placeholder]"

    fig = go.Figure()

    # NPSHr threshold marker
    fig.add_trace(go.Scatter(
        x=[npshr], y=[0],
        mode="markers+text",
        marker=dict(symbol="triangle-up", size=16, color=npshr_color),
        text=[npshr_label],
        textposition="top center",
        textfont=dict(size=11),
        showlegend=False,
    ))

    # NPSHa operating point marker
    fig.add_trace(go.Scatter(
        x=[npsha], y=[0],
        mode="markers+text",
        marker=dict(symbol="circle", size=13, color=_COLOR_OPERATING),
        text=[f"NPSHa<br>{npsha:.1f} ft"],
        textposition="bottom center",
        textfont=dict(size=11),
        showlegend=False,
    ))

    mid_x = (npshr + npsha) / 2
    annotations = [
        dict(
            ax=npshr, ay=0,
            x=npsha, y=0,
            axref="x", ayref="y",
            xref="x", yref="y",
            arrowhead=2,
            arrowcolor=arrow_color,
            arrowwidth=2,
        ),
        dict(
            x=mid_x, y=0.22,
            xref="x", yref="paper",
            text=f"{margin:+.1f} ft",
            showarrow=False,
            font=dict(size=11, color=arrow_color),
        ),
        dict(
            x=0.99, y=1.10,
            xref="paper", yref="paper",
            text=trust,
            showarrow=False,
            font=dict(size=10, color=_COLOR_TRUST),
            xanchor="right",
        ),
    ]

    # Secondary proxy note
    if "proxy" in trust.lower():
        annotations.append(dict(
            x=0.99, y=0.98,
            xref="paper", yref="paper",
            text="NPSHa may be understated (P_b proxy used as vapor pressure)",
            showarrow=False,
            font=dict(size=9, color="#F39C12"),
            xanchor="right",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(title="Head (ft)", range=[0, x_max]),
    )
    return fig
