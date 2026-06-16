"""Affinity Law panel chart for the ML Recommendation page, tab6."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _agreement_color(label: str) -> str:
    return {
        "Aligned": "#2ECC71",
        "Review": "#F39C12",
        "Divergent": "#E74C3C",
    }.get(label, "#95A5A6")


def _xref(i: int) -> str:
    return "x" if i == 1 else f"x{i}"


def _yref(i: int) -> str:
    return "y" if i == 1 else f"y{i}"


def _render_available_row(
    fig: go.Figure,
    row_idx: int,
    cur_val: float,
    pred_val: float,
    rec_val: float,
    diff_pct: float | None,
    agreement_label: str,
    unit: str,
    row_label: str,
    is_power: bool,
    power_source_label: str | None,
) -> None:
    xr = _xref(row_idx)
    yr = _yref(row_idx)
    color = _agreement_color(agreement_label)

    fig.add_trace(
        go.Scatter(
            x=[cur_val], y=[0],
            mode="markers",
            marker=dict(symbol="circle-open", color="#95A5A6", size=11),
            showlegend=False,
            hovertemplate=f"Current: {{x:.2f}}{' ' + unit if unit else ''}<extra></extra>",
        ),
        row=row_idx, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[pred_val], y=[0],
            mode="markers",
            marker=dict(symbol="diamond", color="#F39C12", size=11),
            showlegend=False,
            hovertemplate=f"Affinity predicted: {{x:.2f}}{' ' + unit if unit else ''}<extra></extra>",
        ),
        row=row_idx, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[rec_val], y=[0],
            mode="markers",
            marker=dict(symbol="circle", color=color, size=12),
            showlegend=False,
            hovertemplate=f"ML recommended: {{x:.2f}}{' ' + unit if unit else ''}<extra></extra>",
        ),
        row=row_idx, col=1,
    )

    fig.add_annotation(
        x=rec_val, y=0,
        ax=pred_val, ay=0,
        xref=xr, yref=yr,
        axref=xr, ayref=yr,
        arrowhead=2, arrowwidth=2, arrowcolor=color,
        showarrow=True, text="",
    )

    diff_text = f"{diff_pct:.1f}%  {agreement_label}" if diff_pct is not None else agreement_label
    fig.add_annotation(
        x=1.04, y=0,
        xref=f"{xr} domain", yref=yr,
        text=diff_text,
        font=dict(color=color, size=10),
        showarrow=False, xanchor="left",
    )

    if is_power and str(power_source_label or "").strip().lower() in {"bhp_proxy", "amp_x_volt"}:
        fig.add_annotation(
            x=-0.05, y=0,
            xref=f"{xr} domain", yref=yr,
            text="[proxy]",
            font=dict(color="#3498DB", size=9),
            showarrow=False, xanchor="right",
        )

    vals = [cur_val, pred_val, rec_val]
    span = max(vals) - min(vals) or abs(max(vals)) * 0.1 or 1.0
    fig.update_xaxes(
        range=[min(vals) - span * 0.15, max(vals) + span * 0.35],
        ticksuffix=f" {unit}" if unit else "",
        showgrid=False, zeroline=False,
        row=row_idx, col=1,
    )
    fig.update_yaxes(
        tickvals=[0], ticktext=[row_label],
        showgrid=False, zeroline=False,
        row=row_idx, col=1,
    )


def _render_unavailable_row(
    fig: go.Figure,
    row_idx: int,
    row_label: str,
    check: dict,
) -> None:
    xr = _xref(row_idx)
    yr = _yref(row_idx)
    reason = check.get("reason_unavailable") or "missing inputs"
    fig.add_annotation(
        x=0.5, y=0,
        xref=f"{xr} domain", yref=yr,
        text=f"Unavailable — {reason}",
        font=dict(color="#95A5A6", size=10),
        showarrow=False,
    )
    fig.update_xaxes(range=[0, 1], showticklabels=False, showgrid=False, row=row_idx, col=1)
    fig.update_yaxes(
        tickvals=[0], ticktext=[row_label],
        showgrid=False, zeroline=False,
        row=row_idx, col=1,
    )


def build_affinity_law_panel(diagnostic: dict) -> go.Figure:
    """Return a compact Plotly strip chart summarising the Affinity Law diagnostic."""
    if not diagnostic.get("available"):
        fig = go.Figure()
        fig.add_annotation(
            text="Affinity Law diagnostic unavailable",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="#95A5A6", size=13),
        )
        fig.update_layout(
            height=120,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_white",
            showlegend=False,
        )
        return fig

    flow_check = diagnostic.get("flow_check") or {}
    pressure_check = diagnostic.get("pressure_check") or {}
    power_check = diagnostic.get("power_check") or {}

    rows = [
        {
            "label": "Flow",
            "check": flow_check,
            "cur_key": "current_liquid_rate_bpd",
            "pred_key": "affinity_predicted_liquid_rate_bpd",
            "rec_key": "ml_recommended_liquid_rate_bpd",
            "unit": "bpd",
            "is_power": False,
        },
        {
            "label": "ΔP",
            "check": pressure_check,
            "cur_key": "current_delta_p_psi",
            "pred_key": "affinity_predicted_delta_p_psi",
            "rec_key": "ml_recommended_delta_p_psi",
            "unit": "psi",
            "is_power": False,
        },
        {
            "label": "Power",
            "check": power_check,
            "cur_key": "current_power",
            "pred_key": "affinity_predicted_power",
            "rec_key": "recommended_power",
            "unit": "",
            "is_power": True,
        },
    ]

    n_rows = 3
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
    )

    for i, row in enumerate(rows, start=1):
        check = row["check"]
        if check.get("available"):
            _render_available_row(
                fig=fig,
                row_idx=i,
                cur_val=check[row["cur_key"]],
                pred_val=check[row["pred_key"]],
                rec_val=check[row["rec_key"]],
                diff_pct=check.get("difference_pct"),
                agreement_label=check.get("agreement_label", "Unavailable"),
                unit=row["unit"],
                row_label=row["label"],
                is_power=row["is_power"],
                power_source_label=check.get("power_source_label"),
            )
        else:
            _render_unavailable_row(fig=fig, row_idx=i, row_label=row["label"], check=check)

    speed_ratio = diagnostic.get("speed_ratio") or 0.0
    freq_delta = diagnostic.get("frequency_delta_hz") or 0.0
    mode = diagnostic.get("mode") or ""
    trust = diagnostic.get("trust_label") or ""

    fig.add_annotation(
        x=0, y=1.08,
        xref="paper", yref="paper",
        text=(
            f"Speed ratio: {speed_ratio:.3f}×  |  "
            f"Δf: {freq_delta:+.1f} Hz  |  "
            f"Mode: {mode}  |  Trust: {trust}"
        ),
        font=dict(color="#2C3E50", size=12),
        showarrow=False, xanchor="left",
    )

    if diagnostic.get("frequency_change_label") == "Minimal frequency change":
        fig.add_annotation(
            x=0, y=1.02,
            xref="paper", yref="paper",
            text="⚠ Minimal frequency change — interpret with caution",
            font=dict(color="#F39C12", size=11),
            showarrow=False, xanchor="left",
        )

    fig.update_layout(
        height=120 + 80 * n_rows,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        showlegend=False,
        margin=dict(t=60, r=120, b=20, l=60),
    )
    return fig
