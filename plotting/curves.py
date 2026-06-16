"""Plotting builders for temporal analysis and pump-curve visualizations."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compute.core_calcs import (
    SG_OIL,
    SG_WATER,
    WELL_DEPTH_FT,
    detect_amp_column,
    detect_frequency_column,
    detect_volt_column,
    engineer_features,
)


MIN_POINTS_PER_FREQ = 20


def build_dual_axis_time_plot(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    title: str,
    left_label: str,
    right_label: str,
) -> go.Figure:
    """Build a dual-axis temporal plot for two columns against timestamp."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[left_col],
            name=left_col,
            mode="lines",
            line=dict(color="#1f77b4", width=2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[right_col],
            name=right_col,
            mode="lines",
            line=dict(color="#d62728", width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=left_label, secondary_y=False)
    fig.update_yaxes(title_text=right_label, secondary_y=True)
    fig.update_layout(
        title=title,
        width=1400,
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def generate_temporal_plots_after_merge(
    merged_data: pd.DataFrame,
    well_depth_ft: float = WELL_DEPTH_FT,
    sg_oil: float = SG_OIL,
    sg_water: float = SG_WATER,
    resample_hours: int = 6,
) -> Tuple[List[Dict], List[str]]:
    """Generate temporal dual-axis plots immediately after merge (before cleaning)."""
    plot_df = merged_data.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce")
    plot_df = plot_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    plot_df = engineer_features(
        plot_df,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
    )

    freq_col = detect_frequency_column(plot_df)
    amp_col = detect_amp_column(plot_df)
    volt_col = detect_volt_column(plot_df)

    if resample_hours and resample_hours > 0:
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
        plot_df = (
            plot_df[["timestamp"] + numeric_cols]
            .set_index("timestamp")
            .resample(f"{resample_hours}h")
            .median()
            .reset_index()
        )

    temporal_plots = []
    notes = []

    requested_specs = [
        (
            "liquid_rate_bbl_day",
            freq_col,
            "Temporal: Liquid Rate and Frequency",
            "Liquid Rate (bbl/day)",
            "Frequency (Hz)",
        ),
        (
            "water_cut",
            freq_col,
            "Temporal: Water Cut and Frequency",
            "Water Cut (fraction)",
            "Frequency (Hz)",
        ),
        (
            "delta_P_pump_psi",
            freq_col,
            "Temporal: ΔP Pump and Frequency",
            "ΔP Pump (psi)",
            "Frequency (Hz)",
        ),
        (amp_col, freq_col, "Temporal: Amp and Frequency", "Current (A)", "Frequency (Hz)"),
        (volt_col, freq_col, "Temporal: Voltage and Frequency", "Voltage (V)", "Frequency (Hz)"),
        (
            "delta_P_pump_psi",
            "amp_x_volt",
            "Temporal: ΔP Pump and Amp×Volt",
            "ΔP Pump (psi)",
            "Amp×Volt",
        ),
    ]

    for left_col, right_col, title, left_label, right_label in requested_specs:
        if not left_col or not right_col:
            notes.append(f"Skipped '{title}' (missing required column detection).")
            continue
        if left_col not in plot_df.columns or right_col not in plot_df.columns:
            notes.append(f"Skipped '{title}' (missing columns: {left_col}, {right_col}).")
            continue

        fig = build_dual_axis_time_plot(
            plot_df,
            left_col=left_col,
            right_col=right_col,
            title=title,
            left_label=left_label,
            right_label=right_label,
        )
        temporal_plots.append({"title": title, "figure": fig})

    if not freq_col:
        notes.append("Frequency column was not detected in merged data.")
    if not amp_col:
        notes.append("Amp/current column was not detected in merged data.")
    if not volt_col:
        notes.append("Voltage column was not detected in merged data.")

    return temporal_plots, notes


def generate_pump_curves(outlier_df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Generate pump-curve subplots for top segments by row count."""
    plot_df = outlier_df.copy()

    req = ["segment", "motor_frequency_hz", "delta_P_pump_psi"]
    for c in req:
        if c not in plot_df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "liquid_rate_bbl_day" not in plot_df.columns:
        plot_df["liquid_rate_bbl_day"] = (
            pd.to_numeric(plot_df["oil_flow_rate"], errors="coerce").fillna(0)
            + pd.to_numeric(plot_df["water_flow_rate"], errors="coerce").fillna(0)
        )

    plot_df = plot_df[
        plot_df["segment"].notna()
        & plot_df["motor_frequency_hz"].notna()
        & plot_df["delta_P_pump_psi"].notna()
        & plot_df["liquid_rate_bbl_day"].notna()
        & (plot_df["liquid_rate_bbl_day"] > 0)
    ].copy()

    plot_df["motor_frequency_hz"] = pd.to_numeric(plot_df["motor_frequency_hz"], errors="coerce")
    top_segments = plot_df["segment"].value_counts().head(top_n).index.tolist()

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[f"{s}" for s in top_segments] + [""],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    for idx, seg in enumerate(top_segments):
        r = idx // 2 + 1
        c = idx % 2 + 1
        seg_data = plot_df[plot_df["segment"] == seg]

        for f in sorted(seg_data["motor_frequency_hz"].dropna().unique()):
            d = seg_data[seg_data["motor_frequency_hz"] == f].sort_values("liquid_rate_bbl_day")
            if len(d) < MIN_POINTS_PER_FREQ:
                continue

            fig.add_trace(
                go.Scattergl(
                    x=d["liquid_rate_bbl_day"],
                    y=d["delta_P_pump_psi"],
                    mode="markers",
                    name=f"{f:g} Hz",
                    marker=dict(size=4, opacity=0.35),
                    legendgroup=f"{f:g}",
                    showlegend=(idx == 0),
                ),
                row=r,
                col=c,
            )

        fig.update_xaxes(title_text="Liquid rate (bbl/day)", row=r, col=c)
        fig.update_yaxes(title_text="ΔP_pump (psi)", row=r, col=c)

    fig.update_layout(
        height=1100,
        width=1300,
        title="Top 5 Segments: ΔP_pump vs Liquid Rate (After Outlier Treatment)",
        hovermode="closest",
    )

    return fig
