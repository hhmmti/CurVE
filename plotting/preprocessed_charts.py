"""Plotting layer for preprocessed analysis visualizations.

Builds Plotly charts for the 4 non-ideal analysis tabs following STEP16 Section G.

Tab structure:
- Tab 1: Production and Fluid History (V1 + V2)
- Tab 2: Pressure and Frequency Behavior (V3 + V4 + V10)
- Tab 3: Electrical Diagnostics (V8 + V9)
- Tab 4: Segmentation (V16)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional


def _get_last_observation_row(
    df: pd.DataFrame,
    flow_col: str = "liquid_rate_bbl_day",
    require_positive_flow: bool = True,
) -> Optional[pd.Series]:
    """Return the last observation row, optionally constrained to positive flow."""
    if df is None or df.empty:
        return None

    work = df.copy()
    if require_positive_flow and flow_col in work.columns:
        flow = pd.to_numeric(work[flow_col], errors="coerce")
        work = work[flow > 0].copy()
        if work.empty:
            return None

    if "observation_day" in work.columns:
        obs_day = pd.to_datetime(work["observation_day"], errors="coerce")
        if obs_day.notna().any():
            return work.loc[obs_day.idxmax()]

    return work.iloc[-1]


def build_allocation_temporal(
    df: pd.DataFrame,
    title: str = "Allocation Temporal Analysis",
) -> go.Figure:
    """
    Build V1: Allocation temporal analysis chart.

    Purpose: Production behavior over time (daily).
    X-axis: observation_day
    Y-axis left: alloc_oil_vol, alloc_water_vol
    Y-axis right: alloc_gas_vol

    Args:
        df: DataFrame with observation_day, alloc_oil_vol, alloc_water_vol, alloc_gas_vol

    Returns:
        Plotly Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    df_sorted = df.sort_values("observation_day")

    # Left axis: Oil and Water allocation
    fig.add_trace(
        go.Scatter(
            x=df_sorted["observation_day"],
            y=df_sorted["alloc_oil_vol"],
            name="Oil Volume",
            mode="lines",
            line=dict(color="darkgreen", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_sorted["observation_day"],
            y=df_sorted["alloc_water_vol"],
            name="Water Volume",
            mode="lines",
            line=dict(color="blue", width=2),
        ),
        secondary_y=False,
    )

    # Right axis: Gas allocation
    fig.add_trace(
        go.Scatter(
            x=df_sorted["observation_day"],
            y=df_sorted["alloc_gas_vol"],
            name="Gas Volume",
            mode="lines",
            line=dict(color="red", width=2),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Oil / Water Volume (bbls/day)", secondary_y=False)
    fig.update_yaxes(title_text="Gas Volume (mcf/day)", secondary_y=True)
    fig.update_layout(title_text=title, hovermode="x unified", height=500)

    return fig


def build_water_cut_gor_analysis(
    df: pd.DataFrame,
    title: str = "Water Cut vs GOR Analysis",
) -> go.Figure:
    """
    Build V2: Water cut vs GOR with liquid-flow background.

    Purpose: Fluid composition and gas behavior trends (daily).
    X-axis: observation_day
    Y-axis left: water_cut
    Y-axis right: gor
    Background: normalized liquid_rate_bbl_day band

    Args:
        df: DataFrame with observation_day, water_cut, gor, liquid_rate_bbl_day

    Returns:
        Plotly Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    df_sorted = df.sort_values("observation_day")

    # Normalized liquid rate background band
    liquid_rate_normalized = (
        (df_sorted["liquid_rate_bbl_day"] - df_sorted["liquid_rate_bbl_day"].min())
        / (df_sorted["liquid_rate_bbl_day"].max() - df_sorted["liquid_rate_bbl_day"].min())
        * 100  # Scale to 0-100 for visibility
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["observation_day"],
            y=liquid_rate_normalized,
            name="Liquid Rate (normalized)",
            mode="lines",
            line=dict(color="rgba(200,200,200,0.3)", width=0),
            fill="tozeroy",
            fillcolor="rgba(200,200,200,0.2)",
        ),
        secondary_y=False,
    )

    # Left axis: Water cut
    fig.add_trace(
        go.Scatter(
            x=df_sorted["observation_day"],
            y=df_sorted["water_cut"] * 100,  # Convert to percentage
            name="Water Cut (%)",
            mode="lines",
            line=dict(color="blue", width=2),
        ),
        secondary_y=False,
    )

    # Right axis: GOR
    fig.add_trace(
        go.Scatter(
            x=df_sorted["observation_day"],
            y=df_sorted["gor"],
            name="GOR (scf/bbl)",
            mode="lines",
            line=dict(color="red", width=2),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Water Cut (%) / Liquid Rate Norm", secondary_y=False)
    fig.update_yaxes(title_text="GOR (scf/bbl)", secondary_y=True)
    fig.update_layout(title_text=title, hovermode="x unified", height=500)

    return fig


def build_delta_p_pump_vs_frequency(
    df: pd.DataFrame,
    title: str = "Delta P Pump vs Motor Frequency",
) -> go.Figure:
    """
    Build V3: Delta P pump vs frequency temporal plot.

    Purpose: Pressure-lift response to operating frequency.
    X-axis: timestamp_telem
    Y-axis left: delta_p_pump_psi
    Y-axis right: motor_frequency_hz

    Args:
        df: DataFrame with timestamp_telem, delta_p_pump_psi, motor_frequency_hz

    Returns:
        Plotly Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    df_sorted = df.sort_values("timestamp_telem")

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["delta_p_pump_psi"],
            name="ΔP Pump (psi)",
            mode="lines",
            line=dict(color="darkblue", width=1),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["motor_frequency_hz"],
            name="Motor Frequency (Hz)",
            mode="lines",
            line=dict(color="orange", width=1),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="ΔP Pump (psi)", secondary_y=False)
    fig.update_yaxes(title_text="Motor Frequency (Hz)", secondary_y=True)
    fig.update_layout(
        title_text=title,
        hovermode="x unified",
        height=500,
    )

    return fig


def build_discharge_pressure_vs_frequency(
    df: pd.DataFrame,
    title: str = "Discharge Pressure Downhole vs Motor Frequency",
) -> go.Figure:
    """
    Build V4: Discharge pressure downhole vs frequency temporal plot.

    Purpose: Hydraulic response over time.
    X-axis: timestamp_telem
    Y-axis left: p_dis_downhole_psi
    Y-axis right: motor_frequency_hz

    Args:
        df: DataFrame with timestamp_telem, p_dis_downhole_psi, motor_frequency_hz

    Returns:
        Plotly Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    df_sorted = df.sort_values("timestamp_telem")

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["p_dis_downhole_psi"],
            name="P Discharge Downhole (psi)",
            mode="lines",
            line=dict(color="darkgreen", width=1),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["motor_frequency_hz"],
            name="Motor Frequency (Hz)",
            mode="lines",
            line=dict(color="orange", width=1),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="P Discharge Downhole (psi)", secondary_y=False)
    fig.update_yaxes(title_text="Motor Frequency (Hz)", secondary_y=True)
    fig.update_layout(
        title_text=title,
        hovermode="x unified",
        height=500,
    )

    return fig


def build_delta_p_composition(
    df: pd.DataFrame,
    title: str = "Temporal ΔP Composition",
) -> go.Figure:
    """
    Build V10: Temporal Delta P composition.

    Purpose: Explicit pressure-component decomposition.
    X-axis: timestamp_telem
    Y-axis: pump_intake_pressure_psi, delta_p_hyd_psi, tubing_pressure_psi, p_dis_downhole_psi

    Args:
        df: DataFrame with required pressure columns and timestamp_telem

    Returns:
        Plotly Figure
    """
    df_sorted = df.sort_values("timestamp_telem")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["pump_intake_pressure_psi"],
            name="Pump Intake (psi)",
            mode="lines",
            line=dict(color="purple", width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["delta_p_hyd_psi"],
            name="ΔP Hydrostatic (psi)",
            mode="lines",
            line=dict(color="brown", width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["tubing_pressure_psi"],
            name="Tubing Pressure (psi)",
            mode="lines",
            line=dict(color="cyan", width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["p_dis_downhole_psi"],
            name="P Discharge Downhole (psi)",
            mode="lines",
            line=dict(color="red", width=2),
        )
    )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Pressure (psi)")
    fig.update_layout(
        title_text=title,
        hovermode="x unified",
        height=500,
    )

    return fig


def build_motor_amps_vs_volts(
    df: pd.DataFrame,
    title: str = "Motor Amps vs Volts",
) -> go.Figure:
    """
    Build V8: Motor amps vs motor volts temporal plot.

    Purpose: Electrical-side stability and drift checks.
    X-axis: timestamp_telem
    Y-axis left: motor_amps
    Y-axis right: motor_volts

    Args:
        df: DataFrame with timestamp_telem, motor_amps, motor_volts

    Returns:
        Plotly Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    df_sorted = df.sort_values("timestamp_telem")

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["motor_amps"],
            name="Motor Amps",
            mode="lines",
            line=dict(color="red", width=1),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["motor_volts"],
            name="Motor Volts",
            mode="lines",
            line=dict(color="green", width=1),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Motor Amps (A)", secondary_y=False)
    fig.update_yaxes(title_text="Motor Volts (V)", secondary_y=True)
    fig.update_layout(
        title_text=title,
        hovermode="x unified",
        height=500,
    )

    return fig


def build_discharge_pressure_with_amp_volt(
    df: pd.DataFrame,
    title: str = "Discharge Pressure with Electrical Load Context",
) -> go.Figure:
    """
    Build V9: Discharge pressure vs frequency with AmpXVolt highlight.

    Purpose: Pressure trend with electrical-load context.
    X-axis: timestamp_telem
    Y-axis left: p_dis_downhole_psi
    Y-axis right: motor_frequency_hz
    Background: scaled amp_x_volt band

    Args:
        df: DataFrame with required columns and timestamp_telem

    Returns:
        Plotly Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    df_sorted = df.sort_values("timestamp_telem")

    # Background: scaled amp_x_volt
    amp_volt_min = df_sorted["amp_x_volt"].min()
    amp_volt_max = df_sorted["amp_x_volt"].max()
    amp_volt_normalized = (
        (df_sorted["amp_x_volt"] - amp_volt_min) / (amp_volt_max - amp_volt_min) * 100
        if (amp_volt_max - amp_volt_min) > 0
        else pd.Series([50] * len(df_sorted))
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=amp_volt_normalized,
            name="Electrical Load (normalized)",
            mode="lines",
            line=dict(color="rgba(200,200,200,0.3)", width=0),
            fill="tozeroy",
            fillcolor="rgba(200,200,200,0.2)",
        ),
        secondary_y=False,
    )

    # Pressure
    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["p_dis_downhole_psi"],
            name="P Discharge Downhole (psi)",
            mode="lines",
            line=dict(color="darkgreen", width=1),
        ),
        secondary_y=False,
    )

    # Frequency
    fig.add_trace(
        go.Scatter(
            x=df_sorted["timestamp_telem"],
            y=df_sorted["motor_frequency_hz"],
            name="Motor Frequency (Hz)",
            mode="lines",
            line=dict(color="orange", width=1),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Pressure (psi) / Load Norm", secondary_y=False)
    fig.update_yaxes(title_text="Motor Frequency (Hz)", secondary_y=True)
    fig.update_layout(
        title_text=title,
        hovermode="x unified",
        height=500,
    )

    return fig


def build_segmentation_heatmap(
    df: pd.DataFrame,
    title: str = "Operating Occupancy by Segment (Days)",
) -> go.Figure:
    """
    Build V16: Segmentation heatmap - days per segment.

    Purpose: Summarize operating occupancy by segment.
    Rows: water-cut categories (seg_liquid_comp)
    Columns: GOR categories (seg_gas)
    Cell value: number of unique days

    Args:
        df: DataFrame with seg_liquid_comp, seg_gas, and observation_day columns

    Returns:
        Plotly Figure (heatmap)
    """
    # Count unique days per segment
    segment_counts = df.groupby(["seg_liquid_comp", "seg_gas"])["observation_day"].nunique().reset_index()
    segment_counts.columns = ["water_cut_seg", "gor_seg", "unique_days"]

    # Pivot to matrix form
    pivot = segment_counts.pivot(index="water_cut_seg", columns="gor_seg", values="unique_days").fillna(0)

    # Order rows and columns for better readability
    row_order = ["mainly_oil", "mixed", "mainly_water", "undefined"]
    col_order = ["low", "medium", "high", "undefined"]

    pivot = pivot.reindex(
        index=[r for r in row_order if r in pivot.index],
        columns=[c for c in col_order if c in pivot.columns],
        fill_value=0,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="YlOrRd",
            text=pivot.values.astype(int),
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )

    fig.update_layout(
        title_text=title,
        xaxis_title="GOR Segment",
        yaxis_title="Water Cut Segment",
        height=500,
    )

    return fig


def build_ideal_single_frequency_delta_p(
    observed_daily: pd.DataFrame,
    ideal_curve: pd.DataFrame,
    title: str = "Single Frequency: DeltaP vs Flow + Ideal Overlay",
) -> go.Figure:
    """Observed vs ideal DeltaP map with efficiency overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    delta_blue = "#1D4ED8"
    eff_green = "#16A34A"

    obs = observed_daily.copy()
    fig.add_trace(
        go.Scatter(
            x=obs["liquid_rate_bbl_day"],
            y=obs["delta_p_pump_psi"],
            mode="markers",
            name="Observed Daily Points",
            marker=dict(size=7, color="rgba(29,78,216,0.60)"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=ideal_curve["flow_bpd"],
            y=ideal_curve["delta_p_psi"],
            mode="lines",
            name="Ideal DeltaP Curve",
            line=dict(color=delta_blue, width=3),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=obs["liquid_rate_bbl_day"],
            y=obs["eff_real_proxy_ratio"],
            mode="markers",
            name="Real efficiency proxy",
            marker=dict(size=6, color="rgba(22,163,74,0.70)"),
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=ideal_curve["flow_bpd"],
            y=ideal_curve["eff_ideal_ratio"],
            mode="lines",
            name="Ideal efficiency",
            line=dict(color=eff_green, width=2, dash="dash"),
        ),
        secondary_y=True,
    )

    last_row = _get_last_observation_row(obs, flow_col="liquid_rate_bbl_day", require_positive_flow=True)
    if last_row is not None:
        x_last = pd.to_numeric(pd.Series([last_row.get("liquid_rate_bbl_day")]), errors="coerce").iloc[0]
        y_last = pd.to_numeric(pd.Series([last_row.get("delta_p_pump_psi")]), errors="coerce").iloc[0]
        if pd.notna(x_last) and pd.notna(y_last):
            fig.add_trace(
                go.Scatter(
                    x=[x_last],
                    y=[y_last],
                    mode="markers",
                    name="Last Observation Day",
                    marker=dict(
                        symbol="star",
                        size=16,
                        color="#FACC15",
                        line=dict(color="#A16207", width=1.5),
                    ),
                ),
                secondary_y=False,
            )

    fig.update_xaxes(title_text="Flow (bpd)")
    fig.update_yaxes(title_text="DeltaP (psi)", secondary_y=False)
    fig.update_yaxes(title_text="Efficiency (real proxy / ideal)", range=[0, 1.5], secondary_y=True)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=560,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def build_ideal_single_frequency_bhp(
    observed_daily: pd.DataFrame,
    ideal_curve: pd.DataFrame,
    title: str = "Single Frequency: BHP vs Flow + Ideal Overlay",
) -> go.Figure:
    """Observed BHP proxy vs ideal BHP map with efficiency overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bhp_red = "#DC2626"
    eff_green = "#16A34A"

    obs = observed_daily.copy()
    fig.add_trace(
        go.Scatter(
            x=obs["liquid_rate_bbl_day"],
            y=obs["bhp_proxy"],
            mode="markers",
            name="Observed BHP Proxy",
            marker=dict(size=7, color="rgba(220,38,38,0.60)"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=ideal_curve["flow_bpd"],
            y=ideal_curve["bhp_hp"],
            mode="lines",
            name="Ideal BHP Curve",
            line=dict(color=bhp_red, width=3),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=obs["liquid_rate_bbl_day"],
            y=obs["eff_real_proxy_ratio"],
            mode="markers",
            name="Real efficiency proxy",
            marker=dict(size=6, color="rgba(22,163,74,0.70)"),
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=ideal_curve["flow_bpd"],
            y=ideal_curve["eff_ideal_ratio"],
            mode="lines",
            name="Ideal efficiency",
            line=dict(color=eff_green, width=2, dash="dash"),
        ),
        secondary_y=True,
    )

    last_row = _get_last_observation_row(obs, flow_col="liquid_rate_bbl_day", require_positive_flow=True)
    if last_row is not None:
        x_last = pd.to_numeric(pd.Series([last_row.get("liquid_rate_bbl_day")]), errors="coerce").iloc[0]
        y_last = pd.to_numeric(pd.Series([last_row.get("bhp_proxy")]), errors="coerce").iloc[0]
        if pd.notna(x_last) and pd.notna(y_last):
            fig.add_trace(
                go.Scatter(
                    x=[x_last],
                    y=[y_last],
                    mode="markers",
                    name="Last Observation Day",
                    marker=dict(
                        symbol="star",
                        size=16,
                        color="#FACC15",
                        line=dict(color="#A16207", width=1.5),
                    ),
                ),
                secondary_y=False,
            )

    fig.update_xaxes(title_text="Flow (bpd)")
    fig.update_yaxes(title_text="BHP (hp)", secondary_y=False)
    fig.update_yaxes(title_text="Efficiency (real proxy / ideal)", range=[0, 1.5], secondary_y=True)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=560,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def build_ideal_multi_frequency_delta_p(
    observed_daily: pd.DataFrame,
    family_curves: dict,
    selected_frequency_hz: float,
    title: str = "Multi Frequency: DeltaP Family + Observed",
) -> go.Figure:
    """Plot DeltaP family curves with observed cloud and efficiency overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    obs = observed_daily.copy()
    delta_blue = "#1D4ED8"
    delta_blue_soft = "rgba(29,78,216,0.35)"
    eff_green = "#16A34A"

    for freq, curve in sorted(family_curves.items()):
        width = 4 if abs(float(freq) - float(selected_frequency_hz)) < 1e-9 else 2
        color = delta_blue if width == 4 else delta_blue_soft
        fig.add_trace(
            go.Scatter(
                x=curve["flow_bpd"],
                y=curve["delta_p_psi"],
                mode="lines",
                name=f"{freq:.0f} Hz",
                line=dict(color=color, width=width),
                showlegend=True,
            ),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=obs["liquid_rate_bbl_day"],
            y=obs["delta_p_pump_psi"],
            mode="markers",
            name="Observed Daily",
            marker=dict(size=7, color="rgba(29,78,216,0.60)"),
        ),
        secondary_y=False,
    )

    selected_curve = min(
        family_curves.items(),
        key=lambda kv: abs(float(kv[0]) - float(selected_frequency_hz)),
    )[1]
    fig.add_trace(
        go.Scatter(
            x=obs["liquid_rate_bbl_day"],
            y=obs["eff_real_proxy_ratio"],
            mode="markers",
            name="Real efficiency proxy",
            marker=dict(size=7, color="rgba(22,163,74,0.75)", symbol="diamond"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_curve["flow_bpd"],
            y=selected_curve["eff_ideal_ratio"],
            mode="lines",
            name="Ideal efficiency (selected)",
            line=dict(color=eff_green, width=2, dash="dash"),
        ),
        secondary_y=True,
    )

    last_row = _get_last_observation_row(obs, flow_col="liquid_rate_bbl_day", require_positive_flow=True)
    if last_row is not None:
        x_last = pd.to_numeric(pd.Series([last_row.get("liquid_rate_bbl_day")]), errors="coerce").iloc[0]
        y_last = pd.to_numeric(pd.Series([last_row.get("delta_p_pump_psi")]), errors="coerce").iloc[0]
        if pd.notna(x_last) and pd.notna(y_last):
            fig.add_trace(
                go.Scatter(
                    x=[x_last],
                    y=[y_last],
                    mode="markers",
                    name="Last Observation Day",
                    marker=dict(
                        symbol="star",
                        size=16,
                        color="#FACC15",
                        line=dict(color="#A16207", width=1.5),
                    ),
                ),
                secondary_y=False,
            )

    fig.update_layout(
        title=title,
        xaxis_title="Flow (bpd)",
        template="plotly_white",
        height=560,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_yaxes(title_text="Pump DeltaP (psi)", secondary_y=False)
    fig.update_yaxes(title_text="Efficiency (real proxy / ideal)", range=[0, 1.5], secondary_y=True)
    return fig


def build_ideal_multi_frequency_bhp(
    observed_daily: pd.DataFrame,
    family_curves: dict,
    selected_frequency_hz: float,
    title: str = "Multi Frequency: BHP Family + Observed",
) -> go.Figure:
    """Plot BHP family curves with observed proxy cloud and efficiency overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    obs = observed_daily.copy()
    bhp_red = "#DC2626"
    bhp_red_soft = "rgba(220,38,38,0.35)"
    eff_green = "#16A34A"

    for freq, curve in sorted(family_curves.items()):
        width = 4 if abs(float(freq) - float(selected_frequency_hz)) < 1e-9 else 2
        color = bhp_red if width == 4 else bhp_red_soft
        fig.add_trace(
            go.Scatter(
                x=curve["flow_bpd"],
                y=curve["bhp_hp"],
                mode="lines",
                name=f"{freq:.0f} Hz",
                line=dict(color=color, width=width),
                showlegend=True,
            ),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=obs["liquid_rate_bbl_day"],
            y=obs["bhp_proxy"],
            mode="markers",
            name="Observed BHP Proxy",
            marker=dict(size=7, color="rgba(220,38,38,0.60)"),
        ),
        secondary_y=False,
    )

    selected_curve = min(
        family_curves.items(),
        key=lambda kv: abs(float(kv[0]) - float(selected_frequency_hz)),
    )[1]
    fig.add_trace(
        go.Scatter(
            x=obs["liquid_rate_bbl_day"],
            y=obs["eff_real_proxy_ratio"],
            mode="markers",
            name="Real efficiency proxy",
            marker=dict(size=7, color="rgba(22,163,74,0.75)", symbol="diamond"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_curve["flow_bpd"],
            y=selected_curve["eff_ideal_ratio"],
            mode="lines",
            name="Ideal efficiency (selected)",
            line=dict(color=eff_green, width=2, dash="dash"),
        ),
        secondary_y=True,
    )

    last_row = _get_last_observation_row(obs, flow_col="liquid_rate_bbl_day", require_positive_flow=True)
    if last_row is not None:
        x_last = pd.to_numeric(pd.Series([last_row.get("liquid_rate_bbl_day")]), errors="coerce").iloc[0]
        y_last = pd.to_numeric(pd.Series([last_row.get("bhp_proxy")]), errors="coerce").iloc[0]
        if pd.notna(x_last) and pd.notna(y_last):
            fig.add_trace(
                go.Scatter(
                    x=[x_last],
                    y=[y_last],
                    mode="markers",
                    name="Last Observation Day",
                    marker=dict(
                        symbol="star",
                        size=16,
                        color="#FACC15",
                        line=dict(color="#A16207", width=1.5),
                    ),
                ),
                secondary_y=False,
            )

    fig.update_layout(
        title=title,
        xaxis_title="Flow (bpd)",
        template="plotly_white",
        height=560,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_yaxes(title_text="BHP (proxy and ideal)", secondary_y=False)
    fig.update_yaxes(title_text="Efficiency (real proxy / ideal)", range=[0, 1.5], secondary_y=True)
    return fig
