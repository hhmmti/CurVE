"""Pure compute functions for preprocessed telemetry + production analysis.

This module contains all engineering calculations for the preprocessed data workflow.
All functions are pure (no side effects, deterministic).

Feature Engineering References:
- STEP16 Checkpoint 2 Analysis Pipeline and Visualization Design, Stage 4
- Uses canonical defaults: WELL_DEPTH_FT=5000, SG_OIL=0.85, SG_WATER=1.00
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd

from compute.physics_common import (
    DEFAULT_WELL_DEPTH_FT,
    DEFAULT_SG_OIL,
    DEFAULT_SG_WATER,
    PSI_PER_FT_PER_SG,
    calc_mixture_sg,
    calc_hydrostatic_pressure_psi,
)


# Canonical defaults (kept as module-level aliases for backward compatibility)
WELL_DEPTH_FT = DEFAULT_WELL_DEPTH_FT
SG_OIL = DEFAULT_SG_OIL
SG_WATER = DEFAULT_SG_WATER
MIN_SEGMENT_ROWS = 20


def engineer_features(
    df: pd.DataFrame,
    well_depth_ft: float = WELL_DEPTH_FT,
    sg_oil: float = SG_OIL,
    sg_water: float = SG_WATER,
) -> pd.DataFrame:
    """
    Engineer physical features from preprocessed telemetry + production data.

    Implements STEP16 Stage 4 feature engineering.

    Args:
        df: Merged telemetry + production DataFrame
        well_depth_ft: Well depth in feet. Defaults to canonical fallback depth
            when well metadata is unavailable.
        sg_oil: Oil specific gravity (default 0.85).
        sg_water: Water specific gravity (default 1.00).

    Returns:
        DataFrame with engineered features added

        Notes:
                - amp_x_volt is intentionally preserved as a stable public output name.
                - amp_x_volt is an electrical proxy (V x I), not a validated three-phase
                    power measurement.
                - This function does not apply intake-pressure fallback; it expects
                    pump_intake_pressure_psi to be already populated by the caller.
    """
    result = df.copy()

    # 1. Liquid rate (bbls/day)
    result["liquid_rate_bbl_day"] = (
        pd.to_numeric(result.get("alloc_oil_vol", 0), errors="coerce").fillna(0)
        + pd.to_numeric(result.get("alloc_water_vol", 0), errors="coerce").fillna(0)
    )

    # 2. Gas-oil ratio (scf/bbl)
    oil_vol = pd.to_numeric(result.get("alloc_oil_vol", 0), errors="coerce").fillna(0)
    gas_vol = pd.to_numeric(result.get("alloc_gas_vol", 0), errors="coerce").fillna(0)
    result["gor"] = np.where(oil_vol > 0, gas_vol / oil_vol, np.nan)

    # 3. Water cut (fraction)
    result["water_cut"] = np.where(
        result["liquid_rate_bbl_day"] > 0,
        pd.to_numeric(result.get("alloc_water_vol", 0), errors="coerce").fillna(0)
        / result["liquid_rate_bbl_day"],
        np.nan,
    )

    # 4. Mixture specific gravity (weighted average via shared helper)
    result["sg_mixture"] = result["water_cut"].map(
        lambda wc: calc_mixture_sg(wc, sg_oil=sg_oil, sg_water=sg_water)
    )

    # 5. Hydraulic pressure drop across well depth (shared helper)
    result["delta_p_hyd_psi"] = result["sg_mixture"].map(
        lambda sg: calc_hydrostatic_pressure_psi(sg, depth_ft=well_depth_ft)
    )

    # 6. Discharge pressure downhole (tubing pressure + hydrostatic)
    result["p_dis_downhole_psi"] = (
        pd.to_numeric(result.get("tubing_pressure_psi", np.nan), errors="coerce")
        + result["delta_p_hyd_psi"]
    )

    # 7. Delta P across pump (discharge - intake)
    intake_pressure = pd.to_numeric(result.get("pump_intake_pressure_psi", np.nan), errors="coerce")
    result["delta_p_pump_psi"] = result["p_dis_downhole_psi"] - intake_pressure

    # 8. Electrical proxy only (stable public name preserved for downstream use).
    # Not true 3-phase motor power; use direct power telemetry when available.
    motor_amps = pd.to_numeric(result.get("motor_amps", np.nan), errors="coerce")
    motor_volts = pd.to_numeric(result.get("motor_volts", np.nan), errors="coerce")
    result["amp_x_volt"] = motor_amps * motor_volts

    return result


def create_segmentation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create fluid and operating-point segmentation buckets.

    Implements STEP16 Stage 5 segmentation.

    Water-cut buckets:
        - 'mainly_oil': water_cut < 0.3
        - 'mixed': 0.3 <= water_cut < 0.7
        - 'mainly_water': water_cut >= 0.7
        - 'undefined': missing values

    GOR buckets by quantiles:
        - 'low': <= Q33
        - 'medium': Q33 < gor <= Q67
        - 'high': > Q67
        - 'undefined': missing values

    Args:
        df: DataFrame with 'water_cut' and 'gor' columns

    Returns:
        DataFrame with added segmentation columns
    """
    result = df.copy()

    # Water-cut segmentation
    def segment_water_cut(wc):
        if pd.isna(wc):
            return "undefined"
        if wc < 0.3:
            return "mainly_oil"
        elif wc < 0.7:
            return "mixed"
        else:
            return "mainly_water"

    result["seg_liquid_comp"] = result["water_cut"].apply(segment_water_cut)

    # GOR segmentation by quantiles
    gor_valid = result["gor"].dropna()
    if len(gor_valid) > 0:
        q33 = gor_valid.quantile(0.33)
        q67 = gor_valid.quantile(0.67)
    else:
        q33, q67 = np.nan, np.nan

    def segment_gor(g):
        if pd.isna(g):
            return "undefined"
        if g <= q33:
            return "low"
        elif g <= q67:
            return "medium"
        else:
            return "high"

    result["seg_gas"] = result["gor"].apply(segment_gor)

    # Combined segmentation label
    result["segment"] = result["seg_liquid_comp"] + " | " + result["seg_gas"]

    return result


def profile_data_quality(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Profile data quality metrics for specified columns.

    Implements STEP16 Stage 6 data quality profiling.

    Args:
        df: Input DataFrame
        columns: List of columns to profile. If None, profiles key columns.

    Returns:
        DataFrame with quality metrics
    """
    if columns is None:
        columns = [
            "motor_frequency_hz",
            "tubing_pressure_psi",
            "pump_intake_pressure_psi",
            "motor_amps",
            "motor_volts",
            "alloc_oil_vol",
            "alloc_water_vol",
            "alloc_gas_vol",
        ]

    audit_rows = []
    for col in columns:
        if col not in df.columns:
            audit_rows.append(
                {
                    "column": col,
                    "status": "missing",
                    "rows": len(df),
                    "null_pct": 100.0,
                    "zero_pct": np.nan,
                    "availability_pct": 0.0,
                }
            )
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        n = len(s)
        null_count = s.isna().sum()
        zero_count = (s == 0).sum(skipna=True)
        
        # Availability = not null and not zero
        availability = ((~s.isna()) & (s != 0)).sum()
        availability_pct = 100.0 * availability / n if n > 0 else 0.0

        audit_rows.append(
            {
                "column": col,
                "rows": n,
                "null_count": int(null_count),
                "null_pct": round(100.0 * null_count / n, 2) if n > 0 else np.nan,
                "zero_count": int(zero_count),
                "zero_pct": round(100.0 * zero_count / n, 2) if n > 0 else np.nan,
                "availability_pct": round(availability_pct, 2),
                "status": "ok",
            }
        )

    return pd.DataFrame(audit_rows)


def check_pump_intake_fallback_needed(df: pd.DataFrame, threshold: float = 0.45) -> bool:
    """
    Determine if pump intake pressure fallback is needed.

    Fallback activates when null+zero exceeds threshold (default 45%).

    Args:
        df: DataFrame with 'pump_intake_pressure_psi' column
        threshold: Fallback activation threshold (default 0.45)

    Returns:
        Boolean indicating if fallback is needed
    """
    intake_col = df.get("pump_intake_pressure_psi", pd.Series(dtype=float))
    if len(intake_col) == 0:
        return True

    intake = pd.to_numeric(intake_col, errors="coerce")
    invalid_count = (intake.isna() | (intake == 0)).sum()
    invalid_pct = invalid_count / len(intake) if len(intake) > 0 else 1.0

    return invalid_pct > threshold
