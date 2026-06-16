"""Pure compute/core functions for ESP analysis transformations."""

from typing import Tuple

import numpy as np
import pandas as pd

from compute.physics_common import (
    DEFAULT_WELL_DEPTH_FT,
    DEFAULT_SG_OIL,
    DEFAULT_SG_WATER,
    calc_mixture_sg,
    calc_hydrostatic_pressure_psi,
)


# Canonical defaults reused across pipeline stages (kept as module-level
# aliases for backward compatibility with callers that read these directly)
WELL_DEPTH_FT = DEFAULT_WELL_DEPTH_FT
SG_OIL = DEFAULT_SG_OIL
SG_WATER = DEFAULT_SG_WATER
MIN_SEGMENT_ROWS = 20


def merge_flow_telemetry(
    df_flow: pd.DataFrame,
    df_telem: pd.DataFrame,
    tolerance_minutes: int = 15,
) -> pd.DataFrame:
    """Merge flowmeter and telemetry data using nearest-time merge."""
    flow_df = df_flow.copy()
    telem_df = df_telem.copy()

    flow_df["timestamp"] = pd.to_datetime(flow_df["timestamp"], errors="coerce")
    telem_df["timestamp"] = pd.to_datetime(telem_df["timestamp"], errors="coerce")

    merged_data = pd.merge_asof(
        telem_df.dropna(subset=["timestamp"]).sort_values("timestamp"),
        flow_df.dropna(subset=["timestamp"]).sort_values("timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta(f"{tolerance_minutes}min"),
        direction="nearest",
        suffixes=("_telem", "_flow"),
    )
    return merged_data


def flow_quality_audit(merged_data: pd.DataFrame) -> pd.DataFrame:
    """Generate data quality audit for flow parameters."""
    flow_cols = ["oil_flow_rate", "water_flow_rate", "gas_flow_rate"]
    audit_rows = []

    for col in flow_cols:
        if col not in merged_data.columns:
            audit_rows.append({"column": col, "status": "missing", "rows": len(merged_data)})
            continue

        s = pd.to_numeric(merged_data[col], errors="coerce")
        n = len(s)
        null_count = s.isna().sum()
        zero_count = (s == 0).sum(skipna=True)
        neg_count = (s < 0).sum(skipna=True)

        s_valid = s.dropna()
        if len(s_valid) > 0:
            q1 = s_valid.quantile(0.25)
            q3 = s_valid.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (s_valid < lower) | (s_valid > upper)
            outlier_count = outlier_mask.sum()
        else:
            lower, upper, outlier_count = np.nan, np.nan, np.nan

        audit_rows.append(
            {
                "column": col,
                "rows": n,
                "null_count": int(null_count),
                "null_pct": round(100 * null_count / n, 2) if n else np.nan,
                "zero_count": int(zero_count),
                "zero_pct": round(100 * zero_count / n, 2) if n else np.nan,
                "negative_count": int(neg_count),
                "negative_pct": round(100 * neg_count / n, 2) if n else np.nan,
                "iqr_lower": lower,
                "iqr_upper": upper,
                "outlier_count": int(outlier_count) if pd.notna(outlier_count) else np.nan,
                "outlier_pct": round(100 * outlier_count / n, 2)
                if pd.notna(outlier_count) and n
                else np.nan,
                "status": "ok",
            }
        )

    return pd.DataFrame(audit_rows)


def clean_flow_data(merged_data: pd.DataFrame) -> pd.DataFrame:
    """Clean flow data by removing invalid liquid production rows."""
    oil = pd.to_numeric(merged_data["oil_flow_rate"], errors="coerce").fillna(0)
    wat = pd.to_numeric(merged_data["water_flow_rate"], errors="coerce").fillna(0)

    oil_not_null = merged_data["oil_flow_rate"].notna()
    wat_not_null = merged_data["water_flow_rate"].notna()
    liquid_positive = (oil + wat) > 0
    valid_mask = (oil_not_null | wat_not_null) & liquid_positive

    return merged_data[valid_mask].copy()


def engineer_features(
    merged_data: pd.DataFrame,
    well_depth_ft: float = WELL_DEPTH_FT,
    sg_oil: float = SG_OIL,
    sg_water: float = SG_WATER,
) -> pd.DataFrame:
    """Create derived flow, pressure, and fluid property features.

    Notes:
        - Public column names are intentionally preserved, including legacy
          capitalized names used by downstream plotting/service paths.
        - amp_x_volt is an electrical proxy (V x I), not a validated
          three-phase motor power measurement.
        - well_depth_ft, sg_oil, and sg_water may be caller-provided inputs;
          module defaults are fallback assumptions, not per-well measurements.
    """
    df = merged_data.copy()

    oil = df["oil_flow_rate"]
    wat = df["water_flow_rate"]

    df["liquid_rate_bbl_day"] = oil + wat
    df["GOR_scf_bbl"] = np.where(df["oil_flow_rate"] > 0, df["gas_flow_rate"] / df["oil_flow_rate"], np.nan)
    df["water_cut"] = np.where(df["liquid_rate_bbl_day"] > 0, wat / df["liquid_rate_bbl_day"], np.nan).astype(float)
    df["water_cut"] = df["water_cut"].clip(0, 1)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    amp_candidates = [c for c in numeric_cols if ("amp" in c.lower() or "current" in c.lower())]
    volt_candidates = [c for c in numeric_cols if "volt" in c.lower()]
    if amp_candidates and volt_candidates:
        # Stable proxy output name retained for compatibility with existing
        # pipeline and plotting usage.
        df["amp_x_volt"] = df[amp_candidates[0]] * df[volt_candidates[0]]
    else:
        df["amp_x_volt"] = np.nan

    df["well_depth_ft"] = well_depth_ft
    df["SG"] = np.where(
        df["water_cut"].notna(),
        df["water_cut"].map(lambda wc: calc_mixture_sg(wc, sg_oil=sg_oil, sg_water=sg_water)),
        np.nan,
    )
    df["delta_P_hyd_psi"] = df["SG"].map(
        lambda sg: calc_hydrostatic_pressure_psi(sg, depth_ft=well_depth_ft)
    )
    df["P_dis_downhole_psi"] = df["tubing_pressure_psi"] + df["delta_P_hyd_psi"]
    df["delta_P_pump_psi"] = df["P_dis_downhole_psi"] - df["pump_intake_pressure_psi"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def create_segments(merged_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create 2-axis segmentation (water_cut x GOR)."""
    seg_df = merged_data.copy()

    # Keep category labels strictly string-typed for NumPy compatibility.
    seg_df["seg_liquid_comp"] = np.select(
        [
            seg_df["water_cut"].isna(),
            seg_df["water_cut"] < 0.2,
            seg_df["water_cut"].between(0.2, 0.8, inclusive="both"),
            seg_df["water_cut"] > 0.8,
        ],
        ["undefined water cut", "mainly oil", "mixed oil-water", "mainly water"],
        default="undefined water cut",
    )

    q20 = seg_df["GOR_scf_bbl"].dropna().quantile(0.20)
    q80 = seg_df["GOR_scf_bbl"].dropna().quantile(0.80)

    seg_df["seg_gas"] = np.select(
        [
            seg_df["GOR_scf_bbl"].isna(),
            seg_df["GOR_scf_bbl"] < q20,
            seg_df["GOR_scf_bbl"].between(q20, q80, inclusive="both"),
            seg_df["GOR_scf_bbl"] > q80,
        ],
        ["undefined gor", "low gas", "medium gas", "high gas"],
        default="undefined gor",
    )

    seg_df["segment"] = seg_df["seg_liquid_comp"].astype(str) + " | " + seg_df["seg_gas"]

    segment_summary = (
        seg_df.groupby(["seg_liquid_comp", "seg_gas"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )

    return seg_df, segment_summary


def detect_frequency_column(df: pd.DataFrame) -> str:
    """Detect frequency column in numeric telemetry columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    freq_col = next((c for c in numeric_cols if "frequency" in c.lower() and "motor" in c.lower()), None)
    if not freq_col:
        freq_col = next((c for c in numeric_cols if "freq" in c.lower()), None)
    return freq_col


def detect_amp_column(df: pd.DataFrame) -> str:
    """Detect amp/current column in numeric telemetry columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    amp_col = next((c for c in numeric_cols if "amp" in c.lower() or "current" in c.lower()), None)
    return amp_col


def detect_volt_column(df: pd.DataFrame) -> str:
    """Detect voltage column in numeric telemetry columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    volt_col = next((c for c in numeric_cols if "volt" in c.lower()), None)
    return volt_col


def treat_outliers(
    seg_df: pd.DataFrame,
    well_depth_ft: float = WELL_DEPTH_FT,
    sg_oil: float = SG_OIL,
    sg_water: float = SG_WATER,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply segment-based outlier treatment using IQR method."""
    outlier_df = seg_df.copy()
    flow_cols = ["oil_flow_rate", "water_flow_rate", "gas_flow_rate"]
    outlier_stats = []

    for seg in outlier_df["segment"].dropna().unique():
        seg_mask = outlier_df["segment"] == seg
        seg_data = outlier_df[seg_mask]

        if len(seg_data) < MIN_SEGMENT_ROWS:
            continue

        for col in flow_cols:
            s = pd.to_numeric(seg_data[col], errors="coerce")

            if s.notna().sum() < MIN_SEGMENT_ROWS:
                continue

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_mask = (s < lower) | (s > upper)
            n_outliers = outlier_mask.sum()
            outlier_pct = 100 * n_outliers / s.notna().sum() if s.notna().sum() > 0 else 0

            outlier_df.loc[seg_mask, col] = np.where(
                (outlier_df.loc[seg_mask, col] < lower) | (outlier_df.loc[seg_mask, col] > upper),
                np.nan,
                outlier_df.loc[seg_mask, col],
            )

            outlier_stats.append(
                {
                    "segment": seg,
                    "column": col,
                    "n_rows": len(seg_data),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "n_outliers": int(n_outliers),
                    "outlier_pct": round(outlier_pct, 2),
                }
            )

    outlier_summary = pd.DataFrame(outlier_stats).sort_values(["segment", "column"])

    outlier_df = engineer_features(
        outlier_df,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
    )

    return outlier_df, outlier_summary
