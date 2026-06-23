"""Services layer for preprocessed data analysis workflow orchestration.

Coordinates data loading, joining, feature engineering, and analysis for the
preprocessed telemetry + production workflow.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

from compute import preprocessed_calcs


def join_telemetry_production(
    df_telemetry: pd.DataFrame,
    df_production: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Join telemetry and production data.

    Implements STEP16 Stage 2: telemetry-grain left join with production on
    organization_id, well_id, and observation_day.

    Telemetry timestamp is primary grain. Production values repeated across
    intraday telemetry rows.

    Args:
        df_telemetry: Preprocessed telemetry data
        df_production: Preprocessed production data (daily grain)

    Returns:
        Tuple of (joined_dataframe, join_metadata_dict)
    """
    telem = df_telemetry.copy()
    prod = df_production.copy()

    # Ensure timestamp is datetime
    telem["timestamp_telem"] = pd.to_datetime(
        telem.get(
            "timestamp_telem",
            telem.get("timestamp", telem.get("observation_day", None)),
        ),
        errors="coerce",
    )
    
    # Extract observation_day from telemetry timestamp
    telem["observation_day"] = telem["timestamp_telem"].dt.date

    # Ensure production observation_day is datetime.date
    prod["observation_day"] = pd.to_datetime(prod.get("observation_day", None), errors="coerce").dt.date

    # Avoid pandas merge collisions by renaming overlapping non-key production columns.
    # Keep telemetry names as canonical; preserve production overlaps with `_prod` suffix.
    join_keys = {"organization_id", "well_id", "observation_day"}
    overlap_cols = [c for c in prod.columns if c in telem.columns and c not in join_keys]
    if overlap_cols:
        rename_map = {}
        taken_names = set(telem.columns).union(set(prod.columns))
        for col in overlap_cols:
            base_name = f"{col}_prod"
            new_name = base_name
            i = 1
            while new_name in taken_names or new_name in rename_map.values():
                i += 1
                new_name = f"{base_name}_{i}"
            rename_map[col] = new_name
            taken_names.add(new_name)
        prod = prod.rename(columns=rename_map)

    # Left join on organization_id, well_id, observation_day
    merged = telem.merge(
        prod,
        on=["organization_id", "well_id", "observation_day"],
        how="left",
    )

    # Metadata
    metadata = {
        "telemetry_rows": len(telem),
        "production_rows": len(prod),
        "joined_rows": len(merged),
        "telemetry_min_date": telem["observation_day"].min(),
        "telemetry_max_date": telem["observation_day"].max(),
        "production_min_date": prod["observation_day"].min(),
        "production_max_date": prod["observation_day"].max(),
    }

    return merged, metadata


def run_preprocessed_analysis(
    df_telemetry: pd.DataFrame,
    df_production: pd.DataFrame,
    well_depth_ft: float = preprocessed_calcs.WELL_DEPTH_FT,
    sg_oil: float = preprocessed_calcs.SG_OIL,
    sg_water: float = preprocessed_calcs.SG_WATER,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full preprocessed analysis pipeline.

    Implements STEP16 Stages 2-6:
    1. Join telemetry + production
    2. Engineer features
    3. Create segmentation
    4. Profile data quality

    Args:
        df_telemetry: Preprocessed telemetry data
        df_production: Preprocessed production data
        well_depth_ft: Well depth in feet
        sg_oil: Oil specific gravity
        sg_water: Water specific gravity

    Returns:
        Tuple of (analyzed_dataframe, analysis_metadata_dict)
    """
    # Stage 2: Join
    merged, join_meta = join_telemetry_production(df_telemetry, df_production)

    return run_preprocessed_analysis_from_joined(
        merged,
        join_meta,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
    )


def run_preprocessed_analysis_from_joined(
    joined_df: pd.DataFrame,
    join_meta: Optional[Dict] = None,
    well_depth_ft: float = preprocessed_calcs.WELL_DEPTH_FT,
    sg_oil: float = preprocessed_calcs.SG_OIL,
    sg_water: float = preprocessed_calcs.SG_WATER,
) -> Tuple[pd.DataFrame, Dict]:
    """Run analysis pipeline starting from an already-joined dataframe."""
    merged = joined_df.copy()
    join_meta = join_meta or {
        "telemetry_rows": len(merged),
        "production_rows": np.nan,
        "joined_rows": len(merged),
        "telemetry_min_date": merged.get("observation_day", pd.Series(dtype="object")).min()
        if "observation_day" in merged.columns
        else np.nan,
        "telemetry_max_date": merged.get("observation_day", pd.Series(dtype="object")).max()
        if "observation_day" in merged.columns
        else np.nan,
        "production_min_date": np.nan,
        "production_max_date": np.nan,
    }

    # PIP is measured-or-missing: the legacy tubing/0.45 intake proxy was removed
    # after the PIP ~ tubing regression was rejected (weak; LOWO R² ~= -0.01,
    # ~37% error — see pip_reg_report). Null/zero intake is marked missing (NaN),
    # never backfilled with a constant; engineer_features() then yields NaN delta-P,
    # which the readiness layer surfaces as a missing, operator-suppliable input.
    if "pump_intake_pressure_psi" in merged.columns:
        intake = pd.to_numeric(merged["pump_intake_pressure_psi"], errors="coerce")
        merged["pump_intake_pressure_psi"] = intake.where(intake > 0, np.nan)

    # Stage 4: Feature engineering
    analyzed = preprocessed_calcs.engineer_features(
        merged,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
    )

    # Stage 5: Segmentation
    analyzed = preprocessed_calcs.create_segmentation(analyzed)

    # Stage 6: Data quality profiling
    quality_profile = preprocessed_calcs.profile_data_quality(analyzed)

    metadata = {
        **join_meta,
        "pump_intake_fallback_applied": False,  # PIP intake proxy removed (regression rejected); measured-or-missing.
        "quality_profile": quality_profile,
    }

    return analyzed, metadata


def prepare_temporal_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare temporal data for plotting (resample 6-hour telemetry).

    Args:
        df: Analyzed DataFrame with timestamp_telem

    Returns:
        DataFrame resampled to 6-hour intervals
    """
    df_temp = df.copy()
    df_temp["timestamp_telem"] = pd.to_datetime(
        df_temp.get(
            "timestamp_telem",
            df_temp.get("timestamp", df_temp.get("observation_day", None)),
        ),
        errors="coerce",
    )
    df_temp = df_temp.dropna(subset=["timestamp_telem"]).set_index("timestamp_telem")

    # Convert non-datetime columns to numeric where possible to keep median robust.
    for col in df_temp.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_temp[col]):
            df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce")

    # Resample to 6-hour intervals with median aggregation
    df_6h = df_temp.resample("6H").median(numeric_only=True)

    return df_6h.reset_index()


def prepare_daily_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare daily aggregated data for non-temporal plots.

    Uses daily medians for each day to reduce noise.

    Args:
        df: Analyzed DataFrame with observation_day

    Returns:
        DataFrame aggregated to daily medians
    """
    df_daily = df.copy()
    df_daily["observation_day"] = pd.to_datetime(df_daily.get("observation_day", None), errors="coerce")

    agg_columns = [
        "motor_frequency_hz",
        "tubing_pressure_psi",
        "pump_intake_pressure_psi",
        "delta_p_pump_psi",
        "p_dis_downhole_psi",
        "motor_amps",
        "motor_volts",
        "amp_x_volt",
        "liquid_rate_bbl_day",
        "alloc_oil_vol",
        "alloc_water_vol",
        "alloc_gas_vol",
        "gor",
        "water_cut",
        "sg_mixture",
        "delta_p_hyd_psi",
    ]

    # Keep only available columns and coerce them to numeric.
    available_cols = [c for c in agg_columns if c in df_daily.columns]
    for col in available_cols:
        df_daily[col] = pd.to_numeric(df_daily[col], errors="coerce")

    # Group by observation_day and calculate medians
    daily_agg = df_daily.groupby("observation_day")[available_cols].median(numeric_only=True).reset_index()

    return daily_agg


def filter_by_frequency_coverage(
    df: pd.DataFrame,
    min_days: int = 10,
) -> Dict[float, pd.DataFrame]:
    """
    Group daily data by frequency and filter out sparse frequencies.

    Args:
        df: Daily aggregated DataFrame
        min_days: Minimum number of days required per frequency (default 10)

    Returns:
        Dictionary mapping frequency -> filtered daily data
    """
    freq_groups = {}

    for freq, group in df.groupby("motor_frequency_hz"):
        if len(group) >= min_days:
            freq_groups[freq] = group.copy()

    return freq_groups
