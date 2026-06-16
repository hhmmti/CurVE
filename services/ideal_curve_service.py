"""Service orchestration for ideal-curve overlays in Page 2."""

import re
from typing import Dict, List, Optional

import boto3
import pandas as pd

from compute import ideal_curve_overlay
from data.ideal_pump_catalog import load_ideal_catalog


def load_catalog(
    session: Optional[boto3.Session] = None,
    profile_name: str = "roam-ai",
    region_name: str = "us-east-1",
    catalog: str = "roam_dev_products",
    database: str = "esp_ideal_pump_dev",
    table: str = "ideal_pump_library_v1",
    s3_output: str = "s3://esp-athena-results-v2-411237692998/",
) -> pd.DataFrame:
    return load_ideal_catalog(
        session=session,
        profile_name=profile_name,
        region_name=region_name,
        catalog=catalog,
        database=database,
        table=table,
        s3_output=s3_output,
    )


def selector_options(catalog_df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        "manufacturers": sorted(catalog_df["manufacturer"].dropna().astype(str).unique().tolist()),
    }


def models_for_manufacturer(catalog_df: pd.DataFrame, manufacturer: str) -> List[str]:
    df = catalog_df[catalog_df["manufacturer"] == manufacturer]
    return sorted(df["esp_model"].dropna().astype(str).unique().tolist())


def series_for_manufacturer_model(catalog_df: pd.DataFrame, manufacturer: str, esp_model: str) -> List[str]:
    df = catalog_df[(catalog_df["manufacturer"] == manufacturer) & (catalog_df["esp_model"] == esp_model)]
    return sorted(df["series"].dropna().astype(str).unique().tolist())


def narrow_catalog(
    catalog: pd.DataFrame,
    liquid_rate_bpd: float,
    bep_tolerance: float = 0.25,
) -> pd.DataFrame:
    """Filter catalog to pumps compatible with the well's liquid rate.

    Two-stage hard filter:
      Stage 1 — flow-range containment:
          min_recommended_bpd <= liquid_rate_bpd <= max_recommended_bpd
      Stage 2 — BEP proximity:
          bep_bpd * (1 - bep_tolerance) <= liquid_rate_bpd <= bep_bpd * (1 + bep_tolerance)

    Returns the filtered DataFrame. If no rows survive, returns an empty DataFrame.
    """
    rate = pd.to_numeric(pd.Series([liquid_rate_bpd]), errors="coerce").iloc[0]
    if pd.isna(rate) or rate <= 0:
        return catalog.iloc[0:0].copy()

    min_col = pd.to_numeric(catalog["min_recommended_bpd"], errors="coerce")
    max_col = pd.to_numeric(catalog["max_recommended_bpd"], errors="coerce")
    bep_col = pd.to_numeric(catalog["bep_bpd"], errors="coerce")

    stage1 = (min_col <= rate) & (rate <= max_col)
    stage2 = (bep_col * (1 - bep_tolerance) <= rate) & (rate <= bep_col * (1 + bep_tolerance))

    return catalog[stage1 & stage2].copy()


def build_narrowing_message(
    catalog_before: pd.DataFrame,
    catalog_after: pd.DataFrame,
    liquid_rate_bpd: float,
    bep_tolerance: float,
    well_config_pump_text: Optional[str] = None,
) -> str:
    """Build a human-readable narrowing report to display alongside the pump dropdowns.

    Returns a markdown string. Caller renders it via st.info() or st.warning().
    """
    n_before = len(catalog_before)
    n_after = len(catalog_after)

    # Zero-match guard — return early with a short prompt
    if n_after == 0:
        return (
            f"⚠️ **No pumps matched** the current liquid rate of **{liquid_rate_bpd:.0f} bpd** "
            f"with BEP tolerance **±{bep_tolerance * 100:.0f}%**.\n\n"
            "Increase the BEP tolerance or verify the liquid rate for this well."
        )

    pct_eliminated = 100.0 * (n_before - n_after) / n_before if n_before > 0 else 0.0

    parts = []

    # Guidance item A — narrowing summary
    parts.append(
        f"🔍 **Pump Narrowing** &nbsp;|&nbsp; "
        f"{n_before} → **{n_after} pumps** &nbsp;({pct_eliminated:.0f}% eliminated) &nbsp;|&nbsp; "
        f"Liquid rate: **{liquid_rate_bpd:.0f} bpd** &nbsp;|&nbsp; "
        f"BEP tolerance: **±{bep_tolerance * 100:.0f}%**"
    )

    # Guidance item B — operating zone and efficiency hint
    min_vals = pd.to_numeric(catalog_after["min_recommended_bpd"], errors="coerce")
    max_vals = pd.to_numeric(catalog_after["max_recommended_bpd"], errors="coerce")
    bep_vals = pd.to_numeric(catalog_after["bep_bpd"], errors="coerce")

    overall_min = min_vals.min()
    overall_max = max_vals.max()
    median_bep = bep_vals.median()

    span = overall_max - overall_min
    if span > 0:
        position = (liquid_rate_bpd - overall_min) / span
        if position < 0.33:
            zone_label = "🔵 Low end — consider a smaller pump"
        elif position < 0.67:
            zone_label = "🟢 Middle — good operating zone"
        else:
            zone_label = "🟠 High end — consider a larger pump"
    else:
        zone_label = "within matched range"

    if not pd.isna(median_bep) and median_bep > 0:
        bep_dev_pct = abs(liquid_rate_bpd - median_bep) / median_bep * 100
        if bep_dev_pct <= 10:
            eff_icon, eff_text = "🟢", f"Excellent — {bep_dev_pct:.1f}% from BEP"
        elif bep_dev_pct <= 20:
            eff_icon, eff_text = "🟡", f"Good — {bep_dev_pct:.1f}% from BEP"
        else:
            eff_icon, eff_text = "🟠", f"Moderate — {bep_dev_pct:.1f}% from BEP"
        direction = "below" if liquid_rate_bpd < median_bep else "above"
        bep_summary = (
            f"**Operating zone:** {zone_label} &nbsp;|&nbsp; "
            f"**Median BEP:** {median_bep:.0f} bpd ({bep_dev_pct:.1f}% {direction}) &nbsp;|&nbsp; "
            f"**Efficiency:** {eff_icon} {eff_text}"
        )
    else:
        bep_summary = f"**Operating zone:** {zone_label}"

    parts.append(bep_summary)

    # Guidance item C — well config pump text hint (display only)
    if well_config_pump_text and str(well_config_pump_text).strip():
        raw_text = str(well_config_pump_text).strip()
        numeric_tokens = re.findall(r"\d+", raw_text)
        hint = f"💡 **Well config pump:** `{raw_text}`"
        if numeric_tokens:
            hint += f" &nbsp;|&nbsp; **Numeric hints:** {', '.join(numeric_tokens)} — look for these in the model dropdown"
        parts.append(hint)

    return "  \n".join(parts)


def get_selected_pump_row(
    catalog_df: pd.DataFrame,
    manufacturer: str,
    esp_model: str,
    series: str,
) -> pd.Series:
    subset = catalog_df[
        (catalog_df["manufacturer"] == manufacturer)
        & (catalog_df["esp_model"] == esp_model)
        & (catalog_df["series"] == series)
    ]
    if subset.empty:
        raise ValueError("No matching pump found for selected manufacturer/model/series")
    return subset.iloc[0]


def get_frequency_options(daily_df: pd.DataFrame, min_days: int = 10) -> List[str]:
    """Build notebook-like frequency options (only bins with > min_days entries)."""
    if "motor_frequency_hz" not in daily_df.columns:
        return ["All"]

    freq = pd.to_numeric(daily_df["motor_frequency_hz"], errors="coerce")
    freq_rounded = freq.round(2)
    counts = freq_rounded.value_counts(dropna=True)
    eligible = sorted([float(v) for v in counts[counts > min_days].index.tolist()])
    return ["All"] + [f"{x:.2f}" for x in eligible]


def build_ideal_payload(
    daily_df: pd.DataFrame,
    pump_row: pd.Series,
    selected_frequency: str,
    stages: int,
    sg_for_dp: float = 1.0,
) -> Dict:
    df = daily_df.copy()
    df["motor_frequency_hz"] = pd.to_numeric(df.get("motor_frequency_hz", pd.Series(dtype=float)), errors="coerce")
    df["frequency_option"] = df["motor_frequency_hz"].round(2)

    if selected_frequency == "All":
        observed_source = df.copy()
        if observed_source["motor_frequency_hz"].notna().any():
            selected_frequency_hz = float(observed_source["motor_frequency_hz"].median())
        else:
            selected_frequency_hz = 60.0
    else:
        selected_frequency_hz = float(selected_frequency)
        observed_source = df[df["frequency_option"] == round(selected_frequency_hz, 2)].copy()

    observed = ideal_curve_overlay.compute_observed_proxies(observed_source)

    single_curve = ideal_curve_overlay.build_ideal_curve_for_frequency(
        pump_row,
        frequency_hz=selected_frequency_hz,
        stages=stages,
        sg_for_dp=sg_for_dp,
    )

    family_freqs = [45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
    family_curves = ideal_curve_overlay.build_multi_frequency_curves(
        pump_row,
        stages=stages,
        frequencies=family_freqs,
        sg_for_dp=sg_for_dp,
    )

    return {
        "pump": {
            "pump_id": str(pump_row.get("pump_id", "")),
            "manufacturer": str(pump_row.get("manufacturer", "")),
            "esp_model": str(pump_row.get("esp_model", "")),
            "series": str(pump_row.get("series", "")),
        },
        "selected_frequency_hz": float(selected_frequency_hz),
        "selected_frequency_option": selected_frequency,
        "stages": int(stages),
        "observed_daily": observed,
        "single_curve": single_curve,
        "family_curves": family_curves,
    }
