"""Compute helpers for ideal curve overlays in CP3 page.

Implements polynomial evaluation + frequency/stage scaling and proxy metrics.

Proxy terminology in this module means estimated/relative indicator, not
validated field measurement. Direct measured power should be preferred when
available.
"""

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from compute.physics_common import (
    FT_PER_PSI_WATER,
    WATTS_PER_HP,
    DEFAULT_POWER_FACTOR,
    HHP_DENOMINATOR,
    PUMP_EFF_DENOMINATOR,
    calc_pressure_psi_from_head_ft,
)


def _safe_float(value):
    """Return float(value) when possible, else None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _bep_position_label(abs_distance_pct: float) -> str:
    """Map absolute BEP distance percent to a fixed diagnostic band."""
    if abs_distance_pct <= 10.0:
        return "Near BEP"
    if abs_distance_pct <= 25.0:
        return "Acceptable deviation"
    return "Far from BEP"


def _range_status(flow_bpd: float, min_recommended_bpd: float, max_recommended_bpd: float) -> str:
    """Classify flow position relative to catalog recommended flow range."""
    if flow_bpd < min_recommended_bpd:
        return "Below recommended range"
    if flow_bpd > max_recommended_bpd:
        return "Above recommended range"
    return "Inside recommended range"


def _unavailable_bep_diagnostic(reason: str, pump_label=None, pump_source=None) -> Dict:
    """Build a stable unavailable response schema for BEP diagnostic."""
    unavailable_point = {
        "point_label": None,
        "flow_bpd": None,
        "distance_from_bep_pct": None,
        "abs_distance_from_bep_pct": None,
        "bep_position_label": None,
        "range_status": None,
    }
    return {
        "available": False,
        "reason_unavailable": reason,
        "pump_label": pump_label,
        "pump_source": pump_source,
        "bep_bpd": None,
        "min_recommended_bpd": None,
        "max_recommended_bpd": None,
        "current": dict(unavailable_point),
        "recommended": dict(unavailable_point),
        "movement": {
            "current_abs_distance_from_bep_pct": None,
            "recommended_abs_distance_from_bep_pct": None,
            "delta_abs_distance_pct": None,
            "movement_label": "Unavailable",
        },
    }


def compute_bep_position_diagnostic(
    current_flow_bpd: float,
    recommended_flow_bpd: float,
    bep_bpd: float,
    min_recommended_bpd: float,
    max_recommended_bpd: float,
    pump_label: str = None,
    pump_source: str = None,
) -> Dict:
    """Evaluate current and recommended flow positions against catalog BEP/range.

    Diagnostic metric:
        distance_from_bep_pct = ((point_flow_bpd - bep_bpd) / bep_bpd) * 100

    This is a catalog-based BEP position diagnostic and not a field efficiency
    measurement.
    """
    cur_q = _safe_float(current_flow_bpd)
    rec_q = _safe_float(recommended_flow_bpd)
    bep_q = _safe_float(bep_bpd)
    min_q = _safe_float(min_recommended_bpd)
    max_q = _safe_float(max_recommended_bpd)

    if cur_q is None or cur_q <= 0:
        return _unavailable_bep_diagnostic(
            reason="Missing or invalid current flow (must be > 0).",
            pump_label=pump_label,
            pump_source=pump_source,
        )
    if rec_q is None or rec_q <= 0:
        return _unavailable_bep_diagnostic(
            reason="Missing or invalid recommended flow (must be > 0).",
            pump_label=pump_label,
            pump_source=pump_source,
        )
    if bep_q is None or bep_q <= 0:
        return _unavailable_bep_diagnostic(
            reason="Missing or invalid BEP flow from selected pump catalog row.",
            pump_label=pump_label,
            pump_source=pump_source,
        )
    if min_q is None:
        return _unavailable_bep_diagnostic(
            reason="Missing catalog min_recommended_bpd for selected pump.",
            pump_label=pump_label,
            pump_source=pump_source,
        )
    if max_q is None:
        return _unavailable_bep_diagnostic(
            reason="Missing catalog max_recommended_bpd for selected pump.",
            pump_label=pump_label,
            pump_source=pump_source,
        )
    if min_q > max_q:
        return _unavailable_bep_diagnostic(
            reason="Invalid catalog recommended range: min_recommended_bpd is greater than max_recommended_bpd.",
            pump_label=pump_label,
            pump_source=pump_source,
        )

    def _point(point_label: str, flow_bpd: float) -> Dict:
        dist_pct = ((flow_bpd - bep_q) / bep_q) * 100.0
        abs_dist_pct = abs(dist_pct)
        return {
            "point_label": point_label,
            "flow_bpd": float(flow_bpd),
            "distance_from_bep_pct": float(dist_pct),
            "abs_distance_from_bep_pct": float(abs_dist_pct),
            "bep_position_label": _bep_position_label(abs_dist_pct),
            "range_status": _range_status(flow_bpd, min_q, max_q),
        }

    current = _point("Current", cur_q)
    recommended = _point("Recommended", rec_q)

    cur_abs = float(current["abs_distance_from_bep_pct"])
    rec_abs = float(recommended["abs_distance_from_bep_pct"])
    delta_abs = rec_abs - cur_abs

    if abs(delta_abs) <= 1.0:
        movement_label = "Recommended point stays about the same distance from BEP"
    elif delta_abs < 0:
        movement_label = "Recommended point moves closer to BEP"
    else:
        movement_label = "Recommended point moves farther from BEP"

    return {
        "available": True,
        "reason_unavailable": None,
        "pump_label": pump_label,
        "pump_source": pump_source,
        "bep_bpd": float(bep_q),
        "min_recommended_bpd": float(min_q),
        "max_recommended_bpd": float(max_q),
        "current": current,
        "recommended": recommended,
        "movement": {
            "current_abs_distance_from_bep_pct": cur_abs,
            "recommended_abs_distance_from_bep_pct": rec_abs,
            "delta_abs_distance_pct": float(delta_abs),
            "movement_label": movement_label,
        },
    }


def _poly6(q: np.ndarray, coeffs: Iterable[float]) -> np.ndarray:
    c1, c2, c3, c4, c5, c6 = coeffs
    return c1 + c2 * q + c3 * q**2 + c4 * q**3 + c5 * q**4 + c6 * q**5


def _build_base_q_grid(pump_row: pd.Series, n_points: int = 120) -> np.ndarray:
    """Build a notebook-like flow grid with anchor-aware segmentation."""
    required = ["min_recommended_bpd", "bep_bpd", "max_recommended_bpd", "max_plotted_bpd"]
    if all(k in pump_row for k in required):
        min_q = float(pump_row["min_recommended_bpd"])
        bep_q = float(pump_row["bep_bpd"])
        max_q = float(pump_row["max_recommended_bpd"])
        max_plot_q = float(pump_row["max_plotted_bpd"])
        anchors = [0.0, min_q, bep_q, max_q, max_plot_q]
        if np.all(np.isfinite(anchors)) and anchors == sorted(anchors):
            q = [0.0]
            div = 4
            for a, b in zip(anchors[:-1], anchors[1:]):
                step = (b - a) / div if div > 0 else 0.0
                for k in range(1, div + 1):
                    q.append(a + k * step)
            return np.array(q, dtype=float)

    return np.linspace(0.0, float(pump_row["max_plotted_bpd"]), int(n_points))


def build_ideal_curve_for_frequency(
    pump_row: pd.Series,
    frequency_hz: float,
    stages: int,
    sg_for_dp: float = 1.0,
    n_points: int = 120,
) -> pd.DataFrame:
    """Build ideal curve arrays at a selected frequency and stage count.

    Args:
        sg_for_dp: Specific gravity used for head-to-pressure conversion.
            Defaults to 1.0 when no fluid-specific SG is provided.
    """
    q0 = _build_base_q_grid(pump_row, n_points=n_points)

    head_coeffs = [float(pump_row[f"ideal_head_c{i}"]) for i in range(1, 7)]
    power_coeffs = [float(pump_row[f"ideal_power_c{i}"]) for i in range(1, 7)]

    head_60 = _poly6(q0, head_coeffs)
    bhp_60 = _poly6(q0, power_coeffs)

    freq_scale = float(frequency_hz) / 60.0

    q = q0 * freq_scale
    head = head_60 * (freq_scale**2) * max(int(stages), 1)
    bhp = bhp_60 * (freq_scale**3) * max(int(stages), 1)

    delta_p_psi = calc_pressure_psi_from_head_ft(head, sg=float(sg_for_dp))
    eff_ratio = np.where(bhp > 0, (q * head / bhp / PUMP_EFF_DENOMINATOR), np.nan)

    return pd.DataFrame(
        {
            "flow_bpd": q,
            "head_ft": head,
            "delta_p_psi": delta_p_psi,
            "bhp_hp": bhp,
            "eff_ideal_ratio": eff_ratio,
        }
    )


def build_multi_frequency_curves(
    pump_row: pd.Series,
    stages: int,
    frequencies: Iterable[float],
    sg_for_dp: float = 1.0,
) -> Dict[float, pd.DataFrame]:
    """Build ideal curves for multiple frequencies."""
    return {
        float(f): build_ideal_curve_for_frequency(
            pump_row,
            frequency_hz=float(f),
            stages=stages,
            sg_for_dp=sg_for_dp,
        )
        for f in frequencies
    }


def compute_observed_proxies(daily_df: pd.DataFrame, pf: float = DEFAULT_POWER_FACTOR) -> pd.DataFrame:
    """Compute observed power/efficiency proxies for overlay plots.

    Output columns (stable names retained):
        - hhp_proxy: hydraulic horsepower proxy from flow and delta-P.
        - bhp_proxy: electrical horsepower proxy from amp_x_volt and PF.
        - eff_real_proxy_ratio: proxy efficiency ratio (hhp_proxy / bhp_proxy).

    Notes:
        - pf defaults to DEFAULT_POWER_FACTOR (0.90) when a measured PF is not
          available.
        - These outputs are proxy indicators and should not be interpreted as
          validated pump or motor efficiency measurements.
    """
    df = daily_df.copy()
    for c in ["liquid_rate_bbl_day", "delta_p_pump_psi", "amp_x_volt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Hydraulic HP proxy and electrical HP proxy.
    # Keep column names unchanged to avoid breaking downstream consumers.
    df["hhp_proxy"] = (df["liquid_rate_bbl_day"] * df["delta_p_pump_psi"]) / HHP_DENOMINATOR
    df["bhp_proxy"] = (np.sqrt(3.0) * df["amp_x_volt"] * float(pf)) / WATTS_PER_HP
    df["eff_real_proxy_ratio"] = np.where(
        df["bhp_proxy"] > 0,
        df["hhp_proxy"] / df["bhp_proxy"],
        np.nan,
    )

    return df
