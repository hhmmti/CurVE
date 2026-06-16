"""Gas-Interference Trend Screen (screening prototype).

Pure compute module for trend-based gas-interference risk screening using
historical preprocessed data. This module intentionally does not implement
confirmed gas-lock detection, bubble-point calculations, NPSH, or IPR logic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_MIN_ROWS = 14
DEFAULT_MAX_ROWS = 120
STABLE_PERCENT_THRESHOLD = 3.0
STABLE_SLOPE_EPS = 1e-9


def _to_datetime_series(df: pd.DataFrame) -> pd.Series:
    if "timestamp_telem" in df.columns:
        ts = pd.to_datetime(df["timestamp_telem"], errors="coerce")
        if ts.notna().any():
            return ts
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            return ts
    if "observation_day" in df.columns:
        ts = pd.to_datetime(df["observation_day"], errors="coerce")
        if ts.notna().any():
            return ts
    return pd.Series(pd.NaT, index=df.index)


def _usable_numeric_count(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").notna().sum())


def _direction_from_trend(percent_change: Optional[float], slope: Optional[float]) -> str:
    if percent_change is None or slope is None:
        return "unknown"
    if abs(percent_change) < STABLE_PERCENT_THRESHOLD or abs(slope) <= STABLE_SLOPE_EPS:
        return "stable"
    return "increasing" if slope > 0 else "decreasing"


def _compute_signal_trend(signal: pd.Series, timestamps: pd.Series) -> Dict:
    values = pd.to_numeric(signal, errors="coerce")
    ts = pd.to_datetime(timestamps, errors="coerce")
    work = pd.DataFrame({"value": values, "ts": ts}).dropna(subset=["value", "ts"]).copy()

    if work.empty:
        return {
            "available": False,
            "availability_count": 0,
            "first_value": None,
            "last_value": None,
            "absolute_change": None,
            "percent_change": None,
            "slope": None,
            "direction": "unknown",
            "volatility_std": None,
            "volatility_cv": None,
        }

    work = work.sort_values("ts")
    y = work["value"].to_numpy(dtype=float)
    n = len(y)
    first_val = float(y[0])
    last_val = float(y[-1])
    abs_change = float(last_val - first_val)

    if abs(first_val) > 0:
        pct_change = float((abs_change / abs(first_val)) * 100.0)
    else:
        pct_change = None

    if n >= 2:
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    std = float(np.nanstd(y, ddof=0)) if n > 1 else 0.0
    mean_abs = float(abs(np.nanmean(y))) if n > 0 else 0.0
    cv = float(std / mean_abs) if mean_abs > 0 else None

    return {
        "available": True,
        "availability_count": int(n),
        "first_value": first_val,
        "last_value": last_val,
        "absolute_change": abs_change,
        "percent_change": pct_change,
        "slope": slope,
        "direction": _direction_from_trend(pct_change, slope),
        "volatility_std": std,
        "volatility_cv": cv,
    }


def _delta_p_volatility_stats(series: pd.Series, timestamps: pd.Series) -> Dict:
    values = pd.to_numeric(series, errors="coerce")
    ts = pd.to_datetime(timestamps, errors="coerce")
    work = pd.DataFrame({"value": values, "ts": ts}).dropna(subset=["value", "ts"]).copy()
    if len(work) < 8:
        return {
            "available": False,
            "count": int(len(work)),
            "first_half_std": None,
            "second_half_std": None,
            "volatility_ratio": None,
            "volatility_change": None,
            "increasing": False,
        }

    work = work.sort_values("ts")
    y = work["value"].to_numpy(dtype=float)
    mid = len(y) // 2
    first = y[:mid]
    second = y[mid:]

    first_std = float(np.nanstd(first, ddof=0))
    second_std = float(np.nanstd(second, ddof=0))
    ratio = None
    if first_std > 0:
        ratio = float(second_std / first_std)
    delta = float(second_std - first_std)
    increasing = bool((ratio is not None and ratio >= 1.15) or (ratio is None and delta > 0))

    return {
        "available": True,
        "count": int(len(y)),
        "first_half_std": first_std,
        "second_half_std": second_std,
        "volatility_ratio": ratio,
        "volatility_change": delta,
        "increasing": increasing,
    }


def _normalize_for_chart(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)
    if s_min is None or s_max is None or not np.isfinite(s_min) or not np.isfinite(s_max):
        return pd.Series(np.nan, index=series.index)
    if s_max - s_min == 0:
        return pd.Series(0.5, index=series.index)
    return (s - s_min) / (s_max - s_min)


def _build_daily_history_window(historical_df: pd.DataFrame, max_days: int) -> pd.DataFrame:
    """Aggregate raw joined history to daily rows before screening.

    The screen combines telemetry and daily production-derived signals. Using the
    last N raw telemetry rows can collapse the window to a few recent hours and
    exclude otherwise valid production days. The screening window should be
    recent daily history, not recent raw telemetry rows.
    """
    work = historical_df.copy()
    work["_ts"] = _to_datetime_series(work)
    work = work.dropna(subset=["_ts"]).copy()
    if work.empty:
        return work

    work["_day"] = work["_ts"].dt.floor("D")

    numeric_cols = [
        col for col in work.columns if col not in {"_ts", "_day"} and pd.api.types.is_numeric_dtype(work[col])
    ]
    agg_map = {col: "median" for col in numeric_cols}
    agg_map["_ts"] = "max"

    daily = work.groupby("_day", as_index=False).agg(agg_map)
    daily = daily.sort_values("_day")
    daily["_ts"] = pd.to_datetime(daily["_day"], errors="coerce")

    if max_days > 0:
        daily = daily.tail(int(max_days)).copy()
    return daily.reset_index(drop=True)


def compute_gas_interference_trend_screen(
    historical_df: pd.DataFrame,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> Dict:
    """Compute trend-based Gas-Interference Risk Screening payload.

    Required core signals:
    - enough historical rows,
    - pump_intake_pressure_psi,
    - at least one gas indicator (gor OR alloc_gas_vol+alloc_oil_vol),
    - at least one pump behavior signal (delta_p_pump_psi OR liquid_rate_bbl_day).
    """
    if historical_df is None or historical_df.empty:
        return {
            "available": False,
            "reason_unavailable": "Historical preprocessed data is empty.",
            "mode": "Insufficient data",
            "risk_label": "Insufficient data",
            "trust_label": "Screening prototype",
            "time_window_summary": {"rows_total": 0, "rows_used": 0, "min_rows_required": int(min_rows)},
            "trend_statistics": {},
            "evidence_table": [],
            "triggered_evidence": [],
            "missing_optional_signals": [],
            "notes": [
                "This is a trend-based gas-interference screening diagnostic.",
                "It does not confirm gas lock or calculate true downhole gas volume fraction.",
            ],
        }

    work = _build_daily_history_window(historical_df=historical_df, max_days=max_rows)

    rows_total = int(len(historical_df))
    rows_used = int(len(work))

    if rows_used < min_rows:
        return {
            "available": False,
            "reason_unavailable": f"Not enough historical rows for screening (rows={rows_used}, min_rows={min_rows}).",
            "mode": "Insufficient data",
            "risk_label": "Insufficient data",
            "trust_label": "Screening prototype",
            "time_window_summary": {
                "rows_total": rows_total,
                "rows_used": rows_used,
                "min_rows_required": int(min_rows),
                "start_timestamp": None,
                "end_timestamp": None,
            },
            "trend_statistics": {},
            "evidence_table": [],
            "triggered_evidence": [],
            "missing_optional_signals": [],
            "notes": [
                "This is a trend-based gas-interference screening diagnostic.",
                "It does not confirm gas lock or calculate true downhole gas volume fraction.",
            ],
        }

    if "gor" not in work.columns and {"alloc_gas_vol", "alloc_oil_vol"}.issubset(work.columns):
        oil = pd.to_numeric(work["alloc_oil_vol"], errors="coerce")
        gas = pd.to_numeric(work["alloc_gas_vol"], errors="coerce")
        work["gor"] = np.where(oil > 0, gas / oil, np.nan)

    has_pip = "pump_intake_pressure_psi" in work.columns and _usable_numeric_count(work["pump_intake_pressure_psi"]) >= min_rows
    has_gas = "gor" in work.columns and _usable_numeric_count(work["gor"]) >= min_rows
    has_delta_p = "delta_p_pump_psi" in work.columns and _usable_numeric_count(work["delta_p_pump_psi"]) >= min_rows
    has_liquid = "liquid_rate_bbl_day" in work.columns and _usable_numeric_count(work["liquid_rate_bbl_day"]) >= min_rows

    if not has_pip or not has_gas or not (has_delta_p or has_liquid):
        missing = []
        if not has_pip:
            missing.append("pump_intake_pressure_psi")
        if not has_gas:
            missing.append("gas indicator (gor or alloc_gas_vol/alloc_oil_vol)")
        if not (has_delta_p or has_liquid):
            missing.append("pump behavior signal (delta_p_pump_psi or liquid_rate_bbl_day)")
        return {
            "available": False,
            "reason_unavailable": "Missing required core trend signals: " + ", ".join(missing),
            "mode": "Insufficient data",
            "risk_label": "Insufficient data",
            "trust_label": "Screening prototype",
            "time_window_summary": {
                "rows_total": rows_total,
                "rows_used": rows_used,
                "min_rows_required": int(min_rows),
                "start_timestamp": str(work["_ts"].min()) if work["_ts"].notna().any() else None,
                "end_timestamp": str(work["_ts"].max()) if work["_ts"].notna().any() else None,
            },
            "trend_statistics": {},
            "evidence_table": [],
            "triggered_evidence": [],
            "missing_optional_signals": [],
            "notes": [
                "This is a trend-based gas-interference screening diagnostic.",
                "It does not confirm gas lock or calculate true downhole gas volume fraction.",
            ],
        }

    signal_map = {
        "pump_intake_pressure_psi": "PIP",
        "gor": "GOR",
        "water_cut": "Water cut",
        "liquid_rate_bbl_day": "Liquid rate",
        "delta_p_pump_psi": "Pump delta-P",
    }
    optional_signal_candidates = {
        "motor_amps": "Motor amps",
        "amp_x_volt": "Amp x Volt proxy",
        "pump_intake_temperature_f": "Pump intake temperature",
        "motor_temperature_f": "Motor temperature",
    }

    trend_statistics: Dict[str, Dict] = {}
    evidence_table: List[Dict] = []
    for column, label in signal_map.items():
        if column in work.columns:
            stat = _compute_signal_trend(work[column], work["_ts"])
            trend_statistics[column] = stat
            evidence_table.append(
                {
                    "signal": column,
                    "label": label,
                    "availability_count": stat.get("availability_count"),
                    "first_value": stat.get("first_value"),
                    "last_value": stat.get("last_value"),
                    "absolute_change": stat.get("absolute_change"),
                    "percent_change": stat.get("percent_change"),
                    "slope": stat.get("slope"),
                    "direction": stat.get("direction"),
                    "volatility_cv": stat.get("volatility_cv"),
                }
            )

    motor_signal_key = None
    for candidate in ["motor_amps", "amp_x_volt"]:
        if candidate in work.columns and _usable_numeric_count(work[candidate]) >= min_rows:
            motor_signal_key = candidate
            stat = _compute_signal_trend(work[candidate], work["_ts"])
            trend_statistics[candidate] = stat
            evidence_table.append(
                {
                    "signal": candidate,
                    "label": optional_signal_candidates[candidate],
                    "availability_count": stat.get("availability_count"),
                    "first_value": stat.get("first_value"),
                    "last_value": stat.get("last_value"),
                    "absolute_change": stat.get("absolute_change"),
                    "percent_change": stat.get("percent_change"),
                    "slope": stat.get("slope"),
                    "direction": stat.get("direction"),
                    "volatility_cv": stat.get("volatility_cv"),
                }
            )
            break

    temp_reliable = False
    if "pump_intake_temperature_f" in work.columns:
        temp_series = pd.to_numeric(work["pump_intake_temperature_f"], errors="coerce")
        realistic = temp_series[(temp_series >= 20.0) & (temp_series <= 400.0)]
        if len(temp_series.dropna()) > 0:
            reliability_ratio = len(realistic) / len(temp_series.dropna())
            temp_reliable = reliability_ratio >= 0.80 and len(realistic) >= min_rows
        if temp_reliable:
            stat = _compute_signal_trend(realistic, work.loc[realistic.index, "_ts"])
            trend_statistics["pump_intake_temperature_f"] = stat
            evidence_table.append(
                {
                    "signal": "pump_intake_temperature_f",
                    "label": optional_signal_candidates["pump_intake_temperature_f"],
                    "availability_count": stat.get("availability_count"),
                    "first_value": stat.get("first_value"),
                    "last_value": stat.get("last_value"),
                    "absolute_change": stat.get("absolute_change"),
                    "percent_change": stat.get("percent_change"),
                    "slope": stat.get("slope"),
                    "direction": stat.get("direction"),
                    "volatility_cv": stat.get("volatility_cv"),
                }
            )

    if "delta_p_pump_psi" in work.columns:
        delta_p_vol = _delta_p_volatility_stats(work["delta_p_pump_psi"], work["_ts"])
        trend_statistics["delta_p_volatility"] = delta_p_vol
    else:
        delta_p_vol = {"available": False, "increasing": False}

    triggered_evidence: List[Dict] = []

    pip_stat = trend_statistics.get("pump_intake_pressure_psi", {})
    if pip_stat.get("available") and (pip_stat.get("percent_change") or 0.0) <= -5.0:
        triggered_evidence.append(
            {
                "id": "pip_declining",
                "label": "Pump intake pressure is declining",
                "score": 1,
                "details": f"PIP percent change={pip_stat.get('percent_change'):.2f}%",
            }
        )

    gor_stat = trend_statistics.get("gor", {})
    if gor_stat.get("available") and (gor_stat.get("percent_change") or 0.0) >= 8.0:
        triggered_evidence.append(
            {
                "id": "gor_increasing",
                "label": "GOR is increasing",
                "score": 1,
                "details": f"GOR percent change={gor_stat.get('percent_change'):.2f}%",
            }
        )

    wc_stat = trend_statistics.get("water_cut", {})
    if wc_stat.get("available") and (wc_stat.get("percent_change") or 0.0) >= 5.0:
        triggered_evidence.append(
            {
                "id": "water_cut_increasing",
                "label": "Water cut is increasing",
                "score": 1,
                "details": f"Water cut percent change={wc_stat.get('percent_change'):.2f}%",
            }
        )

    if delta_p_vol.get("available") and delta_p_vol.get("increasing"):
        ratio = delta_p_vol.get("volatility_ratio")
        ratio_txt = f"{ratio:.2f}" if ratio is not None else "N/A"
        triggered_evidence.append(
            {
                "id": "delta_p_volatility_increasing",
                "label": "Pump delta-P volatility is increasing",
                "score": 1,
                "details": f"Delta-P volatility ratio(second/first)={ratio_txt}",
            }
        )

    liq_stat = trend_statistics.get("liquid_rate_bbl_day", {})
    if liq_stat.get("available"):
        liq_declining = (liq_stat.get("percent_change") or 0.0) <= -5.0
        liq_unstable = (liq_stat.get("volatility_cv") or 0.0) >= 0.15
        if liq_declining or liq_unstable:
            reason = "declining" if liq_declining else "unstable"
            triggered_evidence.append(
                {
                    "id": "liquid_rate_declining_or_unstable",
                    "label": "Liquid rate is declining or unstable",
                    "score": 1,
                    "details": (
                        f"Liquid rate status={reason}; "
                        f"percent_change={liq_stat.get('percent_change')}, cv={liq_stat.get('volatility_cv')}"
                    ),
                }
            )

    if motor_signal_key is not None:
        motor_stat = trend_statistics.get(motor_signal_key, {})
        motor_declining = (motor_stat.get("percent_change") or 0.0) <= -5.0
        instability_present = any(
            item["id"] in {"delta_p_volatility_increasing", "liquid_rate_declining_or_unstable"}
            for item in triggered_evidence
        )
        if motor_declining and instability_present:
            triggered_evidence.append(
                {
                    "id": "motor_load_decreasing_with_instability",
                    "label": "Motor load/current is decreasing while instability is increasing",
                    "score": 1,
                    "details": f"{motor_signal_key} percent change={motor_stat.get('percent_change'):.2f}%",
                }
            )

    if temp_reliable:
        temp_stat = trend_statistics.get("pump_intake_temperature_f", {})
        if temp_stat.get("available") and (temp_stat.get("absolute_change") or 0.0) >= 5.0:
            triggered_evidence.append(
                {
                    "id": "intake_temperature_rising",
                    "label": "Pump intake temperature is rising",
                    "score": 1,
                    "details": f"Intake temperature change={temp_stat.get('absolute_change'):.2f} F",
                }
            )

    score = int(sum(int(item.get("score", 0)) for item in triggered_evidence))
    if score <= 1:
        risk_label = "Low"
    elif score <= 3:
        risk_label = "Watch"
    else:
        risk_label = "Elevated"

    missing_optional_signals = []
    for signal in ["water_cut", "motor_amps", "amp_x_volt", "pump_intake_temperature_f", "motor_temperature_f"]:
        if signal == "pump_intake_temperature_f" and temp_reliable:
            continue
        if signal in {"motor_amps", "amp_x_volt"} and motor_signal_key in {"motor_amps", "amp_x_volt"}:
            continue
        if signal not in work.columns or _usable_numeric_count(work[signal]) < min_rows:
            missing_optional_signals.append(signal)

    mode = "Full" if len(missing_optional_signals) == 0 else "Reduced"

    chart_df = pd.DataFrame({"timestamp": work["_ts"]})
    for col in ["pump_intake_pressure_psi", "gor", "water_cut", "liquid_rate_bbl_day", "delta_p_pump_psi"]:
        if col in work.columns:
            chart_df[f"normalized_{col}"] = _normalize_for_chart(work[col])

    return {
        "available": True,
        "reason_unavailable": None,
        "mode": mode,
        "risk_label": risk_label,
        "risk_score": score,
        "trust_label": "Screening prototype",
        "time_window_summary": {
            "rows_total": rows_total,
            "rows_used": rows_used,
            "min_rows_required": int(min_rows),
            "start_timestamp": str(work["_ts"].min()) if work["_ts"].notna().any() else None,
            "end_timestamp": str(work["_ts"].max()) if work["_ts"].notna().any() else None,
        },
        "trend_statistics": trend_statistics,
        "evidence_table": evidence_table,
        "triggered_evidence": triggered_evidence,
        "missing_optional_signals": missing_optional_signals,
        "normalized_trend_series": chart_df.to_dict(orient="records"),
        "notes": [
            "This is a trend-based gas-interference screening diagnostic.",
            "It does not confirm gas lock or calculate true downhole gas volume fraction.",
            "Risk labels are provisional and intended for screening support only.",
        ],
    }
