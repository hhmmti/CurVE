"""Compute helpers for ML recommendation analysis page."""

import json
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from compute import ideal_curve_overlay
from compute.physics_common import (
    DEFAULT_SG_OIL,
    DEFAULT_SG_WATER,
    DEFAULT_WELL_DEPTH_FT,
    PSI_PER_FT_PER_SG,
    calc_mixture_sg,
    calc_hydrostatic_pressure_psi,
    calc_discharge_pressure_downhole_psi,
    calc_pump_delta_p_psi,
)


def _find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_by_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in cols_by_lower:
            return cols_by_lower[key]
    return None


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        try:
            return bool(int(v))
        except Exception:
            return False
    text = str(v).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _key_float(v) -> str:
    num = _to_float(v)
    if num is None:
        return "na"
    if not np.isfinite(num):
        return "na"
    return f"{float(num):.6f}"


def normalize_recommendation_surface_rows(surface_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize recommendation_surfaces rows into a stable plotting schema."""
    if surface_df is None or surface_df.empty:
        return pd.DataFrame(
            columns=[
                "organization_id",
                "well_id",
                "recommendation_uuid",
                "inserted_at",
                "motor_frequency_hz",
                "tubing_pressure_psi",
                "total_economics",
                "in_bounds",
                "violated_boundaries",
                "scenario_flow_bpd",
                "scenario_delta_p_pump_psi",
                "scenario_id",
                "is_selectable",
            ]
        )

    df = surface_df.copy()

    column_aliases = {
        "organization_id": ["organization_id", "organization", "org_id", "org"],
        "well_id": ["well_id", "well", "well_name"],
        "recommendation_uuid": ["recommendation_uuid", "run_uuid", "recommendation_id"],
        "inserted_at": ["inserted_at", "created_at", "timestamp"],
        "motor_frequency_hz": ["motor_frequency_hz", "frequency_hz", "frequency"],
        "tubing_pressure_psi": ["tubing_pressure_psi", "tubing_pressure", "target_tubing_pressure_psi"],
        "total_economics": ["total_economics", "economics", "objective", "score"],
        "in_bounds": ["in_bounds", "is_in_bounds", "within_bounds"],
        "violated_boundaries": ["violated_boundaries", "violated_boundary", "boundary_violations"],
        "scenario_flow_bpd": [
            "liquid_rate_bpd",
            "total_liquid_bpd",
            "predicted_liquid_bpd",
            "flow_bpd",
            "total_fluid"
        ],
        "scenario_delta_p_pump_psi": [
            "delta_p_pump_psi",
            "pump_delta_p_psi",
            "predicted_delta_p_pump_psi",
        ],
    }

    out = pd.DataFrame(index=df.index)

    for canonical, aliases in column_aliases.items():
        source_col = _find_first_column(df, aliases)
        out[canonical] = df[source_col] if source_col is not None else np.nan

    out["organization_id"] = out["organization_id"].astype(str).str.strip()
    out["well_id"] = out["well_id"].astype(str).str.strip()
    out["recommendation_uuid"] = out["recommendation_uuid"].astype(str).str.strip()
    out["inserted_at"] = out["inserted_at"].astype(str).str.strip()

    out["motor_frequency_hz"] = pd.to_numeric(out["motor_frequency_hz"], errors="coerce")
    out["tubing_pressure_psi"] = pd.to_numeric(out["tubing_pressure_psi"], errors="coerce")
    out["total_economics"] = pd.to_numeric(out["total_economics"], errors="coerce")
    out["scenario_flow_bpd"] = pd.to_numeric(out["scenario_flow_bpd"], errors="coerce")
    out["scenario_delta_p_pump_psi"] = pd.to_numeric(out["scenario_delta_p_pump_psi"], errors="coerce")

    out["in_bounds"] = out["in_bounds"].map(_to_bool)
    out["violated_boundaries"] = out["violated_boundaries"].fillna("").astype(str).str.strip()
    out["is_selectable"] = out["in_bounds"].astype(bool)

    out["scenario_id"] = out.apply(
        lambda r: "|".join(
            [
                str(r.get("well_id", "")),
                str(r.get("recommendation_uuid", "")),
                _key_float(r.get("motor_frequency_hz")),
                _key_float(r.get("tubing_pressure_psi")),
            ]
        ),
        axis=1,
    )

    out = out.drop_duplicates(subset=["scenario_id"], keep="first").reset_index(drop=True)
    return out


def build_recommendation_surface_grid_payload(surface_df: pd.DataFrame) -> Dict:
    """Build plotting/selection payload for ML grid scenario exploration."""
    normalized = normalize_recommendation_surface_rows(surface_df)

    if normalized.empty:
        return {
            "points_df": normalized,
            "points": [],
            "stats": {
                "n_points": 0,
                "n_in_bounds": 0,
                "n_out_of_bounds": 0,
            },
            "selectable_scenario_ids": [],
            "in_bounds_default_only": True,
            "key_fields": [
                "well_id",
                "recommendation_uuid",
                "motor_frequency_hz",
                "tubing_pressure_psi",
            ],
            "axes": {
                "x": "tubing_pressure_psi",
                "y": "motor_frequency_hz",
                "color": "total_economics",
            },
        }

    points = normalized.to_dict(orient="records")
    n_points = int(len(normalized))
    n_in_bounds = int(normalized["in_bounds"].sum())

    return {
        "points_df": normalized,
        "points": points,
        "stats": {
            "n_points": n_points,
            "n_in_bounds": n_in_bounds,
            "n_out_of_bounds": int(n_points - n_in_bounds),
        },
        "selectable_scenario_ids": normalized.loc[
            normalized["is_selectable"], "scenario_id"
        ].astype(str).tolist(),
        "in_bounds_default_only": True,
        "key_fields": [
            "well_id",
            "recommendation_uuid",
            "motor_frequency_hz",
            "tubing_pressure_psi",
        ],
        "axes": {
            "x": "tubing_pressure_psi",
            "y": "motor_frequency_hz",
            "color": "total_economics",
        },
    }


def filter_selectable_surface_points_by_ids(
    grid_payload: Dict,
    selected_scenario_ids: Iterable[str],
) -> pd.DataFrame:
    """Return only selected, in-bounds scenario points."""
    df = grid_payload.get("points_df")
    if df is None or df.empty:
        return pd.DataFrame()

    selected_ids = {str(x) for x in (selected_scenario_ids or [])}
    if not selected_ids:
        return pd.DataFrame(columns=df.columns)

    out = df[df["scenario_id"].astype(str).isin(selected_ids) & df["is_selectable"]].copy()
    return out.reset_index(drop=True)


def enrich_surface_points_for_panel2(
    points_df: pd.DataFrame,
    compare_row: Optional[Dict] = None,
    physical_inputs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Fill/engineer scenario fields needed for Panel 2 projection.

        This computes scenario delta-P when surface rows do not provide it directly.

        Assumption/fallback behavior:
                - Default SG and depth are used when not supplied in physical_inputs.
                - rec_pump_intake_pressure_psi is sourced from compare_row; if absent,
                    cur_pump_intake_pressure_psi is used.
                - scenario_delta_p_pump_psi is treated as authoritative when present;
                    engineered_delta_p is only a fallback estimate.
    """
    if points_df is None or points_df.empty:
        return pd.DataFrame() if points_df is None else points_df.copy()

    compare_row = compare_row or {}
    physical_inputs = physical_inputs or {}

    out = points_df.copy()

    # Normalize optional scenario columns when present in source payload.
    for col in ["scenario_water_cut", "scenario_oil_bpd", "scenario_water_bpd"]:
        if col not in out.columns:
            out[col] = np.nan

    out["scenario_flow_bpd"] = pd.to_numeric(out.get("scenario_flow_bpd"), errors="coerce")
    out["scenario_delta_p_pump_psi"] = pd.to_numeric(out.get("scenario_delta_p_pump_psi"), errors="coerce")
    out["scenario_water_cut"] = pd.to_numeric(out.get("scenario_water_cut"), errors="coerce")
    out["scenario_oil_bpd"] = pd.to_numeric(out.get("scenario_oil_bpd"), errors="coerce")
    out["scenario_water_bpd"] = pd.to_numeric(out.get("scenario_water_bpd"), errors="coerce")
    out["tubing_pressure_psi"] = pd.to_numeric(out.get("tubing_pressure_psi"), errors="coerce")

    # Defaults from current recommendation context.
    default_wc = _to_float(compare_row.get("rec_water_cut"))
    if default_wc is None or not np.isfinite(default_wc):
        default_wc = _to_float(compare_row.get("cur_water_cut"))

    sg_oil = _to_float(physical_inputs.get("sg_oil"))
    sg_water = _to_float(physical_inputs.get("sg_water"))
    depth = _to_float(physical_inputs.get("well_depth_ft"))
    # Canonical fallback fluid/geometry assumptions.
    if sg_oil is None:
        sg_oil = DEFAULT_SG_OIL
    if sg_water is None:
        sg_water = DEFAULT_SG_WATER
    if depth is None:
        depth = DEFAULT_WELL_DEPTH_FT

    intake = _to_float(compare_row.get("rec_pump_intake_pressure_psi"))
    if intake is None or not np.isfinite(intake):
        # Recommendation-state intake fallback: assume unchanged from current.
        intake = _to_float(compare_row.get("cur_pump_intake_pressure_psi"))

    # Build scenario water-cut using best available source.
    scenario_total = out["scenario_oil_bpd"].fillna(0) + out["scenario_water_bpd"].fillna(0)
    scenario_wc_from_rates = np.where(
        scenario_total > 0,
        out["scenario_water_bpd"] / scenario_total,
        np.nan,
    )
    wc = out["scenario_water_cut"].copy()
    wc = wc.where(wc.notna(), pd.Series(scenario_wc_from_rates, index=out.index))
    if default_wc is not None and np.isfinite(default_wc):
        wc = wc.fillna(float(default_wc))
    wc = wc.clip(lower=0, upper=1)

    sg_mix = wc.map(lambda wc_val: calc_mixture_sg(wc_val, sg_oil=sg_oil, sg_water=sg_water))
    delta_p_hyd = sg_mix.map(lambda sg: calc_hydrostatic_pressure_psi(sg, depth_ft=depth))
    p_dis = out["tubing_pressure_psi"] + delta_p_hyd
    engineered_delta_p = (p_dis - intake) if intake is not None and np.isfinite(intake) else np.nan

    # Keep direct scenario delta-P when present; fallback to engineered estimate otherwise.
    missing_mask = out["scenario_delta_p_pump_psi"].isna() | ~np.isfinite(out["scenario_delta_p_pump_psi"])
    out.loc[missing_mask, "scenario_delta_p_pump_psi"] = pd.Series(engineered_delta_p, index=out.index)[missing_mask]

    return out


def _to_float(v):
    """Safely convert value to float."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return None


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except Exception:
        return False


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> float:
    if _is_missing(numerator) or _is_missing(denominator) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def parse_json_summary_data(summary_text: Optional[str]) -> Dict:
    """Parse json_summary_data text, tolerating NaN literals."""
    if summary_text is None:
        return {}

    if isinstance(summary_text, dict):
        return summary_text

    text = str(summary_text).strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        # Athena payload can contain bare NaN which is not strict JSON.
        cleaned = re.sub(r"\bNaN\b", "null", text)
        try:
            return json.loads(cleaned)
        except Exception:
            return {}


def parse_setpoint_like_map(text: Optional[str]) -> Dict:
    """Parse map-like strings such as {a=1, b={c=2}} into nested dictionaries."""
    if text is None:
        return {}

    s = str(text).strip()
    if not s or s == "{}":
        return {}

    n = len(s)

    def skip_ws(idx: int) -> int:
        while idx < n and s[idx].isspace():
            idx += 1
        return idx

    def parse_key(idx: int):
        idx = skip_ws(idx)
        start = idx
        while idx < n and s[idx] not in ["=", ",", "{", "}"]:
            idx += 1
        return s[start:idx].strip(), idx

    def parse_value(idx: int):
        idx = skip_ws(idx)
        if idx < n and s[idx] == "{":
            return parse_object(idx)

        start = idx
        while idx < n and s[idx] not in [",", "}"]:
            idx += 1
        raw = s[start:idx].strip()

        if re.fullmatch(r"[-+]?\\d+(\\.\\d+)?", raw or ""):
            return float(raw), idx
        return raw, idx

    def parse_object(idx: int):
        obj = {}
        idx = skip_ws(idx)
        if idx < n and s[idx] == "{":
            idx += 1

        while idx < n:
            idx = skip_ws(idx)
            if idx < n and s[idx] == "}":
                idx += 1
                break

            key, idx = parse_key(idx)
            idx = skip_ws(idx)
            if idx < n and s[idx] == "=":
                idx += 1

            value, idx = parse_value(idx)
            if key:
                obj[key] = value

            idx = skip_ws(idx)
            if idx < n and s[idx] == ",":
                idx += 1

        return obj, idx

    parsed, _ = parse_object(0)
    return parsed


def extract_compare_row(optimal_text: str, current_text: str, method: str = "max_oil") -> Dict:
    """Extract recommendation values from method branch and compare against current setpoint."""
    optimal = parse_setpoint_like_map(optimal_text)
    current = parse_setpoint_like_map(current_text)

    rec = optimal.get(method, {})
    rec_prod = rec.get("production", {})
    cur_prod = current.get("production", {})

    def d(a, b):
        """Calculate delta, ensuring both operands are numeric."""
        a_num = _to_float(a)
        b_num = _to_float(b)
        if a_num is None or b_num is None:
            return np.nan
        return a_num - b_num

    rec_freq = _to_float(rec.get("motor_frequency_hz"))
    cur_freq = _to_float(current.get("motor_frequency_hz"))
    rec_tp = _to_float(rec.get("tubing_pressure_psi"))
    cur_tp = _to_float(current.get("tubing_pressure_psi"))

    rec_oil = _to_float(rec_prod.get("oil"))
    cur_oil = _to_float(cur_prod.get("oil"))
    rec_water = _to_float(rec_prod.get("water"))
    cur_water = _to_float(cur_prod.get("water"))
    rec_gas = _to_float(rec_prod.get("gas"))
    cur_gas = _to_float(cur_prod.get("gas"))

    cur_q = (cur_oil or 0.0) + (cur_water or 0.0)
    rec_q = (rec_oil or 0.0) + (rec_water or 0.0)

    return {
        "method_used": method,
        "cur_motor_frequency_hz": cur_freq,
        "rec_motor_frequency_hz": rec_freq,
        "delta_motor_frequency_hz": d(rec_freq, cur_freq),
        "cur_tubing_pressure_psi": cur_tp,
        "rec_tubing_pressure_psi": rec_tp,
        "delta_tubing_pressure_psi": d(rec_tp, cur_tp),
        "cur_oil": cur_oil,
        "rec_oil": rec_oil,
        "delta_oil": d(rec_oil, cur_oil),
        "cur_water": cur_water,
        "rec_water": rec_water,
        "delta_water": d(rec_water, cur_water),
        "cur_gas": cur_gas,
        "rec_gas": rec_gas,
        "delta_gas": d(rec_gas, cur_gas),
        "cur_liquid_rate_bpd": cur_q,
        "rec_liquid_rate_bpd": rec_q,
        "delta_liquid_rate_bpd": d(rec_q, cur_q),
    }


def augment_with_delta_p_pump(
    compare_row: Dict,
    json_summary_text: Optional[str],
    well_depth_ft: float,
    sg_oil: float,
    sg_water: float,
) -> Dict:
    """Augment compare row with current/recommended pump delta-P metrics.

        Recommended intake pressure is assumed unchanged from current 1d average intake.

        Assumption/fallback behavior:
                - PIP is measured-or-missing: if current intake is missing it is NOT
                    proxied. The legacy tubing/0.45 fallback was removed after the
                    PIP ~ tubing regression was rejected (weak; see pip_reg_report);
                    missing intake is flagged a missing required input instead.
                - If recommendation intake is missing, recommended intake is assumed
                    equal to current intake.
                - sg_oil, sg_water, and well_depth_ft are caller-provided inputs and
                    should be treated as assumptions unless sourced from validated metadata.
    """
    out = dict(compare_row)
    summary = parse_json_summary_data(json_summary_text)

    # Current-state stats from json_summary_data.
    cur_intake = _to_float(summary.get("pump_intake_pressure_psi_1d_avg"))
    cur_tubing = _to_float(summary.get("tubing_pressure_psi_1d_avg"))
    cur_oil_alloc = _to_float(summary.get("alloc_oil_vol_1d_ago"))
    cur_water_alloc = _to_float(summary.get("alloc_water_vol_1d_ago"))

    # PIP is measured-or-missing: the legacy tubing/0.45 intake proxy was removed
    # after the PIP ~ tubing regression was rejected (weak; LOWO R² ~= -0.01,
    # ~37% error — see pip_reg_report). Missing intake stays missing — never
    # backfilled with a constant — so delta-P is NaN and the readiness fields below
    # mark PIP a missing, operator-suppliable required input.
    intake_fallback_used = False

    # Recommended-state values from model recommendation payload.
    rec_tubing = _to_float(out.get("rec_tubing_pressure_psi"))
    rec_oil = _to_float(out.get("rec_oil"))
    rec_water = _to_float(out.get("rec_water"))

    cur_o = np.nan if _is_missing(cur_oil_alloc) else cur_oil_alloc
    cur_w = np.nan if _is_missing(cur_water_alloc) else cur_water_alloc
    rec_o = np.nan if _is_missing(rec_oil) else rec_oil
    rec_w = np.nan if _is_missing(rec_water) else rec_water

    cur_total = cur_o + cur_w
    rec_total = rec_o + rec_w

    cur_wc = _safe_div(cur_water_alloc, cur_total)
    rec_wc = _safe_div(rec_water, rec_total)

    sg_o = _to_float(sg_oil)
    sg_w = _to_float(sg_water)
    depth = _to_float(well_depth_ft)

    if np.isnan(cur_wc) or sg_o is None or sg_w is None:
        cur_sg_mix = np.nan
    else:
        cur_sg_mix = calc_mixture_sg(cur_wc, sg_oil=sg_o, sg_water=sg_w)

    if np.isnan(rec_wc) or sg_o is None or sg_w is None:
        rec_sg_mix = np.nan
    else:
        rec_sg_mix = calc_mixture_sg(rec_wc, sg_oil=sg_o, sg_water=sg_w)

    cur_hyd = np.nan if np.isnan(cur_sg_mix) or _is_missing(depth) else calc_hydrostatic_pressure_psi(cur_sg_mix, depth_ft=depth)
    rec_hyd = np.nan if np.isnan(rec_sg_mix) or _is_missing(depth) else calc_hydrostatic_pressure_psi(rec_sg_mix, depth_ft=depth)

    cur_p_dis = np.nan if cur_tubing is None or np.isnan(cur_hyd) else calc_discharge_pressure_downhole_psi(cur_tubing, cur_hyd)
    rec_p_dis = np.nan if rec_tubing is None or np.isnan(rec_hyd) else calc_discharge_pressure_downhole_psi(rec_tubing, rec_hyd)

    # Recommendation-state fallback assumption: intake unchanged from current.
    rec_intake = cur_intake

    cur_delta_p = np.nan if _is_missing(cur_intake) or np.isnan(cur_p_dis) else calc_pump_delta_p_psi(cur_p_dis, cur_intake)
    rec_delta_p = np.nan if _is_missing(rec_intake) or np.isnan(rec_p_dis) else calc_pump_delta_p_psi(rec_p_dis, rec_intake)
    delta_delta_p = np.nan if np.isnan(cur_delta_p) or np.isnan(rec_delta_p) else rec_delta_p - cur_delta_p

    missing_inputs = []

    if _is_missing(cur_intake):
        missing_inputs.append("json_summary_data.pump_intake_pressure_psi_1d_avg")
    if _is_missing(cur_tubing):
        missing_inputs.append("json_summary_data.tubing_pressure_psi_1d_avg")
    if _is_missing(cur_oil_alloc):
        missing_inputs.append("json_summary_data.alloc_oil_vol_1d_ago")
    if _is_missing(cur_water_alloc):
        missing_inputs.append("json_summary_data.alloc_water_vol_1d_ago")
    if _is_missing(rec_tubing):
        missing_inputs.append("model_setpoint_recommendations.max_oil.tubing_pressure_psi")
    if _is_missing(rec_oil):
        missing_inputs.append("model_setpoint_recommendations.max_oil.production.oil")
    if _is_missing(rec_water):
        missing_inputs.append("model_setpoint_recommendations.max_oil.production.water")
    if _is_missing(depth):
        missing_inputs.append("user_input.well_depth_ft")
    if _is_missing(sg_o):
        missing_inputs.append("user_input.sg_oil")
    if _is_missing(sg_w):
        missing_inputs.append("user_input.sg_water")
    if (not _is_missing(cur_oil_alloc)) and (not _is_missing(cur_water_alloc)) and (cur_total == 0):
        missing_inputs.append("current allocation total is zero (alloc_oil_vol_1d_ago + alloc_water_vol_1d_ago)")
    if (not _is_missing(rec_oil)) and (not _is_missing(rec_water)) and (rec_total == 0):
        missing_inputs.append("recommended allocation total is zero (rec_oil + rec_water)")

    out.update(
        {
            "cur_water_cut": cur_wc,
            "rec_water_cut": rec_wc,
            "cur_sg_mix": cur_sg_mix,
            "rec_sg_mix": rec_sg_mix,
            "cur_delta_p_hyd_psi": cur_hyd,
            "rec_delta_p_hyd_psi": rec_hyd,
            "cur_p_dis_downhole_psi": cur_p_dis,
            "rec_p_dis_downhole_psi": rec_p_dis,
            "cur_pump_intake_pressure_psi": cur_intake,
            "rec_pump_intake_pressure_psi": rec_intake,
            "cur_delta_p_pump_psi": cur_delta_p,
            "rec_delta_p_pump_psi": rec_delta_p,
            "delta_delta_p_pump_psi": delta_delta_p,
            "delta_p_missing_inputs": missing_inputs,
            "delta_p_ready": len(missing_inputs) == 0 and not np.isnan(delta_delta_p),
            "delta_p_intake_fallback_used": intake_fallback_used,
            "delta_p_intake_source": "measured" if not _is_missing(cur_intake) else "missing",
            "delta_p_summary_extract": {
                "pump_intake_pressure_psi_1d_avg": summary.get("pump_intake_pressure_psi_1d_avg"),
                "tubing_pressure_psi_1d_avg": summary.get("tubing_pressure_psi_1d_avg"),
                "alloc_oil_vol_1d_ago": summary.get("alloc_oil_vol_1d_ago"),
                "alloc_water_vol_1d_ago": summary.get("alloc_water_vol_1d_ago"),
            },
        }
    )
    return out


def build_summary_table(compare_row: Dict) -> pd.DataFrame:
    """Create summary table for current vs recommendation deltas."""
    return pd.DataFrame(
        [
            {
                "metric": "motor_frequency_hz",
                "current": compare_row.get("cur_motor_frequency_hz"),
                "recommended": compare_row.get("rec_motor_frequency_hz"),
                "delta": compare_row.get("delta_motor_frequency_hz"),
            },
            {
                "metric": "tubing_pressure_psi",
                "current": compare_row.get("cur_tubing_pressure_psi"),
                "recommended": compare_row.get("rec_tubing_pressure_psi"),
                "delta": compare_row.get("delta_tubing_pressure_psi"),
            },
            {
                "metric": "liquid_rate_bpd (oil+water)",
                "current": compare_row.get("cur_liquid_rate_bpd"),
                "recommended": compare_row.get("rec_liquid_rate_bpd"),
                "delta": compare_row.get("delta_liquid_rate_bpd"),
            },
            {
                "metric": "oil_rate",
                "current": compare_row.get("cur_oil"),
                "recommended": compare_row.get("rec_oil"),
                "delta": compare_row.get("delta_oil"),
            },
            {
                "metric": "water_rate",
                "current": compare_row.get("cur_water"),
                "recommended": compare_row.get("rec_water"),
                "delta": compare_row.get("delta_water"),
            },
            {
                "metric": "gas_rate",
                "current": compare_row.get("cur_gas"),
                "recommended": compare_row.get("rec_gas"),
                "delta": compare_row.get("delta_gas"),
            },
            {
                "metric": "delta_p_pump_psi",
                "current": compare_row.get("cur_delta_p_pump_psi"),
                "recommended": compare_row.get("rec_delta_p_pump_psi"),
                "delta": compare_row.get("delta_delta_p_pump_psi"),
            },
        ]
    )


def build_curve_payload(
    compare_row: Dict,
    pump_row: pd.Series,
    stages: int,
    sg_for_dp: float,
    sweep_freqs: Optional[Iterable[float]] = None,
) -> Dict:
    """Build payload required for current-vs-recommended overlay chart."""
    if sweep_freqs is None:
        sweep_freqs = [45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]

    selected_freq = compare_row.get("rec_motor_frequency_hz")
    if selected_freq is None or not np.isfinite(selected_freq):
        selected_freq = 60.0

    family_curves = ideal_curve_overlay.build_multi_frequency_curves(
        pump_row,
        stages=int(max(1, stages)),
        frequencies=sweep_freqs,
        sg_for_dp=float(sg_for_dp),
    )

    selected_curve = ideal_curve_overlay.build_ideal_curve_for_frequency(
        pump_row,
        frequency_hz=float(selected_freq),
        stages=int(max(1, stages)),
        sg_for_dp=float(sg_for_dp),
    )

    return {
        "family_curves": family_curves,
        "selected_curve": selected_curve,
        "selected_frequency_hz": float(selected_freq),
        "current_point": {
            "flow_bpd": compare_row.get("cur_liquid_rate_bpd"),
            "delta_p_pump_psi": compare_row.get("cur_delta_p_pump_psi"),
        },
        "recommended_point": {
            "flow_bpd": compare_row.get("rec_liquid_rate_bpd"),
            "delta_p_pump_psi": compare_row.get("rec_delta_p_pump_psi"),
        },
    }
