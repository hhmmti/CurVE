"""Service orchestration for ML recommendation analysis page."""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from compute.affinity_validator import compute_affinity_law_validator
from compute.energy_efficiency import compute_energy_efficiency_diagnostic
from compute.gas_interference_screen import compute_gas_interference_trend_screen
from compute import ideal_curve_overlay
from compute import ml_recommendation_calcs
from compute.physics_common import DEFAULT_POWER_FACTOR, WATTS_PER_HP
from data.preprocessed_db import PreprocessedDataDB
from services.data_availability_gate import run_data_availability_gate
from services import preprocessed_pipeline_service


def _extract_json_summary_payload(latest_row: Dict):
    """Return (used_key, payload) for json summary data from latest row."""
    if not latest_row:
        return None, None

    for exact_key in ["json_summary_data", "summary_data_json"]:
        if exact_key in latest_row:
            return exact_key, latest_row.get(exact_key)

    # Fallback for naming variations/case differences.
    for k in latest_row.keys():
        k_l = str(k).strip().lower()
        if (
            "json_summary_data" in k_l
            or "summary_data_json" in k_l
            or "json_summary" in k_l
            or ("summary" in k_l and "json" in k_l)
        ):
            return str(k), latest_row.get(k)

    return None, None


def _to_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _first_float(*candidates):
    for value in candidates:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _extract_surface_power_inputs(surface_row: Dict) -> Dict:
    """Extract direct/proxy-capable power fields from one surface row."""
    if not isinstance(surface_row, dict):
        return {
            "motor_power_kw": None,
            "bhp_proxy": None,
            "amp_x_volt": None,
            "motor_amps": None,
            "motor_volts": None,
        }

    motor_amps = _first_float(
        surface_row.get("motor_amps"),
        surface_row.get("scenario_motor_amps"),
        surface_row.get("predicted_motor_amps"),
    )
    motor_volts = _first_float(
        surface_row.get("motor_volts"),
        surface_row.get("scenario_motor_volts"),
        surface_row.get("predicted_motor_volts"),
    )
    amp_x_volt = _first_float(
        surface_row.get("amp_x_volt"),
        surface_row.get("scenario_amp_x_volt"),
        (motor_amps * motor_volts) if motor_amps is not None and motor_volts is not None else None,
    )

    return {
        "motor_power_kw": _first_float(
            surface_row.get("motor_power_kw"),
            surface_row.get("scenario_motor_power_kw"),
            surface_row.get("predicted_motor_power_kw"),
        ),
        "bhp_proxy": _first_float(
            surface_row.get("bhp_proxy"),
            surface_row.get("scenario_bhp_proxy"),
        ),
        "amp_x_volt": amp_x_volt,
        "motor_amps": motor_amps,
        "motor_volts": motor_volts,
    }


def _resolve_recommended_surface_power_inputs(
    ml_db,
    latest_row: Dict,
    compare_row: Dict,
) -> Dict:
    """Resolve recommended-point power inputs from matched recommendation_surfaces row."""
    if ml_db is None:
        return {"row": {}, "reason": "recommended surface lookup unavailable (ml_db not provided)."}

    org = latest_row.get("organization_id")
    well = latest_row.get("well_id")
    recommendation_uuid = latest_row.get("uuid") or latest_row.get("recommendation_uuid")
    rec_freq = _to_float(compare_row.get("rec_motor_frequency_hz"))
    rec_tubing = _to_float(compare_row.get("rec_tubing_pressure_psi"))

    if not org or not well:
        return {"row": {}, "reason": "recommended surface lookup skipped: missing organization_id/well_id."}
    if not recommendation_uuid:
        return {"row": {}, "reason": "recommended surface lookup skipped: missing uuid from latest recommendation row."}
    if rec_freq is None or rec_tubing is None:
        return {
            "row": {},
            "reason": "recommended surface lookup skipped: missing recommended frequency or tubing pressure.",
        }

    row = ml_db.get_recommended_surface_point_row(
        organization_id=str(org),
        well_id=str(well),
        recommendation_uuid=str(recommendation_uuid),
        motor_frequency_hz=float(rec_freq),
        tubing_pressure_psi=float(rec_tubing),
    )
    if not row:
        return {
            "row": {},
            "reason": "recommended surface row not found for uuid + recommended frequency + recommended tubing pressure.",
        }

    power_inputs = _extract_surface_power_inputs(row)
    has_power = any(
        _to_float(power_inputs.get(k)) is not None
        for k in ["motor_power_kw", "bhp_proxy", "amp_x_volt"]
    )

    return {
        "row": row,
        "reason": None if has_power else "recommended surface row found but recommended power fields are missing.",
        "power_inputs": power_inputs,
    }


def _extract_affinity_power_inputs(
    compare_row: Dict,
    current_text: Optional[str],
    optimal_text: Optional[str],
    summary_payload,
    method: str,
    recommended_surface_power: Optional[Dict] = None,
):
    """Resolve optional current/recommended power values for affinity check.

    Preference order:
    1) direct motor_power_kw (current + recommended)
    2) bhp_proxy (current + recommended)
    3) amp_x_volt (current + recommended)
    """
    current_map = ml_recommendation_calcs.parse_setpoint_like_map(current_text)
    optimal_map = ml_recommendation_calcs.parse_setpoint_like_map(optimal_text)
    rec_map = optimal_map.get(method, {}) if isinstance(optimal_map, dict) else {}
    summary_map = ml_recommendation_calcs.parse_json_summary_data(summary_payload)

    cur_power_direct = _first_float(
        compare_row.get("cur_motor_power_kw"),
        current_map.get("motor_power_kw") if isinstance(current_map, dict) else None,
        summary_map.get("motor_power_kw_1d_avg") if isinstance(summary_map, dict) else None,
        summary_map.get("motor_power_kw") if isinstance(summary_map, dict) else None,
    )
    rec_power_direct = _first_float(
        compare_row.get("rec_motor_power_kw"),
        rec_map.get("motor_power_kw") if isinstance(rec_map, dict) else None,
        (recommended_surface_power or {}).get("motor_power_kw"),
    )
    if cur_power_direct is not None and rec_power_direct is not None:
        return {
            "current_power": cur_power_direct,
            "recommended_power": rec_power_direct,
            "power_source_label": "motor_power_kw",
        }

    cur_bhp_proxy = _first_float(
        compare_row.get("cur_bhp_proxy"),
        summary_map.get("cur_bhp_proxy") if isinstance(summary_map, dict) else None,
        summary_map.get("bhp_proxy") if isinstance(summary_map, dict) else None,
        current_map.get("bhp_proxy") if isinstance(current_map, dict) else None,
    )
    rec_bhp_proxy = _first_float(
        compare_row.get("rec_bhp_proxy"),
        rec_map.get("bhp_proxy") if isinstance(rec_map, dict) else None,
        (recommended_surface_power or {}).get("bhp_proxy"),
    )
    if cur_bhp_proxy is not None and rec_bhp_proxy is not None:
        return {
            "current_power": cur_bhp_proxy,
            "recommended_power": rec_bhp_proxy,
            "power_source_label": "bhp_proxy",
        }

    cur_amp = _first_float(
        compare_row.get("cur_amp_x_volt"),
        current_map.get("amp_x_volt") if isinstance(current_map, dict) else None,
        _first_float(
            summary_map.get("motor_amps_1h_avg") if isinstance(summary_map, dict) else None,
            summary_map.get("motor_amps_1d_avg") if isinstance(summary_map, dict) else None,
            summary_map.get("motor_amps") if isinstance(summary_map, dict) else None,
        )
        * _first_float(
            summary_map.get("motor_volts_1h_avg") if isinstance(summary_map, dict) else None,
            summary_map.get("motor_volts_1d_avg") if isinstance(summary_map, dict) else None,
            summary_map.get("motor_volts") if isinstance(summary_map, dict) else None,
        )
        if isinstance(summary_map, dict)
        and _first_float(
            summary_map.get("motor_amps_1h_avg"),
            summary_map.get("motor_amps_1d_avg"),
            summary_map.get("motor_amps"),
        )
        is not None
        and _first_float(
            summary_map.get("motor_volts_1h_avg"),
            summary_map.get("motor_volts_1d_avg"),
            summary_map.get("motor_volts"),
        )
        is not None
        else None,
    )
    rec_amp = _first_float(
        compare_row.get("rec_amp_x_volt"),
        rec_map.get("amp_x_volt") if isinstance(rec_map, dict) else None,
        (recommended_surface_power or {}).get("amp_x_volt"),
    )
    if cur_amp is not None and rec_amp is not None:
        return {
            "current_power": cur_amp,
            "recommended_power": rec_amp,
            "power_source_label": "amp_x_volt",
        }

    return {
        "current_power": None,
        "recommended_power": None,
        "power_source_label": None,
    }


def _proxy_kw_from_amp_x_volt(amp_x_volt: Optional[float]) -> Optional[float]:
    """Convert amp_x_volt proxy to kW using fixed PF assumptions."""
    amp_volt = _to_float(amp_x_volt)
    if amp_volt is None:
        return None
    if amp_volt <= 0:
        return None
    proxy_hp = (np.sqrt(3.0) * amp_volt * DEFAULT_POWER_FACTOR) / WATTS_PER_HP
    if proxy_hp <= 0:
        return None
    return proxy_hp * WATTS_PER_HP / 1000.0


def _extract_energy_state_inputs(
    compare_row: Dict,
    current_text: Optional[str],
    optimal_text: Optional[str],
    summary_payload,
    method: str,
    state: str,
    recommended_surface_power: Optional[Dict] = None,
) -> Dict:
    """Resolve per-state direct/proxy power candidates for energy diagnostic."""
    current_map = ml_recommendation_calcs.parse_setpoint_like_map(current_text)
    optimal_map = ml_recommendation_calcs.parse_setpoint_like_map(optimal_text)
    rec_map = optimal_map.get(method, {}) if isinstance(optimal_map, dict) else {}
    summary_map = ml_recommendation_calcs.parse_json_summary_data(summary_payload)

    if state == "current":
        liquid_rate_bpd = _first_float(compare_row.get("cur_liquid_rate_bpd"))
        oil_rate_bpd = _first_float(compare_row.get("cur_oil"))
        delta_p_pump_psi = _first_float(compare_row.get("cur_delta_p_pump_psi"))
        motor_power_kw = _first_float(
            compare_row.get("cur_motor_power_kw"),
            current_map.get("motor_power_kw") if isinstance(current_map, dict) else None,
            summary_map.get("motor_power_kw_1d_avg") if isinstance(summary_map, dict) else None,
            summary_map.get("motor_power_kw") if isinstance(summary_map, dict) else None,
        )
        bhp_proxy_hp = _first_float(
            compare_row.get("cur_bhp_proxy"),
            current_map.get("bhp_proxy") if isinstance(current_map, dict) else None,
            summary_map.get("bhp_proxy") if isinstance(summary_map, dict) else None,
            summary_map.get("cur_bhp_proxy") if isinstance(summary_map, dict) else None,
        )
        amp_x_volt = _first_float(
            compare_row.get("cur_amp_x_volt"),
            current_map.get("amp_x_volt") if isinstance(current_map, dict) else None,
            _first_float(
                summary_map.get("motor_amps_1h_avg") if isinstance(summary_map, dict) else None,
                summary_map.get("motor_amps_1d_avg") if isinstance(summary_map, dict) else None,
                summary_map.get("motor_amps") if isinstance(summary_map, dict) else None,
            )
            * _first_float(
                summary_map.get("motor_volts_1h_avg") if isinstance(summary_map, dict) else None,
                summary_map.get("motor_volts_1d_avg") if isinstance(summary_map, dict) else None,
                summary_map.get("motor_volts") if isinstance(summary_map, dict) else None,
            )
            if isinstance(summary_map, dict)
            and _first_float(
                summary_map.get("motor_amps_1h_avg"),
                summary_map.get("motor_amps_1d_avg"),
                summary_map.get("motor_amps"),
            )
            is not None
            and _first_float(
                summary_map.get("motor_volts_1h_avg"),
                summary_map.get("motor_volts_1d_avg"),
                summary_map.get("motor_volts"),
            )
            is not None
            else None,
        )
    else:
        liquid_rate_bpd = _first_float(compare_row.get("rec_liquid_rate_bpd"))
        oil_rate_bpd = _first_float(compare_row.get("rec_oil"))
        delta_p_pump_psi = _first_float(compare_row.get("rec_delta_p_pump_psi"))
        motor_power_kw = _first_float(
            compare_row.get("rec_motor_power_kw"),
            rec_map.get("motor_power_kw") if isinstance(rec_map, dict) else None,
            (recommended_surface_power or {}).get("motor_power_kw"),
        )
        bhp_proxy_hp = _first_float(
            compare_row.get("rec_bhp_proxy"),
            rec_map.get("bhp_proxy") if isinstance(rec_map, dict) else None,
            (recommended_surface_power or {}).get("bhp_proxy"),
        )
        amp_x_volt = _first_float(
            compare_row.get("rec_amp_x_volt"),
            rec_map.get("amp_x_volt") if isinstance(rec_map, dict) else None,
            (recommended_surface_power or {}).get("amp_x_volt"),
        )

    proxy_power_kw = None
    power_source_label = None
    if motor_power_kw is not None:
        power_source_label = "motor_power_kw"
    elif bhp_proxy_hp is not None:
        proxy_power_kw = bhp_proxy_hp * WATTS_PER_HP / 1000.0
        power_source_label = "bhp_proxy"
    else:
        proxy_power_kw = _proxy_kw_from_amp_x_volt(amp_x_volt)
        if proxy_power_kw is not None:
            power_source_label = "amp_x_volt_pf_proxy"

    return {
        "liquid_rate_bpd": liquid_rate_bpd,
        "oil_rate_bpd": oil_rate_bpd,
        "delta_p_pump_psi": delta_p_pump_psi,
        "motor_power_kw": motor_power_kw,
        "proxy_power_kw": proxy_power_kw,
        "bhp_proxy_hp": bhp_proxy_hp,
        "amp_x_volt": amp_x_volt,
        "power_source_label": power_source_label,
    }


def _build_energy_efficiency_state_diagnostic(
    state_inputs: Dict,
    state_label: str,
) -> Dict:
    """Build one state (current/recommended) energy diagnostic plus gate details."""
    proxy_power_kw = _to_float(state_inputs.get("proxy_power_kw"))
    gate_context = {
        "field_values": {
            "liquid_rate_bpd": state_inputs.get("liquid_rate_bpd"),
            "delta_p_pump_psi": state_inputs.get("delta_p_pump_psi"),
            "motor_power_kw": state_inputs.get("motor_power_kw"),
            "bhp_proxy": state_inputs.get("bhp_proxy_hp"),
            "amp_x_volt": state_inputs.get("amp_x_volt"),
            "energy_proxy_power_kw": proxy_power_kw,
        },
        "proxies_available": (
            {"energy_power_from_proxy"} if proxy_power_kw is not None and proxy_power_kw > 0 else set()
        ),
    }
    report = run_data_availability_gate(
        context=gate_context,
        calculation_keys=["energy_efficiency_diagnostic"],
    )[0]

    diagnostic = compute_energy_efficiency_diagnostic(
        liquid_rate_bpd=state_inputs.get("liquid_rate_bpd"),
        delta_p_psi=state_inputs.get("delta_p_pump_psi"),
        motor_power_kw=state_inputs.get("motor_power_kw"),
        proxy_power_kw=state_inputs.get("proxy_power_kw"),
        oil_rate_bpd=state_inputs.get("oil_rate_bpd"),
        power_source_label=state_inputs.get("power_source_label"),
    )
    if (not diagnostic.get("available", False)) and state_inputs.get("unavailable_hint"):
        diagnostic["reason_unavailable"] = (
            f"{diagnostic.get('reason_unavailable')} {state_inputs.get('unavailable_hint')}"
        ).strip()
    diagnostic["state"] = state_label
    diagnostic["gate"] = {
        "calculation": report.calculation,
        "ready": report.ready,
        "output_label": report.output_label,
        "missing_fields": report.missing_fields,
        "fallback_or_proxy_fields": report.fallback_or_proxy_fields,
        "field_resolutions": [
            {
                "required_field": res.required_field,
                "status": res.status,
                "source_or_resolution": res.source_or_resolution,
                "required": res.required,
                "ready": res.ready,
                "notes": res.notes,
            }
            for res in report.field_resolutions
        ],
    }
    return diagnostic


def _build_energy_efficiency_diagnostic(
    compare_row: Dict,
    latest_row: Dict,
    summary_payload,
    method: str,
    recommended_surface_power: Optional[Dict] = None,
    recommended_surface_reason: Optional[str] = None,
) -> Dict:
    """Build current/recommended Energy/Efficiency diagnostic payload."""
    current_inputs = _extract_energy_state_inputs(
        compare_row=compare_row,
        current_text=latest_row.get("current_setpoint"),
        optimal_text=latest_row.get("model_setpoint_recommendations"),
        summary_payload=summary_payload,
        method=method,
        state="current",
        recommended_surface_power=recommended_surface_power,
    )
    recommended_inputs = _extract_energy_state_inputs(
        compare_row=compare_row,
        current_text=latest_row.get("current_setpoint"),
        optimal_text=latest_row.get("model_setpoint_recommendations"),
        summary_payload=summary_payload,
        method=method,
        state="recommended",
        recommended_surface_power=recommended_surface_power,
    )
    if recommended_surface_reason:
        recommended_inputs["unavailable_hint"] = recommended_surface_reason

    return {
        "current": _build_energy_efficiency_state_diagnostic(current_inputs, "current"),
        "recommended": _build_energy_efficiency_state_diagnostic(recommended_inputs, "recommended"),
    }


def _build_gas_interference_trend_screen(
    ml_db,
    organization_id: Optional[str],
    well_id: Optional[str],
    well_depth_ft: float,
    sg_oil: float,
    sg_water: float,
) -> Dict:
    """Build screening-only gas-interference trend diagnostic from preprocessed history."""
    unavailable_base = {
        "available": False,
        "reason_unavailable": "Gas-interference trend screen unavailable.",
        "mode": "Insufficient data",
        "risk_label": "Insufficient data",
        "trust_label": "Screening prototype",
        "time_window_summary": {},
        "trend_statistics": {},
        "evidence_table": [],
        "triggered_evidence": [],
        "missing_optional_signals": [],
        "notes": [
            "This is a trend-based gas-interference screening diagnostic.",
            "It does not confirm gas lock or calculate true downhole gas volume fraction.",
        ],
    }

    if not organization_id or not well_id:
        out = dict(unavailable_base)
        out["reason_unavailable"] = "Missing organization_id or well_id for preprocessed-history lookup."
        return out

    try:
        profile_name = getattr(ml_db, "profile_name", "roam-ai")
        region_name = getattr(ml_db, "region_name", "us-east-1")
        s3_output = getattr(ml_db, "s3_output", "s3://esp-athena-results-v2-411237692998/")
        session = getattr(ml_db, "session", None)

        preprocessed_db = PreprocessedDataDB(
            profile_name=profile_name,
            region_name=region_name,
            s3_output=s3_output,
        )
        if session is not None:
            preprocessed_db.session = session
            preprocessed_db.client = session.client("athena")
        else:
            ok, msg = preprocessed_db.connect()
            if not ok:
                out = dict(unavailable_base)
                out["reason_unavailable"] = f"Failed to connect preprocessed data source: {msg}"
                return out

        df_telemetry = preprocessed_db.fetch_telemetry_for_well(organization_id=organization_id, well_id=well_id)
        df_production = preprocessed_db.fetch_production_for_well(organization_id=organization_id, well_id=well_id)
        if df_telemetry is None or df_telemetry.empty:
            out = dict(unavailable_base)
            out["reason_unavailable"] = "No historical preprocessed telemetry rows found for selected well."
            return out
        if df_production is None or df_production.empty:
            out = dict(unavailable_base)
            out["reason_unavailable"] = "No historical preprocessed production rows found for selected well."
            return out

        analyzed_df, _ = preprocessed_pipeline_service.run_preprocessed_analysis(
            df_telemetry=df_telemetry,
            df_production=df_production,
            well_depth_ft=well_depth_ft,
            sg_oil=sg_oil,
            sg_water=sg_water,
        )

        gate_context = {
            "field_values": {
                "historical_row_count": len(analyzed_df),
            },
            "dataframes": {
                "historical_preprocessed": analyzed_df,
            },
            "fallbacks_available": {
                "derive_gor_from_alloc",
                "pump_behavior_from_liquid_rate",
            },
        }
        gate_report = run_data_availability_gate(
            context=gate_context,
            calculation_keys=["gas_interference_trend_screen"],
        )[0]

        diagnostic = compute_gas_interference_trend_screen(historical_df=analyzed_df)
        diagnostic["gate"] = {
            "calculation": gate_report.calculation,
            "ready": gate_report.ready,
            "output_label": gate_report.output_label,
            "missing_fields": gate_report.missing_fields,
            "fallback_or_proxy_fields": gate_report.fallback_or_proxy_fields,
            "field_resolutions": [
                {
                    "required_field": res.required_field,
                    "status": res.status,
                    "source_or_resolution": res.source_or_resolution,
                    "required": res.required,
                    "ready": res.ready,
                    "notes": res.notes,
                }
                for res in gate_report.field_resolutions
            ],
        }

        if not gate_report.ready and diagnostic.get("available"):
            blocked_required = [
                res.required_field for res in gate_report.field_resolutions if res.status == "Blocked"
            ]
            diagnostic["available"] = False
            diagnostic["mode"] = "Insufficient data"
            diagnostic["risk_label"] = "Insufficient data"
            diagnostic["reason_unavailable"] = (
                "Data availability gate blocked required inputs: " + ", ".join(blocked_required)
            )
        return diagnostic
    except Exception as exc:
        out = dict(unavailable_base)
        out["reason_unavailable"] = f"Gas-interference trend screen failed: {exc}"
        return out


def _build_affinity_law_diagnostic(
    compare_row: Dict,
    latest_row: Dict,
    summary_payload,
    method: str,
    recommended_surface_power: Optional[Dict] = None,
):
    """Build Affinity Law diagnostic payload with gate readiness context."""
    current_freq = _to_float(compare_row.get("cur_motor_frequency_hz"))
    recommended_freq = _to_float(compare_row.get("rec_motor_frequency_hz"))
    current_liquid = _to_float(compare_row.get("cur_liquid_rate_bpd"))
    recommended_liquid = _to_float(compare_row.get("rec_liquid_rate_bpd"))
    current_delta_p = _to_float(compare_row.get("cur_delta_p_pump_psi"))
    recommended_delta_p = _to_float(compare_row.get("rec_delta_p_pump_psi"))

    power_inputs = _extract_affinity_power_inputs(
        compare_row=compare_row,
        current_text=latest_row.get("current_setpoint"),
        optimal_text=latest_row.get("model_setpoint_recommendations"),
        summary_payload=summary_payload,
        method=method,
        recommended_surface_power=recommended_surface_power,
    )

    gate_context = {
        "field_values": {
            "current_motor_frequency_hz": current_freq,
            "recommended_motor_frequency_hz": recommended_freq,
            "current_liquid_rate_bpd": current_liquid,
            "recommended_liquid_rate_bpd": recommended_liquid,
            "current_delta_p_pump_psi": current_delta_p,
            "recommended_delta_p_pump_psi": recommended_delta_p,
            "motor_power_kw": (
                power_inputs.get("current_power")
                if power_inputs.get("power_source_label") == "motor_power_kw"
                else None
            ),
            "bhp_proxy": (
                power_inputs.get("current_power")
                if power_inputs.get("power_source_label") == "bhp_proxy"
                else None
            ),
            "amp_x_volt": (
                power_inputs.get("current_power")
                if power_inputs.get("power_source_label") == "amp_x_volt"
                else None
            ),
        },
        "proxies_available": {"bhp_proxy", "amp_x_volt"},
    }
    report = run_data_availability_gate(
        context=gate_context,
        calculation_keys=["affinity_law_validator"],
    )[0]

    if not report.ready:
        blocked_required = [
            res.required_field for res in report.field_resolutions if res.status == "Blocked"
        ]
        diagnostic = {
            "available": False,
            "reason_unavailable": "Missing required affinity inputs: " + ", ".join(blocked_required),
            "mode": "Unavailable",
            "trust_label": "Diagnostic",
            "current_frequency_hz": current_freq,
            "recommended_frequency_hz": recommended_freq,
            "frequency_delta_hz": None,
            "speed_ratio": None,
            "frequency_change_label": None,
            "flow_check": {"available": False, "agreement_label": "Unavailable"},
            "pressure_check": {"available": False, "agreement_label": "Unavailable"},
            "power_check": {
                "available": False,
                "agreement_label": "Unavailable",
                "power_source_label": power_inputs.get("power_source_label"),
            },
            "overall_label": "Unavailable",
            "notes": [
                "This is a first-order Affinity Law sanity check. It is not a full multiphase pump simulation and should be interpreted as diagnostic evidence only.",
                "Required inputs were blocked, so the diagnostic did not run.",
            ],
        }
    else:
        diagnostic = compute_affinity_law_validator(
            current_frequency_hz=current_freq,
            recommended_frequency_hz=recommended_freq,
            current_liquid_rate_bpd=current_liquid,
            recommended_liquid_rate_bpd=recommended_liquid,
            current_delta_p_psi=current_delta_p,
            recommended_delta_p_psi=recommended_delta_p,
            current_power=power_inputs.get("current_power"),
            recommended_power=power_inputs.get("recommended_power"),
            power_source_label=power_inputs.get("power_source_label"),
        )

    diagnostic["gate"] = {
        "calculation": report.calculation,
        "ready": report.ready,
        "output_label": report.output_label,
        "missing_fields": report.missing_fields,
        "fallback_or_proxy_fields": report.fallback_or_proxy_fields,
        "field_resolutions": [
            {
                "required_field": res.required_field,
                "status": res.status,
                "source_or_resolution": res.source_or_resolution,
                "required": res.required,
                "ready": res.ready,
                "notes": res.notes,
            }
            for res in report.field_resolutions
        ],
    }
    return diagnostic


def select_pump_row(catalog_df: pd.DataFrame, manufacturer: str, series: str, esp_model: str) -> pd.Series:
    """Return one pump row from catalog using case-insensitive matching."""
    sel = catalog_df[
        catalog_df["manufacturer"].astype(str).str.strip().str.lower().eq(str(manufacturer).strip().lower())
        & catalog_df["series"].astype(str).str.strip().str.lower().eq(str(series).strip().lower())
        & catalog_df["esp_model"].astype(str).str.strip().str.lower().eq(str(esp_model).strip().lower())
    ]
    if sel.empty:
        raise ValueError(f"Pump not found: {manufacturer} / {series} / {esp_model}")
    return sel.iloc[0]


def _pump_label_from_row(pump_row: pd.Series) -> str:
    """Build a stable display label for selected pump row metadata."""
    manufacturer = str(pump_row.get("manufacturer", "")).strip()
    series = str(pump_row.get("series", "")).strip()
    model = str(pump_row.get("esp_model", "")).strip()
    parts = [p for p in [manufacturer, series, model] if p]
    return " / ".join(parts) if parts else None


def build_analysis_from_latest_row(
    latest_row: Dict,
    pump_row: pd.Series,
    stages: int,
    ml_db=None,
    sg_for_dp: float = 1.0,
    well_depth_ft: float = 5000.0,
    sg_oil: float = 0.85,
    sg_water: float = 1.00,
    method: str = "max_oil",
    sweep_freqs: Optional[Iterable[float]] = None,
) -> Dict:
    """Build full ML recommendation analysis payload from latest Athena row.

    Note:
        sg_for_dp defaults to 1.0 when not provided by the caller. This is an
        SG assumption for pressure conversion, not a per-well validated value.
    """
    if not latest_row:
        raise ValueError("No latest row found for selected organization/well")

    optimal_text = latest_row.get("model_setpoint_recommendations")
    current_text = latest_row.get("current_setpoint")
    if optimal_text is None or current_text is None:
        raise ValueError("Latest row is missing model_setpoint_recommendations or current_setpoint")

    summary_key, summary_payload = _extract_json_summary_payload(latest_row)

    compare_row = ml_recommendation_calcs.extract_compare_row(
        optimal_text=optimal_text,
        current_text=current_text,
        method=method,
    )
    compare_row = ml_recommendation_calcs.augment_with_delta_p_pump(
        compare_row=compare_row,
        json_summary_text=summary_payload,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
    )
    recommended_surface_power = _resolve_recommended_surface_power_inputs(
        ml_db=ml_db,
        latest_row=latest_row,
        compare_row=compare_row,
    )
    summary_table = ml_recommendation_calcs.build_summary_table(compare_row)
    curve_payload = ml_recommendation_calcs.build_curve_payload(
        compare_row=compare_row,
        pump_row=pump_row,
        stages=stages,
        sg_for_dp=sg_for_dp,
        sweep_freqs=sweep_freqs,
    )

    bep_diagnostic = ideal_curve_overlay.compute_bep_position_diagnostic(
        current_flow_bpd=compare_row.get("cur_liquid_rate_bpd"),
        recommended_flow_bpd=compare_row.get("rec_liquid_rate_bpd"),
        bep_bpd=pump_row.get("bep_bpd"),
        min_recommended_bpd=pump_row.get("min_recommended_bpd"),
        max_recommended_bpd=pump_row.get("max_recommended_bpd"),
        pump_label=_pump_label_from_row(pump_row),
        pump_source="selected_catalog_candidate",
    )
    affinity_law_diagnostic = _build_affinity_law_diagnostic(
        compare_row=compare_row,
        latest_row=latest_row,
        summary_payload=summary_payload,
        method=method,
        recommended_surface_power=(recommended_surface_power or {}).get("power_inputs", {}),
    )
    energy_efficiency_diagnostic = _build_energy_efficiency_diagnostic(
        compare_row=compare_row,
        latest_row=latest_row,
        summary_payload=summary_payload,
        method=method,
        recommended_surface_power=(recommended_surface_power or {}).get("power_inputs", {}),
        recommended_surface_reason=(recommended_surface_power or {}).get("reason"),
    )
    gas_interference_trend_screen = _build_gas_interference_trend_screen(
        ml_db=ml_db,
        organization_id=latest_row.get("organization_id"),
        well_id=latest_row.get("well_id"),
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
    )

    timestamp_value = latest_row.get("timestamp")
    if isinstance(timestamp_value, float) and np.isnan(timestamp_value):
        timestamp_value = None

    return {
        "compare_row": compare_row,
        "summary_table": summary_table,
        "curve_payload": curve_payload,
        "bep_diagnostic": bep_diagnostic,
        "affinity_law_diagnostic": affinity_law_diagnostic,
        "energy_efficiency_diagnostic": energy_efficiency_diagnostic,
        "gas_interference_trend_screen": gas_interference_trend_screen,
        "metadata": {
            "organization_id": latest_row.get("organization_id"),
            "well_id": latest_row.get("well_id"),
            "timestamp": timestamp_value,
            "method_used": method,
        },
        "raw_payload": {
            "current_setpoint": current_text,
            "model_setpoint_recommendations": optimal_text,
            "json_summary_data": summary_payload,
            "recommended_surface_lookup": {
                "reason": (recommended_surface_power or {}).get("reason"),
                "row_found": bool((recommended_surface_power or {}).get("row")),
            },
            "json_summary_column_used": summary_key,
            "latest_row_keys": sorted([str(k) for k in latest_row.keys()]),
        },
    }


def _nearest_surface_scenario_id(
    grid_payload: Dict,
    frequency_hz: Optional[float],
    tubing_pressure_psi: Optional[float],
) -> Optional[str]:
    """Return nearest scenario_id by (frequency, tubing pressure) distance."""
    points_df = grid_payload.get("points_df")
    if points_df is None or points_df.empty:
        return None
    if frequency_hz is None or tubing_pressure_psi is None:
        return None

    freq = float(frequency_hz)
    tubing = float(tubing_pressure_psi)

    df = points_df.copy()
    df = df[df["motor_frequency_hz"].notna() & df["tubing_pressure_psi"].notna()].copy()
    if df.empty:
        return None

    df["_dist2"] = (
        (df["motor_frequency_hz"].astype(float) - freq) ** 2
        + (df["tubing_pressure_psi"].astype(float) - tubing) ** 2
    )
    idx = df["_dist2"].idxmin()
    if idx is None:
        return None
    return str(df.loc[idx, "scenario_id"])


def build_grid_analysis_payload(
    ml_db,
    organization_id: str,
    well_id: str,
    pump_row: pd.Series,
    stages: int,
    sg_for_dp: float = 1.0,
    well_depth_ft: float = 5000.0,
    sg_oil: float = 0.85,
    sg_water: float = 1.00,
    method: str = "max_oil",
    sweep_freqs: Optional[Iterable[float]] = None,
) -> Dict:
    """Build V1 ML-grid payload: latest recommendation + latest surface run + anchors.

    This function preserves the existing single-point recommendation analysis output
    and augments it with surface-grid data and current/recommended anchor mapping.
    """
    latest_row = ml_db.get_latest_recommendation_row(organization_id=organization_id, well_id=well_id)
    single_point_payload = build_analysis_from_latest_row(
        latest_row=latest_row,
        pump_row=pump_row,
        stages=stages,
        ml_db=ml_db,
        sg_for_dp=sg_for_dp,
        well_depth_ft=well_depth_ft,
        sg_oil=sg_oil,
        sg_water=sg_water,
        method=method,
        sweep_freqs=sweep_freqs,
    )

    latest_surface_run = ml_db.get_latest_surface_run(organization_id=organization_id, well_id=well_id)
    latest_run_id = latest_surface_run.get("recommendation_uuid")

    surface_rows = ml_db.get_latest_recommendation_surface_rows(
        organization_id=organization_id,
        well_id=well_id,
    )
    grid_payload = ml_recommendation_calcs.build_recommendation_surface_grid_payload(surface_rows)

    fallback_info = {
        "activated": False,
        "reason": None,
        "latest_run": {
            "recommendation_uuid": latest_surface_run.get("recommendation_uuid"),
            "inserted_at": latest_surface_run.get("inserted_at"),
            "inserted_ts": latest_surface_run.get("inserted_ts"),
        },
        "selected_run": {
            "recommendation_uuid": latest_surface_run.get("recommendation_uuid"),
            "inserted_at": latest_surface_run.get("inserted_at"),
            "inserted_ts": latest_surface_run.get("inserted_ts"),
        },
        "latest_run_violation_reasons": [],
    }

    stats = grid_payload.get("stats", {})
    latest_total = int(stats.get("n_points", 0) or 0)
    latest_in_bounds = int(stats.get("n_in_bounds", 0) or 0)

    if latest_run_id and latest_total > 0 and latest_in_bounds == 0:
        violation_df = ml_db.get_surface_violation_breakdown_for_run(
            organization_id=organization_id,
            well_id=well_id,
            recommendation_uuid=str(latest_run_id),
        )
        fallback_info["latest_run_violation_reasons"] = violation_df.to_dict(orient="records")

        viable_run = ml_db.get_latest_viable_surface_run(
            organization_id=organization_id,
            well_id=well_id,
        )

        viable_id = viable_run.get("recommendation_uuid")
        if viable_id and str(viable_id) != str(latest_run_id):
            fallback_rows = ml_db.get_recommendation_surface_rows_for_run(
                organization_id=organization_id,
                well_id=well_id,
                recommendation_uuid=str(viable_id),
            )
            fallback_grid_payload = ml_recommendation_calcs.build_recommendation_surface_grid_payload(
                fallback_rows
            )
            fallback_stats = fallback_grid_payload.get("stats", {})
            if int(fallback_stats.get("n_in_bounds", 0) or 0) > 0:
                grid_payload = fallback_grid_payload
                fallback_info["activated"] = True
                fallback_info["reason"] = "latest_run_all_out_of_bounds"
                fallback_info["selected_run"] = {
                    "recommendation_uuid": viable_run.get("recommendation_uuid"),
                    "inserted_at": viable_run.get("inserted_at"),
                    "inserted_ts": viable_run.get("inserted_ts"),
                }

    compare_row = single_point_payload.get("compare_row", {})
    cur_freq = compare_row.get("cur_motor_frequency_hz")
    cur_tubing = compare_row.get("cur_tubing_pressure_psi")
    rec_freq = compare_row.get("rec_motor_frequency_hz")
    rec_tubing = compare_row.get("rec_tubing_pressure_psi")

    current_anchor_id = _nearest_surface_scenario_id(
        grid_payload=grid_payload,
        frequency_hz=cur_freq,
        tubing_pressure_psi=cur_tubing,
    )
    recommended_anchor_id = _nearest_surface_scenario_id(
        grid_payload=grid_payload,
        frequency_hz=rec_freq,
        tubing_pressure_psi=rec_tubing,
    )

    selectable_ids = set(grid_payload.get("selectable_scenario_ids", []))

    return {
        "single_point": single_point_payload,
        "grid": {
            "payload": grid_payload,
            "latest_surface_run": fallback_info.get("selected_run", {}),
            "fallback": fallback_info,
            "anchors": {
                "current": {
                    "scenario_id": current_anchor_id,
                    "in_bounds": bool(current_anchor_id in selectable_ids) if current_anchor_id else False,
                },
                "recommended": {
                    "scenario_id": recommended_anchor_id,
                    "in_bounds": bool(recommended_anchor_id in selectable_ids) if recommended_anchor_id else False,
                },
            },
        },
    }
