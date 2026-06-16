"""Energy / Efficiency diagnostic helpers for ML recommendation analysis.

Diagnostic-only compute layer:
- No Streamlit, DB, plotting, or session-state dependencies.
- Uses pump delta-P basis (not full TDH).
- Keeps direct-power and proxy-power efficiencies separated.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from compute.physics_common import HHP_DENOMINATOR, WATTS_PER_HP


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _base_unavailable(reason: str) -> Dict:
    return {
        "available": False,
        "reason_unavailable": reason,
        "mode": "Unavailable",
        "power_source_label": None,
        "trust_label": "Unavailable",
        "liquid_rate_bpd": None,
        "oil_rate_bpd": None,
        "delta_p_psi": None,
        "hydraulic_hp_estimate": None,
        "hydraulic_kw_estimate": None,
        "motor_power_kw": None,
        "proxy_power_kw": None,
        "direct_power_efficiency_pct": None,
        "proxy_power_efficiency_pct": None,
        "specific_power_kwh_per_liquid_bbl": None,
        "specific_power_kwh_per_oil_bbl": None,
        "notes": [
            "This diagnostic uses pump delta-P and available power inputs.",
            "Proxy-power efficiency is not a measured ESP efficiency and should be interpreted as a relative diagnostic only.",
        ],
    }


def compute_energy_efficiency_diagnostic(
    liquid_rate_bpd: float,
    delta_p_psi: float,
    motor_power_kw: Optional[float] = None,
    proxy_power_hp: Optional[float] = None,
    proxy_power_kw: Optional[float] = None,
    oil_rate_bpd: Optional[float] = None,
    power_source_label: Optional[str] = None,
) -> Dict:
    """Compute Energy / Efficiency diagnostic with direct/proxy separation.

    Formula basis (V1):
    - hydraulic_hp_estimate = (liquid_rate_bpd * delta_p_psi) / HHP_DENOMINATOR
    - hydraulic_kw_estimate = hydraulic_hp_estimate * WATTS_PER_HP / 1000

    Direct mode (preferred):
    - direct_power_efficiency_pct = hydraulic_kw_estimate / motor_power_kw * 100

    Proxy mode (fallback):
    - proxy_power_efficiency_pct = hydraulic_kw_estimate / proxy_power_kw * 100

    Specific power:
    - specific_power_kwh_per_liquid_bbl = power_kw * 24 / liquid_rate_bpd
    - specific_power_kwh_per_oil_bbl = power_kw * 24 / oil_rate_bpd (optional)
    """
    q_liq = _to_float(liquid_rate_bpd)
    dp = _to_float(delta_p_psi)
    motor_kw = _to_float(motor_power_kw)
    proxy_kw_input = _to_float(proxy_power_kw)
    proxy_hp_input = _to_float(proxy_power_hp)
    oil_q = _to_float(oil_rate_bpd)

    if q_liq is None or q_liq <= 0:
        return _base_unavailable("Missing or invalid liquid_rate_bpd (must be > 0).")

    if dp is None or dp <= 0:
        return _base_unavailable("Missing or invalid delta_p_psi (must be > 0).")

    hydraulic_hp = (q_liq * dp) / HHP_DENOMINATOR
    hydraulic_kw = hydraulic_hp * WATTS_PER_HP / 1000.0

    selected_mode = "Unavailable"
    trust_label = "Unavailable"
    selected_source = None
    selected_power_kw = None
    direct_eff = None
    proxy_eff = None

    # Power-source hierarchy: direct motor kW first, then proxy power.
    if motor_kw is not None:
        if motor_kw <= 0:
            return _base_unavailable("Selected direct power source motor_power_kw is invalid (must be > 0).")
        selected_mode = "Direct power"
        trust_label = "Estimated"
        selected_source = power_source_label or "motor_power_kw"
        selected_power_kw = motor_kw
        direct_eff = (hydraulic_kw / motor_kw) * 100.0
    else:
        resolved_proxy_kw = None
        resolved_proxy_label = None

        if proxy_kw_input is not None:
            resolved_proxy_kw = proxy_kw_input
            resolved_proxy_label = "proxy_power_kw"
        elif proxy_hp_input is not None:
            resolved_proxy_kw = proxy_hp_input * WATTS_PER_HP / 1000.0
            resolved_proxy_label = "proxy_power_hp"

        if resolved_proxy_kw is None:
            return _base_unavailable(
                "No direct or proxy power source available; efficiency cannot be computed."
            )
        if resolved_proxy_kw <= 0:
            return _base_unavailable("Selected proxy power source is invalid (must be > 0).")

        selected_mode = "Proxy power"
        trust_label = "Proxy"
        selected_source = power_source_label or resolved_proxy_label
        selected_power_kw = resolved_proxy_kw
        proxy_eff = (hydraulic_kw / resolved_proxy_kw) * 100.0

    specific_liq = (selected_power_kw * 24.0 / q_liq) if selected_power_kw is not None else None
    specific_oil = None
    if oil_q is not None and oil_q > 0 and selected_power_kw is not None:
        specific_oil = selected_power_kw * 24.0 / oil_q

    notes = [
        "Hydraulic power estimate is pump-delta-P-based and does not represent a full TDH/system model.",
        "This diagnostic uses pump delta-P and available power inputs.",
        "Proxy-power efficiency is not a measured ESP efficiency and should be interpreted as a relative diagnostic only.",
    ]
    if selected_mode == "Proxy power":
        notes.append("Proxy power path was used because direct motor_power_kw was unavailable.")

    return {
        "available": True,
        "reason_unavailable": None,
        "mode": selected_mode,
        "power_source_label": selected_source,
        "trust_label": trust_label,
        "liquid_rate_bpd": float(q_liq),
        "oil_rate_bpd": None if oil_q is None else float(oil_q),
        "delta_p_psi": float(dp),
        "hydraulic_hp_estimate": float(hydraulic_hp),
        "hydraulic_kw_estimate": float(hydraulic_kw),
        "motor_power_kw": None if motor_kw is None else float(motor_kw),
        "proxy_power_kw": None if selected_mode != "Proxy power" else float(selected_power_kw),
        "direct_power_efficiency_pct": None if direct_eff is None else float(direct_eff),
        "proxy_power_efficiency_pct": None if proxy_eff is None else float(proxy_eff),
        "specific_power_kwh_per_liquid_bbl": None if specific_liq is None else float(specific_liq),
        "specific_power_kwh_per_oil_bbl": None if specific_oil is None else float(specific_oil),
        "notes": notes,
    }
