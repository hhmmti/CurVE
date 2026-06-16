"""Field-level Affinity Law sanity-check diagnostic.

This module applies first-order affinity relationships to observed current field
values and compares those expectations to ML-recommended values.

Scope:
- Diagnostic/sanity check only.
- No optimization logic.
- No Streamlit, DB, plotting, or session-state dependencies.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


ALIGNED_THRESHOLD_PCT = 15.0
REVIEW_THRESHOLD_PCT = 30.0
MINIMAL_FREQ_CHANGE_HZ = 0.5


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


def _agreement_label(difference_pct: Optional[float]) -> str:
    if difference_pct is None or not np.isfinite(difference_pct):
        return "Unavailable"
    if difference_pct <= ALIGNED_THRESHOLD_PCT:
        return "Aligned"
    if difference_pct <= REVIEW_THRESHOLD_PCT:
        return "Review"
    return "Divergent"


def _pct_difference(predicted: Optional[float], observed: Optional[float]) -> Optional[float]:
    if predicted is None or observed is None:
        return None
    if not np.isfinite(predicted) or not np.isfinite(observed):
        return None
    if predicted == 0.0:
        return None
    return abs(observed - predicted) / abs(predicted) * 100.0


def _build_check(
    current_value: Optional[float],
    predicted_value: Optional[float],
    recommended_value: Optional[float],
    current_key: str,
    predicted_key: str,
    recommended_key: str,
    difference_key: str,
) -> Dict:
    if (
        current_value is None
        or predicted_value is None
        or recommended_value is None
        or not np.isfinite(current_value)
        or not np.isfinite(predicted_value)
        or not np.isfinite(recommended_value)
    ):
        return {
            "available": False,
            current_key: current_value,
            predicted_key: predicted_value,
            recommended_key: recommended_value,
            difference_key: None,
            "difference_pct": None,
            "agreement_label": "Unavailable",
        }

    difference = recommended_value - predicted_value
    difference_pct = _pct_difference(predicted_value, recommended_value)

    return {
        "available": True,
        current_key: float(current_value),
        predicted_key: float(predicted_value),
        recommended_key: float(recommended_value),
        difference_key: float(difference),
        "difference_pct": None if difference_pct is None else float(difference_pct),
        "agreement_label": _agreement_label(difference_pct),
    }


def compute_affinity_law_validator(
    current_frequency_hz: float,
    recommended_frequency_hz: float,
    current_liquid_rate_bpd: float,
    recommended_liquid_rate_bpd: float,
    current_delta_p_psi: Optional[float] = None,
    recommended_delta_p_psi: Optional[float] = None,
    current_power: Optional[float] = None,
    recommended_power: Optional[float] = None,
    power_source_label: Optional[str] = None,
) -> Dict:
    """Compute field-level Affinity Law diagnostic payload.

    Required inputs:
    - current/recommended frequency (Hz)
    - current/recommended liquid rate (bpd)

    Optional inputs:
    - current/recommended pump delta-P (psi)
    - current/recommended power (kW or proxy units; source labeled externally)
    """
    cur_freq = _to_float(current_frequency_hz)
    rec_freq = _to_float(recommended_frequency_hz)
    cur_q = _to_float(current_liquid_rate_bpd)
    rec_q = _to_float(recommended_liquid_rate_bpd)

    if cur_freq is None or cur_freq <= 0:
        return {
            "available": False,
            "reason_unavailable": "Missing or invalid current_frequency_hz (must be > 0).",
            "mode": "Unavailable",
            "trust_label": "Diagnostic",
            "current_frequency_hz": cur_freq,
            "recommended_frequency_hz": _to_float(recommended_frequency_hz),
            "frequency_delta_hz": None,
            "speed_ratio": None,
            "frequency_change_label": None,
            "flow_check": {"available": False, "agreement_label": "Unavailable"},
            "pressure_check": {"available": False, "agreement_label": "Unavailable"},
            "power_check": {"available": False, "agreement_label": "Unavailable"},
            "overall_label": "Unavailable",
            "notes": [
                "Required inputs are missing; field-level Affinity Law check was not run.",
            ],
        }

    if rec_freq is None or rec_freq <= 0:
        return {
            "available": False,
            "reason_unavailable": "Missing or invalid recommended_frequency_hz (must be > 0).",
            "mode": "Unavailable",
            "trust_label": "Diagnostic",
            "current_frequency_hz": cur_freq,
            "recommended_frequency_hz": rec_freq,
            "frequency_delta_hz": None,
            "speed_ratio": None,
            "frequency_change_label": None,
            "flow_check": {"available": False, "agreement_label": "Unavailable"},
            "pressure_check": {"available": False, "agreement_label": "Unavailable"},
            "power_check": {"available": False, "agreement_label": "Unavailable"},
            "overall_label": "Unavailable",
            "notes": [
                "Required inputs are missing; field-level Affinity Law check was not run.",
            ],
        }

    if cur_q is None or cur_q <= 0:
        return {
            "available": False,
            "reason_unavailable": "Missing or invalid current_liquid_rate_bpd (must be > 0).",
            "mode": "Unavailable",
            "trust_label": "Diagnostic",
            "current_frequency_hz": cur_freq,
            "recommended_frequency_hz": rec_freq,
            "frequency_delta_hz": None,
            "speed_ratio": None,
            "frequency_change_label": None,
            "flow_check": {"available": False, "agreement_label": "Unavailable"},
            "pressure_check": {"available": False, "agreement_label": "Unavailable"},
            "power_check": {"available": False, "agreement_label": "Unavailable"},
            "overall_label": "Unavailable",
            "notes": [
                "Required inputs are missing; field-level Affinity Law check was not run.",
            ],
        }

    if rec_q is None or rec_q <= 0:
        return {
            "available": False,
            "reason_unavailable": "Missing or invalid recommended_liquid_rate_bpd (must be > 0).",
            "mode": "Unavailable",
            "trust_label": "Diagnostic",
            "current_frequency_hz": cur_freq,
            "recommended_frequency_hz": rec_freq,
            "frequency_delta_hz": None,
            "speed_ratio": None,
            "frequency_change_label": None,
            "flow_check": {"available": False, "agreement_label": "Unavailable"},
            "pressure_check": {"available": False, "agreement_label": "Unavailable"},
            "power_check": {"available": False, "agreement_label": "Unavailable"},
            "overall_label": "Unavailable",
            "notes": [
                "Required inputs are missing; field-level Affinity Law check was not run.",
            ],
        }

    speed_ratio = rec_freq / cur_freq
    frequency_delta_hz = rec_freq - cur_freq

    predicted_flow = cur_q * speed_ratio
    flow_check = _build_check(
        current_value=cur_q,
        predicted_value=predicted_flow,
        recommended_value=rec_q,
        current_key="current_liquid_rate_bpd",
        predicted_key="affinity_predicted_liquid_rate_bpd",
        recommended_key="ml_recommended_liquid_rate_bpd",
        difference_key="difference_bpd",
    )

    cur_dp = _to_float(current_delta_p_psi)
    rec_dp = _to_float(recommended_delta_p_psi)
    predicted_dp = None
    if cur_dp is not None:
        predicted_dp = cur_dp * (speed_ratio ** 2)
    pressure_check = _build_check(
        current_value=cur_dp,
        predicted_value=predicted_dp,
        recommended_value=rec_dp,
        current_key="current_delta_p_psi",
        predicted_key="affinity_predicted_delta_p_psi",
        recommended_key="ml_recommended_delta_p_psi",
        difference_key="difference_psi",
    )

    cur_power = _to_float(current_power)
    rec_power = _to_float(recommended_power)
    predicted_power = None
    if cur_power is not None:
        predicted_power = cur_power * (speed_ratio ** 3)
    power_check = _build_check(
        current_value=cur_power,
        predicted_value=predicted_power,
        recommended_value=rec_power,
        current_key="current_power",
        predicted_key="affinity_predicted_power",
        recommended_key="recommended_power",
        difference_key="difference",
    )
    power_check["power_source_label"] = power_source_label

    if power_check.get("available"):
        mode = "Full"
    elif pressure_check.get("available"):
        mode = "Pressure"
    else:
        mode = "Flow-only"

    freq_change_label = (
        "Minimal frequency change"
        if abs(frequency_delta_hz) < MINIMAL_FREQ_CHANGE_HZ
        else "Frequency change is material"
    )

    labels = [flow_check.get("agreement_label")]
    if pressure_check.get("available"):
        labels.append(pressure_check.get("agreement_label"))
    if power_check.get("available"):
        labels.append(power_check.get("agreement_label"))

    if any(lbl == "Divergent" for lbl in labels):
        overall_label = "Divergent"
    elif any(lbl == "Review" for lbl in labels):
        overall_label = "Review"
    elif all(lbl == "Aligned" for lbl in labels if lbl is not None):
        overall_label = "Aligned"
    else:
        overall_label = "Unavailable"

    source_label_lower = str(power_source_label or "").strip().lower()
    if power_check.get("available") and source_label_lower in {"bhp_proxy", "amp_x_volt"}:
        trust_label = "Proxy"
    elif power_check.get("available"):
        trust_label = "Estimated"
    else:
        trust_label = "Diagnostic"

    notes = [
        "This is a first-order Affinity Law sanity check. It is not a full multiphase pump simulation and should be interpreted as diagnostic evidence only.",
    ]

    if abs(frequency_delta_hz) < MINIMAL_FREQ_CHANGE_HZ:
        notes.append("Minimal frequency change; avoid overinterpreting differences.")
    if not pressure_check.get("available"):
        notes.append("Pressure check unavailable: missing current/recommended pump delta-P input.")
    if not power_check.get("available"):
        notes.append("Power check unavailable: missing current/recommended power or proxy inputs.")
    elif trust_label == "Proxy":
        notes.append("Power agreement uses proxy input and should be treated as proxy evidence.")

    return {
        "available": True,
        "reason_unavailable": None,
        "mode": mode,
        "trust_label": trust_label,
        "current_frequency_hz": float(cur_freq),
        "recommended_frequency_hz": float(rec_freq),
        "frequency_delta_hz": float(frequency_delta_hz),
        "speed_ratio": float(speed_ratio),
        "frequency_change_label": freq_change_label,
        "flow_check": flow_check,
        "pressure_check": pressure_check,
        "power_check": power_check,
        "overall_label": overall_label,
        "notes": notes,
    }
