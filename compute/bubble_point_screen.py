"""Bubble-Point / Gas-Breakout Research Prototype (Order 10).

Pure compute module implementing the Standing-style bubble-point correlation
and pressure comparison logic for ESP intake-pressure screening.

This module intentionally does NOT contain:
- Streamlit imports
- database calls
- plotting
- session state

Usage
-----
1. Call ``calculate_standing_bubble_point_psi()`` with user-confirmed inputs.
2. OR accept a user-supplied ``bubble_point_psi`` directly.
3. Call ``compare_bubble_point_to_pressure()`` to evaluate margin vs pump intake.
4. Call ``build_bubble_point_diagnostic()`` for a complete structured payload.

Research-prototype caution
--------------------------
Standing's correlation requires PVT-style inputs.  Producing GOR, default gas
gravity, default temperature, and SG-derived API are proxies or estimates
unless confirmed by an engineer.  Do not interpret the output as a lab PVT
bubble-point result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import math

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid temperature range for candidate temperature selection (°F).
TEMPERATURE_VALID_MIN_F: float = 100.0
TEMPERATURE_VALID_MAX_F: float = 250.0

#: Default gas specific gravity suggestion.
DEFAULT_GAMMA_G: float = 0.75

#: Default temperature fallback suggestion when no valid candidate is found (°F).
DEFAULT_TEMPERATURE_F: float = 150.0

#: Default API suggestion derived from SG_OIL = 0.85 ≈ 35 API.
DEFAULT_API: float = 35.0

#: Margin band threshold (±) for "Near bubble point" classification.
NEAR_BUBBLE_POINT_MARGIN_PCT: float = 10.0

#: Preferred temperature summary keys, in order of preference.
TEMPERATURE_SUMMARY_KEYS: List[str] = [
    "pump_intake_temperature_f_0d_7d_avg",
    "pump_intake_temperature_f_7d_14d_avg",
    "pump_intake_temperature_f_14d_21d_avg",
    "pump_intake_temperature_f_21d_28d_avg",
    "pump_intake_temperature_f_1d_ago_avg",
    "pump_intake_temperature_f_1d_avg",
]

#: Preferred intake pressure summary keys, in order of preference.
INTAKE_PRESSURE_SUMMARY_KEYS: List[str] = [
    "pump_intake_pressure_psi_1d_avg",
    "pump_intake_pressure_psi_0d_7d_avg",
    "pump_intake_pressure_psi_7d_14d_avg",
    "pump_intake_pressure_psi",
]

#: Gas-unit conversion factors used for GOR proxy suggestions.
#: Keys commonly coming from allocation fields are assumed MSCF-like and are
#: converted to SCF before dividing by oil volume.
GOR_PROXY_GAS_TO_SCF_FACTOR: Dict[str, float] = {
    "alloc_gas_vol_1d": 1000.0,
    "alloc_gas_vol_1d_avg": 1000.0,
    "alloc_gas_vol": 1000.0,
    "gas_1d_avg": 1.0,
}


# ---------------------------------------------------------------------------
# A. API from SG_OIL
# ---------------------------------------------------------------------------


def calculate_api_from_sg_oil(sg_oil: Optional[float]) -> Optional[float]:
    """Convert oil specific gravity to API gravity.

    Formula::

        API = (141.5 / sg_oil) - 131.5

    Parameters
    ----------
    sg_oil:
        Oil specific gravity (dimensionless).  Must be > 0.

    Returns
    -------
    float or None
        API gravity in degrees.  Returns ``None`` when ``sg_oil`` is missing
        or non-positive (guarding against division by zero and invalid input).
    """
    if sg_oil is None:
        return None
    try:
        val = float(sg_oil)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val) or val <= 0.0:
        return None
    return (141.5 / val) - 131.5


# ---------------------------------------------------------------------------
# B. Standing correlation
# ---------------------------------------------------------------------------


def calculate_standing_bubble_point_psi(
    R_so: Optional[float],
    gamma_g: Optional[float],
    T_f: Optional[float],
    API: Optional[float],
) -> Dict[str, Any]:
    """Compute bubble-point pressure using the Standing correlation.

    Formula::

        C_pb = (R_so / gamma_g)^0.83 × 10^(0.00091 × T - 0.0125 × API)
        P_b  = 18.2 × (C_pb - 1.4)

    Parameters
    ----------
    R_so:
        Solution gas-oil ratio, SCF/STB.  Must be > 0.
    gamma_g:
        Gas specific gravity (dimensionless).  Must be > 0.
    T_f:
        Reservoir / pump-depth temperature, °F.  Must be finite.
    API:
        Oil API gravity (degrees).  Must be > 0.

    Returns
    -------
    dict
        ``available`` (bool), optional ``bubble_point_psi`` (float), and
        ``reason_unavailable`` (str) when ``available`` is False.
    """
    validation_errors: List[str] = []

    def _to_float(v, name: str) -> Optional[float]:
        if v is None:
            validation_errors.append(f"{name} is missing.")
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            validation_errors.append(f"{name} cannot be converted to a number.")
            return None
        if not math.isfinite(fv):
            validation_errors.append(f"{name} is not a finite number (got {v}).")
            return None
        return fv

    r_so_f = _to_float(R_so, "R_so")
    gamma_g_f = _to_float(gamma_g, "gamma_g")
    t_f_f = _to_float(T_f, "T_f")
    api_f = _to_float(API, "API")

    if r_so_f is not None and r_so_f <= 0.0:
        validation_errors.append(f"R_so must be > 0 (got {r_so_f}).")
        r_so_f = None

    if gamma_g_f is not None and gamma_g_f <= 0.0:
        validation_errors.append(f"gamma_g must be > 0 (got {gamma_g_f}).")
        gamma_g_f = None

    if api_f is not None and api_f <= 0.0:
        validation_errors.append(f"API must be > 0 (got {api_f}).")
        api_f = None

    if validation_errors:
        return {
            "available": False,
            "bubble_point_psi": None,
            "reason_unavailable": "Invalid inputs: " + "; ".join(validation_errors),
        }

    # Standing correlation
    try:
        exponent = 0.00091 * t_f_f - 0.0125 * api_f  # type: ignore[operator]
        c_pb = (r_so_f / gamma_g_f) ** 0.83 * (10.0**exponent)  # type: ignore[operator]
        pb = 18.2 * (c_pb - 1.4)
    except Exception as exc:
        return {
            "available": False,
            "bubble_point_psi": None,
            "reason_unavailable": f"Standing correlation computation failed: {exc}",
        }

    if not math.isfinite(pb):
        return {
            "available": False,
            "bubble_point_psi": None,
            "reason_unavailable": "Standing correlation produced a non-finite result.",
        }

    return {
        "available": True,
        "bubble_point_psi": float(pb),
        "reason_unavailable": None,
    }


# ---------------------------------------------------------------------------
# C. Pressure comparison
# ---------------------------------------------------------------------------


def compare_bubble_point_to_pressure(
    bubble_point_psi: Optional[float],
    pressure_reference_psi: Optional[float],
) -> Dict[str, Any]:
    """Compare bubble-point pressure against a reference pressure.

    Status bands
    ------------
    - margin_pct > +10%  → "Above bubble point"
    - -10% <= margin_pct <= +10% → "Near bubble point"
    - margin_pct < -10%  → "Below bubble point"

    Parameters
    ----------
    bubble_point_psi:
        Calculated or user-supplied bubble-point pressure (psi).
    pressure_reference_psi:
        Pressure to compare against (typically pump intake pressure, psi).

    Returns
    -------
    dict
        ``margin_psi``, ``margin_pct``, ``status_label``.
    """
    if bubble_point_psi is None or pressure_reference_psi is None:
        return {
            "margin_psi": None,
            "margin_pct": None,
            "status_label": "Unavailable",
        }

    try:
        pb = float(bubble_point_psi)
        pr = float(pressure_reference_psi)
    except (TypeError, ValueError):
        return {
            "margin_psi": None,
            "margin_pct": None,
            "status_label": "Unavailable",
        }

    if not math.isfinite(pb) or not math.isfinite(pr):
        return {
            "margin_psi": None,
            "margin_pct": None,
            "status_label": "Unavailable",
        }

    margin_psi = pr - pb

    if abs(pb) > 0.0:
        margin_pct = (margin_psi / pb) * 100.0
    else:
        return {
            "margin_psi": float(margin_psi),
            "margin_pct": None,
            "status_label": "Unavailable",
        }

    if margin_pct > NEAR_BUBBLE_POINT_MARGIN_PCT:
        status_label = "Above bubble point"
    elif margin_pct < -NEAR_BUBBLE_POINT_MARGIN_PCT:
        status_label = "Below bubble point"
    else:
        status_label = "Near bubble point"

    return {
        "margin_psi": float(margin_psi),
        "margin_pct": float(margin_pct),
        "status_label": status_label,
    }


# ---------------------------------------------------------------------------
# Helpers — input suggestion extraction
# ---------------------------------------------------------------------------


def _first_valid_temperature(summary_map: Dict[str, Any]) -> Optional[float]:
    """Return the first realistic temperature candidate from summary_map keys."""
    for key in TEMPERATURE_SUMMARY_KEYS:
        raw = summary_map.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val) and TEMPERATURE_VALID_MIN_F <= val <= TEMPERATURE_VALID_MAX_F:
            return val
    return None


def _first_valid_intake_pressure(summary_map: Dict[str, Any]) -> Optional[float]:
    """Return the first usable pump intake pressure from summary_map keys."""
    for key in INTAKE_PRESSURE_SUMMARY_KEYS:
        raw = summary_map.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val) and val > 0.0:
            return val
    return None


def suggest_inputs_from_summary(
    summary_map: Dict[str, Any],
    sg_oil: Optional[float] = None,
) -> Dict[str, Any]:
    """Derive input suggestions from the json_summary_data map and sg_oil.

    Parameters
    ----------
    summary_map:
        Parsed json_summary_data dict.
    sg_oil:
        Oil specific gravity from physical inputs panel (used to suggest API).

    Returns
    -------
    dict with suggestion keys:
        - ``temperature_candidate_f`` / ``temperature_candidate_source``
        - ``api_suggestion`` / ``api_suggestion_source``
        - ``gor_proxy_candidate`` / ``gor_proxy_available``
        - ``intake_pressure_psi`` / ``intake_pressure_source``
    """
    # Temperature
    temp_val = _first_valid_temperature(summary_map)
    temp_source = "valid candidate from summary stats" if temp_val is not None else "no valid candidate found"

    # API from sg_oil
    api_from_sg = calculate_api_from_sg_oil(sg_oil)
    if api_from_sg is not None:
        api_suggestion = api_from_sg
        api_source = f"calculated from sg_oil={sg_oil:.3f}"
    else:
        api_suggestion = DEFAULT_API
        api_source = "default context value (35 API)"

    # GOR proxy candidate — look for 1d production fields
    gor_proxy: Optional[float] = None
    gor_available = False
    gor_proxy_gas_unit_assumed: Optional[str] = None
    gor_proxy_conversion_factor_to_scf: Optional[float] = None
    for gas_key in [
        "alloc_gas_vol_1d",
        "alloc_gas_vol_1d_avg",
        "gas_1d_avg",
        "alloc_gas_vol",
    ]:
        for oil_key in ["alloc_oil_vol_1d", "alloc_oil_vol_1d_avg", "oil_1d_avg", "alloc_oil_vol"]:
            gas_raw = summary_map.get(gas_key)
            oil_raw = summary_map.get(oil_key)
            if gas_raw is None or oil_raw is None:
                continue
            try:
                gas_val = float(gas_raw)
                oil_val = float(oil_raw)
            except (TypeError, ValueError):
                continue
            if gas_val <= 0.0 or oil_val <= 0.0:
                continue
            gas_to_scf = GOR_PROXY_GAS_TO_SCF_FACTOR.get(gas_key, 1.0)
            gas_val_scf = gas_val * gas_to_scf
            gor_proxy = float(gas_val_scf / oil_val)
            gor_proxy_conversion_factor_to_scf = float(gas_to_scf)
            gor_proxy_gas_unit_assumed = "MSCF/day" if gas_to_scf == 1000.0 else "SCF/day"
            gor_available = True
            break
        if gor_available:
            break

    # Intake pressure
    pip = _first_valid_intake_pressure(summary_map)
    if pip is not None:
        intake_source = "from json_summary_data (1d avg preferred)"
    else:
        intake_source = "not found in summary"

    return {
        "temperature_candidate_f": temp_val,
        "temperature_candidate_source": temp_source,
        "api_suggestion": float(api_suggestion),
        "api_suggestion_source": api_source,
        "gor_proxy_candidate": gor_proxy,
        "gor_proxy_available": gor_available,
        "gor_proxy_gas_unit_assumed": gor_proxy_gas_unit_assumed,
        "gor_proxy_conversion_factor_to_scf": gor_proxy_conversion_factor_to_scf,
        "intake_pressure_psi": pip,
        "intake_pressure_source": intake_source,
    }


# ---------------------------------------------------------------------------
# D. Full diagnostic builder
# ---------------------------------------------------------------------------

FORMULA_TEXT = (
    "C_pb = (R_so / gamma_g)^0.83 × 10^(0.00091 × T − 0.0125 × API)\n"
    "P_b  = 18.2 × (C_pb − 1.4)\n\n"
    "Where:\n"
    "  R_so   = solution gas-oil ratio (SCF/STB) — user-confirmed\n"
    "  gamma_g = gas specific gravity (dimensionless) — user-confirmed\n"
    "  T      = temperature (°F) — user-confirmed\n"
    "  API    = oil API gravity (°) — derived from sg_oil or user-confirmed\n"
    "\n"
    "Primary comparison: P_b vs pump intake pressure (psi)"
)

CAUTION_TEXT_CORRELATION = (
    "This is a research-prototype bubble-point screen. Standing's correlation requires "
    "PVT-style inputs. Producing GOR, default gas gravity, default temperature, and "
    "SG-derived API are proxies or estimates unless confirmed by an engineer. "
    "Do not interpret this as a lab PVT bubble-point result."
)

CAUTION_TEXT_REGIONAL = (
    "Regional Permian bubble-point ranges should be used as background context only, "
    "not as hidden calculation defaults."
)


def build_bubble_point_diagnostic(
    *,
    # Pressure reference
    pressure_reference_psi: Optional[float] = None,
    pressure_reference_name: str = "pump_intake_pressure_psi",
    # User-supplied P_b (bypasses Standing calculation)
    user_supplied_bubble_point_psi: Optional[float] = None,
    # Standing inputs — must be user-confirmed
    R_so: Optional[float] = None,
    R_so_source: str = "not provided",
    gamma_g: Optional[float] = None,
    gamma_g_source: str = "not provided",
    T_f: Optional[float] = None,
    T_f_source: str = "not provided",
    API: Optional[float] = None,
    API_source: str = "not provided",
) -> Dict[str, Any]:
    """Build a complete bubble-point diagnostic payload.

    Supports two modes:
    1. **User-supplied P_b** — compare directly against the pressure reference.
    2. **Standing correlation** — compute P_b from user-confirmed R_so, gamma_g,
       T_f, API; then compare.

    Parameters
    ----------
    pressure_reference_psi:
        Pump intake pressure (primary V1 reference), psi.
    pressure_reference_name:
        Display name for the pressure reference.
    user_supplied_bubble_point_psi:
        If provided, skip Standing calculation and use this P_b directly.
    R_so, gamma_g, T_f, API:
        Standing correlation inputs — must be user-confirmed values.
    R_so_source, gamma_g_source, T_f_source, API_source:
        Source/trust label strings for each input.

    Returns
    -------
    dict with full diagnostic payload.
    """
    # ----- Check pressure reference -----
    pressure_reference_available = (
        pressure_reference_psi is not None
        and math.isfinite(float(pressure_reference_psi))
        and float(pressure_reference_psi) > 0.0
    ) if pressure_reference_psi is not None else False

    # ----- Determine mode -----
    if user_supplied_bubble_point_psi is not None:
        try:
            _ub = float(user_supplied_bubble_point_psi)
            user_mode_valid = math.isfinite(_ub) and _ub > 0.0
        except (TypeError, ValueError):
            user_mode_valid = False
    else:
        user_mode_valid = False

    standing_inputs_complete = (
        R_so is not None
        and gamma_g is not None
        and T_f is not None
        and API is not None
    )

    # ----- Evaluate availability -----
    if not pressure_reference_available:
        return _unavailable(
            reason="Pump intake pressure reference is missing or invalid.",
            pressure_reference_name=pressure_reference_name,
            pressure_reference_psi=pressure_reference_psi,
        )

    if not user_mode_valid and not standing_inputs_complete:
        return _unavailable(
            reason=(
                "Neither a user-supplied P_b nor a complete Standing input set "
                "(R_so, gamma_g, T_f, API) is available."
            ),
            pressure_reference_name=pressure_reference_name,
            pressure_reference_psi=pressure_reference_psi,
        )

    # ----- Mode: User-supplied P_b -----
    if user_mode_valid:
        selected_pb = float(user_supplied_bubble_point_psi)  # type: ignore[arg-type]
        mode = "User-supplied P_b"
        trust_label = "Manual input"
        calculated_pb = None
        input_table = [
            {"parameter": "bubble_point_psi", "value": selected_pb, "source": "user manual input", "trust": "Manual input"},
            {"parameter": pressure_reference_name, "value": float(pressure_reference_psi), "source": "summary data / recommendation context", "trust": "Direct"},  # type: ignore[arg-type]
        ]
        source_notes = [
            "Bubble-point pressure supplied directly by the user.",
            "Pump intake pressure is the primary V1 ESP suction reference.",
        ]

    # ----- Mode: Standing correlation -----
    else:
        standing_result = calculate_standing_bubble_point_psi(R_so, gamma_g, T_f, API)
        if not standing_result["available"]:
            return _unavailable(
                reason="Standing correlation failed: " + (standing_result.get("reason_unavailable") or "unknown error"),
                pressure_reference_name=pressure_reference_name,
                pressure_reference_psi=pressure_reference_psi,
            )

        calculated_pb = standing_result["bubble_point_psi"]
        selected_pb = float(calculated_pb)  # type: ignore[arg-type]
        mode = "Standing correlation"
        trust_label = "Research prototype"
        input_table = [
            {"parameter": "R_so (SCF/STB)", "value": R_so, "source": R_so_source, "trust": _trust_for_source(R_so_source)},
            {"parameter": "gamma_g", "value": gamma_g, "source": gamma_g_source, "trust": _trust_for_source(gamma_g_source)},
            {"parameter": "T_f (°F)", "value": T_f, "source": T_f_source, "trust": _trust_for_source(T_f_source)},
            {"parameter": "API (°)", "value": API, "source": API_source, "trust": _trust_for_source(API_source)},
            {"parameter": pressure_reference_name, "value": float(pressure_reference_psi), "source": "summary data / recommendation context", "trust": "Direct"},  # type: ignore[arg-type]
        ]
        source_notes = [
            "Bubble-point pressure computed using Standing's correlation.",
            "R_so source: " + R_so_source,
            "gamma_g source: " + gamma_g_source,
            "T_f source: " + T_f_source,
            "API source: " + API_source,
            "Pump intake pressure is the primary V1 ESP suction reference.",
        ]

    # ----- Comparison -----
    comparison = compare_bubble_point_to_pressure(
        bubble_point_psi=selected_pb,
        pressure_reference_psi=float(pressure_reference_psi),  # type: ignore[arg-type]
    )

    return {
        "available": True,
        "reason_unavailable": None,
        "mode": mode,
        "trust_label": trust_label,
        "formula_text": FORMULA_TEXT,
        "input_table": input_table,
        "calculated_bubble_point_psi": calculated_pb,
        "user_supplied_bubble_point_psi": float(user_supplied_bubble_point_psi) if user_mode_valid else None,
        "selected_bubble_point_psi": selected_pb,
        "pressure_reference_name": pressure_reference_name,
        "pressure_reference_psi": float(pressure_reference_psi),  # type: ignore[arg-type]
        "margin_psi": comparison["margin_psi"],
        "margin_pct": comparison["margin_pct"],
        "status_label": comparison["status_label"],
        "source_notes": source_notes,
        "caution_notes": [CAUTION_TEXT_CORRELATION, CAUTION_TEXT_REGIONAL],
    }


def _unavailable(
    *,
    reason: str,
    pressure_reference_name: str,
    pressure_reference_psi: Optional[float],
) -> Dict[str, Any]:
    """Return a standard unavailable diagnostic payload."""
    return {
        "available": False,
        "reason_unavailable": reason,
        "mode": "Unavailable",
        "trust_label": "Research prototype",
        "formula_text": FORMULA_TEXT,
        "input_table": [],
        "calculated_bubble_point_psi": None,
        "user_supplied_bubble_point_psi": None,
        "selected_bubble_point_psi": None,
        "pressure_reference_name": pressure_reference_name,
        "pressure_reference_psi": pressure_reference_psi,
        "margin_psi": None,
        "margin_pct": None,
        "status_label": "Unavailable",
        "source_notes": [],
        "caution_notes": [CAUTION_TEXT_CORRELATION, CAUTION_TEXT_REGIONAL],
    }


def _trust_for_source(source: str) -> str:
    """Map a source label to a trust category string."""
    src = source.lower()
    if "user" in src or "manual" in src or "confirmed" in src:
        return "Manual input"
    if "proxy" in src or "producing gor" in src or "default" in src or "context" in src or "estimated" in src:
        return "Proxy / estimated"
    if "calculated" in src or "derived" in src:
        return "Proxy / estimated"
    return "Proxy / estimated"
