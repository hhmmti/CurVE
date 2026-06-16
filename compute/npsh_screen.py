"""NPSH / Cavitation Screening Prototype (Order 11).

Pure compute module implementing ESP V1 Net Positive Suction Head Available
(NPSHa) calculation using pump intake pressure as the suction-pressure reference.

This module intentionally does NOT contain:
- Streamlit imports
- database calls
- plotting
- session state

V1 ESP equation
---------------
    PIP_abs_psi = pip_gauge_psi + 14.7
    NPSHa_ft    = ((PIP_abs_psi - vapor_pressure_psi_abs) × 2.31) / sg_mixture
    Margin_ft   = NPSHa_ft - NPSHr_ft

Intentionally NOT implemented
------------------------------
- Full suction-piping NPSH (surface static head, pipe friction losses)
- Validated NPSH margin without pump-specific NPSHr
- Automatic vapor pressure from temperature/PVT
- IPR / Vogel / nodal analysis

Usage
-----
1. Call ``calculate_npsha_ft()`` with PIP gauge, vapor pressure, and SG.
2. Call ``calculate_npsh_margin_ft()`` with NPSHa and NPSHr.
3. Call ``classify_npsh_margin()`` to get a status label string.
4. Call ``build_npsh_diagnostic()`` for a complete structured payload.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Atmospheric pressure used to convert gauge to absolute psi.
ATMOSPHERIC_PSI: float = 14.7

#: Head/pressure conversion: psi × 2.31 = ft of water at SG=1 (ft/psi).
HEAD_FT_PER_PSI: float = 2.31

#: Default placeholder NPSHr suggestion when pump-specific value is unknown.
DEFAULT_NPSHR_FT: float = 7.5

#: Margin threshold above which status is "Safe" (ft).
SAFE_MARGIN_FT: float = 5.0

FORMULA_TEXT: str = (
    "PIP_abs_psi = pip_gauge_psi + 14.7\n"
    "NPSHa_ft    = ((PIP_abs_psi - vapor_pressure_psi_abs) × 2.31) / sg_mixture\n"
    "Margin_ft   = NPSHa_ft - NPSHr_ft\n\n"
    "Where:\n"
    "  pip_gauge_psi          = pump intake pressure (gauge psi) — from telemetry/summary\n"
    "  vapor_pressure_psi_abs = absolute vapor pressure of fluid (psia) — user input preferred\n"
    "  sg_mixture             = mixture specific gravity (dimensionless) — from well data/user\n"
    "  NPSHr_ft               = NPSH required by pump (ft) — pump-specific; user input preferred\n"
    "\n"
    "V1 note: pump intake pressure is used directly as the suction-pressure reference.\n"
    "This is NOT full suction-piping NPSH (no surface static head or friction terms)."
)

CAUTION_TEXT: str = (
    "This is an NPSH/cavitation screening prototype. It uses pump intake pressure as the "
    "suction-pressure reference. Vapor pressure and NPSHr must be user-confirmed or treated "
    "as proxy/default assumptions. Do not interpret this as a manufacturer-validated NPSH "
    "margin unless pump-specific NPSHr and fluid vapor pressure are known."
)


# ---------------------------------------------------------------------------
# Internal validation helper
# ---------------------------------------------------------------------------


def _validate_positive_float(
    value: Any,
    name: str,
    errors: List[str],
    allow_zero: bool = False,
) -> Optional[float]:
    """Validate and coerce a value to a finite float.

    Appends to *errors* when validation fails and returns None.
    """
    if value is None:
        errors.append(f"{name} is missing.")
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} cannot be converted to a number.")
        return None
    if not math.isfinite(fv):
        errors.append(f"{name} is not a finite number (got {value}).")
        return None
    if allow_zero and fv < 0.0:
        errors.append(f"{name} must be >= 0 (got {fv}).")
        return None
    if not allow_zero and fv <= 0.0:
        errors.append(f"{name} must be > 0 (got {fv}).")
        return None
    return fv


# ---------------------------------------------------------------------------
# A. NPSHa calculation
# ---------------------------------------------------------------------------


def calculate_npsha_ft(
    pip_gauge_psi: Optional[float],
    vapor_pressure_psi_abs: Optional[float],
    sg_mixture: Optional[float],
) -> Dict[str, Any]:
    """Compute NPSHa in feet using pump intake pressure as suction reference.

    Formula::

        PIP_abs_psi = pip_gauge_psi + 14.7
        NPSHa_ft    = ((PIP_abs_psi - vapor_pressure_psi_abs) * 2.31) / sg_mixture

    Parameters
    ----------
    pip_gauge_psi:
        Pump intake pressure, gauge psi. Must be >= 0 and finite.
    vapor_pressure_psi_abs:
        Absolute vapor pressure of the fluid, psia. Must be > 0 and finite.
    sg_mixture:
        Mixture specific gravity, dimensionless. Must be > 0 and finite.

    Returns
    -------
    dict with keys:
        available (bool), npsha_ft (float|None), pip_abs_psi (float|None),
        reason_unavailable (str|None).
    """
    errors: List[str] = []

    pip_f = _validate_positive_float(pip_gauge_psi, "pip_gauge_psi", errors, allow_zero=True)
    vp_f = _validate_positive_float(vapor_pressure_psi_abs, "vapor_pressure_psi_abs", errors, allow_zero=False)
    sg_f = _validate_positive_float(sg_mixture, "sg_mixture", errors, allow_zero=False)

    if errors:
        return {
            "available": False,
            "npsha_ft": None,
            "pip_abs_psi": None,
            "reason_unavailable": "Invalid inputs: " + "; ".join(errors),
        }

    pip_abs = pip_f + ATMOSPHERIC_PSI  # type: ignore[operator]
    try:
        npsha = ((pip_abs - vp_f) * HEAD_FT_PER_PSI) / sg_f  # type: ignore[operator]
    except Exception as exc:
        return {
            "available": False,
            "npsha_ft": None,
            "pip_abs_psi": float(pip_abs),
            "reason_unavailable": f"NPSHa computation failed: {exc}",
        }

    if not math.isfinite(npsha):
        return {
            "available": False,
            "npsha_ft": None,
            "pip_abs_psi": float(pip_abs),
            "reason_unavailable": "NPSHa computation produced a non-finite result.",
        }

    return {
        "available": True,
        "npsha_ft": float(npsha),
        "pip_abs_psi": float(pip_abs),
        "reason_unavailable": None,
    }


# ---------------------------------------------------------------------------
# B. NPSH margin
# ---------------------------------------------------------------------------


def calculate_npsh_margin_ft(
    npsha_ft: Optional[float],
    npshr_ft: Optional[float],
) -> Dict[str, Any]:
    """Compute NPSH margin in feet.

    Margin_ft = NPSHa_ft - NPSHr_ft

    Parameters
    ----------
    npsha_ft:
        Net positive suction head available (ft). Must be finite.
    npshr_ft:
        Net positive suction head required by the pump (ft). Must be >= 0 and finite.

    Returns
    -------
    dict with keys: available (bool), margin_ft (float|None), reason_unavailable (str|None).
    """
    errors: List[str] = []

    a_f: Optional[float] = None
    r_f: Optional[float] = None

    if npsha_ft is None:
        errors.append("npsha_ft is missing.")
    else:
        try:
            a_f = float(npsha_ft)
            if not math.isfinite(a_f):
                errors.append(f"npsha_ft is not finite (got {npsha_ft}).")
                a_f = None
        except (TypeError, ValueError):
            errors.append(f"npsha_ft cannot be converted to a number.")

    if npshr_ft is None:
        errors.append("npshr_ft is missing.")
    else:
        try:
            r_f = float(npshr_ft)
            if not math.isfinite(r_f):
                errors.append(f"npshr_ft is not finite (got {npshr_ft}).")
                r_f = None
            elif r_f < 0.0:
                errors.append(f"npshr_ft must be >= 0 (got {r_f}).")
                r_f = None
        except (TypeError, ValueError):
            errors.append(f"npshr_ft cannot be converted to a number.")

    if errors:
        return {
            "available": False,
            "margin_ft": None,
            "reason_unavailable": "Invalid inputs: " + "; ".join(errors),
        }

    margin = a_f - r_f  # type: ignore[operator]
    return {
        "available": True,
        "margin_ft": float(margin),
        "reason_unavailable": None,
    }


# ---------------------------------------------------------------------------
# C. Status classification
# ---------------------------------------------------------------------------


def classify_npsh_margin(margin_ft: Optional[float]) -> str:
    """Classify NPSH margin into a status label.

    Status bands
    ------------
    - Safe:        margin_ft >= 5 ft
    - Watch:       0 <= margin_ft < 5 ft
    - Risk:        margin_ft < 0 ft
    - Unavailable: margin_ft is None or non-finite

    Parameters
    ----------
    margin_ft:
        NPSH margin in feet.

    Returns
    -------
    str status label.
    """
    if margin_ft is None:
        return "Unavailable"
    try:
        m = float(margin_ft)
    except (TypeError, ValueError):
        return "Unavailable"
    if not math.isfinite(m):
        return "Unavailable"
    if m >= SAFE_MARGIN_FT:
        return "Safe"
    if m >= 0.0:
        return "Watch"
    return "Risk"


# ---------------------------------------------------------------------------
# D. Full diagnostic builder
# ---------------------------------------------------------------------------


def build_npsh_diagnostic(
    *,
    pip_gauge_psi: Optional[float] = None,
    pip_source: str = "not provided",
    vapor_pressure_psi_abs: Optional[float] = None,
    vapor_pressure_source: str = "not provided",
    vapor_pressure_is_proxy: bool = False,
    sg_mixture: Optional[float] = None,
    sg_source: str = "not provided",
    npshr_ft: Optional[float] = None,
    npshr_source: str = "not provided",
    npshr_is_placeholder: bool = False,
) -> Dict[str, Any]:
    """Build a complete NPSH/cavitation screening diagnostic payload.

    Parameters
    ----------
    pip_gauge_psi:
        Pump intake pressure in gauge psi. Primary V1 suction-pressure reference.
    pip_source:
        Source/trust label for the PIP value.
    vapor_pressure_psi_abs:
        Absolute vapor pressure (psia). User input preferred. Bubble-point P_b
        may be used as a conservative proxy only if user-confirmed.
    vapor_pressure_source:
        Source/trust label for the vapor pressure value.
    vapor_pressure_is_proxy:
        True when vapor pressure comes from bubble-point proxy or another
        approximation; triggers "Proxy / Research prototype" trust label.
    sg_mixture:
        Mixture specific gravity. Use current estimated SG if available.
    sg_source:
        Source/trust label for the SG value.
    npshr_ft:
        NPSH required by the pump (ft). User input preferred. A placeholder
        default of DEFAULT_NPSHR_FT may be used only if user-confirmed.
    npshr_source:
        Source/trust label for the NPSHr value.
    npshr_is_placeholder:
        True when npshr_ft is a generic placeholder default; triggers
        "Proxy / Research prototype" trust label.

    Returns
    -------
    dict with full NPSH diagnostic payload.
    """
    # ----- Determine trust label -----
    if vapor_pressure_is_proxy or npshr_is_placeholder:
        trust_label = "Proxy / Research prototype"
        mode = "Proxy / Research prototype"
    else:
        trust_label = "Manual / Estimated"
        mode = "Manual / Estimated"

    # ----- Attempt NPSHa calculation -----
    npsha_result = calculate_npsha_ft(pip_gauge_psi, vapor_pressure_psi_abs, sg_mixture)

    if not npsha_result["available"]:
        return _unavailable_diagnostic(
            reason=npsha_result["reason_unavailable"] or "NPSHa inputs unavailable.",
            pip_gauge_psi=pip_gauge_psi,
            vapor_pressure_psi_abs=vapor_pressure_psi_abs,
            sg_mixture=sg_mixture,
            npshr_ft=npshr_ft,
            trust_label=trust_label,
            pip_source=pip_source,
            vapor_pressure_source=vapor_pressure_source,
            sg_source=sg_source,
            npshr_source=npshr_source,
        )

    npsha = npsha_result["npsha_ft"]
    pip_abs = npsha_result["pip_abs_psi"]

    # ----- Attempt margin calculation -----
    margin_result = calculate_npsh_margin_ft(npsha, npshr_ft)

    if not margin_result["available"]:
        return _unavailable_diagnostic(
            reason=margin_result["reason_unavailable"] or "NPSHr input unavailable.",
            pip_gauge_psi=pip_gauge_psi,
            vapor_pressure_psi_abs=vapor_pressure_psi_abs,
            sg_mixture=sg_mixture,
            npshr_ft=npshr_ft,
            trust_label=trust_label,
            pip_source=pip_source,
            vapor_pressure_source=vapor_pressure_source,
            sg_source=sg_source,
            npshr_source=npshr_source,
        )

    margin = margin_result["margin_ft"]
    status_label = classify_npsh_margin(margin)

    # ----- Trust labels per input -----
    vp_trust = "Proxy / estimated" if vapor_pressure_is_proxy else "Manual input"
    npshr_trust = "Placeholder / proxy" if npshr_is_placeholder else "Manual input"

    input_source_table = [
        {
            "parameter": "pip_gauge_psi",
            "value": pip_gauge_psi,
            "source": pip_source,
            "trust": "Direct / summary data",
        },
        {
            "parameter": "vapor_pressure_psi_abs",
            "value": vapor_pressure_psi_abs,
            "source": vapor_pressure_source,
            "trust": vp_trust,
        },
        {
            "parameter": "sg_mixture",
            "value": sg_mixture,
            "source": sg_source,
            "trust": "Direct / estimated",
        },
        {
            "parameter": "npshr_ft",
            "value": npshr_ft,
            "source": npshr_source,
            "trust": npshr_trust,
        },
    ]

    return {
        "available": True,
        "reason_unavailable": None,
        "mode": mode,
        "trust_label": trust_label,
        "pip_gauge_psi": pip_gauge_psi,
        "pip_abs_psi": pip_abs,
        "vapor_pressure_psi_abs": vapor_pressure_psi_abs,
        "sg_mixture": sg_mixture,
        "npshr_ft": npshr_ft,
        "npsha_ft": npsha,
        "margin_ft": margin,
        "status_label": status_label,
        "input_source_table": input_source_table,
        "caution_notes": [CAUTION_TEXT],
        "formula_text": FORMULA_TEXT,
    }


def _unavailable_diagnostic(
    *,
    reason: str,
    pip_gauge_psi: Optional[float],
    vapor_pressure_psi_abs: Optional[float],
    sg_mixture: Optional[float],
    npshr_ft: Optional[float],
    trust_label: str,
    pip_source: str,
    vapor_pressure_source: str,
    sg_source: str,
    npshr_source: str,
) -> Dict[str, Any]:
    """Return a standard unavailable NPSH diagnostic payload."""
    return {
        "available": False,
        "reason_unavailable": reason,
        "mode": "Unavailable",
        "trust_label": trust_label,
        "pip_gauge_psi": pip_gauge_psi,
        "pip_abs_psi": None,
        "vapor_pressure_psi_abs": vapor_pressure_psi_abs,
        "sg_mixture": sg_mixture,
        "npshr_ft": npshr_ft,
        "npsha_ft": None,
        "margin_ft": None,
        "status_label": "Unavailable",
        "input_source_table": [
            {"parameter": "pip_gauge_psi", "value": pip_gauge_psi, "source": pip_source, "trust": "Direct / summary data"},
            {"parameter": "vapor_pressure_psi_abs", "value": vapor_pressure_psi_abs, "source": vapor_pressure_source, "trust": "N/A"},
            {"parameter": "sg_mixture", "value": sg_mixture, "source": sg_source, "trust": "N/A"},
            {"parameter": "npshr_ft", "value": npshr_ft, "source": npshr_source, "trust": "N/A"},
        ],
        "caution_notes": [CAUTION_TEXT],
        "formula_text": FORMULA_TEXT,
    }
